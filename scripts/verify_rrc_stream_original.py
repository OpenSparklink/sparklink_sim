"""Join completed numeric streams with explicitly limited fresh-quiet proofs.

The checkpoint is reused only when every nonquiet input and receiver source SHA
matches. Quiet internal states are never inferred. Real IQ also runs the actual
MAC streaming adapter independently. No output chooses an RX phase or CRC seed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import rx_stream_frozen_replay as frozen
from audit_capture_hardware import sha, write
from profile_rx_stream import number
from replay_acquisition_regression import checked_manifest, closed_gate_certificate

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def quiet_output_proof(x, q):
    """Only no-candidate output of a fresh whole stream; no numeric state claim."""
    proof = closed_gate_certificate(x, q, 4)
    if proof is not None:
        proof.pop("rrc_component_upper_bound")  # That bound belongs to floating RRC.
        proof.update(
            rrc_state_verified=False,
            tone_state_verified=False,
            range_verified=False,
            scope="raw-energy gate only; tone may activate but cannot start search",
            fresh_whole_stream_only=True,
        )
    return proof


def receive_mac(x, q, profile):
    rx = MacStreamReceiver(
        profile,
        q,
        quiet_batching=False,
        defer_quiet_fir=False,
        rrc_backend="word-full12-c18",
        correlation_backend="scaled",
        correlation_mantissa_bits=10,
        tone_notch=ToneNotchConfig(radius=0.96875),
    )
    events, packets = [], []
    for start in range(0, len(x), 4096):
        result = rx.feed(x[start : start + 4096])
        events += result.diagnostics["events"]
        packets += result.packets
    result = rx.finish()
    events += result.diagnostics["events"]
    packets += result.packets
    return (
        events,
        packets,
        dict(
            stats=rx.phy.capture_stats,
            rrc_stats=dict(rx.phy._rrc_word_model.stats),
            fir_array_bytes=rx.phy.fir_array_bytes,
        ),
    )


def run(args):
    reference_file = checked_manifest(args.reference, "results.jsonl")
    reference = read_rows(reference_file)
    checkpoint_file = args.word_partial / "results.jsonl"
    checkpoint = read_rows(checkpoint_file)  # Partial/truncated JSON lines are errors, not ignored.
    plan = json.loads((args.word_partial / "frozen-plan.json").read_text())
    assert sha(args.word_partial / "driver.executed.py") == plan["driver_sha256"]
    assert all(sha(Path(f)) == h for f, h in plan["sources"].items()), "Receiver source changed"
    a = {r["id"]: r for r in reference if r["input"]["source"] != "quiet"}
    b = {r["id"]: r for r in checkpoint}
    assert len(b) == len(checkpoint) and a.keys() == b.keys(), "Need every unique nonquiet stream"
    for key, old in a.items():
        new = b[key]
        assert old["input"] == new["input"], "Frozen input/expected data changed"
        assert new["cost"]["rrc_state_observed"] and new["cost"]["certificate"] is None
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence").mkdir()
    write(
        args.output / "plan.json",
        dict(
            reference_sha256=sha(reference_file),
            checkpoint_sha256=sha(checkpoint_file),
            checkpoint_plan_sha256=sha(args.word_partial / "frozen-plan.json"),
            receiver_sources=plan["sources"],
            driver_sha256=sha(Path(__file__)),
            checkpoint_complete_nonquiet_only=True,
            quiet_states="unverified unless actually replayed",
            rrc_numeric_gate_passed=False,
            receiver_promotion=False,
        ),
    )
    original = args.evidence_root / "rx-timing-tracking/final"
    original_plan = json.loads(checked_manifest(original, "frozen-plan.json").read_text())
    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=number(original_plan["sync_word"]),
        data_crc_seed=number(original_plan["crc_seed"]),
        pilot_interval=0,
    )
    old_quiet = {r["sample_id"]: r for r in original_plan["quiet_inputs"]}
    quiet = {
        r["input"]["input"]["sample_id"]: r["input"]["input"]
        for r in reference
        if r["input"]["source"] == "quiet"
    }
    archived = args.evidence_root / "rx-chain-hardening/parser-replay"
    results = []
    for row in reference:
        meta = row["input"]
        if meta["source"] not in ("quiet", "real"):
            continue
        x, q = frozen.load_input(
            meta, archived, original, args.dataset, args.quiet_dataset, old_quiet, quiet
        )
        proof = quiet_output_proof(x, q) if meta["source"] == "quiet" else None
        record = dict(
            id=row["id"],
            input=meta,
            source_samples=len(x),
            rrc_charged_source_and_eof_samples=len(x) + 42,
            continuous_rrc_real_products=82 * (len(x) + 42),
            proof=proof,
        )
        if proof is None:
            events, packets, cost = receive_mac(x, q, profile)
            frames = frozen.score(meta, events)
            arrays = {
                f"{i}-{k}": e[k]
                for i, e in enumerate(events)
                for k in [
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                ]
            }
            np.savez_compressed(args.output / "evidence" / (row["id"] + ".npz"), **arrays)
            record.update(
                frames=frames,
                delivered_payloads=[p.mac_payload.hex() for p in packets],
                actual_state_observed=True,
                cost=cost,
            )
            assert len(packets) == sum(f["status"] == "valid" for f in frames)
        else:
            record.update(
                frames=[],
                delivered_payloads=[],
                actual_state_observed=False,
                cost=None,
                internal_numerical_acceptance=None,
            )
        results.append(record)
        with (args.output / "results.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if len(results) % 16 == 0:
            print(f"checked {len(results)} quiet/real streams", flush=True)
    real = [r for r in results if r["input"]["source"] == "real"]
    controls = [r for r in results if r["input"]["source"] == "quiet"]
    losses = [
        dict(id=k, positions=r["lost_success_positions"])
        for k, r in b.items()
        if r["lost_success_positions"]
    ]
    quiet_false = [r["id"] for r in controls if r["frames"] or r["delivered_payloads"]]
    real_losses = []
    for r in real:
        old = a[r["id"]]
        if sum(f["whole_packet_ok"] for f in r["frames"]) != old["correct"]:
            real_losses.append(r["id"])
    summary = dict(
        reference_inputs=len(reference),
        numeric_nonquiet_inputs=len(b),
        numeric_nonquiet_sources=dict(Counter(r["input"]["source"] for r in b.values())),
        numeric_nonquiet_samples=sum(
            r["cost"]["continuous_work_charged_samples"] for r in b.values()
        ),
        nonquiet_correct=sum(r["correct"] for r in b.values()),
        nonquiet_captures=sum(r["captures"] for r in b.values()),
        original_success_losses=losses,
        matched_center_delta_max=max(r["max_matched_center_delta"] for r in b.values()),
        quiet_inputs=len(controls),
        quiet_output_certificates=sum(r["proof"] is not None for r in controls),
        quiet_actual_state_replays=sum(r["actual_state_observed"] for r in controls),
        quiet_false_candidates=quiet_false,
        quiet_internal_numeric_gate=None,
        quiet_charged_samples=sum(r["source_samples"] for r in controls),
        actual_real_mac_inputs=len(real),
        actual_real_mac_packets=sum(len(r["delivered_payloads"]) for r in real),
        actual_real_correct=sum(f["whole_packet_ok"] for r in real for f in r["frames"]),
        actual_real_regressions=real_losses,
        rrc_numeric_gate_passed=False,
        receiver_promotion=False,
        whole_rx_16msps=None,
        scope="no packet rate denominator for real IQ with unknown transmitted count",
    )
    write(args.output / "summary.json", summary)
    write(
        args.output / "failures.json",
        [
            r
            for r in results
            if any(
                f["status"] != "valid" or not f.get("whole_packet_ok", False) for f in r["frames"]
            )
        ],
    )
    assert all(sha(Path(f)) == h for f, h in plan["sources"].items())
    write(
        args.output / "manifest.json",
        {str(f.relative_to(args.output)): sha(f) for f in args.output.rglob("*") if f.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return 1  # Known frozen local RRC failure is never overridden by output equivalence.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in [
        "reference",
        "word-partial",
        "evidence-root",
        "dataset",
        "quiet-dataset",
        "output",
    ]:
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
