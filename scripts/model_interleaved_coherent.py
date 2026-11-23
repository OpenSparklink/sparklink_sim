"""Prospective four-arm RF/word/cycle comparison,not a complete receiver pass."""

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts import model_finite_coherent as previous
from scripts import model_retired_capture as reference
from scripts.model_captured_frame import safe, write

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.coherent_interleaved import ScheduledCoherentCapturedFrameReplay
from nearlink_sdr.sim.coherent_word_flow import CoherentWordService

CANDIDATES = ("serial-signed22x18", "interleaved-signed22x18")
POLICY = dict(
    version=2,
    development_specs=[
        [2, 8, 0, [89, 149], 32, "clean"],
        [3, 12, 16, [43, 239], 24, "tone"],
        [4, 8, 16, [101, 29], 12, "wrong-sync"],
        [2, 12, 8, [271, 17], 16, "noise"],
        [3, 8, 0, [151, 67], 4, "step"],
        [4, 12, 4, [37, 223], 8, "burst"],
    ],
    validation_specs=[
        [4, 12, 16, [181, 23], 12, "clean"],
        [2, 8, 4, [137, 11], 32, "tone"],
        [3, 12, 0, [79, 293], 28, "wrong-sync"],
        [4, 8, 16, [31, 127], 8, "noise"],
        [2, 12, 8, [59, 257], 24, "step"],
        [3, 8, 8, [47, 157], 16, "burst"],
    ],
    seed_bias={"development": 5701021, "validation": 6701063},
    payload="bit-reversed Gray counter,sparse moving bits/block masks,Philox bytes",
    gate_control_cap_bits=8192,
    minimum_node_edge_reduction=.2,
    minimum_word_pairs=4,
    minimum_float_complete=1,
    original_float_success_loss=0,
    additional_negative_confirmations=0,
    false_MAC_and_duplicates=0,
    aggregate_resource_exit_delta_max=0,
    relative_start_samples_budget=.125,
    trajectory_samples_budget=.25,
    word_and_same_selected_symbol_identity="exact",
    selection="qualify all constraints;minimize ALU count,then register bits",
    validation="only frozen selected structure;failure rejects;no alternative held selection",
    new_IQ_RAM=0,
    history=4096,
    owned=3,
    source_ports=2,
    DSP25x18=1,
    MAC_gap_conformance_not_claimed=True,
    old_numeric_and16KiB_failures_retained=True,
    full_RX_16MSps_synthesis_or_default=False,
)


def payload(n, ident, held):
    if ident % 3 == 0:
        return bytes(int(f"{((k ^ (k >> 1)) + ident * 29 + 91 * held) & 255:08b}"[::-1], 2)
                     for k in range(n))
    if ident % 3 == 1:
        return bytes(((1 << ((3 * k + ident) % 8)) ^ ((k // 9) * 17) ^ (67 * held)) & 255
                     for k in range(n))
    return np.random.Generator(np.random.Philox(790447 + ident * 117 + held * 1291)).integers(
        0, 256, n, dtype=np.uint8,
    ).tobytes()


def validate_recipes():
    """Validate both prospective splits before creating files or reading IQ."""
    for split in ("development", "validation"):
        specs = POLICY[split + "_specs"]
        if len(specs) != 6:
            raise ValueError("Need six explicit ordinary profile recipes")
        for ft, mcs, pilots, lengths, _, _ in specs:
            for length in lengths:
                cfg = reference.shared.shared.configuration(ft, mcs, pilots, length)
                PSKReceiveProfile.ordinary(cfg)


class _CheckedOwner(previous._ObserveConfirmation, ScheduledCoherentCapturedFrameReplay):
    def _terminal(self, event):
        receipt = self.word_receipt
        event["word_reference_check"] = None
        if receipt is not None:
            # Diagnostic only:never returns an answer to the actual receiver.
            oracle = CoherentWordService(receipt.selection, receipt.phase_codes)
            while not oracle.ready:
                oracle.step()
            expected = oracle.take()
            a = {k: v for k, v in vars(receipt).items() if k != "available_cycle"}
            b = {k: v for k, v in vars(expected).items() if k != "available_cycle"}
            assert a == b
            event["word_reference_check"] = dict(
                integer_receipt_exact=True, diagnostic_reference_only=True,
                original_edges=oracle.ready_cycle - oracle.started,
                actual_edges=self.word_service.ready_cycle - self.word_service.started,
                N=len(receipt.phase_codes),
            )
        return super()._terminal(event)


def compare(out, root, row, candidates):
    path = root / row["file"]
    if path.parent != root or reference.shared.sha(path) != row["SHA256"]:
        raise ValueError("Frozen IQ path/hash changed")
    profile = replace(PSKReceiveProfile.ordinary(TxConfig(**row["config"])),
                      head_backend=row["head_backend"], allowed_mcs=(row["config"]["mcs_index"],))
    with np.load(path, allow_pickle=False) as values:
        iq, quiet = values["iq"], values["quiet"]
    modes = dict(floating=previous._FloatOwner,
                 original=partial(previous._WordOwner, clocked=True))
    modes |= {c: partial(_CheckedOwner, structure=c) for c in candidates}
    runs, summaries, traces = {}, {}, {}
    for label, owner in modes.items():
        with patch("tests.captured_frame_reference.CapturedFrameReplay", owner):
            runs[label] = reference.run_arm(profile, iq, quiet, True)
        summaries[label], traces[label] = previous._metrics(
            out, row, label, runs[label], len(iq), profile.allowed_mcs,
        )
    base = runs["floating"]
    details = {}
    for label in candidates:
        run = runs[label]
        shared = base[3].keys() & run[3].keys()
        common_integer = all(base[3][k] == run[3][k] and base[4][k] == run[4][k] for k in shared)
        source_exact = all(run[0][0][k] == base[0][0][k] for k in (
            "requests", "input_samples", "live_source_statistics", "live_coarse_statistics",
        ))
        start, trajectory, identical, noncorresponding = [], [], True, []
        for k in traces["floating"].keys() & traces[label].keys():
            a, b = traces["floating"][k], traces[label][k]
            start.append(abs(a["start"] - b["start"]))
            trajectory.append(float(np.max(abs(a["centers"] - b["centers"])))
                              if a["centers"].shape == b["centers"].shape else float("inf"))
        original_events = {e["selection_id"]: e for e, _ in base[0][1]}
        for event, _ in run[0][1]:
            old = original_events.get(event["selection_id"])
            if old is None:
                continue
            arrays = {k for k, v in old.items() if isinstance(v, np.ndarray)}
            if (old["frame"]["status"] != event["frame"]["status"]
                    or arrays != {k for k, v in event.items() if isinstance(v, np.ndarray)}
                    or any(old[k].shape != event[k].shape for k in arrays)):
                noncorresponding.append(dict(
                    selection=event["selection_id"], baseline=old["frame"]["status"],
                    candidate=event["frame"]["status"],
                    reason="different resource/EOF/field consumption,not a paired symbol transaction",
                ))
                continue
            identical &= all(old[k].dtype == event[k].dtype and old[k].shape == event[k].shape
                             and old[k].tobytes() == event[k].tobytes() for k in arrays)
        word_pairs = [e["word_reference_check"] for e, _ in run[0][1]
                      if e.get("word_reference_check") is not None]
        cost = run[0][0]["frame_replay_cost"]
        details[label] = dict(
            word_pairs=word_pairs, exact_common_fine_values=bool(common_integer),
            source_values_and_requests_exact=bool(source_exact),
            same_selected_symbol_arrays_exact=bool(identical),
            noncorresponding_symbol_transactions=noncorresponding,
            max_start_delta_samples=max(start, default=0),
            max_trajectory_delta_samples=max(trajectory, default=0),
            original_float_success_losses=sorted(set(summaries["floating"]["complete"])
                                                - set(summaries[label]["complete"])),
            recovered_vs_float=sorted(set(summaries[label]["complete"])
                                      - set(summaries["floating"]["complete"])),
            recovered_vs_original_serial=sorted(set(summaries[label]["complete"])
                                                - set(summaries["original"]["complete"])),
            lost_vs_original_serial=sorted(set(summaries["original"]["complete"])
                                           - set(summaries[label]["complete"])),
            resource_exit_deltas={k: run[0][0][k] - base[0][0][k]
                                  for k in ("queue_drops", "history_drops", "frame_resource_drops")},
            gate_control_bits=cost["word_gate_control_node_bits"],
            frame_control_bits=cost["combined_owner_register_bits"],
            ALU32=cost["coherent_word_service"]["ALU32"],
            DSP25x18=cost["coherent_word_service"]["DSP25x18"],
        )
    return dict(id=row["id"], negative=row["negative"], input_SHA256=row["SHA256"],
                samples=len(iq), expected_count=len(row["expected"]),
                modes=summaries, candidates=details,
                three_chunkings_exact=True, old_failures_retained=True, full_RX_gate=False)


def qualify(rows, label):
    if len(rows) != 10 or {r["id"] for r in rows} != set(range(10)):
        return False
    word_pairs = [p for r in rows for p in r["candidates"][label]["word_pairs"]]
    if len(word_pairs) < POLICY["minimum_word_pairs"]:
        return False
    if sum(len(r["modes"]["floating"]["complete"]) for r in rows) < POLICY["minimum_float_complete"]:
        return False
    if any(not p["integer_receipt_exact"] or p["actual_edges"] > (
        (1 - POLICY["minimum_node_edge_reduction"]) * p["original_edges"]
    )
           for p in word_pairs):
        return False
    for row in rows:
        d, v = row["candidates"][label], row["modes"][label]
        if d["original_float_success_losses"] or d["lost_vs_original_serial"]:
            return False
        if d["gate_control_bits"] > POLICY["gate_control_cap_bits"] or d["DSP25x18"] != 1:
            return False
        if not all(d[k] for k in ("exact_common_fine_values", "source_values_and_requests_exact",
                                  "same_selected_symbol_arrays_exact")):
            return False
        if (d["max_start_delta_samples"] > POLICY["relative_start_samples_budget"]
                or d["max_trajectory_delta_samples"] > POLICY["trajectory_samples_budget"]):
            return False
        if v["false_MAC"] or v["duplicate_MAC"]:
            return False
        if row["negative"] and v["coherent"] > row["modes"]["floating"]["coherent"]:
            return False
        if not v["all_pulses_accounted"] or v["max_ports"] > 2 or v["max_owned"] > 3:
            return False
    return all(sum(r["candidates"][label]["resource_exit_deltas"][k] for r in rows) <= 0
               for k in ("queue_drops", "history_drops", "frame_resource_drops"))


def select(rows, candidates):
    qualified = [c for c in candidates if qualify(rows, c)]
    return min(qualified, key=lambda c: (rows[0]["candidates"][c]["ALU32"],
                                        rows[0]["candidates"][c]["gate_control_bits"])) if qualified else None


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    source = reference.shared.sources()
    if args.split == "validation":
        if args.freeze is None:
            parser.error("Need prospective selected development freeze")
        frozen = json.loads(args.freeze.read_text())
        if (frozen["source"] != source or frozen["policy"] != POLICY
                or frozen["selected"] not in CANDIDATES):
            parser.error("Source/policy changed or development rejected;no new held inputs")
        candidates = (frozen["selected"],)
    else:
        if args.freeze is not None:
            parser.error("Development cannot import held selection")
        candidates = CANDIDATES
    validate_recipes()
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "pre-run-policy.json", dict(source=source, policy=POLICY,
                                                   evaluated_candidates=candidates))
    inputs = reference.make_inputs(args.output, args.split, specs=POLICY[args.split + "_specs"],
                                   payload_fn=payload, seed_bias=POLICY["seed_bias"][args.split])
    rows = []
    for row in inputs:
        value = safe(compare(args.output, args.output, row, candidates))
        rows.append(value)
        write(args.output / f"case{row['id']}.json", value)
        print(json.dumps(dict(id=row["id"], complete={k: v["complete"]
                         for k, v in value["modes"].items()})), flush=True)
    assert reference.shared.sources() == source
    chosen = select(rows, candidates)
    result = dict(source=source, policy=POLICY, split=args.split, candidates=candidates,
                  selected=chosen, rows=rows,
                  input_manifest_SHA256=reference.shared.sha(args.output / "inputs.json"),
                  unique_negative_samples=sum(r["samples"] for r in rows if r["negative"]),
                  expected_exit=0 if chosen is not None else 1,
                  new_default=False, all_four_RX_gates_passed=False)
    write(args.output / "model.json", result)
    write(args.output / "freeze.json", dict(source=source, policy=POLICY,
                                            selected=chosen if args.split == "development" else None,
                                            model_SHA256=reference.shared.sha(args.output / "model.json")))
    print(json.dumps({k: v for k, v in result.items() if k not in ("source", "policy", "rows")}))
    return result["expected_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
