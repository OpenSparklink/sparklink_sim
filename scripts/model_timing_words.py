"""Finite timing at real interpolant port;exit1 retains four open receiver gates."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import INPUTS, MEMBERS, bridge, verify_input_container
from model_initial_reuse import cases, verify_manifest
from model_selected_frame import TRACE_KEYS, metrics, normalized
from tests.test_selected_frame_replay import prepare

from nearlink_sdr.phy.interpolant_rx import physical_interpolant
from nearlink_sdr.phy.interpolation_word import interpolate_filter_word
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT
from nearlink_sdr.phy.timing import GardnerPIState, PSKTimingConfig
from nearlink_sdr.phy.timing_word import ONE, TimingWordFormat, WordTimingState
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay
from nearlink_sdr.sim.timing_word_flow import TimingWordService

POLICY = Path("tests/data/timing_word_policy.json")


def execute(out, name, profile, words, selection, fmt, pattern, *, flow_class=SelectedFrameReplay,
            host_edges_per_symbol=None):
    if host_edges_per_symbol is not None and (
        type(host_edges_per_symbol) is not int or not 1 <= host_edges_per_symbol <= 1024
    ):
        raise ValueError("Need a bounded host verification guard,not a hardware deadline")
    prepared, pos = prepare(profile, words, selection)
    observations = []
    flow = flow_class(
        profile,
        prepared,
        reuse_stencil=True,
        word_timing_format=fmt,
        observer=observations.append,
        **selection,
    )
    j = 0
    host_edge_limit = (8_000_000 if host_edges_per_symbol is None
                       else 65536 + host_edges_per_symbol * flow.consumer.max_frame_symbols)
    while not flow.done:
        if flow.pending_service is not None:
            flow.resolve_functional()  # Remaining carrier/normalization/decoder is off-clock.
        else:
            if flow.wants_replay_input:
                if pos == len(words):
                    flow.end_input()
                else:
                    used = flow.feed_replay_words(words[pos : pos + int(pattern[j % len(pattern)])])
                    assert used == 1
                    pos += used
                    j += 1
            flow.step()
        assert flow.cycle < host_edge_limit
    event = flow.event
    write(
        out / (name + ".json"), normalized({k: v for k, v in event.items() if k not in TRACE_KEYS})
    )
    np.savez_compressed(out / (name + "-trajectory.npz"), **{k: event[k] for k in TRACE_KEYS})
    np.savez_compressed(
        out / (name + "-interpolants.npz"),
        **{
            k: np.array([x[k] for x in observations], np.uint64 if k == "word" else np.int64)
            for k in ["symbol", "role", "cell", "phase", "word", "transport_edge"]
        },
    )
    assert len(flow.take_mac_result().packets) == int(event["frame"]["status"] == "valid")
    assert not flow.take_mac_result().packets
    assert flow.release_source() is flow.history
    return event, flow, observations


def replay_timing(out, name, flow, observations, words):
    """Independent float loop and actual divider/DSP clocks on already chosen word stream."""
    rx = flow.consumer
    node = WordTimingState(
        rx.initial_fit["power"], round(rx.selected_start * 8), len(rx.reference), rx.word_timing.fmt
    )
    service = TimingWordService(node)
    p, e = rx.initial_fit["power"]
    floating = GardnerPIState(4, math.ldexp(p, e), PSKTimingConfig())
    groups = {}
    for x in observations:
        groups.setdefault(x["symbol"], {})[x["role"]] = x
        actual, _ = interpolate_filter_word(
            tuple(int(w) for w in words[x["cell"] - 1 : x["cell"] + 3]), x["phase"], INPUT_FORMAT
        )
        assert x["word"] == actual
    rows = []
    previous = 0j
    for n in range(rx.word_timing.count):
        g = groups[n]
        assert set(g) == ({0} if n < len(rx.reference) else {0, 1})
        for role, q in enumerate(node.queries):
            if q is not None:
                assert q == {k: g[role][k] for k in ["cell", "phase"]}
        current, mid = g[0]["word"], None if 1 not in g else g[1]["word"]
        z = physical_interpolant(current)
        step, error = floating.advance(
            z, previous, None if mid is None else physical_interpolant(mid)
        )
        before = node.next_q32
        start_products, start_div = service.stats["native_products"], service.stats["divider_steps"]
        assert service.issue(current, mid)
        while not service.ready:
            service.step()
        latency = service.ready_cycle - service.first_cycle
        snapshot = service.take()
        assert snapshot["count"] == n + 1
        rows.append(
            dict(
                power_m=node.power_word[0],
                power_e=node.power_word[1],
                power_relative_error=abs(node.power - floating.power) / floating.power,
                error_word=node.error_word,
                TED_absolute_error=abs(node.error_word / 2**node.fmt.error_bits - error),
                step_absolute_error=abs((node.next_q32 - before) / ONE - step),
                next_integer=node.next_q32 >> 32,
                next_fraction=node.next_q32 & (ONE - 1),
                omega_q32=node.omega_q32,
                cycles=latency,
                products=service.stats["native_products"] - start_products,
                divider_steps=service.stats["divider_steps"] - start_div,
            )
        )
        previous = z
        service.step()
    assert node.snapshot() == rx.word_timing.snapshot()
    arrays = {k: np.array([x[k] for x in rows]) for k in rows[0]} if rows else {}
    np.savez_compressed(out / (name + "-timing-service.npz"), **arrays)
    return dict(
        actual_DSP_divider_word_states_exact=True,
        power_relative_max=max((x["power_relative_error"] for x in rows), default=0),
        TED_absolute_max=max((x["TED_absolute_error"] for x in rows), default=0),
        step_absolute_max=max((x["step_absolute_error"] for x in rows), default=0),
        service_cycles_min=min((x["cycles"] for x in rows), default=0),
        service_cycles_max=max((x["cycles"] for x in rows), default=0),
        operations=service.stats,
        cost=service.estimate(),
        symbol_count=len(rows),
    )


def compare(
    out, name, profile, words, selection, payload, baseline, fmt, truth=None, partitions=False
):
    label = f"{name}-p{fmt.power_bits}e{fmt.error_bits}"
    event, flow, obs = execute(out, label, profile, words, selection, fmt, [32])
    n = min(len(event["symbols"]), len(baseline["symbols"]))
    delta = float(
        np.max(
            abs(event["symbol_centers_filtered"][:n] - baseline["symbol_centers_filtered"][:n]),
            initial=0,
        )
    )
    proof = replay_timing(out, label, flow, obs, words)
    if partitions:
        for kind, pattern in [
            ("fixed", [7, 31]),
            ("random", np.random.default_rng(1061187).integers(1, 33, 23)),
        ]:
            other, _, samples = execute(
                out, label + "-" + kind, profile, words, selection, fmt, pattern
            )
            assert normalized(other["frame"]) == normalized(event["frame"])
            assert other["word_timing_state"] == event["word_timing_state"]
            assert samples == obs
            for k in TRACE_KEYS:
                np.testing.assert_array_equal(event[k], other[k])
    result = metrics(event, profile, expected=payload, **(truth or {}))
    before = metrics(baseline, profile, expected=payload, **(truth or {}))
    return dict(
        id=name,
        source_words_sha256=hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
        format=asdict(fmt),
        before=before,
        after=result,
        center_delta_max=delta,
        compared_symbols=n,
        total_symbols=len(event["symbols"]),
        success_loss=before["payload_exact"] and not result["payload_exact"],
        recovery=not before["payload_exact"] and result["payload_exact"],
        false_delivery=result["false_delivery"],
        numerical=proof,
        partitions_checked=partitions,
        complete_latency_cycles=None,
        remaining_carrier_head_decoder_cycles=None,
        source_reads=flow.stats["raw_reads"],
        final_state=event["word_timing_state"],
    )


def eligible(row, policy):
    if not row.get("handoff_available"):
        return False
    b, n = policy["node_budgets"], row["numerical"]
    return (
        not row["success_loss"]
        and not row["false_delivery"]
        and row["center_delta_max"] <= b["additional_closed_loop_center_max_samples"]
        and n["power_relative_max"] <= b["power_relative_max"]
        and n["TED_absolute_max"] <= b["TED_absolute_max"]
        and n["cost"]["DSP25x18"] <= b["extra_DSP_max"]
        and n["cost"]["state_register_bits_upper"] <= b["state_bits_max"]
        and n["cost"]["extra_raw_RAM_bits"] <= b["new_raw_RAM_bits_max"]
    )


def development_eligible(rows, fmt, policy):
    """Only the reached timing node is scored;unchanged upstream rejection is not a rescue."""
    key = str(fmt.power_bits)
    reached = [r["candidates"][key] for r in rows if key in r["candidates"]]
    lost_before_timing = any(
        r.get("baseline", {}).get("payload_exact") and key not in r["candidates"] for r in rows
    )
    return bool(reached) and not lost_before_timing and all(eligible(r, policy) for r in reached)


def fresh(out, split, policy, formats, partitions):
    rows = []
    from model_sync_initial_fit import execute as baseline_execute

    for meta, profile, iq, payload in cases(split, policy):
        name = split + "-" + str(meta["id"])
        filtered = np.convolve(iq, rrc_filter(0.4, 4))
        words, quantization = bridge(filtered)
        np.savez_compressed(
            out / (name + "-input.npz"),
            iq=iq,
            filtered=filtered,
            words=words,
            payload=np.frombuffer(payload or b"", np.uint8),
        )
        row = dict(
            id=name,
            input=meta,
            input_sha256=sha(out / (name + "-input.npz")),
            quantization=quantization,
            candidates={},
        )
        found = search_psk_sync(
            np.array([physical_filter_word(w) for w in words]), 4, profile.sync_symbols(), 8
        )
        if found is None:
            row["upstream_no_candidate"] = True
        else:
            selection = dict(
                start_tick=round(8 * (found["phase"] + 4 * found["offset"])),
                delta=found["delta"],
                differential_score=found["score"],
                coherent_threshold=0.45,
                selected_sync=found["phase_symbols"][
                    found["offset"] : found["offset"] + len(profile.sync_symbols())
                ],
            )
            np.savez_compressed(
                out / (name + "-selection.npz"), selected_sync=selection["selected_sync"]
            )
            reference, rx = baseline_execute(
                out, name + "-reference", profile, words, selection, 30, [32]
            )
            row["baseline"] = metrics(reference, profile, expected=payload)
            if rx._sync_fit_result is None or not rx._sync_fit_result["valid"]:
                row["upstream_rejected"] = True
            else:
                truth = (
                    dict(truth_start=meta["truth_start_filtered"], truth_step=meta["truth_step"])
                    if not meta.get("negative")
                    else None
                )
                for fmt in formats:
                    key = str(fmt.power_bits)
                    candidate = compare(
                        out,
                        name,
                        profile,
                        words,
                        selection | dict(sample_limit=len(words)),
                        payload,
                        reference,
                        fmt,
                        truth,
                        partitions,
                    )
                    candidate.update(
                        handoff_available=True,
                        input_archive_sha256=row["input_sha256"],
                        input_archive_file=name + "-input.npz",
                    )
                    row["candidates"][key] = candidate
        rows.append(row)
        write(out / "rows.json", rows)
    return rows


def fixed_input_provenance(capture, stream, original_sha):
    """Retain original SHA;accept derived containers only with exact frozen NPY bytes."""
    index = json.loads(MEMBERS.read_text())
    if index["source_index_sha256"] != sha(INPUTS):
        raise ValueError("Original input index does not match frozen member provenance")
    expected = index["entries"][stream]
    if expected["original_archive_sha256"] != original_sha:
        raise ValueError("Original archive provenance mismatch")
    protocol = json.loads((capture / "protocol.json").read_text())
    mode = protocol["input_mode"]
    if mode not in ("frozen", "derived-container-original-members-verified"):
        raise ValueError("Unknown input container mode")
    if protocol["input_index_sha256"] != sha(INPUTS) or protocol["input_members_sha256"] != sha(
        MEMBERS
    ):
        raise ValueError("Capture input provenance index mismatch")
    actual = verify_input_container(
        capture / (stream + "-input.npz"), expected, require_original=mode == "frozen"
    )
    recorded = json.loads((capture / "input-containers.json").read_text())[stream]
    if actual != recorded:
        raise ValueError("Capture container receipt mismatch")
    return actual


def fixed(out, parent, capture, selected_root, fmt, policy):
    verify_manifest(parent)
    verify_manifest(capture)
    parents = json.loads((parent / "model.json").read_text())["parent_manifests"]
    for root in (selected_root, capture):
        assert str(root) in parents and verify_manifest(root) == parents[str(root)]
    rows = []
    for old in json.loads((parent / "fixed-regressions.json").read_text()):
        record = dict(id=old["id"], original=old)
        if old.get("handoff_available"):
            r, meta = old["original_request"], old["input"]
            path = capture / (r["stream"] + "-input.npz")
            provenance = fixed_input_provenance(capture, r["stream"], old["original_input_sha256"])
            profile = validate_profiles(
                [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
            )[0]
            with np.load(capture / (r["stream"] + "-filtered.npz")) as z:
                words = z["words"][r["origin"] :, r["request"]["path"]]
            with np.load(path) as z:
                payload = z["payload"].tobytes() if meta["positive"] else None
            old_label = old["id"] + "-word-reference"
            reference = json.loads((parent / (old_label + ".json")).read_text())
            if isinstance(reference["frame"]["payload"], dict):
                reference["frame"]["payload"] = bytes.fromhex(reference["frame"]["payload"]["hex"])
            with np.load(parent / (old_label + "-trajectory.npz")) as z:
                reference.update({k: z[k].copy() for k in TRACE_KEYS})
            with np.load(selected_root / (old["id"] + "-full-trajectory.npz")) as z:
                selected_sync = z["selected_sync"].copy()
            selection = old["selection"] | dict(selected_sync=selected_sync)
            candidate = compare(
                out, old["id"], profile, words, selection, payload, reference, fmt, partitions=True
            )
            candidate.update(
                handoff_available=True,
                input_archive_sha256=provenance["current_archive_sha256"],
                original_input_archive_sha256=provenance["original_archive_sha256"],
                input_container_provenance=provenance,
                input_archive_file=str(path),
                original_window_origin=r["origin"],
                original_path=r["request"]["path"],
            )
            record["candidate"] = candidate
            record["eligible"] = eligible(candidate, policy)
        rows.append(record)
    write(out / "fixed-regressions.json", rows)
    return rows


def run(a):
    policy = json.loads(POLICY.read_text())
    for split in ["development", "validation"]:
        validate_profiles(policy[split]["profiles"], policy[split]["pid"])
    source = {
        str(p): sha(p)
        for p in sorted(
            {
                POLICY,
                Path("uv.lock"),
                *Path("src").rglob("*.py"),
                *Path("tests").rglob("*.py"),
                *Path("scripts").glob("*.py"),
            }
        )
    }
    freeze = json.loads(a.freeze.read_text()) if a.freeze else None
    if freeze:
        assert freeze["source"] == source and freeze["policy"] == policy
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "protocol.json",
        dict(
            source=source,
            policy=policy,
            split=a.split,
            truth_and_expected_payload="postdecision metrics only",
            upstream="whole-record differential maximum fixture,not bounded capture",
            old_RRC_interpolation_and16KiB_failures_retained=True,
        ),
    )
    formats = (
        [TimingWordFormat(*x) for x in policy["formats"]]
        if freeze is None
        else [TimingWordFormat(**freeze["selected_format"])]
    )
    rows = fresh(out, a.split, policy, formats, partitions=freeze is not None)
    selected = None
    if freeze is None:
        legal = [r for r in rows if not r["input"].get("negative")]
        options = [f for f in formats if development_eligible(legal, f, policy)]
        if options:
            selected = options[0]
    else:
        selected = formats[0]
    fixed_rows = (
        fixed(out, a.handoff_root, a.capture_root, a.selected_root, selected, policy)
        if a.split == "development" and selected
        else []
    )
    new_rows = [
        r["candidates"][str(selected.power_bits)] | dict(handoff_available=True)
        for r in rows
        if selected and str(selected.power_bits) in r["candidates"]
    ]
    selected_rows = new_rows + [r["candidate"] for r in fixed_rows if "candidate" in r]
    accepted = (
        selected is not None
        and bool(selected_rows)
        and all(eligible(r, policy) for r in selected_rows)
    )
    summary = dict(
        stage=policy["stage"],
        split=a.split,
        selected_format=asdict(selected) if selected else None,
        experimental_numeric_node_eligible=accepted,
        target_live_adopted=False,
        baseline_full=sum(r.get("baseline", {}).get("payload_exact", False) for r in rows),
        candidate_full=sum(r["after"]["payload_exact"] for r in new_rows),
        fixed_original_full=sum(
            r["candidate"]["before"]["payload_exact"] for r in fixed_rows if "candidate" in r
        ),
        fixed_candidate_full=sum(
            r["candidate"]["after"]["payload_exact"] for r in fixed_rows if "candidate" in r
        ),
        successes_lost=sum(r["success_loss"] for r in selected_rows),
        false_deliveries=sum(r["false_delivery"] for r in selected_rows),
        negatives_observed_samples=sum(r["input"].get("observation_samples", 0) for r in rows),
        original_requests=len(fixed_rows),
        service_cycles_max=max(
            (r["numerical"]["service_cycles_max"] for r in selected_rows), default=0
        ),
        service_critical_path_budget=policy["cycle_budget"],
        all_RX_gates=dict(
            streaming=False, full_fixed=False, system_resources=False, continuous16MSps=False
        ),
        default_RX_changed=False,
        old_failures_retained=True,
    )
    write(out / "model.json", summary)
    if freeze is None and selected:
        write(
            out / "freeze.json",
            dict(
                source=source,
                policy=policy,
                selected_format=asdict(selected),
                numeric_node_eligible=accepted,
                target_live_adopted=False,
            ),
        )
    write(
        out / "failure-associations.json",
        [r for r in selected_rows if not r["after"]["payload_exact"] or not eligible(r, policy)],
    )
    assert source == {k: sha(Path(k)) for k in source}
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["development", "validation"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--handoff-root", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--selected-root", type=Path)
    args = p.parse_args()
    if (args.split == "validation") != (args.freeze is not None):
        p.error("validation needs a prior freeze;development must precede it")
    if args.split == "development" and (
        args.handoff_root is None or args.capture_root is None or args.selected_root is None
    ):
        p.error("development needs frozen handoff and original capture roots")
    raise SystemExit(run(args))
