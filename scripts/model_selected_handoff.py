"""Prepared actual words through shared frame gate;unknown arithmetic blocks cycle proof."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import bridge
from model_initial_reuse import cases, verify_manifest
from model_selected_frame import TRACE_KEYS, metrics, normalized
from model_sync_initial_fit import execute as word_execute
from tests.test_selected_frame_replay import prepare, same_values

from nearlink_sdr.phy.interpolation_word import interpolate_filter_word
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, TURN, SyncFitFormat, compile_sync_phases
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay

POLICY = Path("tests/data/selected_handoff_policy.json")


def execute(out, name, profile, words, selection, reuse, pattern):
    observed = []
    prepared, pos = prepare(profile, words, selection)
    flow = SelectedFrameReplay(
        profile, prepared, reuse_stencil=reuse, observer=observed.append, **selection
    )
    j = 0
    while not flow.done:
        if flow.pending_service is not None:
            flow.resolve_functional()  # Explicit reference work;timing evidence becomes invalid.
        else:
            if flow.wants_replay_input:
                if pos < len(words):
                    used = flow.feed_replay_words(words[pos : pos + int(pattern[j % len(pattern)])])
                    assert used > 0
                    pos += used
                    j += 1
                else:
                    flow.end_input()
            flow.step()
        assert flow.cycle < 8_000_000
    event = flow.event
    write(
        out / (name + ".json"), normalized({k: v for k, v in event.items() if k not in TRACE_KEYS})
    )
    np.savez_compressed(
        out / (name + "-trajectory.npz"), **{k: event[k] for k in TRACE_KEYS if k in event}
    )
    np.savez_compressed(
        out / (name + "-interpolants.npz"),
        **{
            k: np.array([r[k] for r in observed], np.uint64 if k == "word" else np.int64)
            for k in [
                "symbol",
                "role",
                "cell",
                "phase",
                "word",
                "transport_edge",
                "clock_evidence_valid",
            ]
        },
    )
    packets = flow.take_mac_result().packets
    assert len(packets) == int(event["frame"]["status"] == "valid")
    if packets:
        assert packets[0].mac_payload == event["frame"]["payload"]
    assert flow.release_source() is flow.history and flow.release_source() is None
    return event, flow, observed


def check_words(observed, reference, words, n_sync):
    centers = reference["symbol_centers_filtered"]
    committed = unused = 0
    roles = set()
    for row in observed:
        n, role = row["symbol"], row["role"]
        assert (n, role) not in roles
        roles.add((n, role))
        cell, phase = row["cell"], row["phase"]
        expected, _ = interpolate_filter_word(
            tuple(int(w) for w in words[cell - 1 : cell + 3]), phase, INPUT_FORMAT
        )
        assert expected == row["word"]
        if n >= len(centers):
            unused += 1
            continue
        t = centers[n] if role == 0 else (centers[n - 1] + centers[n]) / 2
        assert math.floor(t) == cell and round((t - cell) * 65536) == phase
        assert role == 0 or n >= n_sync
        committed += 1
    assert committed == min(n_sync, len(centers)) + 2 * max(0, len(centers) - n_sync)
    return dict(committed_interpolants_exact=committed, uncommitted_results_preserved=unused)


def evaluate(out, name, profile, words, selection, payload, *, compare_reference, truth=None):
    selection = selection | dict(sample_limit=len(words))
    base_selection = {k: v for k, v in selection.items() if k != "sample_limit"}
    reference, rx = word_execute(
        out, name + "-word-reference", profile, words, base_selection, 30, [32]
    )
    row = dict(
        id=name,
        selection=normalized({k: v for k, v in selection.items() if k != "selected_sync"}),
        baseline=metrics(reference, profile, expected=payload, **(truth or {})),
    )
    if rx._sync_fit_result is None or not rx._sync_fit_result["valid"]:
        row.update(
            handoff_available=False, reason="upstream selection rejected or finite fit invalid"
        )
        return row, reference
    row["handoff_available"] = True
    event, flow, observed = execute(out, name + "-reuse", profile, words, selection, True, [32])
    same_values(reference, event)
    assert (
        reference["delivered_after_filtered_samples"] == event["delivered_after_filtered_samples"]
    )
    word_proof = check_words(observed, reference, words, len(profile.sync_symbols()))
    for kind, pattern in [
        ("fixed", [7, 31]),
        ("random", np.random.default_rng(954129).integers(1, 33, 23)),
    ]:
        other, _, queries = execute(
            out, name + "-" + kind, profile, words, selection, True, pattern
        )
        same_values(event, other)
        assert observed == queries and event["replay_transport"] == other["replay_transport"]
    tracked = max(0, len(event["symbols"]) - len(profile.sync_symbols()))
    old_reads = 8 * tracked
    measured = None
    if compare_reference:
        old, previous, _ = execute(
            out, name + "-current-first", profile, words, selection, False, [32]
        )
        same_values(old, event)
        measured = dict(
            raw_reads=previous.stats["raw_reads"],
            transport_edges=previous.cycle,
            data_queries=previous.stats["data_queries"],
        )
        old_reads = previous.stats["raw_reads"]
        assert old_reads == flow.stats["raw_reads"] + flow.stats["overlap_words_reused"]
    cost = flow.estimate()
    row.update(
        reuse=metrics(event, profile, expected=payload, **(truth or {})),
        numerical_and_frame_exact=True,
        partitions_exact=True,
        word_proof=word_proof,
        baseline_READ_count_is_formula=not compare_reference,
        baseline_raw_reads=old_reads,
        reference_measured=measured,
        reuse_stats=flow.stats,
        cost=cost,
        transport_edges=flow.cycle,
        complete_latency_cycles=None,
        remaining_arithmetic_service_cycles=None,
        unconsumed_suffix=flow.history.live,
        success_loss=row["baseline"]["payload_exact"] and event["frame"]["payload"] != payload,
        false_delivery=payload is None and event["frame"]["status"] == "valid",
    )
    return row, reference


def fresh(out, split, policy):
    rows = []
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
            result, _ = evaluate(
                out,
                name,
                profile,
                words,
                selection,
                payload,
                compare_reference=split == "development",
                truth=dict(truth_start=meta["truth_start_filtered"], truth_step=meta["truth_step"])
                if not meta.get("negative")
                else None,
            )
            row.update(result)
        rows.append(row)
    return rows


def fixed(out, args):
    parents = {
        str(p): verify_manifest(p) for p in [args.selected_root, args.capture_root, args.fit_root]
    }
    records = json.loads((args.selected_root / "regressions.json").read_text())
    metas = {r["id"]: r["input"] for r in json.loads((args.capture_root / "rows.json").read_text())}
    rows = []
    for record in records:
        name = record["id"]
        if record.get("selection_not_available"):
            rows.append(dict(id=name, geometry_unavailable=True, original_request=record))
            continue
        meta = metas[record["stream"]]
        assert meta["input_sha256"] == record["original_archive_sha256"]
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        selection = json.loads((args.selected_root / (name + "-full.json")).read_text())[
            "selection"
        ]
        for k in ["sample_limit", "timing_backend", "history_capacity"]:
            selection.pop(k, None)
        with np.load(args.selected_root / (name + "-full-trajectory.npz")) as z:
            selection["selected_sync"] = z["selected_sync"].copy()
        with np.load(args.capture_root / (record["stream"] + "-filtered.npz")) as z:
            words = z["words"][record["origin"] :, record["request"]["path"]]
        with np.load(args.capture_root / (record["stream"] + "-input.npz")) as z:
            payload = z["payload"].tobytes() if meta["positive"] else None
        row, reference = evaluate(
            out, name, profile, words, selection, payload, compare_reference=False
        )
        old = json.loads((args.fit_root / (name + "-p30.json")).read_text())
        for k in ["status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload"]:
            assert normalized(reference["frame"].get(k)) == old["frame"].get(k)
        with np.load(args.fit_root / (name + "-p30-trajectory.npz")) as z:
            for k in TRACE_KEYS:
                if k in z.files:
                    np.testing.assert_array_equal(reference[k], z[k])
        row.update(
            input=meta,
            original_request=record,
            original_input_sha256=meta["input_sha256"],
            prior_finite_receiver_frames_and_traces_exact=True,
        )
        rows.append(row)
    return rows, parents


def live_barriers(out):
    from tests.test_selected_rx import ordinary, selected_input
    from tests.test_selected_word import quantize_input

    rows = []
    for ft in [2, 3, 4]:
        profile = ordinary(ft, 8, 0)
        values, _, selection = selected_input(profile, length=19)
        words = quantize_input(np.r_[values, np.zeros(2048)])
        selection["sample_limit"] = len(words)
        cell = selection["start_tick"] // 8
        origin = cell - 1
        for hz in [100_000_000, 160_000_000, 200_000_000]:
            source = ReceiveHistoryPort(first_index=origin)
            init = InitialSyncFitService(
                source,
                compile_sync_phases(profile.sync_symbols()),
                round(selection["delta"] * TURN / math.tau),
                cell,
                (selection["start_tick"] % 8) * 8192,
                fmt=SyncFitFormat(30),
            )
            pos = origin
            overflow = None
            while not init.ready:
                c = init.cycle
                if c == math.ceil((pos - origin) * hz / 16_000_000):
                    if not source.write(pos, words[pos], c):
                        overflow = dict(index=pos, relative_index=pos - origin, cycle=c)
                        init.abort()
                        break
                    pos += 1
                init.step()
            if overflow is not None:
                rows.append(
                    dict(
                        ft=ft,
                        hz=hz,
                        status="initial_source_overflow",
                        overflow=overflow,
                        full_receiver16MSps=False,
                    )
                )
                continue
            flow = SelectedFrameReplay(profile, init, reuse_stencil=True, **selection)
            first_barrier = None
            while not flow.done:
                c = flow.cycle
                if c == math.ceil((pos - origin) * hz / 16_000_000):
                    if not flow.write_input(words[pos]):
                        overflow = dict(index=pos, relative_index=pos - origin, cycle=c)
                    pos += 1
                flow.step()
                if first_barrier is None and flow.pending_service is not None:
                    first_barrier = dict(flow.pending_service, cycle=flow.cycle)
                assert c < 30000
            assert overflow is not None and flow.stats["functional_resolutions"] == 0
            assert flow.consumer.last_event["sync_symbols_consumed"] == 0
            rows.append(
                dict(
                    ft=ft,
                    hz=hz,
                    status="unknown_service_blocked_then_bounded_overflow",
                    first_barrier=first_barrier,
                    overflow=overflow,
                    counters=flow.stats,
                    remainder=flow.history.live,
                    all_unknown_arithmetic_not_executed=True,
                    full_receiver16MSps=False,
                )
            )
    write(out / "live-barriers.json", rows)
    return rows


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY.read_text())
    for split in ["development", "validation"]:
        validate_profiles(policy[split]["profiles"], policy[split]["pid"])
    sources = [
        POLICY,
        Path("uv.lock"),
        *Path("src").rglob("*.py"),
        *Path("tests").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    ]
    source = {str(p): sha(p) for p in sorted(set(sources))}
    freeze = json.loads(args.freeze.read_text()) if args.freeze else None
    if freeze is not None:
        assert freeze["source"] == source and freeze["policy"] == policy
    eligible_rf = freeze is None or freeze["reuse_eligible"]
    write(
        out / "protocol.json",
        dict(
            source=source,
            policy=policy,
            split=args.split,
            initial_input_pausable=True,
            full_record_upstream_maximum_is_fixture=True,
            no_loop_or_sync_algorithm_change=True,
            unknown_arithmetic_not_zero=True,
        ),
    )
    rows = fresh(out, args.split, policy) if eligible_rf else []
    regressions, parents = fixed(out, args) if freeze is None else ([], {})
    actual = [r for r in rows + regressions if r.get("handoff_available")]
    budget = policy["node_budget"]
    resource = bool(actual) and all(
        r["cost"]["new_controller_state_bits"] <= budget["controller_state_upper_bits"]
        and r["cost"]["source_staging_words"] <= budget["staging_words_max"]
        and r["cost"]["pending_interpolants"] <= budget["pending_interpolants_max"]
        and r["cost"]["new_raw_RAM_bits"] == r["cost"]["extra_DSP"] == 0
        and r["cost"]["source_read_ports"] == 1
        and r["cost"]["within_symbol_compaction_mux2_upper"] <= budget["compaction_data_mux2_upper"]
        for r in actual
    )
    eligible = resource and all(
        r["numerical_and_frame_exact"] and not r["success_loss"] and not r["false_delivery"]
        for r in actual
    )
    live = live_barriers(out)
    model = dict(
        stage=policy["stage"],
        split=args.split,
        reuse_eligible=eligible if eligible_rf else None,
        RF_evaluated=eligible_rf,
        resource_node_budget_passed=resource,
        baseline_full=sum(r.get("baseline", {}).get("payload_exact", False) for r in rows),
        replay_full=sum(r.get("reuse", {}).get("payload_exact", False) for r in rows),
        fixed_baseline_full=sum(
            r.get("baseline", {}).get("payload_exact", False) for r in regressions
        ),
        fixed_replay_full=sum(r.get("reuse", {}).get("payload_exact", False) for r in regressions),
        successes_lost=[r["id"] for r in actual if r["success_loss"]],
        false_deliveries=[r["id"] for r in actual if r["false_delivery"]],
        negative_observation_samples=sum(policy["negative_samples"][args.split])
        if eligible_rf
        else 0,
        measured_reference_raw_reads=sum(
            r["reference_measured"]["raw_reads"]
            for r in actual
            if r["reference_measured"] is not None
        ),
        measured_same_rows_reuse_raw_reads=sum(
            r["reuse_stats"]["raw_reads"] for r in actual if r["reference_measured"] is not None
        ),
        actual_handoffs=len(actual),
        parent_manifests=parents,
        live_scenarios=len(live),
        default_RX_changed=False,
        full_receiver16MSps=False,
        full_RX_gates={k: False for k in ["streaming", "fixed", "system_resources", "16MSps"]},
        old_interpolation_RRC_and16KiB_failures_retained=True,
    )
    write(out / "rows.json", rows)
    write(out / "fixed-regressions.json", regressions)
    write(out / "model.json", model)
    write(
        out / "failure-associations.json",
        dict(
            legal_receive=[
                r
                for r in rows + regressions
                if r.get("handoff_available")
                and not (
                    r.get("input", {}).get("negative")
                    or r.get("input", {}).get("positive") is False
                )
                and not r["reuse"]["payload_exact"]
            ],
            upstream=[r for r in rows + regressions if not r.get("handoff_available")],
            resources=live,
            numerical_or_success_regressions=[r for r in actual if r["success_loss"]],
            interpretation=(
                "observed failure or explicit service gap,not theoretical irrecoverability"
            ),
        ),
    )
    if freeze is None:
        write(
            out / "freeze.json",
            dict(
                source=source,
                policy=policy,
                reuse_eligible=eligible,
                held_not_generated=True,
                default_not_promoted=True,
            ),
        )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(model, indent=2))
    return 1  # Functional pipeline/module success does not close four whole RX gates.


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--split", choices=["development", "validation"], required=True)
    p.add_argument("--selected-root", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--fit-root", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.split == "development" and any(
        x is None for x in [args.selected_root, args.capture_root, args.fit_root]
    ):
        p.error("Need three frozen roots")
    if args.split == "validation" and args.freeze is None:
        p.error("Need development source/policy freeze")
    raise SystemExit(run(args))
