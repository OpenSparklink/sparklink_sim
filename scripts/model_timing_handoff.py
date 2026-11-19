"""Actual finite timing/query handoff;exit1 retains the four open whole-RX gates."""

import argparse
import hashlib
import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import bridge
from model_initial_reuse import cases, verify_manifest
from model_selected_frame import TRACE_KEYS, metrics, normalized
from model_sync_initial_fit import execute as preflight_execute
from model_timing_words import execute, fixed_input_provenance

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.phy.sync_fit_word import TURN, SyncFitFormat, compile_sync_phases
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.finite_timing_replay import FiniteTimingFrameReplay
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

POLICY = Path("tests/data/timing_handoff_policy.json")
FMT = TimingWordFormat(30, 32)


def identical(before, after, old_words, new_words):
    assert normalized(before["frame"]) == normalized(after["frame"])
    assert before["word_timing_state"] == after["word_timing_state"]
    for key in TRACE_KEYS:
        np.testing.assert_array_equal(before[key], after[key])
    keys = ["symbol", "role", "cell", "phase", "word"]
    assert [{k: r[k] for k in keys} for r in old_words] == [
        {k: r[k] for k in keys} for r in new_words
    ]


def pair(out, name, profile, words, selection, payload, policy, *, partitions):
    before, _, old_words = execute(out, name + "-functional", profile, words, selection, FMT, [32])
    services = []
    after, flow, new_words = execute(
        out, name + "-actual", profile, words, selection, FMT, [32],
        flow_class=partial(FiniteTimingFrameReplay, service_observer=services.append),
    )
    identical(before, after, old_words, new_words)
    write(out / (name + "-actual-services.json"), services)
    n = flow.consumer.word_timing.count
    assert flow.stats["timing_requests"] == flow.stats["timing_completions"] == n
    assert flow.stats["query_requests"] == flow.stats["query_replies"]
    terminal_queries = flow.stats["query_replies"] - n
    assert terminal_queries in (0, 1)
    if terminal_queries:
        assert after["frame"]["status"] in ("truncated_header", "truncated_data")
        assert after["frame"]["payload"] is None
    assert flow.stats["symbol_operations"] == flow.stats["normalization_operations"] == n
    grouped = {}
    for word in new_words:
        grouped.setdefault(word["symbol"], {})[word["role"]] = word
    for r in services:
        assert r["taken_edge"] >= r["ready_edge"] == r["issued_edge"] + r["latency"]
        if r["kind"] == "query":
            assert r["latency"] == policy["structural_budgets"]["query_cycles"]
            count = r["values"]["count"]
            for role, q in enumerate(r["values"]["queries"]):
                if q is not None:
                    observed = grouped.get(count, {}).get(role)
                    if observed is None:
                        assert count == n and terminal_queries == 1
                    else:
                        assert q == {k: observed[k] for k in ["cell", "phase"]}
    if partitions:
        for label, pattern in [
            ("fixed", [7, 31]),
            ("random", np.random.default_rng(1071587).integers(1, 33, 23)),
        ]:
            other, _f, obs = execute(
                out, name + "-" + label, profile, words, selection, FMT, pattern,
                flow_class=FiniteTimingFrameReplay,
            )
            identical(after, other, new_words, obs)
            assert other["finite_timing_services"] == after["finite_timing_services"]
            assert obs == new_words
    cost = flow.estimate()
    b = policy["structural_budgets"]
    assert cost["controller_increment_over_previous_bits"] <= b["controller_increment_bits_max"]
    assert cost["query_service"]["logical_register_bits_upper"] <= b["query_service_state_bits_max"]
    assert cost["query_service"]["new_raw_RAM_bits"] == cost["new_raw_RAM_bits"] == 0
    assert cost["query_service"]["extra_DSP"] == cost["extra_DSP"] == 0
    assert cost["timing_service"]["DSP25x18"] == 4
    assert after["finite_timing_services"]["complete_latency_cycles"] is None
    a, z = [metrics(e, profile, expected=payload) for e in (before, after)]
    return dict(
        id=name, before=a, after=z, decisions_words_trajectories_and_states_exact=True,
        source_words_sha256=hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
        timing_latency_histogram=dict(Counter(
            r["latency"] for r in services if r["kind"] == "timing")),
        service_trace_file=name + "-actual-services.json", partitions_checked=partitions,
        terminal_queries_without_symbol_commit=terminal_queries,
        success_loss=a["payload_exact"] and not z["payload_exact"],
        recovery=z["payload_exact"] and not a["payload_exact"],
        false_delivery=z["false_delivery"] and not a["false_delivery"],
        actual_services=after["finite_timing_services"], cost=cost,
        complete_latency_cycles=None, capture_producer_and_remaining_arithmetic_unmodeled=True,
    )


def fresh(out, split, policy, partitions):
    rows = []
    for meta, profile, iq, payload in cases(split, policy):
        name = split + "-" + str(meta["id"])
        filtered = np.convolve(iq, rrc_filter(0.4, 4))
        words, quantization = bridge(filtered)
        path = out / (name + "-input.npz")
        np.savez_compressed(path, iq=iq, filtered=filtered, words=words,
                            payload=np.frombuffer(payload or b"", np.uint8))
        row = dict(id=name, input=meta, input_sha256=sha(path), quantization=quantization)
        found = search_psk_sync(
            np.array([physical_filter_word(w) for w in words]), 4, profile.sync_symbols(), 8
        )
        if found is None:
            row["upstream_no_candidate"] = True
        else:
            selection = dict(
                start_tick=round(8 * (found["phase"] + 4 * found["offset"])),
                delta=found["delta"], differential_score=found["score"], coherent_threshold=0.45,
                selected_sync=found["phase_symbols"][
                    found["offset"]:found["offset"] + len(profile.sync_symbols())],
            )
            preflight, rx = preflight_execute(
                out, name + "-upstream", profile, words, selection, 30, [32]
            )
            row["upstream"] = metrics(preflight, profile, expected=payload)
            if rx._sync_fit_result is None or not rx._sync_fit_result["valid"]:
                row["upstream_rejected"] = True
            else:
                row["comparison"] = pair(
                    out, name, profile, words, selection | dict(sample_limit=len(words)),
                    payload, policy, partitions=partitions,
                )
                np.savez_compressed(out / (name + "-selection.npz"),
                                    selected_sync=selection["selected_sync"])
        rows.append(row)
        write(out / "rows.json", rows)
    return rows


def fixed(out, timing_root, capture_root, selected_root, policy):
    parents = {str(p): verify_manifest(p) for p in (timing_root, capture_root, selected_root)}
    rows = []
    for old in json.loads((timing_root / "fixed-regressions.json").read_text()):
        row = dict(id=old["id"], original=old)
        if "candidate" in old:
            r, meta = old["original"]["original_request"], old["original"]["input"]
            provenance = fixed_input_provenance(capture_root, r["stream"],
                                                old["original"]["original_input_sha256"])
            profile = validate_profiles(
                [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
            )[0]
            with np.load(capture_root / (r["stream"] + "-filtered.npz")) as z:
                words = z["words"][r["origin"]:, r["request"]["path"]]
            assert hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest() == (
                old["candidate"]["source_words_sha256"])
            with np.load(capture_root / (r["stream"] + "-input.npz")) as z:
                payload = z["payload"].tobytes() if meta["positive"] else None
            with np.load(selected_root / (old["id"] + "-full-trajectory.npz")) as z:
                sync = z["selected_sync"].copy()
            selection = old["original"]["selection"] | dict(selected_sync=sync)
            row["comparison"] = pair(out, old["id"], profile, words, selection, payload,
                                     policy, partitions=True)
            assert row["comparison"]["before"] == old["candidate"]["after"]
            row["input_container_provenance"] = provenance
        rows.append(row)
    write(out / "fixed-regressions.json", rows)
    return rows, parents


def blocked_prefixes(out):
    """Real periodic source until an explicit blocked unmodeled service;not full RX latency."""
    from tests.test_selected_rx import ordinary, selected_input
    from tests.test_selected_word import quantize_input

    rows = []
    for ft in [2, 3, 4]:
        profile = ordinary(ft, 8, 0)
        values, _payload, selection = selected_input(profile, length=31)
        words = quantize_input(np.r_[values, np.zeros(8192)])
        selection["sample_limit"] = len(words)
        source_file = out / f"periodic-ft{ft}-input.npz"
        np.savez_compressed(source_file, words=words, selected_sync=selection["selected_sync"])
        cell = selection["start_tick"] // 8
        origin = cell - 1
        for hz in [100_000_000, 160_000_000, 200_000_000]:
            source = ReceiveHistoryPort(first_index=origin)
            init = InitialSyncFitService(
                source, compile_sync_phases(profile.sync_symbols()),
                round(selection["delta"] * TURN / math.tau),
                cell, (selection["start_tick"] % 8) * 8192, fmt=SyncFitFormat(30),
            )
            pos, flow, stop, initial_snapshot = origin, None, None, None
            while stop is None:
                cycle = init.cycle if flow is None else flow.cycle
                due = math.ceil((pos - origin) * hz / 16_000_000)
                assert due >= cycle or pos == len(words), "No suppressed ADC arrivals"
                if pos < len(words) and due == cycle:
                    accepted = (source.write(pos, words[pos], cycle) if flow is None
                                else flow.write_input(words[pos]))
                    pos += 1  # ADC has no backpressure,including the rejected word.
                    if not accepted:
                        stop = "initial_history_overflow" if flow is None else "history_overflow"
                        if flow is None:
                            init.abort(reason=stop)
                        break
                if flow is None:
                    init.step()
                    if init.ready:
                        initial_snapshot = dict(
                            edge=init.cycle, head=source.head, end=source.end,
                            live=source.live, remaining_words=source.capacity - source.live,
                        )
                        flow = FiniteTimingFrameReplay(
                            profile, init, word_timing_format=FMT, reuse_stencil=True, **selection
                        )
                else:
                    # Never execute any floating service or replace its cost with zero.
                    flow.step()
                    if flow.done:
                        stop = flow.event["frame"]["status"]
                assert cycle < 100_000 and pos < len(words)
            row = dict(
                frame_type=ft, clock_hz=hz, source_samples_per_second=16_000_000,
                source_file=source_file.name, source_sha256=sha(source_file),
                source_origin=origin, source_samples_arrived=pos - origin,
                unadmitted_suffix_samples=len(words) - pos, stop=stop,
                initialization_ready=flow is not None,
                history_peak_live=source.stats["peak_live"], history_capacity=source.capacity,
                terminal_edge=cycle, no_functional_resolution=True,
                unmodeled_service_blocked="symbol_normalization" if flow else None,
                complete_latency_cycles=None, whole_receiver16MSps=False,
                drop_unit="one selected context under blocked downstream;not algorithm miss",
                algorithm_misses=None, capture_producer_not_modeled=True,
            )
            if flow is not None:
                assert flow.stats["functional_resolutions"] == 0
                assert not flow.take_mac_result().packets
                row["actual_services"] = flow.event["finite_timing_services"]
                n = len(profile.sync_symbols())
                last_hold_retire = cell + 4 * (n - 1) - 1
                assert last_hold_retire <= initial_snapshot["head"]
                reject_index = initial_snapshot["head"] + source.capacity
                reject_edge = math.ceil((reject_index - origin) * hz / 16_000_000)
                margin = reject_edge - initial_snapshot["edge"]
                row["initial_snapshot"] = initial_snapshot
                row["known_sync_retention_necessary_condition"] = dict(
                    last_hold_query_retire_before=last_hold_retire,
                    no_raw_retirement_before_last_hold_query=True,
                    first_overflow_without_retirement_edge=reject_edge,
                    remaining_edges_before_raw_capacity_exhausted=margin,
                    actual_hold_power_cycles_each=18,
                    timing_only_known_sync_lower_bound_edges=18 * n,
                    current_serial_hold_timing_alone_exceeds_margin=18 * n > margin,
                    cache_queries_control_and_unknown_work_not_included=True,
                    qualification="necessary bound on this fixed replay/retirement structure only",
                )
            rows.append(row)
    write(out / "blocked-periodic-prefixes.json", rows)
    return rows


def run(a):
    policy = json.loads(POLICY.read_text())
    for split in ["development", "validation"]:
        validate_profiles(policy[split]["profiles"], policy[split]["pid"])
        assert len(policy["negative_samples"][split]) == 4
    source = {str(p): sha(p) for p in sorted({POLICY, Path("uv.lock"),
              *Path("src").rglob("*.py"), *Path("tests").rglob("*.py"),
              *Path("scripts").glob("*.py")})}
    freeze = json.loads(a.freeze.read_text()) if a.freeze else None
    if freeze:
        assert freeze["source"] == source and freeze["policy"] == policy
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", dict(source=source, policy=policy, split=a.split,
          upstream="whole-record differential max fixture,not bounded capture producer",
          actual_timing_and_query_services=True, truth_and_payload="postdecision only",
          remaining_float_services_have_unknown_cycles=True))
    rows = fresh(out, a.split, policy, partitions=freeze is not None)
    fixed_rows, parents = (fixed(out, a.timing_root, a.capture_root, a.selected_root, policy)
                           if freeze is None else ([], {}))
    prefixes = blocked_prefixes(out) if freeze is None else []
    reached = [r["comparison"] for r in rows + fixed_rows if "comparison" in r]
    assert reached
    summary = dict(
        stage=policy["stage"], split=a.split, parent_manifests=parents,
        structural_value_handoff_eligible=all(
            r["decisions_words_trajectories_and_states_exact"] for r in reached),
        before_full=sum(r["comparison"]["before"]["payload_exact"] for r in rows
                        if "comparison" in r),
        after_full=sum(r["comparison"]["after"]["payload_exact"] for r in rows
                       if "comparison" in r),
        fixed_before_full=sum(r["comparison"]["before"]["payload_exact"] for r in fixed_rows
                              if "comparison" in r),
        fixed_after_full=sum(r["comparison"]["after"]["payload_exact"] for r in fixed_rows
                             if "comparison" in r),
        original_requests=len(fixed_rows), fixed_reached=sum("comparison" in r for r in fixed_rows),
        success_losses=sum(r["success_loss"] for r in reached),
        recoveries=sum(r["recovery"] for r in reached),
        new_false_deliveries=sum(r["false_delivery"] for r in reached),
        negative_observation_samples=sum(r["input"].get("observation_samples", 0) for r in rows),
        all_RX_gates=dict(streaming=False, full_fixed=False, system_resources=False,
                          continuous16MSps=False),
        default_changed=False, old_RRC_interpolation_and16KiB_failures_retained=True,
        blocked_periodic_prefixes=prefixes,
    )
    write(out / "model.json", summary)
    if freeze is None:
        write(out / "freeze.json", dict(source=source, policy=policy, parent_manifests=parents,
              adoption="value handoff only;unmodeled work and serial scheduling not promoted"))
    write(out / "failure-associations.json", [r for r in rows + fixed_rows
          if "comparison" not in r or not r["comparison"]["after"]["payload_exact"]])
    assert all(sha(Path(k)) == v for k, v in source.items())
    write(out / "manifest.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                                  if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["development", "validation"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--timing-root", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--selected-root", type=Path)
    a = p.parse_args()
    if (a.split == "validation") != (a.freeze is not None):
        p.error("validation requires prior freeze;development must precede it")
    if a.split == "development" and any(
        r is None for r in (a.timing_root, a.capture_root, a.selected_root)
    ):
        p.error("development requires original frozen timing,capture and selection evidence")
    raise SystemExit(run(a))
