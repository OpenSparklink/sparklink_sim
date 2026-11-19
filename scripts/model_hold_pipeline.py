"""Exact bounded hold pipeline and selected-port deadlines;full RX remains unaccepted."""

import argparse
import hashlib
import json
import math
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_selected_frame import metrics
from model_timing_handoff import fixed, fresh, identical
from model_timing_words import execute

from nearlink_sdr.phy.sync_fit_word import TURN, SyncFitFormat, compile_sync_phases
from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.finite_timing_replay import FiniteTimingFrameReplay
from nearlink_sdr.sim.hold_timing_flow import HoldTimingPipeline
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.pipelined_timing_replay import PipelinedTimingFrameReplay
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from nearlink_sdr.sim.timing_word_flow import TimingWordService

POLICY = Path("tests/data/known_sync_pipeline_policy.json")
FMT = TimingWordFormat(30, 32)


def pair(out, name, profile, words, selection, payload, policy, *, partitions):
    before, old_flow, old_words = execute(
        out,
        name + "-serial",
        profile,
        words,
        selection,
        FMT,
        [32],
        flow_class=FiniteTimingFrameReplay,
    )
    records = []
    after, flow, seen = execute(
        out,
        name + "-pipeline",
        profile,
        words,
        selection,
        FMT,
        [32],
        flow_class=partial(PipelinedTimingFrameReplay, service_observer=records.append),
    )
    identical(before, after, old_words, seen)
    write(out / (name + "-actual-services.json"), records)
    n = len(profile.sync_symbols())
    assert flow.timing_service.hold.stats["accepted"] == n
    assert flow.stats["cache_reads"] == old_flow.stats["cache_reads"] == n
    assert flow.stats["query_replies"] == old_flow.stats["query_replies"] - n
    assert flow.timing_service.stats["completed"] == flow.consumer.word_timing.count
    assert flow.stats["timing_completions"] == flow.stats["timing_requests"]
    if partitions:
        for label, pattern in [
            ("fixed", [7, 31]),
            ("random", np.random.default_rng(1081151).integers(1, 33, 23)),
        ]:
            other, f, obs = execute(
                out,
                name + "-" + label,
                profile,
                words,
                selection,
                FMT,
                pattern,
                flow_class=PipelinedTimingFrameReplay,
            )
            identical(after, other, seen, obs)
            assert other["known_sync_pipeline"] == after["known_sync_pipeline"]
            assert obs == seen and f.event["frame"] == flow.event["frame"]
    cost = flow.estimate()
    hold = cost["timing_service"]["known_sync_pipeline"]
    budget = policy["node_budgets"]
    assert hold["additional_state_bits_upper"] <= budget["extra_state_bits_max"]
    assert hold["new_raw_RAM_bits"] == hold["additional_cache_read_ports"] == 0
    assert cost["timing_service"]["DSP25x18"] == old_flow.estimate()["timing_service"]["DSP25x18"]
    a, b = [metrics(e, profile, expected=payload) for e in [before, after]]
    return dict(
        id=name,
        before=a,
        after=b,
        values_states_trajectories_exact=True,
        source_words_sha256=hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
        success_loss=a["payload_exact"] and not b["payload_exact"],
        recovery=b["payload_exact"] and not a["payload_exact"],
        false_delivery=b["false_delivery"] and not a["false_delivery"],
        cost=cost,
        service_trace_file=name + "-actual-services.json",
        skipped_repeated_known_sync_queries=n,
        cache_passes=1,
        complete_latency_cycles=None,
        packet_normalization_carrier_decoder_cycles=None,
    )


def periodic_prefixes(out):
    """Actual selected source/cache/fit and ready timing sink;not a packet receiver."""
    from tests.test_selected_rx import ordinary, selected_input
    from tests.test_selected_word import quantize_input

    rows = []
    for ft in [2, 3, 4]:
        profile = ordinary(ft, 8, 0)
        iq, _, selection = selected_input(profile, length=31)
        words = quantize_input(np.r_[iq, np.zeros(8192)])
        file = out / f"prefix-ft{ft}-input.npz"
        np.savez_compressed(file, words=words, selected_sync=selection["selected_sync"])
        cell, phase = divmod(selection["start_tick"], 8)
        origin = cell - 1
        n = len(profile.sync_symbols())
        for hz in [100_000_000, 160_000_000, 200_000_000]:
            for variant in ["serial", "pipeline"]:
                source = ReceiveHistoryPort(first_index=origin)
                init = InitialSyncFitService(
                    source,
                    compile_sync_phases(profile.sync_symbols()),
                    round(selection["delta"] * TURN / math.tau),
                    cell,
                    phase * 8192,
                    fmt=SyncFitFormat(30),
                )
                pos, service, epoch, accepted = origin, None, 0, 0
                reference = expected = None
                cache_requested = False
                snapshot = None
                status = None
                while status is None:
                    c = init.cycle if service is None else epoch + service.cycle
                    if c == math.ceil((pos - origin) * hz / 16_000_000):
                        ok = source.write(pos, int(words[pos]), c)
                        pos += 1  # Every arrived ADC sample is counted,including the rejected one.
                        if not ok:
                            status = "initial_overflow" if service is None else "hold_overflow"
                            if service is None:
                                init.abort()
                            else:
                                service.abort()
                            break
                    if service is None:
                        init.step()
                        if init.ready:
                            fit = init.fit.take()
                            assert fit["valid"]
                            epoch = init.cycle
                            expected = tuple(init.stencil.output.cache.words[:n])  # Oracle only.
                            state = WordTimingState(fit["power"], selection["start_tick"], n, FMT)
                            reference = WordTimingState(
                                fit["power"], selection["start_tick"], n, FMT
                            )
                            service = (
                                HoldTimingPipeline(state, init.stencil.output, epoch)
                                if variant == "pipeline"
                                else TimingWordService(state)
                            )
                            reject = math.ceil((source.head + 512 - origin) * hz / 16_000_000)
                            snapshot = dict(
                                edge=epoch,
                                head=source.head,
                                end=source.end,
                                margin_edges=reject - epoch,
                            )
                    else:
                        got = service.take()
                        if got is not None:
                            assert got == reference.advance_words(expected[accepted], None)
                            accepted += 1
                            if accepted == n:
                                status = "hold_ready"
                                break
                        if variant == "serial":
                            port = init.stencil.output
                            if service.input_ready and not cache_requested and port.pending is None:
                                cache_requested = port.request(cell + 4 * accepted, phase * 8192, c)
                            reply = port.take(c)
                            if reply is not None:
                                assert reply[0] == accepted and service.issue(reply[1], None)
                                cache_requested = False
                        service.step()
                    assert c < 100_000 and pos < len(words) and source.live <= 512
                rows.append(
                    dict(
                        frame_type=ft,
                        clock_hz=hz,
                        variant=variant,
                        status=status,
                        accepted_known_words=accepted,
                        known_words=n,
                        terminal_edge=c,
                        input_samples=pos - origin,
                        source_sha256=sha(file),
                        history_peak_live=source.stats["peak_live"],
                        initial_snapshot=snapshot,
                        hold_edges=None if snapshot is None else c - epoch,
                        actual_word_states_exact=None if service is None else True,
                        entire_known_sync_verified=status == "hold_ready",
                        sink="timing results only,always ready",
                        normalization_carrier_header_decoder_cycles=None,
                        no_MAC_delivery=True,
                        complete_latency_cycles=None,
                        full_receiver16MSps=False,
                    )
                )
    write(out / "periodic-prefixes.json", rows)
    return rows


def run(a):
    policy = json.loads(POLICY.read_text())
    for split in ["development", "validation"]:
        validate_profiles(policy[split]["profiles"], policy[split]["pid"])
        t = policy[split]["payload_transform"]
        assert len(t) == 3 and all(type(x) is int and 0 <= x < 256 for x in t) and t[0] % 2
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
    frozen = json.loads(a.freeze.read_text()) if a.freeze else None
    if frozen:
        assert frozen["source"] == source and frozen["policy"] == policy
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    write(
        out / "protocol.json",
        dict(
            source=source,
            policy=policy,
            split=a.split,
            upstream="unchanged whole-record max fixture,not bounded capture producer",
            independent_pulse_reference=True,
            truth_and_payload="postdecision only",
        ),
    )
    rows = fresh(out, a.split, policy, partitions=frozen is not None, pair_fn=pair)
    regressions, parents = (
        fixed(out, a.timing_root, a.capture_root, a.selected_root, policy, pair_fn=pair)
        if frozen is None
        else ([], {})
    )
    prefixes = periodic_prefixes(out) if frozen is None else []
    reached = [r["comparison"] for r in rows + regressions if "comparison" in r]
    assert reached and all(r["values_states_trajectories_exact"] for r in reached)
    summary = dict(
        stage=policy["stage"],
        split=a.split,
        before_full=sum(
            r["comparison"]["before"]["payload_exact"] for r in rows if "comparison" in r
        ),
        after_full=sum(
            r["comparison"]["after"]["payload_exact"] for r in rows if "comparison" in r
        ),
        fixed_before_full=sum(
            r["comparison"]["before"]["payload_exact"] for r in regressions if "comparison" in r
        ),
        fixed_after_full=sum(
            r["comparison"]["after"]["payload_exact"] for r in regressions if "comparison" in r
        ),
        original_requests=len(regressions),
        fixed_reached=sum("comparison" in r for r in regressions),
        success_losses=sum(r["success_loss"] for r in reached),
        recoveries=sum(r["recovery"] for r in reached),
        new_false_deliveries=sum(r["false_delivery"] for r in reached),
        negative_observation_samples=sum(r["input"].get("observation_samples", 0) for r in rows),
        node_value_eligible=True,
        periodic_prefixes=prefixes,
        parent_manifests=parents,
        all_RX_gates=policy["receiver_gates"],
        default_changed=False,
        old_RRC_interpolation16KiB_controller2048_and_node4096_failures_retained=True,
    )
    write(out / "model.json", summary)
    write(
        out / "failure-associations.json",
        [
            r
            for r in rows + regressions
            if "comparison" not in r or not r["comparison"]["after"]["payload_exact"]
        ],
    )
    if frozen is None:
        write(out / "freeze.json", dict(source=source, policy=policy, parent_manifests=parents))
    assert all(sha(Path(k)) == v for k, v in source.items())
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return 1  # Four whole-RX gates remain open,even if the selected timing node passes.


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["development", "validation"], default="development")
    p.add_argument("--freeze", type=Path)
    p.add_argument("--timing-root", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--selected-root", type=Path)
    raise SystemExit(run(p.parse_args()))
