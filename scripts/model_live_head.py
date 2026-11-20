"""Actual head-owner integration and bounded suffix resource exits; expected exit1.

Viewed fixtures are fixed regressions only. Word-port calendars are distinct from
continuous ADC/RRC/acquisition and from the unmodeled payload decoder service.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from model_hard_header import plain
from model_header_words import source_identity
from tests.test_b1_wire_length import BASELINE, ROOT, profile
from tests.test_head_word_replay import run as replay
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input

from nearlink_sdr.common import polar
from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay
from nearlink_sdr.sim.overlapped_normalization_replay import OverlappedNormalizationFrameReplay

POLICY = Path("tests/data/live_head_policy.json")
TRACE = ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error")
FIELDS = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")
GEOMETRY_FIELDS = (
    "header_end_symbol",
    "data_start_symbol",
    "frame_end_symbol",
    "coded_bits",
    "data_symbols",
    "wire_data_symbols",
    "payload_length_bytes",
    "data_crc_present",
)


def selected(profile, values):
    """One actual differential maximum; no payload,CRC or truth supplied."""
    found = search_psk_sync(values, 4, profile.sync_symbols(), 8)
    if found is None:
        raise ValueError("Fixed regression no longer has its upstream sync candidate")
    i, n = found["offset"], len(profile.sync_symbols())
    return dict(
        start_tick=round(8 * (found["phase"] + 4 * i)),
        delta=found["delta"],
        differential_score=found["score"],
        coherent_threshold=0.45,
        selected_sync=found["phase_symbols"][i : i + n],
        sample_limit=len(values),
    )


def cases():
    """Historical selected-port fixtures plus six SHA-frozen standard B1/A2 inputs."""
    for ft, mcs, pilot, length in [(2, 8, 8, 19), (3, 12, 4, 47), (4, 10, 16, 23)]:
        p = ordinary(ft, mcs, pilot)
        values, expected, selection = selected_input(p, length=length)
        yield (
            f"legacy-ft{ft}",
            p,
            values,
            expected,
            selection,
            dict(kind="viewed selected-port regression", length_contract=p.length_contract),
        )
    for row in BASELINE["rows"]:
        path = ROOT / f"ft{row['frame_type']}-crc{row['CRC_width']}-input.npz"
        if sha(path) != row["input_sha256"]:
            raise ValueError("Original frozen standard-length input SHA differs")
        with np.load(path) as z:
            iq, expected = z["iq"].copy(), z["payload"].tobytes()
        p = profile(row["frame_type"], row["CRC_width"])
        values = np.convolve(iq, rrc_filter(0.4, 4))
        yield (
            f"standard-ft{row['frame_type']}-crc{row['CRC_width']}",
            p,
            values,
            expected,
            selected(p, values),
            dict(
                kind="original SHA-frozen standard-length regression",
                path=str(path),
                original_SHA256=row["input_sha256"],
                length_contract=p.length_contract,
            ),
        )


def save_event(output, name, event):
    write(output / (name + ".json"), plain({k: v for k, v in event.items() if k not in TRACE}))
    np.savez_compressed(output / (name + "-trace.npz"), **{k: event[k] for k in TRACE})


def functional(output, name, p, words, selection, expected):
    before, old, barriers = replay(
        p,
        words,
        selection,
        partial(OverlappedNormalizationFrameReplay, timing_divider="bounded-prefix"),
    )
    save_event(output, name + "-reference", before)
    result = []
    patterns = [
        ("32", (32,)),
        ("1-7-31", (1, 7, 31)),
        ("random-1-32", tuple(map(int, np.random.default_rng(265031).integers(1, 33, 23)))),
    ]
    for layout in ("dense", "depth-parity"):
        first_calendar = None
        for label, pattern in patterns:
            calendar = []
            cls = partial(HeadWordsFrameReplay, head_layout=layout, head_observer=calendar.append)
            event, flow, pending = replay(p, words, selection, cls, pattern)
            save_event(output, f"{name}-{layout}-{label}", event)
            for field in FIELDS + GEOMETRY_FIELDS:
                assert event["frame"].get(field) == before["frame"].get(field), (name, field)
            for field in TRACE:
                np.testing.assert_array_equal(event[field], before[field])
            assert flow.consumer.word_timing.snapshot() == old.consumer.word_timing.snapshot()
            assert flow.consumer.normalization.snapshot() == old.consumer.normalization.snapshot()
            assert "header" not in pending and "header" in barriers
            assert flow.stats["finite_head_completions"] == 1
            assert flow.head_owner.header.stats["accepted"] == 1
            assert flow.stats["functional_resolutions"] == old.stats["functional_resolutions"] - 1
            assert event["frame"]["payload"] == expected
            if first_calendar is None:
                first_calendar = calendar
                write(output / f"{name}-{layout}-head-calendar.json", calendar)
            else:
                assert calendar == first_calendar
            # Verify exact suffix words as well as source indices at ownership return.
            assert flow.release_source() is flow.history
            for i in range(flow.history.head, flow.history.end):
                assert flow.history.words[i % flow.history.capacity] == words[i]
            assert flow.history.end == old.history.end and flow.history.head == old.history.head
            mac = flow.take_mac_result().packets
            assert [packet.mac_payload for packet in mac] == [expected]
            assert not flow.take_mac_result().packets
            fields = event["frame"]
            result.append(
                dict(
                    id=name,
                    layout=layout,
                    group=label,
                    frame={k: fields.get(k) for k in FIELDS},
                    geometry={k: fields.get(k) for k in GEOMETRY_FIELDS},
                    frame_end=fields["frame_end_symbol"],
                    payload_length_bytes=fields["payload_length_bytes"],
                    payload_bits=8 * len(expected),
                    payload_bit_errors=0,
                    head_and_data_CRC_checked=True,
                    exact_trajectories=True,
                    source_suffix_exact=True,
                    source_end=flow.history.end,
                    source_retired_before=flow.history.head,
                    only_payload_functional_barrier=pending == ["payload"],
                    complete_latency_cycles=None,
                    finite_head_edges=flow.stats["finite_head_edges"],
                    cost=flow.estimate(),
                    actual_epoch_at_completion=flow.cycle_evidence_valid,
                )
            )
    return plain(result)


def suffix(output, p, words, selection, *, input_id, layout, clock, continue_data,
           timing_divider="bounded-prefix", flow_class=HeadWordsFrameReplay):
    """Finite word-port arrival after the head boundary; earlier source was pausable."""
    prepared, pos = prepare(p, words, selection)
    calendar = []
    flow = flow_class(
        p,
        prepared,
        normalization_format=NormalizationFormat(),
        word_timing_format=TimingWordFormat(30, 32),
        head_layout=layout,
        head_observer=calendar.append,
        timing_divider=timing_divider,
        reuse_stencil=True,
        **selection,
    )
    while flow.mode != "word_header":
        assert not flow.done and flow.pending_service is None
        if flow.wants_replay_input:
            used = flow.feed_replay_words(words[pos : pos + 32])
            assert used == 1
            pos += used
        flow.step()
        assert flow.cycle < 1_000_000
    start, origin, initial_live = flow.cycle, pos, flow.history.live
    refusals, head_receipt, head_edge = [], None, None
    while not flow.done:
        last_active_mode = flow.mode
        due = start + ((pos - origin) * clock + 15_999_999) // 16_000_000
        if flow.cycle >= due:
            assert pos < len(words), "Diagnostic source must include the finite suffix under test"
            if flow.write_input(words[pos]):
                pos += 1
            else:
                refusals.append(dict(index=pos, cycle=flow.cycle))
        flow.step()
        if head_receipt is None and flow.stats["finite_head_completions"]:
            head_receipt = dict(flow.consumer._active["header"])
            head_edge = flow.cycle
        if not continue_data and head_receipt is not None:
            break
        assert flow.cycle - start < 20000
    frame = None if flow.event is None else flow.event["frame"]
    for i in range(flow.history.head, flow.history.end):
        assert flow.history.words[i % 512] == words[i]
    assert flow.history.end == pos and flow.history.stats["peak_live"] <= 512
    if flow.done:
        assert frame["status"] == "history_overflow" and len(refusals) == 1
        assert frame["data_crc"] is None and frame["payload"] is None
        assert frame["header_crc"] is (None if head_receipt is None else True)
        assert not flow.take_mac_result().packets
    scope = "head-and-data" if continue_data else "head-only"
    row = dict(
        input_id=input_id,
        source_words_SHA256=hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
        layout=layout,
        clock_Hz=clock,
        scope=scope,
        continuous_input_begins_at="head wait only;previous source was pausable",
        continuous_input_domain="matched-filtered46bit word port;RRC not included",
        input_rate_Hz=16000000,
        start_cycle=start,
        source_start=origin,
        initial_live=initial_live,
        source_capacity_words=flow.history.capacity,
        source_end=flow.history.end,
        source_retired_before=flow.history.head,
        source_live=flow.history.live,
        source_peak=flow.history.stats["peak_live"],
        edges=flow.cycle - start,
        checked_head_receipt=head_receipt,
        checked_head_edge=head_edge,
        frame=frame,
        resource_refusals=refusals,
        unaccepted_input_suffix_words=len(words) - pos,
        stops_at_first_resource_exit=flow.done,
        exact_source_suffix=True,
        pending_service=flow.pending_service,
        actual_epoch_valid=flow.cycle_evidence_valid,
        actual_head_statistics=flow.head_owner.header.stats.copy(),
        last_active_mode=last_active_mode,
        timing_state_commits=flow.consumer.word_timing.count,
        normalized_state_commits=flow.consumer.normalization.count,
        symbols_stored=(
            len(flow.event["symbols"]) if flow.done else flow.consumer._active["count"]
        ),
        timing_state=flow.consumer.word_timing.snapshot(),
        normalization_state=flow.consumer.normalization.snapshot(),
        controller_statistics=flow.stats.copy(),
        cost=flow.estimate(),
        full_RX16MSps=False,
    )
    prefix = f"{input_id}-{layout}-{clock}-{scope}"
    write(output / (prefix + "-head-calendar.json"), calendar)
    if flow.event is not None:
        save_event(output, prefix + "-resource-event", flow.event)
    np.savez_compressed(
        output / (prefix + "-retained.npz"),
        indices=np.arange(flow.history.head, flow.history.end, dtype=np.int64),
        words=np.asarray(
            [flow.history.words[i % 512] for i in range(flow.history.head, flow.history.end)],
            dtype=np.uint64,
        ),
    )
    return plain(row)


def run(output):
    policy = json.loads(POLICY.read_text())
    if output.exists():
        raise ValueError("Use a fresh evidence directory")
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        dict(
            policy=policy,
            source=source_identity(),
            frozen_standard_fixture_metadata_SHA256=sha(ROOT / "baseline.json"),
            new_RF_parameter_selection=False,
            new_RF_holdout=False,
            mathematical_or_loop_parameters_changed=False,
        ),
    )
    rows, inputs, workloads = [], [], {}
    for name, p, values, expected, selection, provenance in cases():
        # Explicit derived representation,not a change to frozen original IQ.
        words = quantize_input(np.r_[values, np.zeros(8000)])
        selection = selection | dict(sample_limit=len(words))
        np.savez_compressed(
            output / (name + "-input.npz"),
            words=words,
            payload=np.frombuffer(expected, np.uint8),
            **selection,
        )
        descriptor = asdict(p)
        descriptor["config"] = {
            k: v for k, v in descriptor["config"].items() if not k.startswith("_")
        }
        inputs.append(
            dict(
                id=name,
                provenance=provenance,
                profile=plain(descriptor),
                derived_input_SHA256=sha(output / (name + "-input.npz")),
                word_bytes_SHA256=hashlib.sha256(
                    np.asarray(words, dtype="<u8").tobytes()
                ).hexdigest(),
            )
        )
        rows.extend(functional(output, name, p, words, selection, expected))
        write(output / "functional.json", rows)
        write(output / "inputs.json", inputs)
        if name in policy["resource_inputs"]:
            workloads[name] = (p, words, selection)
    resources = []
    assert set(workloads) == set(policy["resource_inputs"])
    for name, workload in workloads.items():
        for layout in policy["head_layouts"]:
            for clock in policy["clocks_Hz"]:
                for continue_data in (False, True):
                    resources.append(
                        suffix(
                            output,
                            *workload,
                            input_id=name,
                            layout=layout,
                            clock=clock,
                            continue_data=continue_data,
                        )
                    )
                    write(output / "resource-prefixes.json", resources)
    assert len(rows) == 54 and len(inputs) == 9
    assert all(
        r["cost"]["head_controller_increment_bits"] <= policy["controller_increment_bits_max"]
        for r in rows
    )
    assert all(r["cost"]["new_raw_history"] == 0 for r in rows)
    assert all(not r["cost"]["whole_receiver16MSps"] for r in rows)
    assert all(r["source_capacity_words"] == policy["history_capacity_words"] for r in resources)
    complete = [r for r in resources if r["scope"] == "head-and-data"]
    assert len(complete) == 12 and all(r["frame"]["status"] == "history_overflow" for r in complete)
    summary = dict(
        stage=policy["stage"],
        actual_reference_rust=polar._HAS_RUST_ACCEL,
        no_Rust_head_word_implementation=True,
        input_cases=9,
        exact_comparisons=54,
        input_and_source_suffix_exact=True,
        recovered=0,
        original_success_losses=0,
        frozen_numeric_failures_changed=False,
        component_functional_pass=True,
        head_wait_completed=sum(
            r["checked_head_receipt"] is not None for r in resources if r["scope"] == "head-only"
        ),
        head_wait_resource_exits=sum(
            r["frame"] is not None for r in resources if r["scope"] == "head-only"
        ),
        head_and_data_resource_exits=len(complete),
        resource_inputs=list(workloads),
        source_ports_increment=0,
        source_RAM_increment_bits=0,
        controller_increment_bits=129,
        complete_latency_cycles=None,
        whole_receiver_resources_passed=False,
        four_RX_gates_passed=False,
        expected_exit=1,
        synthesis=False,
        new_RF_holdout=False,
    )
    write(output / "model.json", summary)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    run(a.output)
    raise SystemExit(1)
