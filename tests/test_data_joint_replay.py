"""Exact paired data services, bounded owners and rejection before partial transfer."""

import importlib
import json
from functools import partial

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.data_joint_replay import DataJointFrameReplay
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay
from tests.test_b1_wire_length import b1_header_wire, profile
from tests.test_head_word_replay import run
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_strict_receive import iq_record

TRACE = ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error")


def at_data_issue(cls=DataJointFrameReplay, *, ft=3, mcs=12, pilot=4):
    p = ordinary(ft, mcs, pilot)
    values, _, selection = selected_input(p, length=47)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    prepared, pos = prepare(p, words, selection)
    flow = cls(
        p, prepared, normalization_format=NormalizationFormat(),
        word_timing_format=TimingWordFormat(30, 32), reuse_stencil=True, **selection
    )
    while flow.mode != "timing_issue":
        assert not flow.done and flow.pending_service is None
        if flow.wants_replay_input:
            pos += flow.feed_replay_words(words[pos:pos + 32])
        flow.step()
    assert flow.timing_service.normal is None
    return flow


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 8, 19), (3, 12, 4, 47), (4, 10, 16, 23)])
@pytest.mark.parametrize("layout", ["dense", "depth-parity"])
def test_same_frame_state_source_and_words_with_one_joint_pair_per_data_symbol(
        ft, mcs, pilot, length, layout):
    p = ordinary(ft, mcs, pilot)
    values, expected, selection = selected_input(p, length=length)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    a, old, _ = run(p, words, selection, partial(HeadWordsFrameReplay, head_layout=layout))
    records = []
    b, new, barriers = run(p, words, selection, partial(
        DataJointFrameReplay, head_layout=layout, service_observer=records.append
    ), (1, 7, 31))
    assert a["frame"] == b["frame"] and b["frame"]["payload"] == expected
    for key in TRACE:
        np.testing.assert_array_equal(a[key], b[key])
    assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
    assert old.consumer.normalization.snapshot() == new.consumer.normalization.snapshot()
    assert old.history.end == new.history.end and old.history.head == new.history.head
    for i in range(new.history.head, new.history.end):
        assert new.history.words[i % 512] == words[i]
    hold = new.consumer.word_timing.hold_symbols
    timings = {r["values"]["count"] - 1: r for r in records if r["kind"] == "timing"}
    normal = {r["values"]["index"]: r for r in records if r["kind"] == "normalization"}
    for k in range(hold, len(b["symbols"])):
        assert timings[k]["issued_edge"] == normal[k]["issued_edge"]
        assert timings[k]["taken_edge"] == normal[k]["taken_edge"]
    pairs = new.stats["data_joint_completions"]
    assert pairs == new.stats["data_joint_requests"] == len(b["symbols"]) - hold
    assert old.cycle - new.cycle == 24 * pairs
    assert [x.mac_payload for x in new.take_mac_result().packets] == [expected]
    assert barriers == ["payload"] and not new.cycle_evidence_valid
    old_bits, new_bits = [x.estimate()["new_controller_state_bits"] for x in (old, new)]
    assert new_bits - old_bits == 128
    assert new.estimate()["data_joint_additional_DSP"] == 0
    assert new.estimate()["data_joint_pair_credits"] == 1


@pytest.mark.parametrize("damage,status", [
    ({"bad_head": True}, "bad_header_crc"),
    ({"bad_data": True}, "bad_data_crc"),
    ({"empty": True}, "invalid_length"),
])
def test_real_CRC_failures_and_illegal_length_remain_failures(damage, status):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, **damage)
    event, flow, _ = run(p, quantize_input(values), selection, DataJointFrameReplay)
    assert event["frame"]["status"] == status
    assert not flow.take_mac_result().packets
    assert flow.head_owner.header.stats["accepted"] == 1


def test_truncated_header_keeps_unchecked_CRC_and_never_redecodes(monkeypatch):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p)
    words = quantize_input(values)
    cut = selection["start_tick"] // 8 + 4 * (len(p.sync_symbols()) + 8)
    selection = selection | dict(sample_limit=cut)

    def forbidden(*args, **kwargs):
        pytest.fail("EOF may not rerun a floating header decoder")

    monkeypatch.setattr("nearlink_sdr.phy.selected_rx.decode_psk_frame", forbidden)
    event, flow, _ = run(p, words[:cut], selection, DataJointFrameReplay)
    assert event["frame"]["status"] == "truncated_header"
    assert event["frame"]["header_crc"] is None and not flow.take_mac_result().packets


def test_truncated_payload_preserves_checked_head_but_never_delivers():
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, length=99)
    cut = selection["start_tick"] // 8 + 4 * (len(p.sync_symbols()) + 80)
    selection = selection | dict(sample_limit=cut)
    event, flow, _ = run(p, quantize_input(values)[:cut], selection, DataJointFrameReplay)
    assert event["frame"]["status"] == "truncated_data"
    assert event["frame"]["header_crc"] is True
    assert event["frame"]["data_crc"] is None and event["frame"]["payload"] is None
    assert not flow.take_mac_result().packets


@pytest.mark.parametrize("ft", [3, 4])
def test_standard_zero_data_and_crc_only_keep_distinct_real_CRC_semantics(ft):
    p = profile(ft)
    wire, _ = b1_header_wire(p, 0)
    pre = np.exp(1j * np.tile([np.pi / 4, 0], {3: 24, 4: 32}[ft]))
    iq, _, _, _ = iq_record(np.r_[pre, wire])
    values = np.convolve(iq, rrc_filter(0.4, 4))
    found = search_psk_sync(values, 4, p.sync_symbols(), 8)
    i, n = found["offset"], len(p.sync_symbols())
    selection = dict(start_tick=round(8 * (found["phase"] + 4 * i)), delta=found["delta"],
                     differential_score=found["score"], coherent_threshold=0.45,
                     selected_sync=found["phase_symbols"][i:i + n], sample_limit=len(values))
    event, flow, pending = run(p, quantize_input(values), selection, DataJointFrameReplay)
    assert event["frame"]["status"] == "valid_empty_control" and not pending
    assert event["frame"]["header_crc"] is True and event["frame"]["data_crc"] is None
    assert not flow.take_mac_result().packets and flow.cycle_evidence_valid
    values, _, selection = selected_input(p, length=0)
    event, flow, pending = run(p, quantize_input(values), selection, DataJointFrameReplay)
    assert event["frame"]["status"] == "valid" and event["frame"]["payload"] == b""
    assert event["frame"]["data_crc"] is True and pending == ["payload"]


@pytest.mark.parametrize("fast_timing", [False, True])
def test_neither_half_can_start_next_query_or_store_symbol_before_the_pair(fast_timing):
    flow = at_data_issue()
    if fast_timing:
        flow.words = [0, 0]  # Diagnostic node vector;zero TED gives the shorter timing path.
    n = flow.consumer._active["count"]
    queries = flow.query_service.stats["accepted"]
    flow.step()
    assert flow.timing_service.normal.first_cycle == flow.normalization_service.last_issue_cycle
    unequal_readiness = False
    while flow.stats["data_joint_completions"] == 0:
        t, norm = flow.timing_service.normal, flow.normalization_service
        unequal_readiness |= t.ready != (norm.peek() is not None)
        assert flow.consumer._active["count"] == n
        assert flow.query_service.stats["accepted"] == queries
        assert flow.consumer.normalization.count == n
        flow.step()
    assert unequal_readiness
    assert flow.consumer._active["count"] == flow.consumer.normalization.count == n + 1
    assert flow.query_service.stats["accepted"] == queries + 1
    assert flow.consumer.word_timing.count == n + 1
    # The current data path is always at least29 edges;the shorter zero-TED
    # vector is still later than24-edge normalization,not the opposite order.
    timing = flow.timing_service.normal
    assert timing.ready_cycle - timing.first_cycle == (29 if fast_timing else 75)
    assert (flow.normalization_service.ready_cycle
            - flow.normalization_service.first_cycle) == 24
    assert flow.abort()["frame"]["header_crc"] is None


def test_delayed_normalization_return_holds_completed_timing_without_query_or_symbol_acceptance():
    baseline, flow = at_data_issue(), at_data_issue()
    for candidate in (baseline, flow):
        candidate.words = [0, 0]
        candidate.step()
    n = flow.consumer._active["count"]
    norm = flow.normalization_service
    while norm.mode != "DSP_wait":
        flow.step()
    # Explicit registered-return delay fault,not a nominal DSP-latency change,
    # additional resource budget or an RF experiment. All service clocks tick.
    norm.job["due"] += 32
    while not flow.timing_service.normal.ready:
        flow.step()
    timing = flow.timing_service.normal
    held = timing.result.copy()
    queries = flow.query_service.stats["accepted"]
    for _ in range(5):
        assert norm.peek() is None and timing.ready and timing.result == held
        assert flow.consumer._active["count"] == norm.state.count == n
        assert flow.query_service.stats["accepted"] == queries
        assert flow.head_owner.stats["word_accepts"] == 0
        flow.step()
    for candidate in (baseline, flow):
        while candidate.consumer._active["count"] == n:
            candidate.step()
    assert baseline.consumer.word_timing.snapshot() == flow.consumer.word_timing.snapshot()
    assert baseline.consumer.normalization.snapshot() == flow.consumer.normalization.snapshot()
    assert norm.ready_cycle - norm.first_cycle == 56
    assert flow.query_service.stats["accepted"] == queries + 1
    assert flow.stats["data_joint_completions"] == flow.stats["data_joint_requests"] == 1
    assert flow.head_owner.stats["word_accepts"] == 1
    assert flow.cycle_evidence_valid
    for candidate in (baseline, flow):
        assert candidate.abort()["frame"]["header_crc"] is None
        assert not candidate.take_mac_result().packets


@pytest.mark.parametrize("bad", ["normal_counter", "controller_counter", "credit", "clock"])
def test_input_preflight_rejects_before_either_bank_is_admitted(bad):
    flow = at_data_issue()
    if bad == "normal_counter":
        flow.normalization_service.stats["accepted"] = (1 << 64) - 1
    elif bad == "controller_counter":
        flow.stats["data_joint_requests"] = (1 << 64) - 1
    elif bad == "credit":
        flow.normalization_service.last_take_cycle = flow.normalization_service.cycle
    else:
        flow.normalization_service.cycle += 1
    before = flow.consumer.word_timing.snapshot(), flow.consumer.normalization.snapshot()
    with pytest.raises((RuntimeError, OverflowError)):
        flow.step()
    assert flow.timing_service.normal is None and flow.normalization_service.job is None
    assert before == (flow.consumer.word_timing.snapshot(), flow.consumer.normalization.snapshot())


@pytest.mark.parametrize("bad", ["word", "index", "count", "phase", "counter"])
def test_joint_output_rejects_stale_words_indices_and_overflow_without_partial_accept(bad):
    flow = at_data_issue()
    flow.step()
    while not (flow.timing_service.normal.ready and flow.normalization_service.peek() is not None):
        flow.step()
    norm = flow.normalization_service
    if bad == "counter":
        flow.stats["data_joint_completions"] = (1 << 64) - 1
    else:
        key = dict(word="raw_word", index="index", count="count", phase="phase_before")[bad]
        norm.registered[key] ^= 1
    head_accepts = flow.head_owner.stats["word_accepts"]
    before = flow.consumer.normalization.snapshot(), flow.consumer._active["count"]
    with pytest.raises((RuntimeError, OverflowError)):
        flow.step()
    assert before == (flow.consumer.normalization.snapshot(), flow.consumer._active["count"])
    assert flow.timing_service.normal.ready and norm.peek() is not None
    assert flow.head_owner.stats["word_accepts"] == head_accepts


def test_changed_source_is_rejected_before_opening_node_validation(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_data_joint")
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(dict(source={"incorrect": "source"},
                                     policy=json.loads(model.POLICY.read_text()))))
    out = tmp_path / "validation"
    with pytest.raises(AssertionError):
        model.run(out, freeze=freeze)
    assert not out.exists()


def test_existing_evidence_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_data_joint")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"preserve original failure")
    with pytest.raises(ValueError, match="fresh evidence"):
        model.run(tmp_path)
    assert sentinel.read_bytes() == b"preserve original failure"
