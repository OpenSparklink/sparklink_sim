"""Actual word/query ownership and source ports;remaining arithmetic stays explicit."""

import math

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.finite_timing_replay import FiniteTimingFrameReplay
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_timing_word_rx import consume

FMT = TimingWordFormat(30, 32)


def run(profile, words, selection, pattern=(32,)):
    prepared, pos = prepare(profile, words, selection)
    observations = []
    flow = FiniteTimingFrameReplay(
        profile, prepared, word_timing_format=FMT, reuse_stencil=True,
        observer=observations.append, **selection,
    )
    j = 0
    while not flow.done:
        if flow.pending_service is not None:
            flow.resolve_functional()
        else:
            if flow.wants_replay_input:
                if pos == len(words):
                    flow.end_input()
                else:
                    pos += flow.feed_replay_words(words[pos:pos + int(pattern[j % len(pattern)])])
                    j += 1
            flow.step()
        assert flow.cycle < 2_000_000
    return flow, observations


def value_trace(observations):
    return [{k: row[k] for k in ["symbol", "role", "cell", "phase", "word"]}
            for row in observations]


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (3, 12, 8), (4, 10, 4)])
def test_real_services_match_functional_node_without_recomputing_state_or_queries(
    ft, mcs, pilot, monkeypatch
):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=19)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    baseline, before = consume(profile, words, selection, FMT)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Functional timing/query recomputation is forbidden")

    monkeypatch.setattr(WordTimingState, "advance_words", forbidden)
    monkeypatch.setattr(WordTimingState, "queries", property(forbidden))
    monkeypatch.setattr(WordTimingState, "required_end", property(forbidden))
    flow, observed = run(profile, words, selection)
    assert value_trace(before) == value_trace(observed)
    assert flow.event["frame"]["payload"] == payload
    assert flow.event["word_timing_state"] == baseline.event["word_timing_state"]
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(flow.event[key], baseline.event[key])
    count = flow.consumer.word_timing.count
    assert flow.stats["timing_requests"] == flow.stats["timing_completions"] == count
    assert flow.stats["symbol_operations"] == count
    assert flow.stats["query_replies"] == flow.stats["query_requests"] == count
    assert flow.timing_service.stats["completed"] == count
    assert flow.timing_service.stats["aborted"] == flow.query_service.stats["aborted"] == 0
    assert flow.timing_service.mode == flow.query_service.mode == "released"
    assert flow.estimate()["controller_increment_over_previous_bits"] == 644
    assert flow.estimate()["new_controller_state_bits"] == 2090
    assert flow.event["finite_timing_services"]["complete_latency_cycles"] is None
    assert not flow.event["replay_transport"]["cycle_evidence_valid"]
    assert flow.take_mac_result().packets[0].mac_payload == payload
    assert not flow.take_mac_result().packets
    assert flow.release_source() is flow.history and flow.release_source() is None


@pytest.mark.parametrize("damage,status", [
    ({"bad_head": True}, "bad_header_crc"),
    ({"bad_data": True}, "bad_data_crc"),
    ({"empty": True}, "invalid_length"),
])
def test_strict_frame_failures_are_retained_and_services_release(damage, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, length=17, **damage)
    words = quantize_input(values)
    flow, _ = run(profile, words, selection)
    assert flow.event["frame"]["status"] == status
    assert not flow.take_mac_result().packets
    assert flow.stats["header_operations"] == 1
    assert flow.stats["payload_operations"] == int(status == "bad_data_crc")
    assert flow.consumer.word_timing.count == len(flow.event["symbols"])
    assert flow.event["finite_timing_services"]["stored_symbols"] == len(flow.event["symbols"])


def test_partition_and_EOF_paths_keep_exact_query_and_sample_order():
    profile = ordinary(4, 12, 0)
    values, _, selection = selected_input(profile, length=49)
    words = quantize_input(values)
    a, oa = run(profile, words, selection)
    b, ob = run(profile, words, selection, (1, 7, 31, 3, 19))
    assert oa == ob and a.event["word_timing_state"] == b.event["word_timing_state"]
    cut = selection["start_tick"] // 8 + 4 * 448
    eof, _ = run(profile, words[:cut], selection)
    assert eof.event["frame"]["status"] == "truncated_data"
    assert eof.eof and not eof.take_mac_result().packets
    assert eof.kernel.live == 0 and eof.history.pending is None


def test_timing_result_waits_for_remaining_float_work_and_cannot_commit_twice():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=11)
    words = quantize_input(np.r_[values, np.zeros(1500)])
    selection["sample_limit"] = len(words)
    prepared, pos = prepare(profile, words, selection)
    with pytest.raises(ValueError):
        FiniteTimingFrameReplay(profile, prepared, word_timing_format=None, **selection)
    assert prepared.ready
    flow = FiniteTimingFrameReplay(profile, prepared, word_timing_format=FMT, **selection)
    with pytest.raises(RuntimeError):
        flow.consumer.claim_timing_state()
    with pytest.raises(RuntimeError):
        flow.consumer.push_symbol(0, None, after_samples=flow.history.end)
    with pytest.raises(RuntimeError):
        flow.resolve_functional()
    while flow.pending_service is None:
        flow.step()
    assert flow.pending_service["kind"] == "symbol_normalization"
    assert flow.consumer.word_timing.count == 1 and flow.consumer._active["count"] == 0
    saved = flow.consumer.word_timing.snapshot()
    for _ in range(23):
        flow.step()
    assert flow.consumer.word_timing.snapshot() == saved
    with pytest.raises(RuntimeError):
        flow.consumer.accept_timing_result(flow.words[0] ^ 1, after_samples=flow.history.end)
    flow.resolve_functional()
    assert flow.consumer.word_timing.snapshot() == saved
    with pytest.raises(RuntimeError):
        flow.consumer.accept_timing_result(saved.get("previous_word", 0),
                                          after_samples=flow.history.end)
    # Continuous16MS/s from this prepared takeover,not proof of earlier ADC admission.
    origin, epoch = pos, flow.cycle
    while not flow.done:
        c = flow.cycle
        if c == epoch + math.ceil((pos - origin) * 200_000_000 / 16_000_000):
            flow.write_input(words[pos])
            pos += 1
        flow.step()
    assert flow.event["frame"]["status"] == "history_overflow"
    assert flow.history.live <= 512 and flow.stats["input_overflow"] == 1
    assert not flow.take_mac_result().packets
    assert flow.event["finite_timing_services"]["complete_latency_cycles"] is None


def test_stale_query_state_is_rejected_before_any_timing_issue():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=3)
    words = quantize_input(values)
    prepared, _ = prepare(profile, words, selection)
    flow = FiniteTimingFrameReplay(profile, prepared, word_timing_format=FMT, **selection)
    flow.consumer.word_timing.count = 1  # Deliberate ownership violation after snapshot issue.
    for _ in range(3):
        flow.step()
    with pytest.raises(RuntimeError, match="expired timing state"):
        flow.step()
    assert flow.stats["stale_query_rejections"] == 1
    assert flow.stats["timing_requests"] == 0
    flow.abort()
    assert not flow.take_mac_result().packets


def test_normal_timing_release_requires_drained_output_and_never_counts_work_abort():
    from nearlink_sdr.sim.timing_word_flow import TimingWordService

    state = WordTimingState((1, 0), 96, 32)
    service = TimingWordService(state)
    assert service.issue(0, None)
    with pytest.raises(RuntimeError, match="drain"):
        service.release()
    while not service.ready:
        service.step()
    with pytest.raises(RuntimeError, match="drain"):
        service.release()
    service.take()
    service.release()
    service.abort()
    assert service.stats["aborted"] == 0
    with pytest.raises(RuntimeError, match="Closed"):
        service.issue(0, None)


def test_observer_mutation_is_isolated_and_cache_identity_guard_precedes_request():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=3)
    words = quantize_input(values)

    def mutate(record):
        if record["kind"] == "query":
            record["values"]["queries"][0]["phase"] ^= 1

    prepared, _ = prepare(profile, words, selection)
    flow = FiniteTimingFrameReplay(
        profile, prepared, word_timing_format=FMT, service_observer=mutate, **selection
    )
    for _ in range(4):
        flow.step()
    assert flow.queries[0]["phase"] == flow.cache.cache.phase
    flow.step()
    assert flow.cache_requested
    flow.abort()
    prepared, _ = prepare(profile, words, selection)
    other = FiniteTimingFrameReplay(profile, prepared, word_timing_format=FMT, **selection)
    for _ in range(4):
        other.step()
    other.queries[0]["phase"] ^= 1  # Deliberate internal identity violation.
    with pytest.raises(RuntimeError, match="disagrees with sealed sync cache"):
        other.step()
    assert not other.cache_requested and other.stats["cache_reads"] == 0
    other.abort()
