"""Same actual selected words and strict MAC result through bounded hold lookahead."""

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.pipelined_timing_replay import PipelinedTimingFrameReplay
from tests.test_finite_timing_replay import FMT, value_trace
from tests.test_finite_timing_replay import run as baseline
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def run(profile, words, selection, pattern=(32,)):
    prepared, pos = prepare(profile, words, selection)
    seen = []
    flow = PipelinedTimingFrameReplay(
        profile,
        prepared,
        word_timing_format=FMT,
        reuse_stencil=True,
        observer=seen.append,
        **selection,
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
                    pos += flow.feed_replay_words(words[pos : pos + pattern[j % len(pattern)]])
                    j += 1
            flow.step()
        assert flow.cycle < 2_000_000
    return flow, seen


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (3, 12, 16), (4, 10, 8)])
def test_unknown_float_work_and_strict_frame_stay_identical_after_exact_hold_pipeline(
    ft, mcs, pilot
):
    profile = ordinary(ft, mcs, pilot)
    iq, payload, selection = selected_input(profile, length=29)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    a, oa = baseline(profile, words, selection)
    b, ob = run(profile, words, selection)
    c, oc = run(profile, words, selection, (1, 7, 31, 3))
    assert value_trace(oa) == value_trace(ob) == value_trace(oc)
    assert b.event["frame"] == a.event["frame"] == c.event["frame"]
    assert b.event["frame"]["payload"] == payload
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(a.event[key], b.event[key])
        np.testing.assert_array_equal(b.event[key], c.event[key])
    assert a.event["word_timing_state"] == b.event["word_timing_state"]
    h = len(profile.sync_symbols())
    assert b.stats["cache_reads"] == b.timing_service.hold.stats["cache_returns"] == h
    assert b.stats["query_replies"] == a.stats["query_replies"] - h
    assert b.event["known_sync_pipeline"]["accepted"] == h
    assert b.event["finite_timing_services"]["complete_latency_cycles"] is None
    assert not b.event["replay_transport"]["cycle_evidence_valid"]
    assert b.take_mac_result().packets[0].mac_payload == payload
    assert b.timing_service.normal.stats["aborted"] == b.timing_service.hold.stats["aborted"] == 0
    assert b.estimate()["timing_service"]["DSP25x18"] == 4


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_bad_frames_are_not_retried_and_no_payload_reaches_MAC(damage, status):
    p = ordinary()
    iq, _, selection = selected_input(p, length=23, **damage)
    words = quantize_input(iq)
    a, _ = baseline(p, words, selection)
    b, _ = run(p, words, selection)
    assert a.event["frame"] == b.event["frame"] and b.event["frame"]["status"] == status
    assert not b.take_mac_result().packets
    assert b.consumer.word_timing.count == len(b.event["symbols"])


def test_EOF_and_blocked_normalization_keep_future_guard_and_credits():
    p = ordinary(4, 12, 0)
    iq, _, selection = selected_input(p, length=49)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    cut = selection["start_tick"] // 8 + 4 * 448
    a, _ = baseline(p, words[:cut], selection)
    b, _ = run(p, words[:cut], selection)
    assert a.event["frame"] == b.event["frame"] and b.event["frame"]["status"] == "truncated_data"
    prepared, _ = prepare(p, words, selection)
    flow = PipelinedTimingFrameReplay(p, prepared, word_timing_format=FMT, **selection)
    while flow.pending_service is None:
        flow.step()
    state = flow.consumer.word_timing.snapshot()
    for _ in range(100):
        flow.step()
    assert flow.consumer.word_timing.snapshot() == state
    hold = flow.timing_service.hold
    assert hold.requested - hold.stats["accepted"] == 4
    assert len(hold.outputs) == 2
    assert flow.stats["functional_resolutions"] == 0
    with pytest.raises(ValueError):
        flow.abort("test_downstream_block")
    flow.abort("history_overflow")
    assert not flow.take_mac_result().packets
    assert flow.event["finite_timing_services"]["cancelled_inflight"]["timing"]


def test_unsupported_format_fails_before_consuming_the_prepared_context():
    p = ordinary()
    iq, _, selection = selected_input(p, length=11)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    prepared, _ = prepare(p, words, selection)
    before = prepared.stencil.history.head
    with pytest.raises(ValueError, match="before claiming"):
        PipelinedTimingFrameReplay(
            p, prepared, word_timing_format=TimingWordFormat(38, 32), **selection
        )
    assert not prepared.fit.output_taken
    assert prepared.stencil.output.cache.read == 0 and prepared.stencil.history.head == before
    good = PipelinedTimingFrameReplay(p, prepared, word_timing_format=FMT, **selection)
    assert good.consumer.word_timing.count == 0 and prepared.fit.output_taken
