"""Real cache fanout and joint state acceptance preserve strict fields and trajectories."""

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.sim.overlapped_normalization_replay import OverlappedNormalizationFrameReplay
from tests.test_finite_timing_replay import FMT, value_trace
from tests.test_normalized_word_replay import run
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (3, 12, 16), (4, 10, 4)])
def test_actual_broadcast_preserves_all_values_fields_and_MAC_across_blocks(ft, mcs, pilot):
    p = ordinary(ft, mcs, pilot)
    iq, payload, selection = selected_input(p, length=23)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    a, oa, sa = run(p, words, selection)
    b, ob, sb = run(p, words, selection, flow_type=OverlappedNormalizationFrameReplay)
    c, oc, _ = run(p, words, selection, (1, 7, 31, 3),
                   flow_type=OverlappedNormalizationFrameReplay)
    assert value_trace(oa) == value_trace(ob) == value_trace(oc)
    assert a.event["frame"] == b.event["frame"] == c.event["frame"]
    assert b.event["frame"]["payload"] == payload
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error",
                "word_timing_state", "word_normalization_state"]:
        if isinstance(a.event[key], np.ndarray):
            np.testing.assert_array_equal(a.event[key], b.event[key])
            np.testing.assert_array_equal(b.event[key], c.event[key])
        else:
            assert a.event[key] == b.event[key] == c.event[key]
    assert a.stats["normalization_requests"] == b.stats["normalization_requests"]
    assert b.stats["normalization_requests"] == b.stats["normalization_completions"]
    assert b.stats["cache_reads"] == a.stats["cache_reads"]
    hold = b.timing_service.hold
    assert hold.stats["cache_returns"] == b.consumer.word_timing.hold_symbols
    assert hold.stats["cache_requests"] == hold.stats["cache_returns"]
    assert b.normalization_service.peak_owned <= 4
    assert [r["values"] for r in sa] == [r["values"] for r in sb]
    assert b.cycle < a.cycle
    assert b.take_mac_result().packets[0].mac_payload == payload
    assert not b.event["normalization_service"]["cancelled_inflight"]
    assert b.normalization_service.pipeline_stats["aborted_contexts"] == 0
    assert b.event["finite_timing_services"]["complete_latency_cycles"] is None
    assert b.estimate()["extra_source_or_cache_read_ports"] == 0


@pytest.mark.parametrize("mode,status", [("bad_head", "bad_header_crc"),
                                         ("bad_data", "bad_data_crc")])
def test_join_does_not_retry_true_header_or_data_CRC_failures(mode, status):
    p = ordinary(3, 8, 0)
    iq, _, selection = selected_input(p, length=37, **{mode: True})
    words = quantize_input(iq)
    a, _, _ = run(p, words, selection)
    b, _, _ = run(p, words, selection, flow_type=OverlappedNormalizationFrameReplay)
    assert a.event["frame"] == b.event["frame"]
    assert b.event["frame"]["status"] == status and not b.take_mac_result().packets
    assert b.consumer.word_timing.count == b.consumer.normalization.count


@pytest.mark.parametrize("counter_owner", ["hold", "normalization", "controller"])
def test_joint_transfer_preflight_cannot_commit_only_one_owner(counter_owner):
    p = ordinary(4, 12, 0)
    iq, _, selection = selected_input(p, length=11)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    prepared, _ = prepare(p, words, selection)
    f = OverlappedNormalizationFrameReplay(p, prepared, normalization_format=NormalizationFormat(),
                                          word_timing_format=FMT, **selection)
    hold, service = f.timing_service.hold, f.normalization_service
    while hold.peek() is None or service.peek() is None:
        f.step()
    states = (f.consumer.word_timing.snapshot(), f.consumer.normalization.snapshot())
    if counter_owner == "hold":
        hold.stats["accepted"] = (1 << 64) - 1
    elif counter_owner == "normalization":
        service.stats["completed"] = (1 << 64) - 1
    else:
        f.stats["normalization_completions"] = (1 << 64) - 1
    with pytest.raises(OverflowError):
        f.step()
    assert (f.consumer.word_timing.snapshot(), f.consumer.normalization.snapshot()) == states
    assert f.consumer._active["count"] == 0
    f.abort("history_overflow")
    assert f.event["normalization_pipeline"]["statistics"]["aborted_contexts"] > 0
    assert not f.take_mac_result().packets


def test_EOF_discards_outstanding_contexts_without_delivering_truncated_payload():
    p = ordinary(4, 12, 0)
    iq, _, selection = selected_input(p, length=49)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    cut = selection["start_tick"] // 8 + 4 * 448
    a, _, _ = run(p, words[:cut], selection)
    b, _, _ = run(p, words[:cut], selection, flow_type=OverlappedNormalizationFrameReplay)
    assert b.event["frame"] == a.event["frame"]
    assert b.event["frame"]["status"] == "truncated_data" and not b.take_mac_result().packets
    assert b.normalization_service.owned == 0
