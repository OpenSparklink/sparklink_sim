"""Finite normalized words preserve actual timing/fields and share strict MAC gates."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.sim.normalized_word_replay import NormalizedWordFrameReplay
from tests.test_finite_timing_replay import FMT, value_trace
from tests.test_pipelined_timing_replay import run as baseline
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def run(profile, words, selection, pattern=(32,), *, flow_type=NormalizedWordFrameReplay):
    prepared, pos = prepare(profile, words, selection)
    seen, services = [], []
    flow = flow_type(
        profile, prepared, normalization_format=NormalizationFormat("linear"),
        word_timing_format=FMT, reuse_stencil=True, observer=seen.append,
        service_observer=services.append, **selection,
    )
    j = 0
    while not flow.done:
        if flow.pending_service is not None:
            assert flow.pending_service["kind"] in ("header", "payload")
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
    return flow, seen, services


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (3, 12, 16), (4, 10, 4)])
def test_actual_finite_rotation_gain_reaches_strict_MAC_without_timing_changes(ft, mcs, pilot):
    p = ordinary(ft, mcs, pilot)
    iq, payload, selection = selected_input(p, length=23)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    a, oa = baseline(p, words, selection)
    b, ob, services = run(p, words, selection)
    c, oc, _ = run(p, words, selection, (1, 7, 31, 3))
    assert value_trace(oa) == value_trace(ob) == value_trace(oc)
    for key in ["status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload"]:
        assert a.event["frame"].get(key) == b.event["frame"].get(key)
    assert b.event["frame"] == c.event["frame"] and b.event["frame"]["payload"] == payload
    for key in ["symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(a.event[key], b.event[key])
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(b.event[key], c.event[key])
    error = np.sqrt(np.sum(abs(a.event["symbols"] - b.event["symbols"])**2)
                    / np.sum(abs(a.event["symbols"])**2))
    assert error < 5e-5
    n = len(b.event["symbols"])
    assert b.consumer.word_timing.count == b.consumer.normalization.count == n
    assert b.stats["functional_resolutions"] == 2  # Only true header and payload barriers.
    assert b.stats["normalization_requests"] == b.stats["normalization_completions"] == n
    assert len([s for s in services if s["kind"] == "normalization"]) == n
    assert b.normalization_service.stats["native_products"] == 10 * n
    assert b.normalization_service.stats["rom_reads"] == 4 * n
    assert not b.event["normalization_service"]["cancelled_inflight"]
    assert b.event["normalization_service"]["residual_carrier_header_LLR_decoder_cycles"] is None
    assert b.event["finite_timing_services"]["complete_latency_cycles"] is None
    assert b.take_mac_result().packets[0].mac_payload == payload
    assert b.release_source() is b.history


@pytest.mark.parametrize("mode,status", [
    ("bad_head", "bad_header_crc"), ("bad_data", "bad_data_crc"),
    ("empty", "invalid_length"), ("wrong_seed", "bad_header_crc"),
    ("length", "invalid_length"), ("mcs", "unsupported_mcs"),
])
def test_real_header_CRC_length_MCS_and_payload_failures_are_not_retried(mode, status):
    p = ordinary(2 if mode == "empty" else 3, 8, 0)
    damage = {mode: True} if mode in ("bad_head", "bad_data", "empty") else {}
    if mode == "mcs":
        damage = dict(mcs=15)
    iq, _, selection = selected_input(p, length=37, **damage)
    if mode == "wrong_seed":
        p = replace(p, config=replace(p.config, head_llid=7))
    elif mode == "length":
        p = replace(p, max_payload_bytes=32)
    words = quantize_input(iq)
    f, _, _ = run(p, words, selection)
    assert f.event["frame"]["status"] == status and not f.take_mac_result().packets
    if mode in ("length", "mcs"):
        assert f.event["frame"]["header_crc"] is True
    if mode != "bad_data":
        assert f.stats["payload_operations"] == 0 and f.consumer.frame_end_symbol is None


def test_EOF_and_inflight_resource_abort_remain_distinct_and_bounded():
    p = ordinary(4, 12, 0)
    iq, _, selection = selected_input(p, length=49)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    cut = selection["start_tick"] // 8 + 4 * 448
    f, _, _ = run(p, words[:cut], selection)
    assert f.event["frame"]["status"] == "truncated_data" and not f.take_mac_result().packets
    prepared, _ = prepare(p, words, selection)
    f = NormalizedWordFrameReplay(p, prepared, normalization_format=NormalizationFormat(),
                                 word_timing_format=FMT, **selection)
    while f.mode != "word_normalize":
        f.step()
    assert f.consumer.word_timing.count == 1 and f.consumer.normalization.count == 0
    f.abort("history_overflow")
    assert f.event["normalization_service"]["cancelled_inflight"]
    assert f.event["normalization_service"]["statistics"]["aborted"] == 1
    assert not f.take_mac_result().packets


def test_invalid_format_preserves_prepared_owner_and_forged_output_cannot_store():
    p = ordinary()
    iq, _, selection = selected_input(p, length=11)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    prepared, _ = prepare(p, words, selection)
    with pytest.raises(ValueError):
        NormalizedWordFrameReplay(p, prepared, normalization_format=None,
                                  word_timing_format=FMT, **selection)
    assert not prepared.fit.output_taken
    f = NormalizedWordFrameReplay(p, prepared, normalization_format=NormalizationFormat(),
                                 word_timing_format=FMT, **selection)
    while f.normalization_service.peek() is None:
        f.step()
    result = f.normalization_service.take()
    with pytest.raises(RuntimeError):
        f.consumer.accept_normalized_timing_result(
            result | dict(normalized_word=result["normalized_word"] ^ 1),
            after_samples=f.history.end,
        )
    assert f.consumer._active["count"] == 0
    f.consumer.accept_normalized_timing_result(result, after_samples=f.history.end)
    with pytest.raises(RuntimeError):
        f.consumer.accept_normalized_timing_result(result, after_samples=f.history.end)
    assert f.consumer._active["count"] == 1
