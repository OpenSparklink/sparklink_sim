"""Finite timing in real cache/source consumption;strict frame/MAC gates unchanged."""

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay
from tests.test_selected_frame_replay import prepare, replay
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def consume(profile, words, selection, fmt, chunks=(32,)):
    initialized, pos = prepare(profile, words, selection)
    observed = []
    flow = SelectedFrameReplay(
        profile,
        initialized,
        reuse_stencil=True,
        word_timing_format=fmt,
        observer=observed.append,
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
                    n = int(chunks[j % len(chunks)])
                    used = flow.feed_replay_words(words[pos : pos + n])
                    assert used == 1
                    pos += used
                    j += 1
            flow.step()
        assert flow.cycle < 2000000
    return flow, observed


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 19), (3, 12, 8, 47), (4, 10, 4, 23)])
def test_finite_timing_real_words_to_MAC_without_answer_search(ft, mcs, pilot, length):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    baseline, _, _ = replay(profile, words, selection, reuse=True)
    for fmt in [TimingWordFormat(), TimingWordFormat(30, 32), TimingWordFormat(38, 32)]:
        flow, observed = consume(profile, words, selection, fmt)
        e = flow.event
        assert e["frame"]["payload"] == payload and e["frame"]["status"] == "valid"
        for k in ["header_crc", "data_crc", "length_ok"]:
            assert e["frame"][k] is True
        assert e["frame"]["mcs"] == mcs
        assert len(flow.take_mac_result().packets) == 1
        assert not flow.take_mac_result().packets
        delta = np.max(abs(e["symbol_centers_filtered"] - baseline["symbol_centers_filtered"]))
        assert delta < 0.001
        assert flow.consumer.reference_memory()["timing_power_TED_PI_NCO_finite"]
        assert flow.consumer.reference_memory()["remaining_carrier_normalization_and_decoder_float"]
        assert e["word_timing_state"]["count"] == e["frame"]["frame_end_symbol"]
        assert e["replay_transport"]["complete_latency_cycles"] is None
        assert e["replay_transport"]["cycle_evidence_valid"] is False
        assert len(observed) > len(e["symbols"])


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_bad_frame_strict_gate_no_CRC_driven_retry(damage, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, length=13, **damage)
    words = quantize_input(np.r_[values, np.zeros(500)])
    selection["sample_limit"] = len(words)
    flow, _ = consume(profile, words, selection, TimingWordFormat())
    assert flow.event["frame"]["status"] == status
    assert not flow.take_mac_result().packets
    assert flow.stats["header_operations"] == 1
    assert flow.stats["payload_operations"] == int(status == "bad_data_crc")


def test_word_query_partition_exact_and_truncation_release_owned_source():
    profile = ordinary(4, 12, 0)
    values, _, selection = selected_input(profile, length=49)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    a, oa = consume(profile, words, selection, TimingWordFormat())
    b, ob = consume(profile, words, selection, TimingWordFormat(), (1, 7, 31, 2, 19))
    assert oa == ob
    assert a.event["word_timing_state"] == b.event["word_timing_state"]
    for k in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(a.event[k], b.event[k])
    end = int(np.ceil(selection["start_tick"] / 8 + 448 * 4))
    cut = words[:end]
    truncated, _ = consume(
        profile, cut, selection | dict(sample_limit=len(cut) + 100), TimingWordFormat()
    )
    assert truncated.event["frame"]["status"] == "truncated_data"
    assert not truncated.take_mac_result().packets
    assert truncated.release_source() is truncated.history
