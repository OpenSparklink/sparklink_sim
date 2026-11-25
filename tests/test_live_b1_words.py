"""Actual filtered-source clock through one B1 head, finite carrier and MAC."""

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import physical_normalized_word
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.sim.live_b1_words import replay_live_b1_words
from tests.test_psk_selected_carrier import _frame
from tests.test_selected_frame_replay import prepare
from tests.test_selected_word import quantize_input
from tests.test_strict_receive import iq_record


def fixture(ft, mcs, length, pilot, *, fault=None):
    payload = np.random.default_rng(237_409 + length).bytes(length)
    profile, normalized, _, costs, loads = _frame(
        ft, mcs, payload, pilot=pilot, width=32, bad_head=fault=="head",
        bad_data="frame" if fault=="crc" else None,
    )
    preamble = np.exp(1j * np.tile([np.pi / 4, 0], {3: 24, 4: 32}[ft]))
    symbols = np.r_[preamble, [physical_normalized_word(word) for word in normalized]]
    iq, _, truth, step = iq_record(symbols, cfo=-27000, ppm=77, offset=.491)
    if fault == "eof":
        iq = iq[:int(truth + (len(symbols)-40-100)*step)]
    filtered = np.convolve(iq, rrc_filter(.4, 4))[:len(iq)]
    match = search_psk_sync(filtered, 4, profile.sync_symbols(), 8)
    start = match["offset"]
    selection = dict(start_tick=round(8 * (match["phase"] + 4 * start)),
                     delta=match["delta"], differential_score=match["score"],
                     selected_sync=match["phase_symbols"][start:start+len(profile.sync_symbols())],
                     sample_limit=len(filtered), coherent_threshold=.45)
    words = quantize_input(filtered)
    prepared, index = prepare(profile, words, selection)
    return profile, prepared, iter(words[index:]), selection, costs, loads, payload


def replay(ft, mcs, length, pilot, *, continuous=True, clk=200, fault=None, **options):
    profile, prepared, words, selection, costs, loads, payload = fixture(
        ft, mcs, length, pilot, fault=fault,
    )
    front, events = [], []
    result = replay_live_b1_words(
        profile, prepared, words, selection=selection, clock_hz=clk*10**6,
        word_cycles=costs, input_load_cycles=loads,
        continuous=continuous, fraction_bits=1, geometry="threshold",
        observer=events.append, front_observer=front.append,
        discard_failed_suffix=True, short_tail_slot=True, short_result_slot=True,
        prefer_short_tail=True, packed_tail_slot=True, mac_ring=True,
        mac_frame_capacity=4, **options,
    )
    return result, front, events, payload


@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("ft,mcs,pilot", [(3, 4, 0), (4, 11, 4)])
def test_one_real_head_live_gain_carrier_crc_and_mac(ft, mcs, pilot, continuous):
    result, front, output, payload = replay(ft, mcs, 127, pilot, continuous=continuous)
    assert result["status"] == "delivered" and result["source_terminal_status"] == (
        "payload_transferred"
    )
    assert result["head_decodes"] == 1 and result["functional_resolutions"] == 0
    head, = [e for e in front if e["kind"] == "header_visible"]
    assert head["header_crc"] and result["header"]["length_ok"]
    assert result["header"]["mcs"] == mcs and result["header"]["payload_length_bytes"] == 127
    assert head["edge"] < result["first_data_input_edge"] < result["first_payload_launch"]
    assert result["first_payload_launch"] - result["first_data_input_edge"] == 12
    assert result["first_payload_output"] - result["first_payload_launch"] == 39
    assert result["sync_wait_peak"] == 1 and result["peak_suffix_words"] <= 8
    assert result["reference_trace_symbols"] == {3: 62, 4: 126}[ft]
    assert result["carrier_payload_outputs"] == result["header"]["wire_data_symbols"]
    assert result["wire_inputs"] == result["normalized_outputs"] == (
        result["header"]["frame_end_symbol"]
    )
    ready, = [e for e in output if e["kind"] == "crc_result"]
    assert ready["frame_crc_ok"] and ready["block_crc_ok"]
    assert bytes(e["value"] for e in output if e["kind"] == "mac_byte") == payload
    assert bool(result["normalized_output_hold_edges"]) is continuous
    assert not result["whole_receiver16MSps"]


@pytest.mark.parametrize("fault,expected", [
    ("head", "bad_header_crc"), ("crc", "payload_rejected"), ("eof", "truncated_data"),
])
def test_bad_head_bad_data_and_actual_raw_eof_never_deliver(fault, expected):
    result, _, output, _ = replay(3, 4, 127, 8, fault=fault)
    assert result["status"] == expected and not result["delivery"].get("mac_complete")
    assert result["head_decodes"] == 1 and result["functional_resolutions"] == 0
    assert not any(e["kind"] == "mac_byte" for e in output)
    if fault == "head":
        assert result["header"]["header_crc"] is False and result["first_payload_output"] is None
    elif fault == "crc":
        crc, = [e for e in output if e["kind"] == "crc_result"]
        assert not crc["frame_crc_ok"]
    else:
        assert result["carrier_payload_outputs"] < result["header"]["wire_data_symbols"]


@pytest.mark.parametrize("clock", [100, 160])
def test_continuous_source_overflow_has_raw_edge_and_no_mac(clock):
    r, _, output, _ = replay(4, 11, 255, 8, clk=clock)
    assert r["status"] == r["source_terminal_status"] == "history_overflow"
    assert r["source_history_peak"] == 512
    edge, address = r["first_rejected_raw"]
    assert edge <= r["failure_edge"] and address > 0
    assert not output


def test_crc_valid_frame_waits_for_mac_sink_on_the_same_absolute_clock():
    result, _, events, payload = replay(3, 5, 99, 8, mac_stall_until=280_000)
    assert result["status"] == "delivered"
    output = [e for e in events if e["kind"] == "mac_byte"]
    assert output[0]["edge"] >= 280_000
    assert bytes(e["value"] for e in output) == payload
    assert result["delivery"]["peak_ram_reserved_bytes"] == len(payload)
