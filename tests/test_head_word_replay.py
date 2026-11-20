"""Actual head clocks at the selected replay port,remaining payload is explicitly off-clock."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.phy.word_head_rx import WordHeadPSKReceiver
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay
from nearlink_sdr.sim.normalized_word_replay import NormalizedWordFrameReplay
from tests.test_b1_wire_length import b1_header_wire, profile
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_strict_receive import iq_record


def run(profile, words, selection, cls, pattern=(32,)):
    prepared, pos = prepare(profile, words, selection)
    flow = cls(
        profile,
        prepared,
        normalization_format=NormalizationFormat(),
        word_timing_format=TimingWordFormat(30, 32),
        reuse_stencil=True,
        **selection,
    )
    services, index = [], 0
    while not flow.done:
        if flow.pending_service is not None:
            services.append(flow.pending_service["kind"])
            flow.resolve_functional()
        else:
            if flow.wants_replay_input:
                if pos == len(words):
                    flow.end_input()
                else:
                    used = flow.feed_replay_words(words[pos : pos + pattern[index % len(pattern)]])
                    assert used == 1
                    pos += used
                    index += 1
            flow.step()
        assert flow.cycle < 2_000_000
    return flow.event, flow, services


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 8, 19), (3, 12, 4, 47), (4, 10, 16, 23)])
@pytest.mark.parametrize("layout", ["dense", "depth-parity"])
def test_actual_head_replaces_only_header_barrier_and_preserves_every_selected_value(
    ft, mcs, pilot, length, layout
):
    p = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(p, length=length)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    before, old, barriers = run(p, words, selection, NormalizedWordFrameReplay)
    assert barriers == ["header", "payload"]
    for pattern in [(32,), (1, 7, 31)]:
        after, new, barriers = run(
            p, words, selection, partial(HeadWordsFrameReplay, head_layout=layout), pattern
        )
        assert barriers == ["payload"]
        for key in ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload"):
            assert after["frame"].get(key) == before["frame"].get(key), key
        assert after["frame"]["payload"] == payload
        for key in (
            "symbols",
            "symbol_centers_filtered",
            "timing_rate_ppm",
            "timing_detector_error",
        ):
            np.testing.assert_array_equal(after[key], before[key])
        assert new.consumer.word_timing.snapshot() == old.consumer.word_timing.snapshot()
        assert new.stats["finite_head_completions"] == new.head_owner.stats["completed"] == 1
        assert new.head_owner.stats["header_issues"] == 1
        assert not after["finite_word_head"]["cancelled"]
        assert new.stats["functional_resolutions"] == old.stats["functional_resolutions"] - 1
        assert len(new.take_mac_result().packets) == 1
        assert not new.take_mac_result().packets
        assert new.release_source() is new.history
        cost = new.estimate()
        assert cost["new_raw_history"] == 0 and cost["head_controller_increment_bits"] == 129
        assert cost["payload_cycles"] is None and not cost["whole_receiver16MSps"]


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_failed_frame_has_one_real_head_decision_and_no_MAC(damage, status):
    p = ordinary(2, 8, 8)
    values, _payload, selection = selected_input(p, **damage)
    words = quantize_input(values)
    after, flow, barriers = run(p, words, selection, HeadWordsFrameReplay)
    assert after["frame"]["status"] == status
    assert not flow.take_mac_result().packets
    assert flow.head_owner.stats["header_issues"] == flow.head_owner.header.stats["accepted"] == 1
    assert barriers == (["payload"] if status == "bad_data_crc" else [])
    assert flow.head_owner.header.geometry.stats["accepted"] == int(status == "bad_data_crc")


def test_missing_header_and_wrong_seed_never_accept_unchecked_fields():
    p = ordinary(2, 8, 8)
    values, _payload, selection = selected_input(p)
    words = quantize_input(values)
    wrong = replace(p, config=replace(p.config, head_crc_seed=0))
    after, flow, _ = run(wrong, words, selection, HeadWordsFrameReplay)
    assert after["frame"]["status"] == "bad_header_crc"
    assert flow.head_owner.header.stats["field_bits"] == 0
    cut = selection["start_tick"] // 8 + 4 * (len(p.sync_symbols()) + 8)
    selection = selection | dict(sample_limit=cut)
    after, flow, barriers = run(p, words[:cut], selection, HeadWordsFrameReplay)
    assert after["frame"]["status"] == "truncated_header"
    assert after["frame"]["header_crc"] is None and not barriers
    assert flow.head_owner.header.stats["accepted"] == 0
    assert after["finite_word_head"]["cancelled"]
    assert not flow.take_mac_result().packets


@pytest.mark.parametrize(
    "mode,status",
    [("mcs", "unsupported_mcs"), ("length", "invalid_length"), ("LLID", "bad_header_crc")],
)
def test_B1_policy_and_real_LLID_are_checked_before_payload(mode, status):
    p = profile(3)
    values, _payload, selection = selected_input(p, **({"mcs": 15} if mode == "mcs" else {}))
    if mode == "length":
        p = replace(p, max_payload_bytes=32)
    elif mode == "LLID":
        p = replace(p, config=replace(p.config, head_llid=7))
    event, flow, barriers = run(p, quantize_input(values), selection, HeadWordsFrameReplay)
    assert event["frame"]["status"] == status and barriers == []
    assert flow.head_owner.header.geometry.stats["accepted"] == 0
    assert not flow.take_mac_result().packets
    assert event["header_decision_after_filtered_samples"] == flow.consumer.input_samples


@pytest.mark.parametrize("ft", [3, 4])
def test_standard_zero_data_is_real_head_only_and_crc_only_remains_payload(ft):
    p = profile(ft)
    symbols, _ = b1_header_wire(p, 0)
    preamble = {3: 48, 4: 64}[ft]
    pre = np.exp(1j * np.tile([np.pi / 4, 0], preamble // 2))
    iq, _quiet, _truth, _step = iq_record(np.r_[pre, symbols])
    values = np.convolve(iq, rrc_filter(0.4, 4))
    chosen = search_psk_sync(values, 4, p.sync_symbols(), 8)
    assert chosen is not None
    n, i = len(p.sync_symbols()), chosen["offset"]
    selection = dict(
        start_tick=round(8 * (chosen["phase"] + 4 * i)),
        delta=chosen["delta"],
        differential_score=chosen["score"],
        coherent_threshold=0.45,
        selected_sync=chosen["phase_symbols"][i : i + n],
        sample_limit=len(values),
    )
    event, flow, barriers = run(p, quantize_input(values), selection, HeadWordsFrameReplay)
    row = event["frame"]
    assert row["status"] == "valid_empty_control" and row["header_crc"] is True
    assert row["data_crc"] is None and row["payload"] is None and not barriers
    assert row["frame_end_symbol"] == len(symbols)
    assert not flow.take_mac_result().packets and flow.cycle_evidence_valid
    assert event["header_decision_after_filtered_samples"] == flow.consumer.input_samples
    # Zero information bytes with an actual dataCRC is a different legal B1 frame.
    values, _payload, selection = selected_input(p, length=0)
    event, flow, barriers = run(p, quantize_input(values), selection, HeadWordsFrameReplay)
    assert event["frame"]["status"] == "valid" and barriers == ["payload"]
    assert event["frame"]["data_crc"] is True and event["frame"]["payload"] == b""
    assert event["frame"]["data_crc_present"] is True


def test_owner_clock_and_parse_guards_do_not_consume_another_answer():
    p = ordinary(2, 8, 8)
    with pytest.raises(ValueError, match="begun finite head"):
        WordHeadPSKReceiver(p, head_owner=None)
    values, _payload, selection = selected_input(p)
    words = quantize_input(values)
    prepared, _pos = prepare(p, words, selection)
    events = []
    flow = HeadWordsFrameReplay(
        p,
        prepared,
        normalization_format=NormalizationFormat(),
        word_timing_format=TimingWordFormat(30, 32),
        head_observer=events.append,
        **selection,
    )
    with pytest.raises(RuntimeError, match="completed finite head"):
        flow.consumer.accept_header_result(after_samples=flow.history.end)
    with pytest.raises(ValueError, match="termination"):
        flow.consumer.finish(reason="silently_pause_ADC")
    flow.head_owner.cycle += 1
    with pytest.raises(RuntimeError, match="same actual edge"):
        flow.step()
    flow.head_owner.cycle -= 1
    flow.abort("cancelled")
    assert flow.consumer.finish() == [] and not flow.take_mac_result().packets
    assert events and flow.history.pending is None


@pytest.mark.parametrize(
    "layout,clock,completes",
    [
        ("dense", 100_000_000, False),
        ("dense", 160_000_000, False),
        ("dense", 200_000_000, True),
        ("depth-parity", 100_000_000, False),
        ("depth-parity", 160_000_000, True),
        ("depth-parity", 200_000_000, True),
    ],
)
def test_head_wait_continuous_suffix_is_bounded_and_exit_never_redecodes(
    layout, clock, completes, monkeypatch
):
    # Only the head-wait suffix is continuous;this does not prove preceding/full RX.
    p = ordinary(3, 8, 8)
    values, _payload, selection = selected_input(p)
    words = quantize_input(np.r_[values, np.zeros(2000)])
    selection["sample_limit"] = len(words)
    prepared, pos = prepare(p, words, selection)
    flow = HeadWordsFrameReplay(
        p,
        prepared,
        normalization_format=NormalizationFormat(),
        word_timing_format=TimingWordFormat(30, 32),
        head_layout=layout,
        reuse_stencil=True,
        **selection,
    )

    def forbidden(*_args, **_options):
        pytest.fail("Resource/EOF path must not functionally re-decode the head")

    monkeypatch.setattr("nearlink_sdr.phy.selected_rx.decode_psk_frame", forbidden)
    monkeypatch.setattr("nearlink_sdr.phy.stream_rx.decode_psk_frame", forbidden)
    while flow.mode != "word_header":
        assert flow.pending_service is None and not flow.done
        if flow.wants_replay_input:
            pos += flow.feed_replay_words(words[pos : pos + 32])
        flow.step()
    start, origin, refusals = flow.cycle, pos, []
    while not flow.done and not flow.stats["finite_head_completions"]:
        # Rational integer clock:ceil(arrival_index * clock / 16 MS/s).
        due = start + ((pos - origin) * clock + 15_999_999) // 16_000_000
        if flow.cycle >= due:
            if flow.write_input(words[pos]):
                pos += 1
            else:
                refusals.append(pos)
        flow.step()
        assert flow.cycle - start < 10000
    assert bool(flow.stats["finite_head_completions"]) == completes
    assert flow.history.stats["peak_live"] <= 512 and flow.cycle_evidence_valid
    assert flow.history.end == pos
    if completes:
        assert not flow.done and not refusals
        assert flow.consumer._active["header"]["header_crc"] is True
        assert flow.head_owner.header.stats["CRC_input_bits"] == 27
        # Cancel after the real receipt:resource diagnostics may retain that checked CRC.
        flow.abort("cancelled")
        assert flow.event["frame"]["header_crc"] is True
    else:
        assert flow.event["frame"]["status"] == "history_overflow" and refusals == [pos]
        assert flow.event["frame"]["header_crc"] is None
        assert flow.event["frame"]["data_crc"] is None
        assert flow.head_owner.header.stats["CRC_input_bits"] == 0
    assert not flow.take_mac_result().packets
    assert flow.release_source() is flow.history
    # Exact absolute indices and every unconsumed word,not just an occupancy count.
    for index in range(flow.history.head, flow.history.end):
        assert flow.history.words[index % 512] == words[index]
