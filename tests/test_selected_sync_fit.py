"""Same selected words→finite initializer→existing tracker/strict header boundary."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.selected_word import WordInterpolatedPSKReceiver
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, SyncFitFormat
from tests.test_selected_rx import equal, ordinary, selected_input
from tests.test_selected_word import quantize_input, word_consume


def consume_fit(profile, words, selection, chunks, fmt=None):
    fmt = SyncFitFormat(30) if fmt is None else fmt
    rx = WordInterpolatedPSKReceiver(
        profile,
        **selection,
        word_format=INPUT_FORMAT,
        reuse_sync_interpolants=True,
        sync_fit_format=fmt,
    )
    pos = k = 0
    events = []
    while pos < len(words) and not rx.done:
        events.extend(rx.feed_words(words[pos : pos + int(chunks[k % len(chunks)])]))
        pos += rx.last_input_consumed
        k += 1
    events.extend(rx.finish())
    assert len(events) == 1
    assert rx.reference_memory()["raw_history"]["live_samples"] == 0
    return events[0], rx


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 79), (3, 12, 8, 217), (4, 0, 4, 83)])
def test_valid_head_controls_legal_tail_without_reading_unrelated_suffix(ft, mcs, pilot, length):
    profile = ordinary(ft, mcs, pilot)
    values, expected, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(10000)])
    selection["sample_limit"] = len(words)
    baseline, _ = word_consume(profile, words, selection, INPUT_FORMAT, [32], reuse=True)
    event, rx = consume_fit(profile, words, selection, [32])
    for key in ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload"):
        assert event["frame"].get(key) == baseline["frame"].get(key), key
    assert event["frame"]["payload"] == expected and event["frame"]["mcs"] == mcs
    assert event["frame"]["header_crc"] and event["frame"]["data_crc"]
    assert rx.input_samples < len(words) - 10000 and len(event["symbols"]) == rx.frame_end_symbol
    scale = max(float(np.sqrt(np.mean(abs(baseline["symbols"]) ** 2))), 1e-30)
    assert np.sqrt(np.mean(abs(event["symbols"] - baseline["symbols"]) ** 2)) / scale <= 1e-4
    for chunks in ([1], [7, 29, 31]):
        other, _ = consume_fit(profile, words, selection, chunks)
        equal(event, other)
    assert rx._sync_fit.used == len(rx.reference)
    assert event["word_interpolation"]["initial_interpolants"]["written"] == len(rx.reference)
    assert event["word_sync_initialization"]["remaining_loop_carrier_and_decoder_float"]
    assert not rx.reference_memory()["all_other_receiver_nodes_fixed"]


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_real_header_rejects_or_payload_fails_without_answer_retries(damage, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, **damage)
    event, rx = consume_fit(profile, quantize_input(values), selection, [1, 32])
    assert event["frame"]["status"] == status and event["frame"]["payload"] is None
    if status == "bad_header_crc":
        assert rx.frame_end_symbol is None and event["frame"]["data_crc"] is None
    assert rx._sync_fit.used == len(rx.reference)


@pytest.mark.parametrize(
    "count,status", [(10, "truncated_sync"), (44, "truncated_header"), (90, "truncated_data")]
)
def test_eof_and_zero_phase_are_explicit_failure_not_fallback(count, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, length=255)
    cut = int(selection["start_tick"] / 8 + 4 * count)
    event, _ = consume_fit(profile, quantize_input(values[:cut]), selection, [32])
    assert event["frame"]["status"] == status and event["frame"]["payload"] is None
    if count == 90:
        assert event["frame"]["header_crc"]
    words = np.zeros(len(values), np.uint64)
    event, rx = consume_fit(profile, words, selection, [32])
    assert event["frame"]["status"] == "invalid_sync_fit"
    assert event["frame"]["header_crc"] is None
    assert event["word_sync_initialization"]["result"]["stats"]["zero_vectors"] == len(rx.reference)
    assert rx.last_input_consumed > 0 and rx.initial_interpolants.state == "released"


def test_experimental_initializer_requires_actual_compiled_input_and_cache():
    profile = ordinary()
    _, _, selection = selected_input(profile)
    for fmt, reuse, fit in [
        (replace(INPUT_FORMAT, output_bits=19), True, SyncFitFormat()),
        (INPUT_FORMAT, False, SyncFitFormat()),
        (INPUT_FORMAT, True, 30),
    ]:
        with pytest.raises(ValueError):
            WordInterpolatedPSKReceiver(
                profile,
                **selection,
                word_format=fmt,
                reuse_sync_interpolants=reuse,
                sync_fit_format=fit,
            )
