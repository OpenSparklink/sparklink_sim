"""Real words→fixed interpolation→existing floating loop/parser;no CRC answer retry."""

import math

import numpy as np
import pytest

from nearlink_sdr.phy.selected_word import WordInterpolatedPSKReceiver, physical_filter_word
from nearlink_sdr.sim.capture_fine import pack_filter_word
from tests.test_interpolation_word import FORMATS
from tests.test_selected_rx import consume, equal, ordinary, selected_input


def quantize_input(values):
    """Independent host boundary;applied identically to float and word receivers."""
    result = []
    for z in values:
        magnitude = max(abs(z.real), abs(z.imag))
        e = math.frexp(magnitude)[1] - 18 if magnitude else 0
        r, i = round(math.ldexp(z.real, -e)), round(math.ldexp(z.imag, -e))
        if max(abs(r), abs(i)) >= 2**18:
            e += 1
            r, i = round(math.ldexp(z.real, -e)), round(math.ldexp(z.imag, -e))
        result.append(pack_filter_word(r, i, e))
    return np.array(result, np.uint64)


def word_consume(profile, words, selection, fmt, chunks):
    selection = {
        k: v for k, v in selection.items() if k not in ("timing_backend", "history_capacity")
    }
    rx = WordInterpolatedPSKReceiver(profile, **selection, word_format=fmt)
    events, pos, k = [], 0, 0
    while pos < len(words) and not rx.done:
        n = int(chunks[k % len(chunks)])
        events.extend(rx.feed_words(words[pos : pos + n]))
        pos += rx.last_input_consumed
        k += 1
    events.extend(rx.finish())
    assert rx.reference_memory()["raw_history"]["live_samples"] == 0
    return events[0], rx


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 1), (3, 12, 8, 197), (4, 0, 4, 73)])
@pytest.mark.parametrize("fmt", FORMATS)
def test_fixed_interpolation_is_causal_and_preserves_valid_frames(ft, mcs, pilot, length, fmt):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    words = quantize_input(values)
    decoded = np.array([physical_filter_word(w) for w in words])
    (reference,), _ = consume(profile, decoded, selection, [32])
    assert reference["frame"]["payload"] == payload
    event, rx = word_consume(profile, words, selection, fmt, [32])
    assert event["frame"]["payload"] == payload
    assert event["frame"]["header_crc"] and event["frame"]["data_crc"]
    assert event["frame"]["length_ok"] and event["frame"]["mcs"] == mcs
    np.testing.assert_allclose(
        reference["symbol_centers_filtered"], event["symbol_centers_filtered"], atol=0.001, rtol=0
    )
    for chunks in ([1], [7, 29, 31]):
        split, _ = word_consume(profile, words, selection, fmt, chunks)
        equal(event, split)
    assert event["word_interpolation"]["other_nodes_fixed"] is False
    assert rx.reference_memory()["additional_word_store_logical_bits"] == 512 * 46


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_bad_frame_is_not_retried_and_empty_flag_requires_zero_length(damage, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, **damage)
    words = quantize_input(values)
    event, _ = word_consume(profile, words, selection, FORMATS[0], [13, 31])
    assert event["frame"]["payload"] is None
    assert event["frame"]["status"] == status
    if status == "invalid_length":
        assert event["frame"]["header_crc"] is True
        assert event["frame"]["length_ok"] is False


def test_horizon_and_EOF_keep_the_real_header_and_do_not_consume_suffix():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=255)
    cut = int(selection["start_tick"] / 8 + 91 * 4)
    words = quantize_input(values)
    eof, _ = word_consume(profile, words[:cut], selection, FORMATS[0], [32])
    limited, _ = word_consume(profile, words, selection | {"sample_limit": cut}, FORMATS[0], [32])
    assert eof["frame"]["status"] == "truncated_data"
    assert limited["frame"]["status"] == "retention_limit"
    assert eof["frame"]["header_crc"] is True and limited["frame"]["header_crc"] is True


def test_word_port_rejects_silent_float_conversion_native_fallback_and_bad_blocks():
    profile = ordinary()
    values, _, selection = selected_input(profile)
    with pytest.raises(ValueError):
        WordInterpolatedPSKReceiver(
            profile, **selection, word_format=FORMATS[0], timing_backend="rust"
        )
    with pytest.raises(ValueError):
        WordInterpolatedPSKReceiver(profile, **selection, word_format=None)
    rx = WordInterpolatedPSKReceiver(profile, **selection, word_format=FORMATS[0])
    for block in ([], [0] * 33, np.zeros((2, 2)), [np.bool_(True)]):
        with pytest.raises(ValueError):
            rx.feed_words(block)
    with pytest.raises(RuntimeError):
        rx.feed_filtered(values)
    with pytest.raises(RuntimeError):
        rx._sample(0)


def test_word_end_keeps_invalid_unconsumed_suffix_for_caller_and_releases_history():
    profile = ordinary()
    values, payload, selection = selected_input(profile, length=11)
    words = quantize_input(values)
    event, rx = word_consume(profile, words, selection, FORMATS[-1], [32])
    assert event["frame"]["payload"] == payload
    assert rx.input_samples < len(words)
    assert rx.feed_words([2**46]) == [] and rx.last_input_consumed == 0
    assert rx.reference_memory()["raw_history"]["live_samples"] == 0
    with pytest.raises(ValueError):
        WordInterpolatedPSKReceiver(
            profile, **selection, word_format=FORMATS[-1], history_capacity=256
        )


def test_rejected_selection_does_not_validate_or_store_an_unconsumed_word():
    profile = ordinary()
    rx = WordInterpolatedPSKReceiver(
        profile,
        word_format=FORMATS[-1],
        start_tick=8,
        delta=0,
        differential_score=0,
        selected_sync=profile.sync_symbols(),
        sample_limit=1000,
        coherent_threshold=None,
    )
    (event,) = rx.feed_words([2**46])
    assert event["frame"]["status"] == "sync_confirmation_rejected"
    assert rx.last_input_consumed == rx.input_samples == 0
    assert rx.reference_memory()["word_store_writes"] == 0


def test_receiver_preserves_exponent_floor_telemetry():
    profile = ordinary()
    rx = WordInterpolatedPSKReceiver(
        profile,
        word_format=FORMATS[-1],
        start_tick=12,
        delta=0,
        differential_score=1,
        selected_sync=profile.sync_symbols(),
        sample_limit=1000,
        coherent_threshold=None,
    )
    for _ in range(5):
        rx.feed_words([pack_filter_word(1, 0, -128)] * 32)
    assert rx.interpolation_stats["output_underflows"] > 0
    assert rx.reference_memory()["word_store_writes"] == rx.input_samples
