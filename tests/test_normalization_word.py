"""Independent word/range/RNE/phase checks, including preserved startup defects."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import (
    MAX_SYMBOLS,
    NormalizationFormat,
    WordNormalizationState,
    pack_normalized_word,
    physical_normalized_word,
    reciprocal_gain,
    rotation_coefficients,
    unpack_normalized_word,
)
from tests.normalization_reference import (
    fraction_round,
    raw_word,
    relative_error,
    rounded_coefficient,
)


def fit(phase=0, frequency=0, gain=(1, 0)):
    return dict(valid=True, phase_turn40=phase, frequency_turn40=frequency, gain=list(gain))


def test_original_quarter_ROM_matches_independent_high_precision_constants():
    for address in range(4096):
        assert rotation_coefficients(address << 28, NormalizationFormat("nearest")) == (
            rounded_coefficient(address << 28)
        )


def test_reciprocal_RNE_full_gain_domain_matches_independent_fraction():
    rng = np.random.default_rng(1100557)
    values = [1, 3, 131071, 1 << 53, (1 << 53) + 1, (1 << 54) - 1]
    values += [int(v) for v in rng.integers(1, 1 << 54, 256)]
    for m in values:
        for exponent in [-512, -47, 0, 511]:
            q, e = reciprocal_gain((m, exponent))
            normalized = m * 2 ** (54 - m.bit_length())
            expected = fraction_round(Fraction(1 << 83, normalized))
            target_exponent = -83 - exponent + (54 - m.bit_length())
            if expected == 1 << 30:
                expected //= 2
                target_exponent += 1
            assert (q, e) == (expected, target_exponent)
            actual = Fraction(q) * Fraction(2) ** e
            reference = Fraction(1, m) * Fraction(2) ** -exponent
            assert abs(actual / reference - 1) <= Fraction(1, 1 << 29)


def test_linear_phase_words_and_gain_satisfy_frozen_development_error_budget():
    rng = np.random.default_rng(1100557)
    errors = []
    phases = [0, 1, (1 << 40) - 1, 1 << 27, (1 << 38) - 2048,
              (1 << 38) + 2048, (1 << 40) - 2048]
    phases += [int(v) for v in rng.integers(0, 1 << 40, 256)]
    for k, phase in enumerate(phases):
        gain = [(1, -512), ((1 << 54) - 1, 511), (1 << 53, -53), (1234567, -13)][k % 4]
        f = fit(phase, 0, gain)
        word = raw_word(int(rng.integers(-(1 << 21), 1 << 21)),
                        int(rng.integers(-(1 << 21), 1 << 21)), [-128, -7, 0, 127][k % 4])
        state = WordNormalizationState(f, NormalizationFormat("linear"))
        result = state.advance_word(np.uint64(word))
        error = relative_error(word, result["normalized_word"], f)
        assert error <= 5e-5, (k, phase, error)
        errors.append(float(error))
    assert np.sqrt(np.mean(np.square(errors))) < 3e-5
    # The lower-cost nearest control is independently rejected, never retried by CRC.
    f = fit(1 << 27)
    word = raw_word(1 << 20, 0, 0)
    nearest = WordNormalizationState(f, NormalizationFormat("nearest")).advance_word(word)
    assert relative_error(word, nearest["normalized_word"], f) > 7e-4


def test_turn40_recurrence_survives_long_packets_without_frequency_requantization():
    frequency = -(1 << 32) + 13
    initial = (1 << 40) - 273
    state = WordNormalizationState(fit(initial, frequency), NormalizationFormat())
    for k in range(MAX_SYMBOLS):
        state.advance_word(0)
        assert state.phase == (initial + (k + 1) * frequency) % (1 << 40)
    assert state.count == MAX_SYMBOLS
    with pytest.raises(RuntimeError, match="exhausted"):
        state.advance_word(0)
    # Modulo events are intentional NCO wraps, not evidence of carrier cycle slips.
    assert state.stats["intentional_modulo_wrap"] > 0


def test_zero_alias_and_invalid_input_do_not_hide_or_partially_prepare_state():
    s = WordNormalizationState(fit(), NormalizationFormat())
    before = s.snapshot()
    for bad in [-1, 1 << 52, True, 1.5, None]:
        with pytest.raises((ValueError, OverflowError)):
            s.advance_word(bad)
        assert s.snapshot() == before
    result = s.advance_word(raw_word(0, 0, 127))
    assert result["normalized_word"] == 0
    assert s.stats["input_zero"] == s.stats["output_zero"] == 1
    copied = s.snapshot()
    copied["statistics"]["accepted"] = 91
    assert s.count == s.stats["accepted"] == 1


def test_explicit55bit_layout_canonical_zero_and_exact_adapter():
    for r, i, e in [(-1, -2, -3), (1 << 20, -(1 << 21), 127),
                    (1, -1, -1024), (7, 19, 1023), (0, 0, 0)]:
        word = pack_normalized_word(r, i, e)
        assert unpack_normalized_word(np.uint64(word)) == (r, i, e)
    assert pack_normalized_word(0, 0, 17) == 0
    assert physical_normalized_word(pack_normalized_word(7, -19, -3)) == complex(7 / 8, -19 / 8)
    with pytest.raises(ValueError, match="canonical"):
        unpack_normalized_word(1 << 44)
    for bad in [-1, 1 << 55, True, 1.1]:
        with pytest.raises((ValueError, OverflowError)):
            unpack_normalized_word(bad)
    with pytest.raises(OverflowError):
        pack_normalized_word(1 << 21, 0, 0)
    with pytest.raises(OverflowError):
        physical_normalized_word(pack_normalized_word(1 << 20, 0, 1023))


def test_invalid_initialization_and_single_owner_contract():
    for update in [dict(valid=False), dict(gain=[0, 0]), dict(gain=[1, -513]),
                   dict(phase_turn40=1 << 40), dict(frequency_turn40=1 << 41)]:
        with pytest.raises(ValueError):
            WordNormalizationState(fit() | update, NormalizationFormat())
    with pytest.raises(ValueError):
        NormalizationFormat("arbitrary")
    with pytest.raises(ValueError):
        reciprocal_gain([True, 0])
    s = WordNormalizationState(fit(), NormalizationFormat())
    s.claim()
    for operation in [s.claim, s.prepare, lambda: s.advance_word(0)]:
        with pytest.raises(RuntimeError):
            operation()
