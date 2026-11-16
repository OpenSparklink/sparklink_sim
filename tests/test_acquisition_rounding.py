"""Arithmetic-floor factorization against rational RNE and independent per-term sums."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_rounding import (
    folded_rounding_cost,
    folded_statistics,
    split_rne_words,
)
from nearlink_sdr.phy.acquisition_scaled import align_words


def test_exhaustive_small_words_and_wide_signed_half_ties_match_rational_rounding():
    words = np.repeat(np.arange(-128, 128, dtype=np.int64), 18)
    shifts = np.tile(np.arange(18, dtype=np.int64), 256)
    floor, bit = split_rne_words(words, shifts)
    expected = [round(Fraction(int(a), 1 << int(s))) for a, s in zip(words, shifts, strict=True)]
    np.testing.assert_array_equal(floor + bit, expected)
    for shift in [0, 1, 2, 27, 58, 59, 60, 64, 160, 320]:
        a = np.array([0, 1, -1, 3, -3, 2**58 - 3, -(2**58 - 3), 2**59 - 1, -(2**59 - 1)])
        base, inc = split_rne_words(a, np.full(len(a), shift))
        np.testing.assert_array_equal(base + inc, [round(Fraction(int(v), 1 << shift)) for v in a])
    for a, s in [
        ([1.0], [1]),
        ([1], [-1]),
        ([1, 2], [1]),
        ([2**59], [1]),
        ([-(2**59)], [1]),
        (np.array([2**64 - 1], np.uint64), [1]),
    ]:
        with pytest.raises(ValueError):
            split_rne_words(a, s)


def test_folded_statistics_equal_reference_term_rounding_across_shifts_and_lengths():
    rng = np.random.default_rng(430117)
    for k in [1, 31, 127]:
        r = rng.integers(-(2**25), 2**25, (32, k), dtype=np.int64)
        i = rng.integers(-(2**25), 2**25, (32, k), dtype=np.int64)
        r[0, 0], i[0, 0] = 2**27, 0
        p = (r * r + i * i).astype(np.uint64)
        s = rng.integers(0, 161, (32, k), dtype=np.int64)
        cr, ci = rng.choice([-1, 0, 1], size=(2, k))
        actual_real, actual_imag, actual_energy, discarded = folded_statistics(r, i, p, s, cr, ci)
        ar, ai, pe = [
            align_words(v.ravel(), shift.ravel()).reshape(v.shape)
            for v, shift in ((r, s), (i, s), (p, 2 * s))
        ]
        np.testing.assert_array_equal(actual_real, ar @ cr + ai @ ci)
        np.testing.assert_array_equal(actual_imag, ai @ cr - ar @ ci)
        np.testing.assert_array_equal(actual_energy, np.sum(pe.astype(np.uint64), axis=1))
        assert discarded == np.count_nonzero((p > 0) & (pe == 0))
    cost = folded_rounding_cost()
    assert cost["wide_result_additions"] == 3
    assert cost["small_count_additions_upper"] == 152
    assert cost["extra_state_bits_upper"] == 188
    assert cost["signed_correction_bits"] == 7 and cost["energy_correction_bits"] == 5


def test_invalid_core_dimensions_words_and_costs_fail_before_wrapping():
    r = np.zeros((1, 31), np.int64)
    p = r.astype(np.uint64)
    c = np.ones(31, np.int64)
    for args in [
        (r[0], r[0], p[0], r[0], c, c),
        (r, r, p, r, c[:1], c),
        (r + 2**28, r, p, r, c, c),
        (r, r - 2**28, p, r, c, c),
        (r, r, p + 2**55, r, c, c),
        (r, r, p, r + 161, c, c),
        (r, r, p, r - 1, c, c),
        (r, r, p, r, c + 1, c),
        (r.astype(float), r, p, r, c, c),
    ]:
        with pytest.raises(ValueError):
            folded_statistics(*args)
    for terms, lanes in [(0, 4), (128, 4), (31, 0), (31, 9)]:
        with pytest.raises(ValueError):
            folded_rounding_cost(terms, lanes)


def test_unsigned_coefficient_port_keeps_every_accumulator_integer():
    r = np.ones((1, 31), np.int64)
    p = np.ones((1, 31), np.uint64)
    s = np.zeros((1, 31), np.uint64)
    coefficient = np.ones(31, np.uint64)
    real, imag, energy, _ = folded_statistics(r, r, p, s, coefficient, coefficient)
    assert real.dtype == imag.dtype == np.dtype("int64")
    assert energy.dtype == np.dtype("uint64")
    np.testing.assert_array_equal(real, [62])
    np.testing.assert_array_equal(imag, [0])
