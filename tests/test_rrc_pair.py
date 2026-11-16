"""Independent pair/product rational reference and shared baseline regression."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.rrc_pair import PairProductRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from tests.test_rrc_word import round_scaled


def oracle(real, imag, exponents, coefficients, fraction, precision):
    r, i, e = [0] * 80, [0] * 80, [0] * 80
    result = []
    for a, b, c in zip(real, imag, exponents, strict=True):
        r.append(int(a))
        i.append(int(b))
        e.append(int(c))
        pairs = []
        for j in range(40):
            k = 80 - j
            native = max(e[j] if r[j] or i[j] else -48, e[k] if r[k] or i[k] else -48)
            pair = [
                round_scaled(v[j], max(0, native - e[j]))
                + round_scaled(v[k], max(0, native - e[k]))
                for v in [r, i]
            ]
            pairs.append((pair[0] * int(coefficients[j]), pair[1] * int(coefficients[j]), native))
        pairs.append((r[40] * int(coefficients[40]), i[40] * int(coefficients[40]), e[40]))
        common = max(
            (
                ex + max(abs(a).bit_length(), abs(b).bit_length()) - (precision - 2)
                for a, b, ex in pairs
                if a or b
            ),
            default=-96,
        )
        rr, ii = [sum(round_scaled(p[part], common - p[2]) for p in pairs) for part in [0, 1]]
        peak = max(abs(rr), abs(ii))
        if peak:
            shift = peak.bit_length() - 17
            exponent = min(48, max(-96, common - fraction + shift))
            result.append(
                (
                    round_scaled(rr, exponent - common + fraction),
                    round_scaled(ii, exponent - common + fraction),
                    exponent,
                )
            )
        else:
            result.append((0, 0, 0))
        r.pop(0)
        i.pop(0)
        e.pop(0)
    return tuple(np.array(a) for a in zip(*result, strict=True))


@pytest.mark.parametrize("coefficient_bits", [14, 16, 18])
def test_pair_product_words_match_fraction_oracle_across_full_native_range(coefficient_bits):
    rng = np.random.default_rng(470118)
    r, i = rng.integers(-2048, 2048, (2, 143))
    e = rng.integers(-48, 33, 143)
    f = PairProductRRC(FixedRRCFormat(12, coefficient_bits))
    expected = oracle(r, i, e, f.coefficients, f.coefficient_fraction, f.product_bits)
    got = [f.feed_words(r[k : k + 17], i[k : k + 17], e[k : k + 17]) for k in range(0, len(r), 17)]
    for part in range(3):
        np.testing.assert_array_equal(np.concatenate([a[part] for a in got]), expected[part])
    assert f.stats["intermediate_overflows"] == 0
    assert f.stats["product_components_erased"] > 0
    assert f.stats["max_accumulator_magnitude"] < 2 ** (f.accumulator_bits - 1)


def test_exact_prefix_and_block_invariance_with_weighted_strong_weak_transitions():
    rng = np.random.default_rng(470119)
    x = (rng.normal(size=513) + 1j * rng.normal(size=513)) * np.r_[
        np.full(171, 1e5), np.full(171, 1e-8), np.ones(171)
    ]
    fmt = FixedRRCFormat(12, 16)
    a, b = PairProductRRC(fmt), PairProductRRC(fmt)
    expected = a.feed(x)
    got = np.concatenate([b.feed(x[k : k + 7]) for k in range(0, len(x), 7)])
    np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(PairProductRRC(fmt).feed(x[:137]), expected[:137])
    np.testing.assert_array_equal(a.finish(), b.finish())
    assert a.stats == b.stats
    assert a.stats["exponent_underflows"] == a.stats["input_saturations"] == 0
    with pytest.raises(ValueError):
        PairProductRRC(FixedRRCFormat(14, 18))


def test_rounding_flags_are_exact_and_fit_single_pair_carry_input():
    f = PairProductRRC(FixedRRCFormat(12, 16))
    # A native max leaves at least one component unshifted, including signed ties.
    left = np.arange(-33, 34, dtype=np.int64)[None, :]
    right = left[:, ::-1]
    for shift in range(8):
        for a, b in [(0, shift), (shift, 0)]:
            got = f._pair_sum(left, right, np.full_like(left, a), np.full_like(left, b))
            expected = [
                round(Fraction(int(x), 2**a)) + round(Fraction(int(y), 2**b))
                for x, y in zip(left[0], right[0], strict=True)
            ]
            np.testing.assert_array_equal(got[0], expected)
    with pytest.raises(AssertionError):
        f._pair_sum(np.array([[3]]), np.array([[3]]), np.array([[1]]), np.array([[1]]))


def test_zero_and_signed_power_product_rounding_are_bounded():
    f = PairProductRRC(FixedRRCFormat(12, 18))
    assert not f.feed(np.zeros(256)).any()
    values = np.array([[-(2**30) + 1, -17, -3, -1, 0, 1, 3, 17, 2**30 - 1]], np.int64)
    for shift in [-1, 0, 1, 7, 31, 81]:
        # Negative shifts must keep every product inside its normalization word.
        active = values // 4 if shift < 0 else values
        got = f._product_sum(active, np.full_like(active, shift))
        assert got[0] == sum(round_scaled(v, shift) for v in active[0])
    assert f.operation_estimate()["pair_preadd_carry_inputs"] == 80
