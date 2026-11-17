"""Independent scalar rational input rounding and unchanged finite integer core."""

import math
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.rrc_input import quantize_full_rrc_words
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC, PairProductRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat


def scalar_word(z):
    r, i = Fraction(float(z.real)), Fraction(float(z.imag))
    p = max(abs(r), abs(i))
    native = math.frexp(float(p))[1] - 11 if p else 0
    exponent = min(32, max(-48, native))
    scale = Fraction(2) ** exponent
    a, b = round(r / scale), round(i / scale)
    if not (-2048 <= a < 2048 and -2048 <= b < 2048) and exponent < 32:
        exponent += 1
        a, b = round(r / (Fraction(2) ** exponent)), round(i / (Fraction(2) ** exponent))
    return min(2047, max(-2048, a)), min(2047, max(-2048, b)), exponent


def test_full_input_signed_ties_extremes_and_double_rounding_control():
    samples = np.array(
        [
            0,
            2047.75 + 2045.25j,
            -2047.75 + 2045.25j,
            2**-70,
            1e20,
            2047.75 * 2**32,
            1 + 0.5j,
            1024.5 + 1025.5j,
            -1024.5 - 1025.5j,
        ]
    )
    stats = {}
    got = quantize_full_rrc_words(samples, stats=stats)
    expected = np.array([scalar_word(z) for z in samples]).T
    np.testing.assert_array_equal(got, expected)
    assert tuple(v[1] for v in got) == (1024, 1023, 1)
    # The negative limit fits; it must not force exponent+1 just by magnitude.
    assert tuple(v[2] for v in got) == (-2048, 2045, 0)
    assert stats == dict(
        exponent_underflows=1, exponent_overflows=1, input_saturations=2, input_renormalizations=1
    )


def test_random_scalar_reference_and_block_causality():
    rng = np.random.default_rng(490118)
    x = (rng.normal(size=421) + 1j * rng.normal(size=421)) * np.exp2(rng.uniform(-30, 29, 421))
    words = [quantize_full_rrc_words(x[k : k + 17], stats={}) for k in range(0, len(x), 17)]
    got = [np.concatenate([v[k] for v in words]) for k in range(3)]
    np.testing.assert_array_equal(got, np.array([scalar_word(z) for z in x]).T)
    f, g = [FullMantissaRRC(FixedRRCFormat(12, 16)) for _ in range(2)]
    y = f.feed(x)
    split = np.concatenate([g.feed(x[k : k + 7]) for k in range(0, len(x), 7)])
    np.testing.assert_array_equal(y, split)
    np.testing.assert_array_equal(FullMantissaRRC(f.format).feed(x[:53]), y[:53])
    np.testing.assert_array_equal(f.finish(), g.finish())
    assert f.stats == g.stats
    a, b = PairProductRRC(f.format), FullMantissaRRC(f.format)
    for k in range(0, len(x), 17):
        args = [v[k : k + 17] for v in got]
        np.testing.assert_array_equal(a.feed_words(*args), b.feed_words(*args))


@pytest.mark.parametrize("values", [[1] * 33, [[1]], [float("inf")]])
def test_input_rejects_unbounded_nonvector_or_nonfinite_batches(values):
    with pytest.raises(ValueError):
        quantize_full_rrc_words(values, stats={})
