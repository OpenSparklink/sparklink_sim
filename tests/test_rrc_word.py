"""Independent causal convolution and rational finite-word RNE references."""

from fractions import Fraction

import numpy as np
import pytest
from scipy.signal import lfilter

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, quantize_scaled_words
from nearlink_sdr.phy.rrc_word import FixedRRC, FixedRRCFormat, SymmetricRRC
from tests.test_rx_timing_audit import audit


def round_scaled(value, shift):
    return round(Fraction(int(value), 2**shift)) if shift >= 0 else int(value) * 2**-shift


def oracle(real, imag, exponents, coefficients, fraction):
    r, i, e = [0] * 80, [0] * 80, [0] * 80
    result = []
    for nr, ni, ne in zip(real, imag, exponents, strict=True):
        r.append(int(nr))
        i.append(int(ni))
        e.append(int(ne))
        common = max((c for a, b, c in zip(r, i, e, strict=True) if a or b), default=-48)
        sums = [
            sum(
                round_scaled(a, common - c) * int(h)
                for a, c, h in zip(values, e, coefficients, strict=True)
            )
            for values in (r, i)
        ]
        peak = max(map(abs, sums))
        if peak:
            shift = peak.bit_length() - 17
            exponent = min(48, max(-96, common - fraction + shift))
            shift = exponent - common + fraction
            result.append((round_scaled(sums[0], shift), round_scaled(sums[1], shift), exponent))
        else:
            result.append((0, 0, 0))
        r.pop(0)
        i.pop(0)
        e.pop(0)
    return tuple(np.array(v) for v in zip(*result, strict=True))


def test_symmetric_reference_is_causal_and_matches_independent_pulse_and_convolution():
    model = SymmetricRRC()
    independent = audit.pulse(np.arange(-40, 41) / 4)
    independent /= np.linalg.norm(independent)
    np.testing.assert_allclose(model.taps, independent, atol=2e-16, rtol=0)
    rng = np.random.default_rng(450118)
    x = rng.normal(size=517) + 1j * rng.normal(size=517)
    expected = np.convolve(x, independent)
    whole = np.r_[model.feed(x), model.finish()]
    np.testing.assert_allclose(whole, expected, atol=2e-15, rtol=0)
    split = SymmetricRRC()
    got = [split.feed(x[k : k + 7]) for k in range(0, len(x), 7)]
    got.append(split.finish())
    np.testing.assert_array_equal(np.concatenate(got), whole)
    causal = SymmetricRRC().feed(x[:53])
    np.testing.assert_array_equal(causal, whole[:53])
    assert model.finish().size == 0 and model.history.size == 80


@pytest.mark.parametrize("input_bits,coefficient_bits", [(10, 14), (12, 16), (14, 18)])
def test_word_fir_matches_direct_rational_oracle_with_dynamic_exponents(
    input_bits, coefficient_bits
):
    model = FixedRRC(FixedRRCFormat(input_bits, coefficient_bits))
    rng = np.random.default_rng(450119)
    limit = 1 << (input_bits - 1)
    r, i = rng.integers(-limit, limit, (2, 117))
    e = rng.integers(-48, 33, 117)
    expected = oracle(r, i, e, model.coefficients, model.coefficient_fraction)
    words = [
        model.feed_words(r[k : k + 13], i[k : k + 13], e[k : k + 13]) for k in range(0, len(r), 13)
    ]
    for part in range(3):
        np.testing.assert_array_equal(np.concatenate([v[part] for v in words]), expected[part])
    assert model.stats["intermediate_overflows"] == 0
    assert model.stats["max_accumulator_magnitude"] < 2 ** (model.accumulator_bits - 1)
    assert model.stats["aligned_components_erased"] > 0
    assert model.operation_estimate()["real_coefficient_products"] == 82


def test_word_blocks_prefix_tail_and_strong_weak_switches_are_exact():
    rng = np.random.default_rng(450120)
    x = (rng.normal(size=321) + 1j * rng.normal(size=321)) * 2.0 ** np.repeat([-20, 10, -12], 107)
    format = FixedRRCFormat(12, 18)
    model, split = FixedRRC(format), FixedRRC(format)
    expected = model.feed(x)
    actual = np.r_[tuple(split.feed(x[k : k + 7]) for k in range(0, len(x), 7))]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(FixedRRC(format).feed(x[:137]), expected[:137])
    np.testing.assert_array_equal(model.finish(), split.finish())
    assert model.stats == split.stats and model.stats["zero_flush_samples"] == 80
    assert model.stats["output_saturations"] == model.stats["intermediate_overflows"] == 0
    assert model._r.size == model._i.size == model._e.size == 80
    reference = lfilter(SymmetricRRC().taps, [1.0], x)
    assert np.linalg.norm(expected - reference) / np.linalg.norm(reference) < 0.002


def test_input_range_is_recorded_not_silently_accepted_and_invalid_ports_reject():
    f = FixedRRCFormat(12, 18)
    model = FixedRRC(f)
    y = model.feed(np.array([1e-100, 1e100 + 1j * 1e100]))
    assert np.isfinite(y).all() and model.stats["exponent_underflows"] == 1
    assert model.stats["exponent_overflows"] == 1 and model.stats["input_saturations"] == 2
    for values in [[np.nan], np.zeros((1, 2))]:
        with pytest.raises(ValueError):
            FixedRRC(f).feed(values)
    for words in [
        ([1.0], [0], [0]),
        ([2048], [0], [0]),
        ([0], [0], [33]),
        ([], [], []),
        (np.array([2**64 - 1], np.uint64), [0], [0]),
    ]:
        with pytest.raises(ValueError):
            FixedRRC(f).feed_words(*words)
    for args in [(8, 18), (12, 15), (True, 18)]:
        with pytest.raises(ValueError):
            FixedRRCFormat(*args)
    with pytest.raises(ValueError):
        FixedRRC(None)
    model.finish()
    assert model.finish().size == 0
    with pytest.raises(RuntimeError):
        model.feed([])
    with pytest.raises(RuntimeError):
        model.feed_words([0], [0], [0])
    exhausted = FixedRRC(f)
    exhausted.stats["samples"] = 2**63 - 1
    with pytest.raises(OverflowError):
        exhausted.feed([0])
    assert FixedRRC(f).finish().size == 0


def test_extracted_quantizer_matches_scalar_binary_rounding_and_explicit_counters():
    samples = np.array([0, 0.5, -0.75, 1e-20, 1e20, 3.03125 + 1.015625j])
    for bits in [10, 12, 14]:
        stats = {}
        r, i, e = quantize_scaled_words(samples, ScaledBankFormat(bits), stats=stats)
        expected = []
        for z in samples:
            peak = max(abs(z.real), abs(z.imag))
            exponent = int(np.frexp(peak)[1]) - (bits - 2)
            exponent = min(32, max(-48, exponent))
            limit = 2 ** (bits - 1)
            expected.append(
                (
                    min(limit - 1, max(-limit, round(z.real * 2.0**-exponent))),
                    min(limit - 1, max(-limit, round(z.imag * 2.0**-exponent))),
                    exponent,
                )
            )
        np.testing.assert_array_equal(np.column_stack([r, i, e]), expected)
        assert stats["exponent_underflows"] > 0 and stats["exponent_overflows"] > 0


def test_seal_after_owner_tail_does_not_add_implicit_samples():
    f = FixedRRC(FixedRRCFormat(12, 18))
    f.feed(np.ones(3))
    before = dict(f.stats)
    f.seal()
    assert f.closed and f.stats == before and len(f.finish()) == 0
    with pytest.raises(RuntimeError):
        f.feed(np.ones(1))
