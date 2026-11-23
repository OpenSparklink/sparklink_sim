"""Independent RNE,full-growth products and compiled full14 RRC regressions."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.rrc_full14 import Full14MantissaRRC, quantize_full14_rrc_words
from nearlink_sdr.phy.rrc_input import quantize_full_rrc_words
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC, PairProductRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from tests.test_rrc_pair import oracle


def quantizer_oracle(value, bits):
    values = [Fraction.from_float(float(v)) for v in (value.real, value.imag)]
    peak = max(map(abs, values))
    if peak:
        power = peak.numerator.bit_length() - peak.denominator.bit_length()
        if peak < Fraction(2) ** power:
            power -= 1
        native = power + 1 - (bits - 1)
    else:
        native = 0
    exponent = min(32, max(-48, native))
    limit = 2 ** (bits - 1)
    words = [round(v / Fraction(2) ** exponent) for v in values]
    renormalize = any(not -limit <= v < limit for v in words) and exponent < 32
    exponent += int(renormalize)
    words = [round(v / Fraction(2) ** exponent) for v in values]
    counts = dict(exponent_underflows=int(bool(peak) and native < -48),
                  exponent_overflows=int(native > 32), input_renormalizations=int(renormalize),
                  input_saturations=sum(not -limit <= v < limit for v in words))
    return (*[min(limit - 1, max(-limit, v)) for v in words], exponent), counts


@pytest.mark.parametrize("bits", [12, 14])
def test_shared_quantizer_matches_independent_fraction_at_ties_and_range_edges(bits):
    rng = np.random.default_rng(714083)
    peak = 2 ** (bits - 1) - .5
    values = [0j, complex(-(2 ** (bits - 1)), .5)]
    for exponent in [-80, -48, -13, 0, 32, 70]:
        for x in [peak, np.nextafter(peak, 0), np.nextafter(peak, np.inf), -peak]:
            values.append(complex(np.ldexp(x, exponent), np.ldexp(.5, exponent)))
    values += list((rng.normal(size=151) + 1j*rng.normal(size=151)) *
                   np.ldexp(np.ones(151), rng.integers(-55, 39, 151)))
    x, stats, expected_counts = np.asarray(values), {}, dict.fromkeys([
        "exponent_underflows", "exponent_overflows", "input_renormalizations", "input_saturations",
    ], 0)
    expected = []
    for value in x:
        words, counts = quantizer_oracle(value, bits)
        expected.append(words)
        for name, count in counts.items():
            expected_counts[name] += count
    converter = quantize_full_rrc_words if bits == 12 else quantize_full14_rrc_words
    got = [converter(x[j:j+17], stats=stats) for j in range(0, len(x), 17)]
    for k in range(3):
        np.testing.assert_array_equal(np.concatenate([a[k] for a in got]),
                                      np.asarray(expected)[:, k])
    assert stats == expected_counts and stats["input_saturations"] > 0


@pytest.mark.parametrize("other", [0, 1, -1, 3])
def test_logical_adc12_codes_are_exactly_represented_without_claiming_extra_adc_bits(other):
    codes = np.arange(-2048, 2048)
    x = (codes + 1j*other) / 2048
    stats, parts = {}, []
    for j in range(0, len(x), 32):
        parts.append(quantize_full14_rrc_words(x[j:j+32], stats=stats))
    r, i, e = [np.concatenate([p[k] for p in parts]) for k in range(3)]
    np.testing.assert_array_equal(np.ldexp(r.astype(float), e) +
                                  1j*np.ldexp(i.astype(float), e), x)
    assert not any(stats.values())


def test_original_input_erasure_control_is_not_relabelled_as_more_analog_information():
    x = np.array([-1 + 1j/2048])
    old = quantize_full_rrc_words(x, stats={})
    new = quantize_full14_rrc_words(x, stats={})
    assert old[1][0] == 0 and new[1][0] != 0
    assert complex(np.ldexp(float(new[0][0]), int(new[2][0])),
                   np.ldexp(float(new[1][0]), int(new[2][0]))) == x[0]


@pytest.mark.parametrize("values", [np.zeros(33), np.zeros((2, 3)), [np.nan], [np.inf]])
def test_nonfinite_and_unbounded_adapter_inputs_reject_before_mutating_telemetry(values):
    stats = {}
    with pytest.raises(ValueError):
        quantize_full14_rrc_words(values, stats=stats)
    assert stats == {}


def test_bad_adapter_counter_and_empty_input_contract():
    with pytest.raises(ValueError):
        quantize_full14_rrc_words([1], stats=[])
    result = quantize_full14_rrc_words([], stats={})
    assert all(len(a) == 0 for a in result)


def test_compiled_coefficients_prove_all_signed14_vertices_fit_existing31bit_slots():
    model = Full14MantissaRRC()
    for coefficient in model.coefficients[:40]:
        for a in [-8192, 8191]:
            for b in [-8192, 8191]:
                assert -(2**30) <= (a+b)*int(coefficient) < 2**30
    assert model.native_product_bounds["pair_abs_upper"] == 1034452992
    assert model.product_bits == 31 and model.accumulator_bits == 37
    assert model.logical_state_bits - FullMantissaRRC(FixedRRCFormat(12, 18)).logical_state_bits == 320
    cost = model.operation_estimate()
    assert cost["native_multiplier_operand_bits"] == [15, 18]
    assert cost["nominal_native_product_bits"] == 33 and cost["stored_product_bits"] == 31
    assert cost["lossless_sign_extension_slice"] and cost["physical_ADC_to_word_cycles"] is None
    with pytest.raises(ValueError, match="12-bit"):
        PairProductRRC(FixedRRCFormat(14, 18))
    with pytest.raises(TypeError):
        Full14MantissaRRC(FixedRRCFormat(12, 18))


def test_coefficient_set_without_static_product_proof_is_rejected(monkeypatch):
    h = np.zeros(81)
    h[0] = h[-1] = .75
    monkeypatch.setattr("nearlink_sdr.phy.rrc_word._kernel", lambda: h)
    with pytest.raises(OverflowError, match="signed31"):
        Full14MantissaRRC()


@pytest.mark.parametrize("kind", ["random", "negative", "mixed-exponents"])
def test_finite_port_matches_independent_pair_fraction_reference_and_full_native_range(kind):
    rng = np.random.default_rng(714091)
    r, i = rng.integers(-8192, 8192, (2, 137))
    e = rng.integers(-48, 33, 137) if kind == "mixed-exponents" else np.zeros(137, int)
    if kind == "negative":
        r[:] = -8192
        i[:] = 8191
    model = Full14MantissaRRC()
    expected = oracle(r, i, e, model.coefficients, 17, 31)
    got = [model.feed_words(r[j:j+13], i[j:j+13], e[j:j+13]) for j in range(0, len(r), 13)]
    for k in range(3):
        np.testing.assert_array_equal(np.concatenate([a[k] for a in got]), expected[k])
    assert model.stats["intermediate_overflows"] == 0
    assert model.stats["max_accumulator_magnitude"] < 2**36
    with pytest.raises(ValueError):
        model.feed_words([8192], [0], [0])


def test_causal_chunk_history_virtual_flush_and_seal_do_not_reset_at_packet_boundaries():
    rng = np.random.default_rng(714097)
    x = (rng.normal(size=643) + 1j*rng.normal(size=643)) * np.resize([.001, 1, 3000, .0002], 643)
    whole, chunks = Full14MantissaRRC(), Full14MantissaRRC()
    expected = np.r_[whole.feed(x), whole.finish()]
    got = np.r_[np.concatenate([chunks.feed(x[j:j+7]) for j in range(0, len(x), 7)]), chunks.finish()]
    np.testing.assert_array_equal(got, expected)
    assert whole.stats == chunks.stats and whole.stats["zero_flush_samples"] == 80
    assert not whole.finish().size
    with pytest.raises(RuntimeError):
        whole.feed([1])
    prefix = Full14MantissaRRC()
    np.testing.assert_array_equal(prefix.feed(x[:137]), expected[:137])
    prefix.seal()
    assert prefix.stats["zero_flush_samples"] == 0 and not prefix.finish().size
