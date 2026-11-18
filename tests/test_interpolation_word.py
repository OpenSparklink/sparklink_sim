"""Independent exact-rational Lagrange,extreme exponents and RNE boundaries."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    interpolate_filter_word,
    unpack_interpolated_word,
)
from nearlink_sdr.sim.capture_fine import pack_filter_word, unpack_filter_word

FORMATS = [
    LagrangeWordFormat(16, 0),
    LagrangeWordFormat(16, 2),
    LagrangeWordFormat(20, 2),
    LagrangeWordFormat(16, 2, 22),
]


def exact_reference(words, phase, fmt):
    u = Fraction(phase, 2**fmt.phase_bits)
    weights = (
        -u * (u - 1) * (u - 2) / 6,
        (u + 1) * (u - 1) * (u - 2) / 2,
        -(u + 1) * u * (u - 2) / 2,
        (u + 1) * u * (u - 1) / 6,
    )
    triples = [unpack_filter_word(w) for w in words]
    return tuple(
        sum(
            weight * Fraction(t[k]) * Fraction(2) ** t[2]
            for weight, t in zip(weights, triples, strict=True)
        )
        for k in (0, 1)
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_independent_fraction_bound_including_extreme_dynamic_range(fmt):
    rng = np.random.default_rng(930117)
    for k in range(160):
        exponents = (
            [-128, 127, -96, 48]
            if k % 4 == 0
            else rng.integers(-96, 49, 4).tolist()
            if k % 4 == 1
            else [-13] * 4
        )
        words = tuple(
            pack_filter_word(int(r), int(i), int(e))
            for (r, i), e in zip(rng.integers(-(2**18), 2**18, (4, 2)), exponents, strict=True)
        )
        phase = int(rng.integers(0, 2**fmt.phase_bits + 1))
        word, info = interpolate_filter_word(words, phase, fmt)
        assert info["output_saturations"] == 0
        r, i, exponent = unpack_interpolated_word(word, fmt)
        reference = exact_reference(words, phase, fmt)
        # Alignment <=.5inputLSB times Lagrange L1<=1.25;three Horner
        # rounds and /6 <=.75guardLSB;last pack <=.5outputLSB.
        bound = Fraction(5, 8) * Fraction(2) ** info["common_exponent"]
        bound += Fraction(3, 4) * Fraction(2) ** (info["common_exponent"] - fmt.guard_bits)
        bound += Fraction(1, 2) * Fraction(2) ** exponent
        assert (
            max(
                abs(Fraction(v) * Fraction(2) ** exponent - y)
                for v, y in zip((r, i), reference, strict=True)
            )
            <= bound
        )


@pytest.mark.parametrize("fmt", FORMATS)
def test_endpoints_and_zero_exponents_cannot_choose_scale(fmt):
    words = tuple(pack_filter_word(v, -v, -7) for v in (31, -31, 71, -13))
    for phase in (0, 2**fmt.phase_bits):
        result, _ = interpolate_filter_word(words, phase, fmt)
        actual = unpack_interpolated_word(result, fmt)
        assert tuple(Fraction(v) * Fraction(2) ** actual[2] for v in actual[:2]) == exact_reference(
            words, phase, fmt
        )
    zeros = [pack_filter_word(0, 0, e) for e in (-128, 127, 18, -96)]
    assert interpolate_filter_word(zeros, 0, fmt)[0] == 0
    zeros[1] = pack_filter_word(37, -9, -96)
    assert interpolate_filter_word(zeros, 0, fmt)[1]["common_exponent"] == -96


def test_saturation_is_explicit_at_output_exponent_ceiling():
    words = [pack_filter_word(v, 0, 127) for v in (-262144, 262143, 262143, -262144)]
    _, info = interpolate_filter_word(words, 32768, FORMATS[0])
    assert info["output_saturations"] == 1


def test_invalid_formats_phase_and_word_types():
    for bits, guard in product((True, 12, 16, 20, 24), (0, 1, 2)):
        if (bits, guard) not in ((16, 0), (16, 2), (20, 2)):
            with pytest.raises(ValueError):
                LagrangeWordFormat(bits, guard)
    for words, phase in (
        ([0] * 3, 0),
        ([0] * 4, -1),
        ([0] * 4, 65537),
        ([True, 0, 0, 0], 0),
        ([np.bool_(True), 0, 0, 0], 0),
        ([2**46, 0, 0, 0], 0),
        ([1.5, 0, 0, 0], 0),
    ):
        with pytest.raises(ValueError):
            interpolate_filter_word(words, phase, FORMATS[0])


def test_explicit_underflow_and_typed_output_port():
    fmt = LagrangeWordFormat(16, 2, 22)
    words = [0, pack_filter_word(1, 0, -128), 0, 0]
    word, info = interpolate_filter_word(words, 32768, fmt)
    assert info["output_underflow"]
    assert not info["output_saturations"]
    assert unpack_interpolated_word(word, fmt)[2] == -128
    for bad in (-1, 2**52, 1.5, True):
        with pytest.raises(ValueError):
            unpack_interpolated_word(bad, fmt)


def test_output_format_requires_a_compiled_model():
    with pytest.raises(ValueError):
        unpack_interpolated_word(0, None)
