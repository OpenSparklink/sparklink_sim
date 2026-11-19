"""Independent rational/Decimal checks,extreme ranges and finite NCO coordinates."""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.interpolant_rx import physical_interpolant
from nearlink_sdr.phy.timing import GardnerPIState, PSKTimingConfig
from nearlink_sdr.phy.timing_word import (
    KI_Q48,
    KP_Q48,
    LOWER_Q32,
    ONE,
    UPPER_Q32,
    TimingWordFormat,
    WordTimingState,
    compare_scaled,
    division_job,
    positive,
    power_update,
)
from tests.test_sync_fit_word import independent_input, packed

FORMATS = [TimingWordFormat(), TimingWordFormat(30, 32), TimingWordFormat(38, 32)]


def dyadic(m, e):
    return Fraction(m) * Fraction(2) ** e


@pytest.mark.parametrize("fmt", FORMATS)
def test_independent_ratio_rounding_clips_zero_huge_shifts_and_ties(fmt):
    rng = np.random.default_rng(1061101)
    vectors = [
        (n, e, p, pe)
        for n in [0, 1, -1, 3, -3, 2**44, -(2**44)]
        for e in [-256, -9, 0, 254]
        for p, pe in [(1, -300), (7, 0), (2**29, 40)]
    ]
    vectors += [
        (
            int(rng.integers(-(2**44), 2**44)),
            int(rng.integers(-250, 251)),
            int(rng.integers(1, 2**fmt.power_bits)),
            int(rng.integers(-300, 301)),
        )
        for _ in range(120)
    ]
    vectors += [
        (n, 0, 1 << (fmt.power_bits - 1), fmt.error_bits + 2 - fmt.power_bits)
        for n in [1, 3, 5, -1, -3, -5]
    ]  # Exactly half-LSB ties,even in both signs.
    for n, e, p, pe in vectors:
        job = division_job(n, e, (p, pe), fmt)
        ratio = dyadic(n, e) / dyadic(p, pe)
        expected = round(max(Fraction(-2), min(Fraction(2), ratio)) * 2**fmt.error_bits)
        actual = job.get("bypass")
        if actual is None:
            actual = round(Fraction(job["numerator"], job["denominator"]))
            if job["negative"]:
                actual = -actual
        assert actual == expected
        assert job["clipped"] == (abs(ratio) > 2)
    for a, ae, b, be in [(0, 127, 0, -128), (1, -512, 1, 511), (15, 0, 30, -1)]:
        x, y = dyadic(a, ae), dyadic(b, be)
        assert compare_scaled(a, ae, b, be) == (x > y) - (x < y)


def test_compiled_gains_and_rate_limits_from_independent_high_precision_equations():
    with localcontext() as ctx:
        ctx.prec = 80
        d = Decimal(".5").sqrt()
        theta = Decimal(".003") / (d + Decimal(".25") / d)
        denominator = 1 + 2 * d * theta + theta * theta
        gains = [
            16 * d * theta / denominator / Decimal("1.2"),
            16 * theta * theta / denominator / Decimal("1.2"),
        ]
        assert [int((x * 2**48).to_integral_value(rounding=ROUND_HALF_EVEN)) for x in gains] == [
            KP_Q48,
            KI_Q48,
        ]
        assert round(Decimal("3.998") * ONE) == LOWER_Q32
        assert round(Decimal("4.002") * ONE) == UPPER_Q32


@pytest.mark.parametrize("fmt", FORMATS)
def test_power_alignment_rne_growth_and_zero_floor_independent_fraction(fmt):
    for value, exponent in [(1, -128), (2**44 - 1, -17), (2**54 - 1, 0)]:
        m, e = positive(value, exponent, fmt.power_bits)
        assert abs(dyadic(m, e) - dyadic(value, exponent)) <= dyadic(1, e - 1)
    old = positive(7 * 2**45, -48, fmt.power_bits)
    for energy, e in [(0, -256), (2**43, -20), (1, 254), (0, 0)]:
        result, erased, floor = power_update(old, energy, e, fmt)
        assert erased >= 0
        assert dyadic(*result) > 0
        # Alignment is an explicit quantization step,not an exact rational IIR claim.
        expected = (63 * dyadic(*old) + dyadic(energy, e)) / 64
        bound = max(expected / 2 ** (fmt.power_bits - 2), Fraction(1, 10**30))
        assert abs(dyadic(*result) - max(expected, Fraction(1, 10**30))) <= bound
        assert isinstance(floor, bool)
        old = result


@pytest.mark.parametrize("fmt", FORMATS)
def test_actual_words_preserve_hold_and_have_bounded_open_loop_errors(fmt):
    rng = np.random.default_rng(1061109)
    z = (0.3 + 0.02 * rng.normal(size=700)) * np.exp(1j * np.arange(700) * 0.31)
    words, actual = independent_input(z)
    state = WordTimingState((1 << 53, -56), 19, 32, fmt)
    floating = GardnerPIState(4, 0.125, PSKTimingConfig())
    previous = 0j
    errors = []
    for k, (word, current) in enumerate(zip(words, actual, strict=True)):
        midword = words[max(0, k - 1)] if k >= 32 else None
        midpoint = None if midword is None else physical_interpolant(midword)
        step, error = floating.advance(current, previous, midpoint)
        old = state.next_q32
        result = state.advance_words(word, midword)
        errors.append(abs(result["error_word"] / 2**fmt.error_bits - error))
        assert abs(state.power - floating.power) / floating.power < 1e-6
        assert abs((state.next_q32 - old) / ONE - step) < 4e-6
        if k < 32:
            assert state.omega_q32 == 4 * ONE and state.error_word == 0
        previous = current
    assert max(errors) < 2e-5


def test_absolute_NCO_keeps_fraction_above_binary64_integer_precision():
    start = (2**60 + 3) * 8 + 5
    s = WordTimingState((1 << 53, -53), start, 32)
    assert s.queries[0] == dict(cell=2**60 + 3, phase=40960)
    for _ in range(32):
        s.advance_words(packed(1 << 20, 0, -20), None)
    assert s.queries[1] == dict(cell=2**60 + 129, phase=40960)
    assert s.required_end == 2**60 + 135
    s.count = 2**64 - 1
    before = s.snapshot()
    with pytest.raises(OverflowError):
        s.advance_words(packed(1 << 20, 0, -20), packed(0, 0, 0))
    assert s.snapshot() == before


def test_malformed_context_and_word_do_not_mutate_packet_state():
    for pair in [(True, 24), (30, 28), (38, 24)]:
        with pytest.raises(ValueError):
            TimingWordFormat(*pair)
    for args in [
        ((0, 0), 0, 32),
        ((1, -513), 0, 32),
        ((1, 0), True, 32),
        ((1, 0), 2**67, 32),
        ((1, 0), 0, 31),
    ]:
        with pytest.raises(ValueError):
            WordTimingState(*args)
    with pytest.raises(ValueError):
        WordTimingState((1, 0), 0, 32, config=PSKTimingConfig(bandwidth=0.004))
    s = WordTimingState((1, 0), 0, 32)
    old = s.snapshot()
    for current, midpoint in [(True, None), (2**52, None), (0, 0)]:
        with pytest.raises(ValueError):
            s.advance_words(current, midpoint)
        assert s.snapshot() == old


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("kind", ["small_mantissas", "guard_endpoint", "zero_previous"])
def test_non_normalized_and_zero_words_do_not_choose_alignment_scale(fmt, kind):
    if kind == "zero_previous":
        initial = (1 << 53, -59)
        word = packed(1 << 20, 0, -23)
        words, mid = [word] * 31 + [0, word], word
    else:
        initial = (1 << 53, -53)
        k, e = (1, 0) if kind == "small_mantissas" else (4, -2)
        mid = packed(k, 0, e)
        words = [mid] * 31 + [packed(3 * k, 0, e - 1), mid]
    s = WordTimingState(initial, 0, 32, fmt)
    p = float(dyadic(*initial))
    previous = 0j
    for n, word in enumerate(words):
        z = physical_interpolant(word)
        p = (63 * p + abs(z) ** 2) / 64
        midpoint = None if n < 32 else mid
        result = s.advance_words(word, midpoint)
        if midpoint is not None:
            numerator = ((previous - z) * physical_interpolant(midpoint).conjugate()).real
            expected = max(-2.0, min(2.0, numerator / p))
            assert abs(result["error_word"] / 2**fmt.error_bits - expected) < 2e-5
        previous = z
        assert abs(s.power - p) / p < 1e-6
