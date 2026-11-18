"""Independent angle/least-squares reference,axes,zero and dynamic-range cases."""

import math
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.sync_fit_constants import ATAN_TURN40, GAIN_FLOOR
from nearlink_sdr.phy.sync_fit_word import (
    HALF,
    TURN,
    PositiveWordSum,
    SyncFitFormat,
    WordSyncFit,
    compile_sync_phases,
    finish_vector,
    fit_as_floats,
    prepare_vector,
    vector_step,
)
from tests.test_selected_rx import ordinary


def packed(real, imag, exponent):
    if not -128 <= exponent <= 127 or not all(-(2**21) <= v < 2**21 for v in (real, imag)):
        raise ValueError("Independent input outside actual52bit layout")
    return (real & ((1 << 22) - 1)) | ((imag & ((1 << 22) - 1)) << 22) | ((exponent & 255) << 44)


def independent_input(values):
    words, decoded = [], []
    for z in values:
        e = math.frexp(max(abs(z.real), abs(z.imag)))[1] - 21
        r, i = round(math.ldexp(z.real, -e)), round(math.ldexp(z.imag, -e))
        if max(abs(r), abs(i)) >= 2**21:
            e += 1
            r, i = round(math.ldexp(z.real, -e)), round(math.ldexp(z.imag, -e))
        words.append(packed(r, i, e))
        decoded.append(complex(math.ldexp(r, e), math.ldexp(i, e)))
    return words, np.array(decoded)


def independent_fit(raw, codes, coarse):
    t = np.arange(len(raw))
    corrected = raw * np.exp(-1j * coarse * math.tau / TURN * t)
    reference = np.exp(1j * np.array(codes) * math.tau / 16)
    slope, phase = np.polyfit(t, np.unwrap(np.angle(corrected * reference.conj())), 1)
    return dict(
        phase=phase,
        slope=slope + coarse * math.tau / TURN,
        gain=max(float(np.mean(abs(corrected))), 1e-30),
        power=float(np.mean(abs(raw) ** 2)),
    )


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("peak", [30, 34])
@pytest.mark.parametrize("wide", [False, True])
def test_actual_template_float_reference_and_independent_error_bound(ft, peak, wide):
    profile = ordinary(ft, 8, 0)
    codes = compile_sync_phases(profile.sync_symbols())
    n = len(codes)
    rng = np.random.default_rng(949119 + ft)
    t = np.arange(n)
    z = (1.1 + rng.random(n)) * np.exp(1j * (np.array(codes) * math.tau / 16 + 0.391 + 0.0197 * t))
    if wide:
        z *= np.ldexp(np.ones(n), np.resize([-90, -40, 0, 30, 110], n))
    words, raw = independent_input(z)
    coarse = round(0.018 * TURN / math.tau)
    fmt = SyncFitFormat(peak)
    fit = WordSyncFit(codes, coarse, fmt)
    for word in words:
        fit.feed_word(word)
    result = fit.finish()
    assert result["valid"]
    actual, expected = fit_as_floats(result), independent_fit(raw, codes, coarse)
    # 32 rounding perturbations,vector growth<2.33;normalization >=2**(peak-1).
    angular_bound = 160 / 2**peak + 2**-30 + 32 * math.pi / TURN
    assert abs(math.remainder(actual["phase"] - expected["phase"], math.tau)) <= 2.5 * angular_bound
    assert (
        abs(actual["slope"] - expected["slope"])
        <= 3 * n / (n * n - 1) * angular_bound + math.tau / TURN
    )
    assert abs(actual["gain"] / expected["gain"] - 1) < 1e-6
    assert abs(actual["power"] / expected["power"] - 1) < 1e-6
    if wide:
        assert result["magnitude_align_erased"] and result["power_align_erased"]
    assert fit.finish() == result
    with pytest.raises(RuntimeError):
        fit.feed_word(words[0])


def test_offline_constants_and_existing_reference_floor_are_independent():
    for k, v in enumerate(ATAN_TURN40):
        assert abs(v - math.atan(2**-k) / math.tau * TURN) <= 0.5001
    assert math.ldexp(*GAIN_FLOOR) == 1e-30


@pytest.mark.parametrize("peak", [30, 34])
def test_CORDIC_axes_and_small_random_vectors_have_bounded_angles(peak):
    fmt = SyncFitFormat(peak)
    rng = np.random.default_rng(949120)
    values = [(1, 0), (-1, 0), (0, 1), (0, -1), (2**21 - 1, -(2**21))]
    values += [(int(a), int(b)) for a, b in rng.integers(-(2**21), 2**21, (80, 2))]
    for r, i in values:
        job = prepare_vector(packed(r, i, -17), fmt)
        for k in range(32):
            job["x"], job["y"], job["angle"] = vector_step(job["x"], job["y"], job["angle"], k, fmt)
        angle, _, _ = finish_vector(job)
        error = math.remainder(angle * math.tau / TURN - math.atan2(i, r), math.tau)
        assert abs(error) <= 160 / 2**peak + 2**-30 + 32 * math.pi / TURN
        if not r or not i:
            assert abs(error) < 1e-14


def test_exact_half_turn_ties_do_not_invent_phase_unwrap_or_CFO_choice():
    n = 32
    fit = WordSyncFit([0] * n, HALF)
    phi = [-HALF if k % 2 == 0 else 0 for k in range(n)]
    for _ in range(n):
        fit.feed_word(packed(-100, 0, 0))
    result = fit.finish()
    w = sum((2 * k - n + 1) * v for k, v in enumerate(phi))
    expected = round(Fraction(w, n * (n * n - 1) // 6))
    assert result["residual_slope_turn40"] == expected
    assert result["frequency_turn40"] == expected + HALF
    assert result["stats"]["unwrap_events"] == 0


def test_power_sum_independent_exact_fraction_and_zero_canonicalization():
    s = PositiveWordSum()
    values = [(1, -256), (12345, -20), (2**43, 254), (1, -256)]
    for v, e in values:
        s.advance(v, e)
    exact = sum(Fraction(v) * Fraction(2) ** e for v, e in values)
    actual = Fraction(s.mantissa) * Fraction(2) ** s.exponent
    assert abs(actual - exact) <= len(values) * Fraction(2) ** s.exponent
    assert s.erased > 0
    with pytest.raises(ValueError):
        s.advance(-1, 0)
    with pytest.raises(OverflowError):
        s.advance(1, 10000)


def test_zero_phase_truncation_and_abort_are_not_success():
    fit = WordSyncFit([0] * 32, 0)
    with pytest.raises(RuntimeError):
        fit.finish()
    for _ in range(32):
        fit.feed_word(0)
    result = fit.finish()
    assert not result["valid"] and result["reason"] == "zero_sync_phase"
    assert result["power"] == (0, 0)
    fit.abort()
    with pytest.raises(RuntimeError):
        fit.finish()
    with pytest.raises(RuntimeError):
        fit.feed_word(0)


def test_compiled_format_context_and_counter_errors():
    for p in (True, 31, 33, None):
        with pytest.raises(ValueError):
            SyncFitFormat(p)
    for codes, cfo, fmt in [
        ([0] * 31, 0, None),
        ([True] * 32, 0, None),
        ([16] * 32, 0, None),
        ([0] * 32, HALF + 1, None),
        ([0] * 32, True, None),
        ([0] * 32, 0, object()),
    ]:
        with pytest.raises(ValueError):
            WordSyncFit(codes, cfo, fmt)
    for template in ([0] * 32, [complex(float("nan"), 0)] * 32, [complex(1, 0.01)] * 32, [1] * 31):
        with pytest.raises(ValueError):
            compile_sync_phases(template)
    fit = WordSyncFit([0] * 32, 0)
    for word in (-1, 2**52, True, 1.5):
        with pytest.raises(ValueError):
            fit.feed_word(word)
    fit.stats["axis_vectors"] = 2**64 - 1
    with pytest.raises(OverflowError):
        fit.feed_word(packed(100, 0, 0))


def test_independent_input_rejects_exponent_wrap_instead_of_changing_truth():
    for r, i, e in [(1, 0, -129), (1, 0, 128), (2**21, 0, 0), (0, -(2**21) - 1, 0)]:
        with pytest.raises(ValueError):
            packed(r, i, e)
