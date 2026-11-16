"""Fold RNE correction bits into small accumulators before the wide C/E result.

No term saturates between alignment and accumulation. Therefore summing arithmetic
floors and signed rounding corrections is exactly the per-term integer RNE model,
including negative half ties. This changes structure, not the represented result.
"""

import math

import numpy as np

__all__ = ["folded_rounding_cost", "folded_statistics", "split_rne_words"]


def split_rne_words(values, shifts):
    """Return arithmetic floor and0/1 correction; no wide increment per term.

    Caller-owned arrays are an emulator convenience. Each hardware term needs a
    signed barrel shift and guard/sticky logic. Shifts≥60 have explicit behavior
    for inputs with magnitude<2^59; no huge mask or modulo shift is implied.
    """
    a, s = np.asarray(values), np.asarray(shifts)
    if (
        a.shape != s.shape
        or a.dtype.kind not in "iu"
        or s.dtype.kind not in "iu"
        or np.any(s < 0)
        or (a.dtype.kind == "u" and np.any(a >= 2**59))
    ):
        raise ValueError("Need integer words and equal nonnegative shift arrays")
    a = a.astype(np.int64)
    if np.any((a <= -(2**59)) | (a >= 2**59)):
        raise ValueError("Word exceeds proven59-bit magnitude")
    shift = np.minimum(s, 59).astype(np.int64)
    floor = a >> shift
    remainder = a - (floor << shift)
    half = np.ones_like(shift) << np.maximum(shift - 1, 0)
    increment = (shift > 0) & ((remainder > half) | ((remainder == half) & ((floor & 1) != 0)))
    large = s >= 60
    floor = np.where(large, np.where(a < 0, -1, 0), floor)
    increment = np.where(large, a < 0, increment)
    return floor, increment


def folded_statistics(real, imag, powers, shifts, coefficient_real, coefficient_imag):
    """Bounded batch of ≤32 new positions and ≤127 terms from the8..14-bit port.

    The physical input bound implies |d|≤2^27 and |d|²≤2^54. Correlation sums
    fit int64 and power sums fit uint64 even for127 terms. Small correction
    counts are separately sized; they are not free hardware or wide adders.
    """
    r, i, p, s = [np.asarray(a) for a in (real, imag, powers, shifts)]
    cr, ci = [np.asarray(a) for a in (coefficient_real, coefficient_imag)]
    if (
        r.ndim != 2
        or not 1 <= r.shape[0] <= 32
        or not 1 <= r.shape[1] <= 127
        or any(a.shape != r.shape for a in (i, p, s))
        or cr.shape != (r.shape[1],)
        or ci.shape != cr.shape
        or any(a.dtype.kind not in "iu" for a in (r, i, p, s, cr, ci))
        or np.any((r < -(2**27)) | (r > 2**27))
        or np.any((i < -(2**27)) | (i > 2**27))
        or np.any(p < 0)
        or np.any(p > 2**54)
        or np.any(s > 160)
        or np.any((cr < -1) | (cr > 1))
        or np.any((ci < -1) | (ci > 1))
    ):
        raise ValueError("Invalid bounded difference/power/coefficient words")
    cr, ci = cr.astype(np.int64), ci.astype(np.int64)
    fr, br = split_rne_words(r, s)
    fi, bi = split_rne_words(i, s)
    fp, bp = split_rne_words(p, 2 * s)
    br, bi = br.astype(np.int64), bi.astype(np.int64)
    correction_real = br @ cr + bi @ ci
    correction_imag = bi @ cr - br @ ci
    corrected_real = fr @ cr + fi @ ci + correction_real
    corrected_imag = fi @ cr - fr @ ci + correction_imag
    energy = np.sum(fp.astype(np.uint64), axis=1) + np.sum(bp, axis=1).astype(np.uint64)
    discarded = int(np.count_nonzero((p > 0) & (fp == 0) & ~bp))
    return corrected_real, corrected_imag, energy, discarded


def folded_rounding_cost(terms=31, lanes=4):
    """Explicit extra small state and operations, not a synthesized resource result."""
    if (
        type(terms) is not int
        or not 1 <= terms <= 127
        or type(lanes) is not int
        or not 1 <= lanes <= 8
    ):
        raise ValueError("Need1..127 terms and1..8 lanes")
    signed_bits = math.ceil(math.log2(2 * terms + 1)) + 1
    power_bits = math.ceil(math.log2(terms + 1))
    counts = 2 * signed_bits + power_bits
    return dict(
        wide_result_additions=3,
        small_count_additions_upper=5 * terms - 3,
        signed_correction_bits=signed_bits,
        energy_correction_bits=power_bits,
        variable_shifts=3 * terms,
        guard_sticky_units_per_lane=3,
        extra_state_bits_upper=3 * terms + (lanes + 1) * counts,
        state_scope="round bits plus lane/final counts; shared wide accumulators and port pipeline separate",
        evidence="integer identity and operation/state graph only; no synthesis or16MS/s acceptance",
    )
