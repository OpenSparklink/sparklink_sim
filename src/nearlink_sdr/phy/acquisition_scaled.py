"""Finite per-complex-sample binary scaling for the acquisition bank (§6.5/6.6).

Exponent tags preserve amplitude through a causal normalization. This is a format
conversion, not an AGC that can improve SIR. Correlation terms are aligned to the
largest nonzero difference exponent at each new position. Cached difference
powers are aligned separately, with explicit nearest/even rounding.
"""

import operator
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.acquisition_bank import CorrelationPoint
from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, FixedDifferentialBank
from nearlink_sdr.phy.acquisition_rounding import folded_rounding_cost, folded_statistics

__all__ = ["ScaledBankFormat", "ScaledDifferentialBank", "align_words", "quantize_scaled_words"]


@dataclass(frozen=True)
class ScaledBankFormat:
    mantissa_bits: int = 10
    minimum_exponent: int = -48
    maximum_exponent: int = 32

    def __post_init__(self):
        if (
            type(self.mantissa_bits) is not int
            or not 8 <= self.mantissa_bits <= 14
            or type(self.minimum_exponent) is not int
            or type(self.maximum_exponent) is not int
            or self.minimum_exponent != -48
            or self.maximum_exponent != 32
        ):
            raise ValueError("Need mantissa8..14 and frozen exponent range[-48,32]")


def align_words(values, shifts):
    """Signed/unsigned words <=59 magnitude bits, positive right shifts, ties/even."""
    a = np.asarray(values)
    s = np.asarray(shifts)
    if (
        a.ndim != 1
        or s.shape != a.shape
        or a.dtype.kind not in "iu"
        or s.dtype.kind not in "iu"
        or np.any(s < 0)
    ):
        raise ValueError("Need integer words and nonnegative equally sized shifts")
    negative = a < 0
    magnitude = np.abs(a).astype(np.uint64)
    if np.any(magnitude >= 2**59):
        raise ValueError("Alignment word exceeds proven59-bit magnitude")
    shift = np.minimum(s, 63).astype(np.uint64)
    base = magnitude >> shift
    mask = (np.ones(len(a), np.uint64) << shift) - 1
    remainder = magnitude & mask
    half = np.ones(len(a), np.uint64) << np.maximum(shift, 1) - np.uint64(1)
    increment = (shift > 0) & ((remainder > half) | ((remainder == half) & ((base & 1) != 0)))
    result = (base + increment).astype(np.int64)
    return np.where(negative, -result, result)


def quantize_scaled_words(samples, format, *, stats):
    """Shared finite-format conversion; preserve the bank's RNE/range semantics.

    Physical value is mantissa*2**exponent. This floating adapter describes an
    input word representation; it is not a measured ADC calibration or SIR gain.
    Caller owns counters and bounds processing batches for hardware modeling.
    """
    if not isinstance(format, ScaledBankFormat) or not isinstance(stats, dict):
        raise ValueError("Need ScaledBankFormat and explicit counter dictionary")
    x = np.asarray(samples, dtype=np.clongdouble)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Need finite one-dimensional input")
    peak = np.maximum(abs(x.real), abs(x.imag))
    exponent = np.frexp(peak)[1] - (format.mantissa_bits - 2)
    stats["exponent_underflows"] = stats.get("exponent_underflows", 0) + int(
        np.count_nonzero((peak > 0) & (exponent < format.minimum_exponent))
    )
    stats["exponent_overflows"] = stats.get("exponent_overflows", 0) + int(
        np.count_nonzero(exponent > format.maximum_exponent)
    )
    exponent = np.clip(exponent, format.minimum_exponent, format.maximum_exponent)
    limit = 1 << (format.mantissa_bits - 1)
    r, i = [np.rint(np.ldexp(v, -exponent)) for v in (x.real, x.imag)]
    stats["input_saturations"] = stats.get("input_saturations", 0) + int(
        np.count_nonzero((r < -limit) | (r >= limit))
        + np.count_nonzero((i < -limit) | (i >= limit))
    )
    return (
        np.clip(r, -limit, limit - 1).astype(np.int64),
        np.clip(i, -limit, limit - 1).astype(np.int64),
        exponent.astype(np.int16),
    )


class ScaledDifferentialBank(FixedDifferentialBank):
    """Finite mantissa/exponent input and integer core; no float gate or ranking.

    Per-sample normalization has no block startup or future-sample requirement.
    Exponents may change within a packet. Window alignment is recomputed, so its
    extra shifters, max-tree and energy additions must be charged to hardware.
    """

    def __init__(
        self,
        reference,
        sps,
        window_samples,
        threshold=0.6,
        *,
        origin=0,
        input_full_scale=1024.0,
        scaled_format=None,
    ):
        self.scaled_format = scaled_format or ScaledBankFormat()
        if not isinstance(self.scaled_format, ScaledBankFormat):
            raise ValueError("Need ScaledBankFormat")
        super().__init__(
            reference,
            sps,
            window_samples,
            threshold,
            origin=origin,
            input_full_scale=input_full_scale,
            format=FixedBankFormat(self.scaled_format.mantissa_bits),
        )
        self._sample_exponent = np.zeros(sps, np.int16)
        self._difference_exponent = np.zeros(self.k * sps, np.int16)
        self._powers = np.zeros(self.k * sps, np.uint64)
        del self._energy
        self.stats.update(exponent_underflows=0, exponent_overflows=0, alignment_discarded_terms=0)

    @property
    def logical_storage_bits(self):
        # Exponent tags and folded rounding flags/counts; other work/FIFO separate.
        return (
            super().logical_storage_bits
            - self.sps * self.widths["window_energy"]
            + self.sps * 7
            + self.k * self.sps * 8
            + folded_rounding_cost(self.k)["extra_state_bits_upper"]
        )

    def quantize(self, samples):
        """RNE conversion with one mantissa headroom bit; exponent clamp is counted.

        Physical value is mantissa*2**exponent, independently for each complex
        input. ``input_full_scale`` is kept only for constructor compatibility;
        this format does not apply the old static normalization shift.
        """
        return quantize_scaled_words(samples, self.scaled_format, stats=self.stats)

    def advance(self, samples, *, inspect=False):
        x = np.asarray(samples)
        if x.ndim != 1 or not 1 <= len(x) <= 8 * self.sps or not np.isfinite(x).all():
            raise ValueError("Need1..8*SPS finite input")
        return self.advance_words(*self.quantize(x), inspect=inspect)

    def advance_words(self, real, imag, exponents, *, inspect=False):
        if (
            np.ndim(real) != 1
            or np.ndim(imag) != 1
            or np.ndim(exponents) != 1
            or not len(real) == len(imag) == len(exponents)
            or not 1 <= len(real) <= 8 * self.sps
        ):
            raise ValueError("Need equally sized1..8*SPS integer word/exponent arrays")
        try:
            r, i, ex = [list(map(operator.index, a)) for a in (real, imag, exponents)]
        except TypeError as error:
            raise ValueError("Integer input cannot round floats") from error
        bound, cfg = 1 << (self.format.input_bits - 1), self.scaled_format
        if any(not -bound <= v < bound for v in r + i) or any(
            not cfg.minimum_exponent <= v <= cfg.maximum_exponent for v in ex
        ):
            raise ValueError("Input exceeds frozen word/exponent range")
        if self.samples + len(r) + abs(self.origin) >= 2**63:
            raise OverflowError("Signed64 sample counter exhausted")
        re, im = np.r_[self._sample_real, r], np.r_[self._sample_imag, i]
        ex = np.r_[self._sample_exponent, ex]
        dr = re[self.sps :] * re[: -self.sps] + im[self.sps :] * im[: -self.sps]
        di = im[self.sps :] * re[: -self.sps] - re[self.sps :] * im[: -self.sps]
        de = ex[self.sps :] + ex[: -self.sps]
        power = (dr * dr + di * di).astype(np.uint64)
        joined = [
            np.r_[a, b]
            for a, b in (
                (self._difference_real, dr),
                (self._difference_imag, di),
                (self._difference_exponent, de),
                (self._powers, power),
            )
        ]
        # Bounded emulator batch; every row still uses only samples through its
        # own position. This does not imply parallel hardware or cheaper ports.
        indices = self.sps + np.arange(len(r))[:, None] + self.sps * np.arange(self.k)
        a, b, exponents, p = [v[indices] for v in joined]
        common = self._window_maxima(exponents, p)
        shifts = np.maximum(common[:, None] - exponents, 0)
        correlations_real, correlations_imag, energies, discarded = folded_statistics(
            a, b, p, shifts, self._coefficient_real, self._coefficient_imag
        )
        self.stats["alignment_discarded_terms"] += discarded
        points, w = [], self.widths
        for j in range(len(r)):
            cr, ci = int(correlations_real[j]), int(correlations_imag[j])
            self._checked(cr, w["correlation"], True)
            self._checked(ci, w["correlation"], True)
            energy = self._checked(int(energies[j]), w["window_energy"])
            numerator = self._checked(cr * cr + ci * ci, w["numerator"])
            left, right = (
                numerator << self.format.threshold_fractional_bits,
                energy * w["threshold_integer"],
            )
            self._checked(left, w["compare"])
            self._checked(right, w["compare"])
            index = self.samples + j
            ready = index >= self.k * self.sps
            self._hits[index % self.window_samples] = ready and energy > 0 and left >= right
            self.stats["zero_energy"] += int(ready and energy == 0)
            self.stats["new_correlations"] += int(ready)
            if inspect and ready:
                points.append(
                    CorrelationPoint(
                        self.origin + index - self.k * self.sps,
                        numerator,
                        energy,
                        self.coefficient_energy,
                    )
                )
        self._sample_real, self._sample_imag = re[-self.sps :].copy(), im[-self.sps :].copy()
        self._sample_exponent = ex[-self.sps :].copy()
        (self._difference_real, self._difference_imag, self._difference_exponent, self._powers) = [
            v[-self.k * self.sps :].copy() for v in joined
        ]
        self.samples += len(r)
        self.stats["samples"] += len(r)
        return points if inspect else None

    def _window_maxima(self, exponents, powers):
        """Reference reduction; structural subclasses must preserve every position."""
        return np.max(np.where(powers > 0, exponents, -96), axis=1)

    def worst_operations_per_sample(self):
        result = super().worst_operations_per_sample()
        result.update(
            add=4 * self.k + 7 + self.k - 1 + 3,
            compare=3 + self.k - 1,
            other=dict(
                variable_rne_shifts=3 * self.k + 2,
                exponent_max_terms=self.k,
                exponent_add=1,
                leading_zero_normalization=1,
            ),
            data_reads=4 * self.k + 8,
            data_writes=11,
            rounding=folded_rounding_cost(self.k),
            scope="continuous bank plus raw energy; per-position exponent alignment; no synthesis",
        )
        return result
