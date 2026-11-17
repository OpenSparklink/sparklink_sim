"""Causal symmetric matched RRC word model for the existing PSK reference.

Existing beta.4/SPS4/span10 coefficients are a project reference, not new RF
limits. The finite input port is mantissa/exponent words; its floating adapter
does not claim measured ADC calibration. Whole-receiver integration is separate.
"""

import operator
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.acquisition_scaled import (
    ScaledBankFormat,
    align_words,
    quantize_scaled_words,
)
from nearlink_sdr.phy.psk import rrc_filter

__all__ = ["FixedRRC", "FixedRRCFormat", "SymmetricRRC"]


def _kernel():
    h = rrc_filter(0.4, 4).copy()
    if len(h) != 81 or not np.allclose(h, h[::-1], atol=1e-15, rtol=0):
        raise ValueError("Expected existing symmetric81-tap reference")
    return (h + h[::-1]) * 0.5


@dataclass(frozen=True)
class FixedRRCFormat:
    input_mantissa_bits: int
    coefficient_bits: int
    output_mantissa_bits: int = 19

    def __post_init__(self):
        if (
            type(self.input_mantissa_bits) is not int
            or self.input_mantissa_bits not in (10, 12, 14)
            or type(self.coefficient_bits) is not int
            or self.coefficient_bits not in (14, 16, 18)
            or type(self.output_mantissa_bits) is not int
            or self.output_mantissa_bits != 19
        ):
            raise ValueError("Need trial input10/12/14, coefficient14/16/18 and output19 bits")


class SymmetricRRC:
    """Finite causal floating structure; continuously retain history across packets."""

    def __init__(self):
        self.taps = _kernel()
        self.history = np.zeros(80, complex)
        self.samples, self.closed = 0, False

    def feed(self, samples):
        x = np.asarray(samples, dtype=complex)
        if self.closed:
            raise RuntimeError("Cannot feed after finish")
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Need finite complex stream")
        if self.samples + len(x) >= 2**63:
            raise OverflowError("Signed64 sample counter exhausted")
        result = np.empty(len(x), complex)
        for start in range(0, len(x), 32):
            chunk = x[start : start + 32]
            joined = np.r_[self.history, chunk]
            windows = np.lib.stride_tricks.sliding_window_view(joined, 81)
            pairs = windows[:, :40] + windows[:, 80:40:-1]
            result[start : start + len(chunk)] = (
                pairs @ self.taps[:40] + windows[:, 40] * self.taps[40]
            )
            self.history[:] = joined[-80:]
        self.samples += len(x)
        return result

    def finish(self):
        if self.closed:
            return np.empty(0, complex)
        result = self.feed(np.zeros(80)) if self.samples else np.empty(0, complex)
        self.closed = True
        return result


class FixedRRC:
    """RNE input alignment → symmetric integer FIR →19-bit output word.

    Each row uses at most81 past/current words. Max exponent ignores zero words.
    Eighty comparisons and all shifters remain part of the structure. Batches
    ≤32 are an emulator convenience, not parallel hardware or future samples.
    Output exponent range[-96,48] and every counter are explicitly bounded.
    """

    def __init__(self, format):
        if not isinstance(format, FixedRRCFormat):
            raise ValueError("Explicit FixedRRCFormat required")
        self.format = format
        self.input_format = ScaledBankFormat(format.input_mantissa_bits)
        self.coefficient_fraction = format.coefficient_bits - 1
        self.coefficients = np.rint(_kernel() * 2**self.coefficient_fraction).astype(np.int64)
        if np.any(abs(self.coefficients) >= 2 ** (format.coefficient_bits - 1)):
            raise OverflowError("Coefficient does not fit selected signed format")
        self.coefficients.setflags(write=False)
        self.accumulator_bits = format.input_mantissa_bits + 1 + format.coefficient_bits + 6
        # Every allowed10/12/14-bit input fits signed16 storage. feed_words
        # promotes joined operands to int64 before all FIR/rounding arithmetic.
        self._r, self._i = np.zeros((2, 80), np.int16)
        self._e = np.zeros(80, np.int8)
        self.closed = False
        self.stats = dict(
            samples=0,
            exponent_underflows=0,
            exponent_overflows=0,
            input_saturations=0,
            aligned_components_erased=0,
            output_underflows=0,
            output_overflows=0,
            output_saturations=0,
            intermediate_overflows=0,
            max_accumulator_magnitude=0,
            zero_flush_samples=0,
            telemetry_saturated=False,
        )

    @property
    def logical_state_bits(self):
        # Functional word history; physical cycle pipeline/ports need extra state.
        return (
            80 * (2 * self.format.input_mantissa_bits + 8)
            + 41 * self.format.coefficient_bits
            + 2 * self.accumulator_bits
            + 11 * 64
            + 2
        )  # Explicit counters, closed and telemetry flags.

    def _pack(self, r, i, common):
        peak = max(abs(r), abs(i))
        if peak == 0:
            return 0, 0, 0
        shift = peak.bit_length() - (self.format.output_mantissa_bits - 2)
        exponent = common - self.coefficient_fraction + shift
        if exponent < -96:
            self.stats["output_underflows"] += 1
        if exponent > 48:
            self.stats["output_overflows"] += 1
        exponent = min(48, max(-96, exponent))
        shift = exponent - common + self.coefficient_fraction
        if shift > 0:
            r, i = map(int, align_words(np.array([r, i]), np.full(2, shift)))
        else:
            r, i = r << -shift, i << -shift
        limit = 1 << (self.format.output_mantissa_bits - 1)
        self.stats["output_saturations"] += int(not -limit <= r < limit) + int(
            not -limit <= i < limit
        )
        return min(limit - 1, max(-limit, r)), min(limit - 1, max(-limit, i)), exponent

    def feed_words(self, real, imag, exponents):
        """Integer input port,1..32 words; output tuple preserves finite words."""
        if self.closed:
            raise RuntimeError("Cannot feed after finish")
        arrays = [np.asarray(a) for a in (real, imag, exponents)]
        if (
            any(a.ndim != 1 for a in arrays)
            or not 1 <= len(arrays[0]) <= 32
            or any(len(a) != len(arrays[0]) or a.dtype.kind not in "iu" for a in arrays)
        ):
            raise ValueError("Need equally sized1..32 integer word arrays")
        r, i, e = [list(map(operator.index, a)) for a in arrays]
        limit = 1 << (self.format.input_mantissa_bits - 1)
        if any(not -limit <= v < limit for v in r + i) or any(not -48 <= v <= 32 for v in e):
            raise ValueError("Input word/exponent out of frozen range")
        r, i, e = [np.array(a, np.int64) for a in (r, i, e)]
        if self.stats["samples"] + len(r) >= 2**63:
            raise OverflowError("Signed64 sample counter exhausted")
        joined = [np.r_[a, b] for a, b in zip((self._r, self._i, self._e), (r, i, e), strict=True)]
        wr, wi, we = [np.lib.stride_tricks.sliding_window_view(a, 81) for a in joined]
        sums, common = self._accumulate(wr, wi, we)
        peak = max(int(np.max(abs(a), initial=0)) for a in sums)
        self.stats["max_accumulator_magnitude"] = max(self.stats["max_accumulator_magnitude"], peak)
        if peak >= 2 ** (self.accumulator_bits - 1):
            self.stats["intermediate_overflows"] += 1
            raise OverflowError("Proven accumulator range exceeded")
        words = [self._pack(int(a), int(b), int(c)) for a, b, c in zip(*sums, common, strict=True)]
        for target, value in zip((self._r, self._i, self._e), joined, strict=True):
            target[:] = value[-80:]
        self.stats["samples"] += len(r)
        self._bound_counters()
        return tuple(np.array(a, np.int64) for a in zip(*words, strict=True))

    def _accumulate(self, wr, wi, we):
        """Original global-input alignment, retained as an explicit reference arm."""
        present = (wr != 0) | (wi != 0)
        common = np.max(np.where(present, we, -48), axis=1)
        shifts = np.maximum(common[:, None] - we, 0)
        ar, ai = [align_words(v.ravel(), shifts.ravel()).reshape(v.shape) for v in (wr, wi)]
        self.stats["aligned_components_erased"] += int(
            np.count_nonzero((wr != 0) & (ar == 0)) + np.count_nonzero((wi != 0) & (ai == 0))
        )
        sums = [
            (v[:, :40] + v[:, 80:40:-1]) @ self.coefficients[:40] + v[:, 40] * self.coefficients[40]
            for v in (ar, ai)
        ]
        return sums, common

    def _bound_counters(self):
        for name, value in self.stats.items():
            if name != "telemetry_saturated" and value > 2**64 - 1:
                self.stats[name] = 2**64 - 1
                self.stats["telemetry_saturated"] = True

    def feed(self, samples):
        if self.closed:
            raise RuntimeError("Cannot feed after finish")
        x = np.asarray(samples)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError("Need finite complex stream")
        result = np.empty(len(x), complex)
        for start in range(0, len(x), 32):
            words = self._quantize(x[start : start + 32])
            r, i, e = self.feed_words(*words)
            result[start : start + len(r)] = np.ldexp(
                r.astype(float), e.astype(int)
            ) + 1j * np.ldexp(i.astype(float), e.astype(int))
        return result

    def _quantize(self, samples):
        return quantize_scaled_words(samples, self.input_format, stats=self.stats)

    def finish(self):
        if self.closed:
            return np.empty(0, complex)
        result = self.feed(np.zeros(80)) if self.stats["samples"] else np.empty(0, complex)
        self.stats["zero_flush_samples"] += len(result)
        self.seal()
        return result

    def seal(self):
        """Close without implicit padding; owner must supply its required tail.

        Standalone finish still supplies80zeros. A centered stream receiver
        supplies its own FIR/interpolation lookahead exactly once, then seals.
        """
        self.closed = True

    def operation_estimate(self):
        return dict(
            real_pre_additions=80,
            real_coefficient_products=82,
            real_accumulation_additions=80,
            gain_max_comparisons=80,
            component_alignments=162,
            conditional_input_rne_increment_additions=162,
            input_rne_increment_width=self.format.input_mantissa_bits,
            history_modulo_address_operations=243,
            coefficient_address_operations=82,
            guard_sticky_tests=162,
            output_roundings=2,
            output_leading_zero_detectors=1,
            accumulator_bits=self.accumulator_bits,
            coefficient_bits=self.format.coefficient_bits,
            coefficient_fraction=self.coefficient_fraction,
            input_exponent_bits=7,
            output_exponent_bits=8,
            logical_functional_state_bits=self.logical_state_bits,
            coefficient_unique_words=41,
            scope="per complex input; operation estimate only, physical ports/pipeline separate",
            output_word_bits=2 * self.format.output_mantissa_bits + 8,
        )
