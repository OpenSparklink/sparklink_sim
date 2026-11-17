"""Finite Q12 normalized coarse-score ring and bounded maximum scan (§6.5/6.6).

Input C/E comes from the existing causal scaled bank. The exact screen gate is
preserved. Ranking truncates squared normalized scores; raw magnitude is never
used as a replacement. Serial restoring division is an executable word model.
"""

import math
import operator

import numpy as np

from nearlink_sdr.phy.acquisition_scaled import ScaledDifferentialBank

__all__ = ["RankedCoarseBank", "normalize_squared"]


def normalize_squared(numerator, energy, coefficient_energy, fractional_bits=12):
    """Restoring fractional divide, truncation and unity clamp; zero energy rejects.

    Denominator preparation precedes this unit. First-part compare and first
    fraction step share a modeled cycle; the finite schedule assumes one-cycle
    wide subtract/compare. That is not synthesis or a throughput guarantee.
    """
    try:
        n, e, a, b = map(operator.index, (numerator, energy, coefficient_energy, fractional_bits))
    except TypeError as error:
        raise ValueError("Need integer ratio words") from error
    if n < 0 or e < 0 or a < 1 or not 8 <= b <= 16:
        raise ValueError(
            "Need nonnegative words, positive coefficient energy and8..16 fraction bits"
        )
    if e == 0:
        return 0
    denominator = a * e
    if n >= denominator:
        return 1 << b
    result, remainder = 0, n
    for _ in range(b):
        remainder <<= 1
        result <<= 1
        if remainder >= denominator:
            remainder -= denominator
            result |= 1
    return result


class RankedCoarseBank(ScaledDifferentialBank):
    """One bounded score word per retained hit position; max scan only on demand."""

    def __init__(self, *args, rank_fractional_bits=12, **kwargs):
        if type(rank_fractional_bits) is not int or not 8 <= rank_fractional_bits <= 16:
            raise ValueError("Need8..16 ranking fraction bits")
        self.rank_fractional_bits = rank_fractional_bits
        super().__init__(*args, **kwargs)
        dtype = np.uint32 if rank_fractional_bits == 16 else np.uint16
        self._rank_scores = np.zeros(self.rank_history, dtype)
        self.last_rank_query = None
        self.stats.update(
            normalization_words=0,
            normalization_zero_energy=0,
            normalization_unity_clamps=0,
            ranking_word_reads=0,
        )

    @property
    def rank_history(self):
        return min(256, self.window_samples)

    @property
    def ranking_storage_bits(self):
        w = self.widths
        denominator = w["window_energy"] + math.ceil(math.log2(self.coefficient_energy))
        normalizer = 2 * denominator + 1 + self.rank_fractional_bits + 4 + 64 + 1
        scanner = (
            self.rank_fractional_bits + 1 + 64 + math.ceil(math.log2(self.rank_history)) + 1 + 128
        )
        return self.rank_history * (self.rank_fractional_bits + 1) + normalizer + scanner

    @property
    def logical_storage_bits(self):
        return super().logical_storage_bits + self.ranking_storage_bits

    def advance_words(self, real, imag, exponents, *, inspect=False):
        points = super().advance_words(real, imag, exponents, inspect=True)
        for point in points:
            slot = (point.start - self.origin + self.k * self.sps) % self.rank_history
            self._rank_scores[slot] = normalize_squared(
                point.numerator, point.energy, self.coefficient_energy, self.rank_fractional_bits
            )
            self.stats["normalization_words"] += 1
            self.stats["normalization_zero_energy"] += int(point.energy == 0)
            self.stats["normalization_unity_clamps"] += int(
                point.energy > 0 and point.numerator > self.coefficient_energy * point.energy
            )
        return points if inspect else None

    def query_best(self, lower_start, upper_start, *, phase_origin=None):
        enabled, count = super().gate(lower_start, upper_start)
        if phase_origin is None:
            phase_origin = self.origin
        try:
            phase_origin = operator.index(phase_origin)
        except TypeError as error:
            raise ValueError("Need integer phase origin") from error
        chosen, best = None, 0
        if enabled:
            hi = min(self.samples - 1, math.floor(upper_start - self.origin) + self.k * self.sps)
            lo = hi - count + 1
            if count > min(131, self.rank_history) or lo < self.samples - self.rank_history:
                raise ValueError("Ranking query exceeds the bounded receive-window contract")
            # Same phase-major tie order as the reference integer search. Scores
            # have finite precision; resulting normalized rank loss is measured.
            for phase in range(self.sps):
                first = (
                    lo + (phase - (self.origin + lo - self.k * self.sps - phase_origin)) % self.sps
                )
                for index in range(first, hi + 1, self.sps):
                    start = self.origin + index - self.k * self.sps
                    value = int(self._rank_scores[index % self.rank_history])
                    self.stats["ranking_word_reads"] += 1
                    if value > best:
                        chosen, best = start, value
        self.last_rank_query = (lower_start, upper_start, chosen)
        return enabled, count, chosen

    def worst_operations_per_sample(self):
        result = super().worst_operations_per_sample()
        result["normalized_ranking"] = dict(
            fractional_bits=self.rank_fractional_bits,
            division="serial restoring; denominator constant multiply upstream",
            iterations_per_word=self.rank_fractional_bits,
            wide_subtract_compare_units=1,
            worst_interval_cycles=self.rank_fractional_bits,
            query_scan_words_max=min(131, self.rank_history),
            extra_state_bits=self.ranking_storage_bits,
            scope="per bank; one-cycle arithmetic/ports assumed, no synthesis",
        )
        return result

    def query_peaks(self, lower_start, upper_start, *, count=1, phase_origin=None):
        """At most2 separated peaks, at most2 scans of the existing score ring.

        Gate and normalized Q12 scores are unchanged. Selection is greedy with
        four-sample suppression and reference phase-major tie order. Additional
        peak registers and scan work must be charged separately by the caller.
        """
        if type(count) is not int or count not in (1, 2):
            raise ValueError("Need1/2 bounded coarse peaks")
        if phase_origin is None:
            phase_origin = self.origin
        try:
            phase_origin = operator.index(phase_origin)
        except TypeError as error:
            raise ValueError("Need integer phase origin") from error
        enabled, queried = super().gate(lower_start, upper_start)
        chosen = []
        if enabled:
            hi = min(self.samples-1, math.floor(upper_start-self.origin)+self.k*self.sps)
            lo = hi-queried+1
            if queried > min(131, self.rank_history) or lo < self.samples-self.rank_history:
                raise ValueError("Ranking query exceeds bounded receive window")
            for _ in range(count):
                best, position = 0, None
                for phase in range(self.sps):
                    first = lo+(phase-(self.origin+lo-self.k*self.sps-phase_origin)) % self.sps
                    for index in range(first, hi+1, self.sps):
                        start = self.origin+index-self.k*self.sps
                        value = int(self._rank_scores[index % self.rank_history])
                        self.stats["ranking_word_reads"] += 1
                        if value > best and all(abs(start-p) >= 4 for p in chosen):
                            position, best = start, value
                if position is None:
                    break
                chosen.append(position)
        self.last_rank_query = (lower_start, upper_start, tuple(chosen))
        return enabled, queried, chosen
