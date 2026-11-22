"""One actual fine maximum and its bounded replayed sync words (§6.2.3).

The word cache is an explicit new allocation. Float angle/score/physical-word
bridges are reference operations,not assigned zero hardware cycles.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["FineSelection"]


@dataclass(frozen=True)
class FineSelection:
    """One credit,one ID,one original window;no payload,CRC or truth inputs."""

    ident: int
    source_end: int
    window: int
    available_cycle: int
    start_tick: int
    exponent: int
    words: tuple
    correlation: tuple
    numerator: int
    energy: int
    template_energy: int

    def __post_init__(self):
        counts = {256: 32, 376: 62, 632: 126}
        if (
            self.window not in counts
            or type(self.window) is not int
            or type(self.ident) is not int
            or not 0 <= self.ident < 2**63
            or type(self.source_end) is not int
            or not self.window <= self.source_end < 2**60
            or type(self.available_cycle) is not int
            or not 0 <= self.available_cycle < 2**63
            or type(self.start_tick) is not int
            or not 8 <= self.start_tick < 8 * self.window
            or type(self.exponent) is not int
            or not -512 <= self.exponent < 512
            or not isinstance(self.words, tuple)
            or len(self.words) != counts[self.window]
            or any(
                not isinstance(w, tuple)
                or len(w) != 2
                or any(type(v) is not int or not -(2**21) <= v < 2**21 for v in w)
                for w in self.words
            )
            or not isinstance(self.correlation, tuple)
            or len(self.correlation) != 2
            or any(type(v) is not int or not -(2**52) <= v < 2**52 for v in self.correlation)
            or type(self.numerator) is not int
            or not 0 <= self.numerator < 2**107
            or self.numerator != sum(v * v for v in self.correlation)
            or type(self.energy) is not int
            or not 0 <= self.energy < 2**98
            or self.template_energy not in (len(self.words) - 1, 2 * (len(self.words) - 1))
            or type(self.template_energy) is not int
        ):
            raise ValueError("Need one bounded actual fine maximum and ordered replay words")

    @property
    def source_start_tick(self):
        return 8 * (self.source_end - self.window) + self.start_tick

    def reference_selection(self):
        """Explicit off-clock bridge to the existing strict floating consumer."""
        values = np.asarray(self.words, dtype=float)
        sync = np.ldexp(values[:, 0], self.exponent) + 1j * np.ldexp(values[:, 1], self.exponent)
        denominator = self.energy * self.template_energy
        score = math.sqrt(self.numerator / denominator) if denominator else 0.0
        if not np.isfinite(sync).all() or not math.isfinite(score):
            raise OverflowError("Selected sync bridge exceeds floating reference range")
        # Quantized finite score can exceed1 by rounding;never clip it to pass a gate.
        if not 0 <= score <= 1:
            raise ValueError("Actual fine normalized score is outside consumer domain")
        return dict(
            start_tick=self.start_tick,
            delta=math.atan2(self.correlation[1], self.correlation[0]),
            differential_score=score,
            selected_sync=sync,
        )

    @staticmethod
    def estimate():
        fields = dict(
            ident=63,
            source_end=63,
            window=10,
            available_cycle=63,
            start_tick=13,
            exponent=10,
            correlation=106,
            numerator=107,
            energy=98,
            template_energy=8,
            count=7,
            write_read_addresses=14,
            valid=1,
        )
        return dict(
            selected_cache_bits=128 * 44,
            selected_cache_BRAM36=1,
            cache_mapping="one SDP72x512 RAMB36;128x44 logical words",
            cache_write_ports=1,
            cache_read_ports=1,
            cache_read_latency=1,
            scalar_fields=fields,
            scalar_register_bits=sum(fields.values()),
            cache_fill="existing selected_sample replay;no second spline selection",
            cache_read_angle_score_and_coherent_cycles=None,
            actual_synthesis=False,
            full_fixed_or16MSps=False,
        )
