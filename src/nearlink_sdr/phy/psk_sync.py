"""Known-sync differential search shared by record and bounded stream receivers."""
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import correlate

__all__ = ["PSKCaptureConfig", "search_psk_sync"]


@dataclass(frozen=True)
class PSKCaptureConfig:
    """Frozen development choices; calibration is explicit, not payload/CRC driven.

    One integer-grid screen precedes the existing fractional maximum search.
    It cannot select an alternative answer based on header or data validity.
    """

    energy_multiplier: float = 2.0
    integer_screen: float | None = .6

    def __post_init__(self):
        if (not np.isscalar(self.energy_multiplier) or not np.isfinite(self.energy_multiplier)
                or not 1 <= self.energy_multiplier <= 4):
            raise ValueError("Energy multiplier must be finite in [1,4]")
        if self.integer_screen is not None and (
                not np.isscalar(self.integer_screen) or not np.isfinite(self.integer_screen)
                or not 0 <= self.integer_screen <= 1):
            raise ValueError("Integer screen must be None or finite in [0,1]")

    @classmethod
    def legacy(cls):
        """Original four-times energy threshold and unfiltered fractional search."""
        return cls(4., None)


def search_psk_sync(filtered: np.ndarray, sps: int, reference: np.ndarray,
                    subdivisions: int, *, min_start: float = 0, lookahead: int = 0,
                    diagnostics: dict | None = None, integer_screen: float | None = None):
    """Select one maximum normalized differential correlation, independent of CRC/data.

    Optional bounds let a stream exclude consumed syncs and delay edge candidates
    until interpolation/sync phase fitting has enough future samples.
    """
    z, ref = np.asarray(filtered), np.asarray(reference)
    if diagnostics is not None:
        diagnostics.update(phase_hypotheses=0, scored_positions=0, best_score=None,
                           integer_phase_hypotheses=0, integer_scored_positions=0,
                           disposition="insufficient_history")
    if len(z) < 4:
        return None
    if subdivisions > 1 and integer_screen is not None:
        coarse = {}
        integer_best = search_psk_sync(z, sps, ref, 1, min_start=min_start,
                                       diagnostics=coarse)
        if diagnostics is not None:
            diagnostics.update(integer_phase_hypotheses=coarse['phase_hypotheses'],
                               integer_scored_positions=coarse['scored_positions'])
        if integer_best is None or integer_best['score'] < integer_screen:
            if diagnostics is not None:
                diagnostics.update(disposition='integer_screen_rejected',
                                   best_score=coarse['best_score'])
            return None
    spline = CubicSpline(np.arange(len(z)), z) if subdivisions > 1 else None
    dr = ref[1:] * ref[:-1].conj()
    ref_energy = np.sum(abs(dr) ** 2)
    best = None
    for tick in range(sps * subdivisions):
        phase = tick / subdivisions if spline is not None else tick
        y = (z[int(phase)::sps] if spline is None else
             spline(np.arange(phase, len(z) - 1, sps)))
        if len(y) < len(ref):
            continue
        dy = y[1:] * y[:-1].conj()
        corr = correlate(dy, dr, mode="valid")
        norm = np.sqrt(np.convolve(abs(dy) ** 2, np.ones(len(dr)), "valid") * ref_energy)
        scores = abs(corr) / np.maximum(norm, 1e-30)
        starts = phase + np.arange(len(scores)) * sps
        legal = starts >= min_start
        if diagnostics is not None:
            diagnostics['phase_hypotheses'] += 1
            diagnostics['scored_positions'] += int(legal.sum())
        if not legal.any():
            continue
        scores[~legal] = -np.inf
        offset = int(scores.argmax())
        score = float(scores[offset])
        if best is None or score > best['score']:
            best = dict(score=score, phase=phase, offset=offset,
                        delta=float(np.angle(corr[offset])), phase_symbols=y)
    if diagnostics is not None:
        diagnostics.update(best_score=None if best is None else best['score'],
                           disposition="no_legal_window" if best is None else "ready")
    if best is not None and lookahead:
        # Readiness is a gate on the observed correlation maximum, not a reason
        # to replace it with an earlier, weaker phase that happens to fit now.
        start = best["phase"] + best["offset"] * sps
        if start + (len(ref) - 1) * sps + lookahead > len(z) - 1:
            if diagnostics is not None:
                diagnostics['disposition'] = 'maximum_needs_future'
            return None
    return best
