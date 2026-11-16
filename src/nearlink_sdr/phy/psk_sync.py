"""Known-sync differential search shared by record and bounded stream receivers."""
from dataclasses import dataclass

import numpy as np
from numpy import correlate
from scipy.interpolate import CubicSpline

try:
    from nearlink_sdr_accel import rust_integer_sync_screen as _RustIntegerScreen
except ImportError:
    _RustIntegerScreen = None

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
                    diagnostics: dict | None = None, integer_screen: float | None = None,
                    integer_backend: str = 'python', _integer_cache: dict | None = None):
    """Select one maximum normalized differential correlation, independent of CRC/data.

    Optional bounds let a stream exclude consumed syncs and delay edge candidates
    until interpolation/sync phase fitting has enough future samples.
    """
    z, ref = np.asarray(filtered), np.asarray(reference)
    if integer_backend not in ('python', 'rust', 'auto'):
        raise ValueError('integer_backend must be python, rust or auto')
    if integer_backend == 'rust' and _RustIntegerScreen is None:
        raise RuntimeError('Rust integer sync screen is not installed')
    if diagnostics is not None:
        diagnostics.update(phase_hypotheses=0, scored_positions=0, best_score=None,
                           integer_phase_hypotheses=0, integer_scored_positions=0,
                           computed_phase_hypotheses=0, reused_phase_hypotheses=0,
                           computed_scored_positions=0, integer_evaluations=0,
                           integer_evaluated_positions=0, integer_native_screens=0,
                           integer_reference_fallbacks=0,
                           disposition="insufficient_history")
    if len(z) < 4:
        return None
    integer_cache = {}
    if subdivisions > 1 and integer_screen is not None:
        coarse = {}
        native = integer_backend != 'python' and _RustIntegerScreen is not None and len(ref) <= 128
        fallback = False
        native_phases = native_positions = 0
        if native:
            value, native_phases, native_positions = _RustIntegerScreen(
                np.ascontiguousarray(z, dtype=complex), sps,
                np.ascontiguousarray(ref, dtype=complex), float(min_start))
            # Bound roundoff conservatively for supported 32/128-symbol syncs.
            # Never change the gate: ambiguous cases use the NumPy reference.
            fallback = value is not None and abs(value-integer_screen) <= 1e-12
            integer_best = None if value is None else dict(score=value)
            coarse.update(phase_hypotheses=native_phases, scored_positions=native_positions,
                          best_score=value)
        if not native or fallback:
            integer_best = search_psk_sync(z, sps, ref, 1, min_start=min_start,
                                           diagnostics=coarse, _integer_cache=integer_cache)
        if diagnostics is not None:
            diagnostics.update(integer_phase_hypotheses=coarse['phase_hypotheses'],
                               integer_scored_positions=coarse['scored_positions'],
                               integer_native_screens=int(native),
                               integer_reference_fallbacks=int(fallback),
                               integer_evaluations=native_phases+(coarse['phase_hypotheses']
                                   if not native or fallback else 0),
                               integer_evaluated_positions=native_positions+(
                                   coarse['scored_positions'] if not native or fallback else 0))
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
        reused = spline is not None and tick % subdivisions == 0 and int(phase) in integer_cache
        if reused:
            cy, cc, cs = integer_cache[int(phase)]
            # Fine interpolation excludes the last raw sample; coarse slicing
            # includes it. Trim that phase's last window rather than reusing an
            # illegal edge maximum.
            ny = max(0, (len(z)-2-int(phase))//sps+1)
            y = cy[:ny]
        else:
            y = (z[int(phase)::sps] if spline is None else
                 spline(np.arange(phase, len(z) - 1, sps)))
        if len(y) < len(ref):
            continue
        if reused:
            count = len(y)-len(ref)+1
            corr, scores = cc[:count], cs[:count]
        else:
            dy = y[1:] * y[:-1].conj()
            corr = correlate(dy, dr, mode="valid")
            norm = np.sqrt(np.convolve(abs(dy) ** 2, np.ones(len(dr)), "valid") * ref_energy)
            scores = abs(corr) / np.maximum(norm, 1e-30)
            if _integer_cache is not None:
                _integer_cache[int(phase)] = (y, corr, scores)
        starts = phase + np.arange(len(scores)) * sps
        legal = starts >= min_start
        legal_count = int(np.count_nonzero(legal))
        if diagnostics is not None:
            diagnostics['phase_hypotheses'] += 1
            diagnostics['scored_positions'] += legal_count
            diagnostics['reused_phase_hypotheses'] += int(reused)
            diagnostics['computed_phase_hypotheses'] += int(not reused)
            diagnostics['computed_scored_positions'] += legal_count if not reused else 0
        if not legal_count:
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
