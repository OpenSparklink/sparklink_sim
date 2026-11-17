"""Experimental bounded search set with the unchanged spline/differential score.

This isolates search geometry (§6.2.3) from interpolation and carrier changes.
The reference remains the full-grid search; this module makes no recall claim.
"""

import numpy as np
from scipy.interpolate import CubicSpline

from nearlink_sdr.phy.psk_sync import search_psk_sync

__all__ = ["search_candidate_grid"]


def search_candidate_grid(
    filtered, sps, reference, subdivisions, *, min_start=0, lookahead=0,
    diagnostics=None, candidate_count=1, coarse_starts=None, interpolator="reference",
    spline_format=None, **kwargs,
):
    """Rank1/2 integer peaks, separate by4 samples, then evaluate9 phases each.

    Optional starts come from the existing bounded normalized-score ring. The
    floating diagnostic path reuses the reference integer-search cache instead.
    Both use the same normalized differential fine score and global spline as
    the reference (explicit finite preparation is optional). CRC/data/truth are
    absent. The greatest evaluated score alone controls future-sample waiting.
    """
    z, ref = np.asarray(filtered), np.asarray(reference)
    if interpolator not in ("reference", "uniform-eighth", "word-eighth"):
        raise ValueError("Unsupported explicit interpolation backend")
    if interpolator != "word-eighth" and spline_format is not None:
        raise ValueError("A word spline format requires explicit word-eighth interpolation")
    if interpolator == "word-eighth":
        from nearlink_sdr.phy.spline_fixed import SplineFixedFormat
        if not isinstance(spline_format, SplineFixedFormat):
            raise ValueError("Need an explicit finite spline word format")
    if (sps != 4 or subdivisions != 8 or type(candidate_count) is not int
            or candidate_count not in (1, 2) or z.ndim != 1 or not 4 <= len(z) <= 640
            or ref.ndim != 1 or not 2 <= len(ref) <= 128
            or not np.isfinite(z).all() or not np.isfinite(ref).all()
            or not np.isfinite([min_start, lookahead]).all() or lookahead < 0):
        raise ValueError("Need bounded SPS4/grid8 window and1/2 candidate peaks")
    if max(np.max(abs(z.real)), np.max(abs(z.imag))) > 1e70:
        raise OverflowError("Candidate search exceeds finite fourth-power range")
    d = diagnostics if diagnostics is not None else {}
    d.update(phase_hypotheses=0, scored_positions=0, best_score=None,
             integer_phase_hypotheses=0, integer_scored_positions=0,
             computed_phase_hypotheses=0, reused_phase_hypotheses=0,
             computed_scored_positions=0, integer_evaluations=0,
             integer_evaluated_positions=0, integer_native_screens=0,
             integer_reference_fallbacks=0, disposition="no_legal_window",
             coarse_ranking_positions=0, requested_peaks=0, local_trials_limit=9*candidate_count)
    if coarse_starts is None:
        cache, stats = {}, {}
        search_psk_sync(z, sps, ref, 1, min_start=min_start, diagnostics=stats,
                        _integer_cache=cache)
        d["coarse_ranking_positions"] = stats["scored_positions"]
        # The stable input order is phase-major, matching the full reference.
        ranked = [(float(score), float(phase + offset*sps))
                  for phase, (_, _, scores) in cache.items()
                  for offset, score in enumerate(scores)
                  if phase + offset*sps >= min_start and np.isfinite(score) and score > 0]
        ranked.sort(key=lambda pair: -pair[0])
        peaks = []
        for _, start in ranked:
            if all(abs(start - p) >= 4 for p in peaks):
                peaks.append(start)
            if len(peaks) == candidate_count:
                break
    else:
        peaks = list(coarse_starts)
        if (len(peaks) > candidate_count or not np.isfinite(peaks).all()
                or any(p != round(p) for p in peaks)
                or any(abs(a-b) < 4 for i, a in enumerate(peaks) for b in peaks[i+1:])):
            raise ValueError("Need bounded, separated integer candidate starts")
    d["requested_peaks"] = len(peaks)
    d["coarse_starts"] = peaks
    if not peaks:
        return None
    if interpolator == "word-eighth":
        from nearlink_sdr.phy.spline_fixed import FixedUniformCubicEighth
        spline = FixedUniformCubicEighth(z, spline_format)
        d.update(interpolation_word_stats=spline.stats,
                 interpolation_common_exponent=spline.exponent)
        if spline.stats['saturations']:
            d['disposition'] = 'interpolation_overflow'
            raise OverflowError("Fixed spline preparation saturated")
    elif interpolator == "uniform-eighth":
        from nearlink_sdr.phy.uniform_spline import UniformCubicEighth
        spline = UniformCubicEighth(z)
    else:
        spline = CubicSpline(np.arange(len(z)), z)
    dr = ref[1:] * ref[:-1].conj()
    coefficient_energy = np.sum(abs(dr)**2)
    # Merge repeated local positions before scoring, preserving reference ties.
    starts = sorted({p + tick/8 for p in peaks for tick in range(-4, 5)},
                    key=lambda start: (start % sps, start // sps))
    best = None
    for start in starts:
        times = start + np.arange(len(ref))*sps
        if start < min_start or times[-1] >= len(z)-1:
            continue
        y = spline(times)
        if interpolator == "word-eighth" and spline.stats['saturations']:
            d['disposition'] = 'interpolation_overflow'
            raise OverflowError("Fixed spline evaluation saturated")
        dy = y[1:]*y[:-1].conj()
        corr = np.correlate(dy, dr, "valid")[0]
        norm = np.sqrt(np.sum(abs(dy)**2)*coefficient_energy)
        value = float(abs(corr)/max(norm, 1e-30))
        if not np.isfinite(value):
            raise OverflowError("Candidate normalized score overflow")
        d["phase_hypotheses"] += 1
        d["computed_phase_hypotheses"] += 1
        d["scored_positions"] += 1
        d["computed_scored_positions"] += 1
        if best is None or value > best["score"]:
            best = dict(score=value, phase=float(start), offset=0,
                        delta=float(np.angle(corr)), phase_symbols=y)
    if best is None:
        return None
    d.update(best_score=best["score"], best_start=best["phase"], disposition="ready")
    if best["phase"] + (len(ref)-1)*sps + lookahead > len(z)-1:
        d["disposition"] = "maximum_needs_future"
        return None
    return best
