"""Bounded one-coarse-peak/four-phase sync refinement (§6.5/6.6), opt-in.

The floating reference remains search_psk_sync. This changes the search set and
interpolator, so packet recall and true timing require separate acceptance.
"""

import numpy as np

from nearlink_sdr.phy.psk_sync import search_psk_sync

__all__ = ["local_cubic_samples", "refine_psk_sync"]


def local_cubic_samples(samples, times):
    """Four-point Lagrange with one-sided edge stencils inside the supplied window."""
    z, t = np.asarray(samples), np.asarray(times)
    if (
        z.ndim != 1
        or len(z) < 4
        or t.ndim != 1
        or not np.isfinite(z).all()
        or not np.isfinite(t).all()
        or np.any(t < 0)
        or np.any(t > len(z) - 1)
    ):
        raise ValueError("Need finite window and interpolation positions inside it")
    base = np.clip(np.floor(t).astype(int) - 1, 0, len(z) - 4)
    u = t - base
    return (
        -(u - 1) * (u - 2) * (u - 3) / 6 * z[base]
        + u * (u - 2) * (u - 3) / 2 * z[base + 1]
        - u * (u - 1) * (u - 3) / 2 * z[base + 2]
        + u * (u - 1) * (u - 2) / 6 * z[base + 3]
    )


def refine_psk_sync(
    filtered,
    sps,
    reference,
    subdivisions,
    *,
    min_start=0,
    lookahead=0,
    diagnostics=None,
    coarse_start=None,
    **kwargs,
):
    """Select by normalized known-sync correlation only; no CRC or payload input.

    Initial local trials are the integer coarse maximum and ±.5 sample. A
    parabola through squared normalized scores predicts at most one extra trial,
    clipped to that interval. The greatest actually evaluated score wins, then
    the unchanged future-sample rule is applied to that winner.
    """
    z, ref = np.asarray(filtered), np.asarray(reference)
    if (
        z.ndim != 1
        or ref.ndim != 1
        or len(z) < 4
        or not 2 <= len(ref) <= 128
        or sps != 4
        or subdivisions != 8
        or not np.isfinite(z).all()
        or not np.isfinite(ref).all()
        or not np.isfinite(min_start)
        or not np.isfinite(lookahead)
        or lookahead < 0
    ):
        raise ValueError("Local model needs finite SPS4/subdivision8 sync/window")
    if np.any(abs(z.real) > 1e70) or np.any(abs(z.imag) > 1e70):
        raise OverflowError("Local search exceeds the finite fourth-power input range")
    diagnostic = diagnostics if diagnostics is not None else {}
    diagnostic.update(
        phase_hypotheses=0,
        scored_positions=0,
        best_score=None,
        integer_phase_hypotheses=0,
        integer_scored_positions=0,
        computed_phase_hypotheses=0,
        reused_phase_hypotheses=0,
        computed_scored_positions=0,
        integer_evaluations=0,
        integer_evaluated_positions=0,
        integer_native_screens=0,
        integer_reference_fallbacks=0,
        disposition="no_legal_window",
        coarse_ranking_positions=0,
    )
    if coarse_start is None:
        coarse = {}
        found = search_psk_sync(z, sps, ref, 1, min_start=min_start, diagnostics=coarse)
        diagnostic["coarse_ranking_positions"] = coarse["scored_positions"]
        if found is None:
            return None
        coarse_start = found["phase"] + found["offset"] * sps
    if not np.isfinite(coarse_start):
        raise ValueError("Coarse start must be finite")
    dr = ref[1:] * ref[:-1].conj()
    coefficient_energy = sum(abs(dr) ** 2)

    def evaluate(start):
        times = start + np.arange(len(ref)) * sps
        if start < min_start or times[-1] >= len(z) - 1:
            return None
        y = local_cubic_samples(z, times)
        difference = y[1:] * y[:-1].conj()
        correlation = np.vdot(dr, difference)
        energy = sum(abs(difference) ** 2)
        squared = float(abs(correlation) ** 2 / max(energy * coefficient_energy, 1e-60))
        if not np.isfinite(squared):
            raise OverflowError("Local correlation exceeded finite arithmetic range")
        diagnostic["phase_hypotheses"] += 1
        diagnostic["computed_phase_hypotheses"] += 1
        diagnostic["scored_positions"] += 1
        diagnostic["computed_scored_positions"] += 1
        return dict(
            score=float(np.sqrt(squared)),
            squared=squared,
            phase=float(start),
            offset=0,
            delta=float(np.angle(correlation)),
            phase_symbols=y,
        )

    initial = [evaluate(coarse_start + delta) for delta in (-0.5, 0, 0.5)]
    candidates = [trial for trial in initial if trial is not None]
    if all(trial is not None for trial in initial):
        minus, center, plus = [trial["squared"] for trial in initial]
        curvature = minus - 2 * center + plus
        if curvature < -1e-12:
            delta = float(np.clip(0.25 * (minus - plus) / curvature, -0.5, 0.5))
            if all(abs(delta - d) > 1e-12 for d in (-0.5, 0, 0.5)):
                extra = evaluate(coarse_start + delta)
                if extra is not None:
                    candidates.append(extra)
    if not candidates:
        return None
    best = max(candidates, key=lambda trial: trial["score"])
    diagnostic.update(best_score=best["score"], disposition="ready")
    if best["phase"] + (len(ref) - 1) * sps + lookahead > len(z) - 1:
        diagnostic["disposition"] = "maximum_needs_future"
        return None
    return best
