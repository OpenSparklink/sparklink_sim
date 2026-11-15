"""Known-sync differential search shared by record and bounded stream receivers."""
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import correlate

__all__ = ["search_psk_sync"]


def search_psk_sync(filtered: np.ndarray, sps: int, reference: np.ndarray,
                    subdivisions: int, *, min_start: float = 0, lookahead: int = 0):
    """Select one maximum normalized differential correlation, independent of CRC/data.

    Optional bounds let a stream exclude consumed syncs and delay edge candidates
    until interpolation/sync phase fitting has enough future samples.
    """
    z, ref = np.asarray(filtered), np.asarray(reference)
    if len(z) < 4:
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
        if lookahead:
            legal &= starts + (len(ref) - 1) * sps + lookahead <= len(z) - 1
        if not legal.any():
            continue
        scores[~legal] = -np.inf
        offset = int(scores.argmax())
        score = float(scores[offset])
        if best is None or score > best['score']:
            best = dict(score=score, phase=phase, offset=offset,
                        delta=float(np.angle(corr[offset])), phase_symbols=y)
    return best
