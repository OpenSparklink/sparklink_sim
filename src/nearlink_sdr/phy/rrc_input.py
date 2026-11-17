"""Alternative RRC input representation, not an ADC or interference gain claim."""

import numpy as np

__all__ = ["quantize_full_rrc_words"]


def quantize_full_rrc_words(samples, *, stats):
    """Causal signed12-bit RNE with exponent[-48,32] and no reserved headroom.

    If rounding would overflow either component, both components are rounded
    directly from the original input at exponent+1. Rounding a previously
    rounded word would double-round ties. At maximum exponent, saturation and
    range events remain explicit. This floating adapter specifies a numerical
    representation; finite ADC-to-word hardware has not been implemented.
    """
    x = np.asarray(samples, dtype=np.clongdouble)
    if not isinstance(stats, dict) or x.ndim != 1 or len(x) > 32 or not np.isfinite(x).all():
        raise ValueError("Need at most32 finite complex samples and explicit counters")
    peak = np.maximum(abs(x.real), abs(x.imag))
    native = np.where(peak > 0, np.frexp(peak)[1] - 11, 0)
    stats["exponent_underflows"] = stats.get("exponent_underflows", 0) + int(
        np.count_nonzero((peak > 0) & (native < -48))
    )
    stats["exponent_overflows"] = stats.get("exponent_overflows", 0) + int(
        np.count_nonzero(native > 32)
    )
    exponent = np.clip(native, -48, 32)
    r, i = [np.rint(np.ldexp(v, -exponent)) for v in (x.real, x.imag)]
    overflow = (r < -2048) | (r >= 2048) | (i < -2048) | (i >= 2048)
    renormalize = overflow & (exponent < 32)
    exponent = exponent + renormalize.astype(np.int32)
    stats["input_renormalizations"] = stats.get("input_renormalizations", 0) + int(
        np.count_nonzero(renormalize)
    )
    # Recompute from the original input, never from r/i.
    r, i = [np.rint(np.ldexp(v, -exponent)) for v in (x.real, x.imag)]
    stats["input_saturations"] = stats.get("input_saturations", 0) + int(
        np.count_nonzero((r < -2048) | (r >= 2048)) + np.count_nonzero((i < -2048) | (i >= 2048))
    )
    return (
        np.clip(r, -2048, 2047).astype(np.int64),
        np.clip(i, -2048, 2047).astype(np.int64),
        exponent.astype(np.int16),
    )
