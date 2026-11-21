"""Finite RRC19+19+8→existing scaled-bank words (§6.2.1.2/6.2.3).

This preserves the existing floating adapter's RNE,range flags and zero exponent.
It changes representation plumbing,not gain,SIR,thresholds or coarse ranking.
"""

import operator

import numpy as np

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat
from nearlink_sdr.phy.interpolation_word import rne_shift
from nearlink_sdr.sim.capture_fine import pack_filter_word, unpack_filter_word

__all__ = ["rrc_to_scaled_word", "rrc_to_scaled_words"]


def _normalize(word, fmt):
    if not isinstance(fmt, ScaledBankFormat):
        raise ValueError("Need the existing compiled scaled-bank format")
    r, i, exponent = unpack_filter_word(word)
    if not -96 <= exponent <= 48:
        raise ValueError("RRC output exponent exceeds frozen[-96,48] range")
    peak = max(abs(r), abs(i))
    native = exponent + peak.bit_length() - (fmt.mantissa_bits - 2)
    if not peak:
        native = -(fmt.mantissa_bits - 2)  # Existing frexp(0) adapter,not input exponent.
    output_exponent = min(32, max(-48, native))
    shift = output_exponent - exponent if peak else 0
    return r, i, output_exponent, shift, native


def _round_components(r, i, shift, bits):
    """Precheck left saturation before the existing bounded signed RNE node."""
    values, saturated = [], 0
    limit = 1 << (bits - 1)
    for v in (r, i):
        bound = limit - int(v >= 0)
        if shift < 0 and abs(v) > bound >> -shift:
            values.append(-limit if v < 0 else limit - 1)
            saturated += 1
        else:
            values.append(rne_shift(v, shift))
    return *values, saturated


def _clip(r, i, exponent, native, left_saturations, nonzero, bits):
    limit = 1 << (bits - 1)
    saturated = left_saturations + int(not -limit <= r < limit) + int(not -limit <= i < limit)
    values = (min(limit - 1, max(-limit, r)), min(limit - 1, max(-limit, i)), exponent)
    events = dict(
        exponent_underflows=int(any(nonzero) and native < -48),
        exponent_overflows=int(native > 32),
        input_saturations=saturated,
        components_erased=sum(was and v == 0 for was, v in zip(nonzero, values[:2], strict=True)),
    )
    return values, events


def rrc_to_scaled_word(word, fmt):
    """One packed46bit word→signed mantissas/exponent and explicit range events."""
    r, i, exponent, shift, native = _normalize(word, fmt)
    rr, ii, saturated = _round_components(r, i, shift, fmt.mantissa_bits)
    return _clip(rr, ii, exponent, native, saturated, (r != 0, i != 0), fmt.mantissa_bits)


def rrc_to_scaled_words(real, imag, exponents, fmt, *, stats):
    """Bounded1..32word port,updating only the three existing quantizer counters.

    Whole-batch validation precedes counter updates. The functional arrays are
    not hardware FIFOs;the registered one-credit service is modeled separately.
    """
    if not isinstance(stats, dict):
        raise ValueError("Need an explicit legacy counter dictionary")
    arrays = [np.asarray(a) for a in (real, imag, exponents)]
    if (
        any(a.ndim != 1 or a.dtype.kind not in "iu" for a in arrays)
        or not 1 <= len(arrays[0]) <= 32
        or any(len(a) != len(arrays[0]) for a in arrays)
    ):
        raise ValueError("Need equally sized1..32 integer RRC output words")
    result, counters = [], dict(exponent_underflows=0, exponent_overflows=0, input_saturations=0)
    for triple in zip(*arrays, strict=True):
        r, i, exponent = map(operator.index, triple)
        values, events = rrc_to_scaled_word(pack_filter_word(r, i, exponent), fmt)
        result.append(values)
        for key in counters:
            counters[key] += events[key]
    updated = {}
    for key, count in counters.items():
        old = stats.get(key, 0)
        if type(old) is not int or not 0 <= old < 2**64:
            raise ValueError("Need unsigned64 legacy conversion counters")
        if old + count >= 2**64:
            raise OverflowError("Finite RRC bridge counter exhausted")
        updated[key] = old + count
    stats.update(updated)
    r, i, e = zip(*result, strict=True)
    return np.asarray(r, np.int64), np.asarray(i, np.int64), np.asarray(e, np.int16)
