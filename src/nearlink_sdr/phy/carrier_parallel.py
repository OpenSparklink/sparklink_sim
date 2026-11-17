"""Experimental parallel-angle carrier (§6.2.1.2); explicit B2 alternative.

Only detector coordinate changes: raw Q10IQ angle minus current NCO phase.
The frozen B2 rotated-word path remains unchanged. No CRC/truth selection.
"""

import numpy as np

from nearlink_sdr.phy.carrier_fixed import (
    _FREQUENCY_BOUNDS,
    CarrierWordFormat,
    FixedCarrierLoop,
    _rne,
    _word,
    cordic_phase_word,
)
from nearlink_sdr.phy.sync_coherent import phase_rotation_words

__all__ = ["ParallelAngleCarrierLoop", "track_parallel_carrier"]

METHOD = "parallel raw-angle integer carrier PI;actual header preroll;no expected bits"


class ParallelAngleCarrierLoop(FixedCarrierLoop):
    """One finite packet; inherited state/limits and exactly18 CORDIC iterations.

    This isolated experimental transition retains B2 as its frozen reference.
    Raw angle bypasses rotation rounding/saturation in the detector only.
    """

    def __init__(self, modulation, start_symbol_index, *, fmt=None):
        super().__init__(modulation, start_symbol_index, fmt=fmt)
        if self.fmt.fraction_bits != 10:
            raise ValueError("Parallel detector study is explicitly Q10/18iterations")

    def feed_words(self, real, imag):
        """No float operations in the transition; output arrays belong to the caller."""
        if self.closed:
            raise RuntimeError("Cannot feed a closed fixed carrier packet")
        if (
            np.ndim(real) != 1
            or np.ndim(imag) != 1
            or not 1 <= len(real) <= 32
            or len(real) != len(imag)
        ):
            raise ValueError("Need1..32 pairs of component words")
        pairs = list(zip(map(_word, real), map(_word, imag), strict=True))
        if self.used + len(pairs) > 2**17:
            raise OverflowError("Fixed carrier packet count exhausted")
        result = {
            k: [] for k in ("real", "imag", "phase_word", "frequency_word", "error_word", "held")
        }
        for a, b in pairs:
            phase, frequency = self.phase, self.frequency
            c, d = phase_rotation_words(phase & ((1 << 32) - 1))
            rotated = (_rne(a * c - b * d, 16), _rne(a * d + b * c, 16))
            z = []
            for word in rotated:
                self.stats["output_saturations"] += int(not -(1 << 17) <= word < 1 << 17)
                z.append(min((1 << 17) - 1, max(-(1 << 17), word)))
            power = _rne(a * a + b * b, 2 * self.fmt.fraction_bits - 16)
            self.stats["power_erased"] += int(a * a + b * b > 0 and power == 0)
            previous = self.power.power
            low, high = power * 16 < previous, power > 4 * previous
            self.power.advance_words([power])
            self.stats["low_power_holds"] += int(low)
            self.stats["high_power_holds"] += int(high)
            error = 0
            if not (low or high):
                try:
                    angle = cordic_phase_word(a, b)
                except ArithmeticError:
                    self.stats["cordic_overflows"] += 1
                    raise
                if angle is None:
                    self.stats["zero_vector_holds"] += 1
                else:
                    angle -= phase
                    if (self.start + self.used) & 1:
                        angle += (1 << 31) // self.m
                    sector = (1 << 32) // self.m
                    error = angle % sector - sector // 2
                    limit = sector // 4
                    self.stats["error_clamps"] += int(abs(error) > limit)
                    error = min(limit, max(-limit, error))
            proposed = frequency + _rne(error, 11)
            bound = _FREQUENCY_BOUNDS[self.m]
            # All allowedM divide powers of two; denominator10M never ties.
            self.frequency = min(bound, max(-bound, proposed))
            self.stats["frequency_clamps"] += int(abs(proposed) > bound)
            following = phase + self.frequency + _rne(error, 5)
            self.stats["phase_wraps"] += int(not -(1 << 31) <= following < 1 << 31)
            self.phase = (following + (1 << 31)) % (1 << 32) - (1 << 31)
            for key, value in zip(result, (*z, phase, frequency, error, low or high), strict=True):
                result[key].append(value)
            self.used += 1
        return {
            k: np.asarray(v, dtype=bool if k == "held" else np.int64) for k, v in result.items()
        }

    def finish(self):
        info = super().finish()
        info.update(method=METHOD, detector="raw CORDIC minus current phase", iterations=18)
        return info


def track_parallel_carrier(
    symbols,
    mod_type,
    start_symbol_index,
    *,
    fraction_bits=10,
    header_symbols=None,
    header_mod_type=None,
):
    """Explicit float/word bridge; normalization and full output/trace remain outside."""
    from nearlink_sdr.phy.carrier_loop import PSKCarrierLoop, _samples

    context = PSKCarrierLoop(mod_type, start_symbol_index)
    fmt = CarrierWordFormat(fraction_bits)
    y = _samples(symbols)
    warm = np.empty(0, complex) if header_symbols is None else _samples(header_symbols)
    if (
        not 1 <= len(y) <= 2**17
        or len(y) + len(warm) > 2**17
        or len(warm) > 512
        or len(warm) > context.start
        or (header_symbols is not None and not len(warm))
        or (header_symbols is None and header_mod_type is not None)
    ):
        raise ValueError("Need bounded actual header/data context")
    loop = ParallelAngleCarrierLoop(
        header_mod_type if len(warm) else mod_type, context.start - len(warm), fmt=fmt
    )

    def feed(values):
        words = []
        for component in (values.real, values.imag):
            raw = np.rint(component * (1 << fmt.fraction_bits))
            loop.stats["input_saturations"] += int(
                np.count_nonzero((raw < -(1 << 17)) | (raw >= 1 << 17))
            )
            loop.stats["input_erased"] += int(np.count_nonzero((component != 0) & (raw == 0)))
            words.append(np.clip(raw, -(1 << 17), (1 << 17) - 1).astype(np.int64))
        return loop.feed_words(*words)

    for k in range(0, len(warm), 32):
        feed(warm[k : k + 32])
    loop.switch_modulation(mod_type)
    parts = [feed(y[k : k + 32]) for k in range(0, len(y), 32)]
    trace = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    info = loop.finish()
    info.update(
        symbols=len(y),
        start_symbol_index=context.start,
        header_symbols=len(warm),
        header_modulation=header_mod_type,
        total_processed_symbols=len(y) + len(warm),
    )
    output = (trace["real"] + 1j * trace["imag"]) / (1 << fmt.fraction_bits)
    return output, info, trace
