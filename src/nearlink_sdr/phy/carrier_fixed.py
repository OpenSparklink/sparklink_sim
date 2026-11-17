"""Integer causal carrier node (§6.2.1.2); normalized float input bridge is outside.

Fixed18-bit input/output components,32-bit turn phase/frequency,18-iteration
vectoring CORDIC and existing causal fixed-power node. No payload/CRC selection.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights
from nearlink_sdr.phy.sync_coherent import phase_rotation_words

__all__ = ["CarrierWordFormat", "FixedCarrierLoop", "cordic_phase_word", "track_fixed_carrier"]

_ATAN_WORDS = tuple(round(math.atan(2**-k) / (2 * math.pi) * 2**32) for k in range(18))
METHOD = "integer causal carrier PI;actual header preroll;no expected bits"
_FREQUENCY_BOUNDS = {2: 214748365, 4: 107374182, 8: 53687091}


def _rne(value, shift):
    floor, remainder = divmod(value, 1 << shift)
    half = 1 << (shift - 1)
    return floor + int(remainder > half or (remainder == half and floor & 1))


def _word(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Need signed18-bit component words")
    try:
        value = operator.index(value)
    except TypeError as error:
        raise ValueError("Need integer component words") from error
    if not -(1 << 17) <= value < 1 << 17:
        raise ValueError("Component exceeds signed18bits")
    return value


def _modulation(modulation):
    if modulation not in ("BPSK", "QPSK", "8PSK"):
        raise ValueError("Need actual alternating PSK field modulation")
    return {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation]


def cordic_phase_word(real, imag):
    """Signed32-bit turn angle, or None for a true zero vector.

    Bounded common shift uses existing integer information, not more ADC bits.
    x/y are signed24bit; angle uses signed34bit scratch before intentional wrap.
    Arithmetic shifts are floor; coefficient generation is RNE offline.
    """
    x, y = _word(real), _word(imag)
    peak = max(abs(x), abs(y))
    if peak == 0:
        return None
    shift = 21 - peak.bit_length()
    x, y = x << shift, y << shift
    angle = 0
    if x < 0:
        x, y, angle = -x, -y, -(1 << 31)
    for k, coefficient in enumerate(_ATAN_WORDS):
        if y == 0:
            break
        sign = 1 if y > 0 else -1
        x, y = x + sign * (y >> k), y - sign * (x >> k)
        angle += sign * coefficient
        if not -(1 << 23) <= x < 1 << 23 or not -(1 << 23) <= y < 1 << 23:
            raise ArithmeticError("CORDIC signed24bit scratch overflow")
        if not -(1 << 33) <= angle < 1 << 33:
            raise ArithmeticError("CORDIC signed34bit angle overflow")
    return (angle + (1 << 31)) % (1 << 32) - (1 << 31)


@dataclass(frozen=True)
class CarrierWordFormat:
    """Two predeclared representation candidates; both use18-bit components."""

    fraction_bits: int = 10

    def __post_init__(self):
        if type(self.fraction_bits) is not int or self.fraction_bits not in (10, 12):
            raise ValueError("Need explicit Q10/Q12 carrier study format")


class FixedCarrierLoop:
    """Stateful exact-word node; one packet, at most131072 actual line symbols."""

    def __init__(self, modulation, start_symbol_index, *, fmt=None):
        self.m = _modulation(modulation)
        if type(start_symbol_index) is not int or not 0 <= start_symbol_index < 2**17:
            raise ValueError("Need17bit actual field start")
        self.fmt = CarrierWordFormat() if fmt is None else fmt
        if not isinstance(self.fmt, CarrierWordFormat):
            raise ValueError("Need CarrierWordFormat")
        self.start, self.used = start_symbol_index, 0
        self.phase = self.frequency = 0
        self.closed = False
        self.power = BoundedCarrierWeights(4, fixed=True)
        self.stats = dict(
            input_saturations=0,
            input_erased=0,
            power_erased=0,
            output_saturations=0,
            low_power_holds=0,
            high_power_holds=0,
            zero_vector_holds=0,
            error_clamps=0,
            frequency_clamps=0,
            phase_wraps=0,
            cordic_overflows=0,
        )

    def switch_modulation(self, modulation):
        if self.closed:
            raise RuntimeError("Cannot reconfigure a closed fixed carrier packet")
        self.m = _modulation(modulation)

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
                    angle = cordic_phase_word(*z)
                except ArithmeticError:
                    self.stats["cordic_overflows"] += 1
                    raise
                if angle is None:
                    self.stats["zero_vector_holds"] += 1
                else:
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
        self.closed = True
        return dict(
            method=METHOD,
            fraction_bits=self.fmt.fraction_bits,
            symbols=self.used,
            start_symbol_index=self.start,
            final_phase_word=self.phase,
            final_frequency_word=self.frequency,
            stats=dict(self.stats),
            power=dict(self.power.stats),
            state_words_exact=True,
            normalized_input_bridge_fixed=False,
            complete_receiver_fixed=False,
            actual_feedback_cycle_verified=False,
        )


def track_fixed_carrier(
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
    loop = FixedCarrierLoop(
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
