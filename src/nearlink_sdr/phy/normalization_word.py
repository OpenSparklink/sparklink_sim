"""Finite selected-symbol rotation/gain bridge (§6.2.1.2); no carrier re-estimation.

turn40 phase, normalized30bit gain reciprocal, existing Q16 ROM, signed18
fused coefficients and signed22+22+11 output. CRC/payload/truth never enter.
"""

import math
import operator
from copy import deepcopy
from dataclasses import dataclass

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.sync_coherent import phase_rotation_words
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, TURN, signed, unsigned

__all__ = [
    "NormalizationFormat", "WordNormalizationState", "fused_coefficients",
    "normalize_complex", "pack_normalized_word", "physical_normalized_word",
    "reciprocal_gain", "rotation_coefficients", "unpack_normalized_word",
]

MAX_SYMBOLS = 1 << 17
MASK = TURN - 1


@dataclass(frozen=True)
class NormalizationFormat:
    """Two compiled phase structures; widths are implementation choices, not user knobs."""

    mode: str = "linear"

    def __post_init__(self):
        if type(self.mode) is not str or self.mode not in ("nearest", "linear"):
            raise ValueError("Need explicit nearest/linear quarter-ROM structure")


def _integer(value):
    if isinstance(value, bool):
        raise ValueError("Need an integer word, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("Need an integer word") from error


def pack_normalized_word(real, imag, exponent):
    """Canonical signed22+22+11, zero exponent0; overflow is rejected, never wrapped."""
    r, i, e = (_integer(v) for v in (real, imag, exponent))
    signed(r, 22)
    signed(i, 22)
    signed(e, 11)
    if r == i == 0:
        return 0
    return (r & ((1 << 22) - 1)) | ((i & ((1 << 22) - 1)) << 22) | ((e & 2047) << 44)


def unpack_normalized_word(word):
    word = unsigned(_integer(word), 55)
    result = []
    for shift, bits in [(0, 22), (22, 22), (44, 11)]:
        value = (word >> shift) & ((1 << bits) - 1)
        result.append(value - (1 << bits) if value & (1 << (bits - 1)) else value)
    if result[0] == result[1] == 0 and result[2] != 0:
        raise ValueError("Normalized zero must have canonical exponent0")
    return tuple(result)


def physical_normalized_word(word):
    """Exact binary64 compatibility view for actual node outputs; not an arithmetic backend."""
    r, i, e = unpack_normalized_word(word)
    return complex(math.ldexp(r, e), math.ldexp(i, e))


def normalize_complex(real, imag, exponent, bits):
    """Shared exponent, RNE and one carry correction; zero is canonical.

    Both components use the same shift. No saturation, gain control or clipping
    is hidden in this representation operation. Exponents remain signed11bit.
    """
    peak = max(abs(real), abs(imag))
    if not peak:
        return 0, 0, 0
    shift = max(0, peak.bit_length() - (bits - 1))
    r, i = rne_shift(real, shift), rne_shift(imag, shift)
    if max(abs(r), abs(i)) >= 1 << (bits - 1):
        shift += 1
        r, i = rne_shift(real, shift), rne_shift(imag, shift)
    return signed(r, bits), signed(i, bits), signed(exponent + shift, 11)


def reciprocal_gain(gain):
    """Functional integer reference: RNE(2**83 / normalized54bit M), positive30+signed11.

    The executable service performs the restoring division on actual edges.
    Normalizing the already known gain does not recover erased ADC information.
    """
    if (not isinstance(gain, (tuple, list)) or len(gain) != 2
            or any(type(v) is not int for v in gain)
            or not 0 < gain[0] < 1 << 54 or not -512 <= gain[1] < 512):
        raise ValueError("Need positive54+signed10 gain")
    m, e = gain
    shift = 54 - m.bit_length()
    m, e = m << shift, e - shift
    q, r = divmod(1 << 83, m)
    q += 2 * r > m or (2 * r == m and q & 1)
    exponent = -83 - e
    if q == 1 << 30:
        q >>= 1
        exponent += 1
    return unsigned(q, 30), signed(exponent, 11)


def rotation_coefficients(phase, fmt):
    """turn40 directly to ROM addresses; never round frequency down to32bit.

    Linear mode uses a28bit phase address (12+16) and two adjacent existing
    Q16 vectors. RNE includes the integer base in half-LSB parity.
    """
    if not isinstance(fmt, NormalizationFormat) or type(phase) is not int:
        raise ValueError("Need compiled phase format and turn40 integer")
    unsigned(phase, 40)
    if fmt.mode == "nearest":
        address = rne_shift(phase, 28) & 4095
        return phase_rotation_words(address << 20)
    phase28 = rne_shift(phase, 12) & ((1 << 28) - 1)
    address, fraction = divmod(phase28, 1 << 16)
    a = phase_rotation_words(address << 20)
    b = phase_rotation_words(((address + 1) & 4095) << 20)
    return tuple(
        signed(rne_shift((x << 16) + (y - x) * fraction, 16), 18)
        for x, y in zip(a, b, strict=True)
    )


def fused_coefficients(real, imag, inverse):
    """Literal30x18 products, shared18bit mantissas and signed11 exponent."""
    m, e = inverse
    return normalize_complex(signed(m * real, 48), signed(m * imag, 48), e - 16, 18)


class WordNormalizationState:
    """One packet's constant selected CFO/gain; no decision-directed loop or answer retry."""

    def __init__(self, initial_fit, fmt):
        if not isinstance(fmt, NormalizationFormat):
            raise ValueError("Need a compiled normalization format")
        if not isinstance(initial_fit, dict) or initial_fit.get("valid") is not True:
            raise ValueError("Need a valid finite initial fit")
        phase = initial_fit.get("phase_turn40")
        frequency = initial_fit.get("frequency_turn40")
        gain = initial_fit.get("gain")
        if (
            type(phase) is not int or not 0 <= phase < TURN
            or type(frequency) is not int or not -(1 << 41) <= frequency < 1 << 41
            or not isinstance(gain, (tuple, list)) or len(gain) != 2
            or any(type(v) is not int for v in gain)
            or not 0 < gain[0] < 1 << 54 or not -512 <= gain[1] < 512
        ):
            raise ValueError("Need turn40/signed42 and positive54+signed10 gain")
        self.fmt = fmt
        self.phase = phase
        self.frequency = frequency
        self.gain = tuple(gain)
        self.inverse = None
        self.count = 0
        self.last_source_word = self.last_normalized_word = None
        self.claimed = False
        self.stats = dict.fromkeys(
            ["gain_preparations", "accepted", "input_zero", "output_zero",
             "erased_real", "erased_imag", "intentional_modulo_wrap"], 0
        )

    def claim(self):
        if self.claimed or self.count or self.inverse is not None:
            raise RuntimeError("Need one untouched normalization owner before actual preparation")
        self.claimed = True
        return self

    def _accept_inverse(self, inverse):
        if self.inverse is not None:
            raise RuntimeError("Packet gain may be prepared exactly once")
        preparations = unsigned(self.stats["gain_preparations"] + 1, 64)
        self.inverse = tuple(inverse)
        self.stats["gain_preparations"] = preparations

    def prepare(self):
        if self.claimed:
            raise RuntimeError("Actual owner must execute gain preparation")
        if self.inverse is None:
            self._accept_inverse(reciprocal_gain(self.gain))

    def prepare_result(self, word, output, erased):
        """Actual service's registered commit macro; owner is unchanged until transfer."""
        if self.inverse is None or self.count >= MAX_SYMBOLS:
            raise RuntimeError("Need prepared gain and an unexhausted packet")
        r, i, _ = unpack_interpolated_word(word, INPUT_FORMAT)
        unpack_normalized_word(output)
        stats = self.stats.copy()
        updates = dict(accepted=1, input_zero=r == i == 0, output_zero=output == 0,
                       erased_real=erased[0], erased_imag=erased[1],
                       intentional_modulo_wrap=not 0 <= self.phase + self.frequency < TURN)
        for key, amount in updates.items():
            stats[key] = unsigned(stats[key] + int(amount), 64)
        return dict(index=self.count, phase_before=self.phase, raw_word=int(word),
                    normalized_word=int(output), phase=(self.phase + self.frequency) & MASK,
                    count=unsigned(self.count + 1, 18), stats=stats)

    def accept_registered(self, result):
        """Copy already registered NCO/counters; no arithmetic or reciprocal recomputation."""
        if result["index"] != self.count or result["phase_before"] != self.phase:
            raise RuntimeError("Normalization result is stale, duplicated or out of order")
        self.phase, self.count = result["phase"], result["count"]
        self.stats = result["stats"].copy()
        self.last_source_word = result["raw_word"]
        self.last_normalized_word = result["normalized_word"]

    def advance_word(self, word):
        if self.claimed:
            raise RuntimeError("Actual normalization owner forbids functional advancement")
        word = unsigned(_integer(word), 52)
        r, i, e = unpack_interpolated_word(word, INPUT_FORMAT)
        self.prepare()
        c, s = rotation_coefficients(self.phase, self.fmt)
        c, s, ce = fused_coefficients(c, s, self.inverse)
        real, imag = signed(r * c - i * s, 41), signed(r * s + i * c, 41)
        nr, ni, ne = normalize_complex(real, imag, e + ce, 22)
        result = self.prepare_result(
            word, pack_normalized_word(nr, ni, ne), (real != 0 and nr == 0, imag != 0 and ni == 0)
        )
        self.accept_registered(result)
        return deepcopy(result)

    def snapshot(self):
        return dict(mode=self.fmt.mode, phase_turn40=self.phase,
                    frequency_turn40=self.frequency, gain=self.gain, inverse=self.inverse,
                    count=self.count, last_source_word=self.last_source_word,
                    last_normalized_word=self.last_normalized_word, statistics=self.stats.copy())
