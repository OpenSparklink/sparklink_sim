"""Finite four-point Lagrange words, matching the stream stencil (§6.2.1.2).

Input is signed19+19+8;compiled output mantissas are19 or22bit. This node quantizes
fractional phase and Horner intermediates; it is not a new timing-loop design.
"""

from dataclasses import dataclass

from numpy import bool_

from nearlink_sdr.sim.capture_fine import unpack_filter_word

__all__ = ["LagrangeWordFormat", "interpolate_filter_word", "unpack_interpolated_word"]


@dataclass(frozen=True)
class LagrangeWordFormat:
    """Compiled experiments,not an arbitrary end-user bit-width interface."""

    phase_bits: int = 16
    guard_bits: int = 0
    output_bits: int = 19
    alignment_guard_first: bool = False

    def __post_init__(self):
        if (self.phase_bits, self.guard_bits, self.output_bits) not in (
            (16, 0, 19),
            (16, 2, 19),
            (20, 2, 19),
            (16, 2, 22),
        ) or any(type(v) is not int for v in (self.phase_bits, self.guard_bits, self.output_bits)):
            raise ValueError("Need compiled phase/guard/output:16/0/19,16/2/19,20/2/19,16/2/22")
        if type(self.alignment_guard_first) is not bool or (
            self.alignment_guard_first
            and (self.phase_bits, self.guard_bits, self.output_bits) != (16, 2, 22)
        ):
            raise ValueError("Guard-before-alignment is a distinct16/2/22 compiled experiment")

    @property
    def state_bits(self):
        return 23 + self.guard_bits


def rne_shift(value, shift):
    """Signed nearest-even;large alignment shifts erase without a large shifter."""
    if shift <= 0:
        return value << -shift
    if shift > abs(value).bit_length():
        return 0
    magnitude, remainder = divmod(abs(value), 1 << shift)
    half = 1 << (shift - 1)
    magnitude += remainder > half or (remainder == half and magnitude & 1)
    return -magnitude if value < 0 else magnitude


def signed_word(value, bits):
    if not -(1 << (bits - 1)) <= value < 1 << (bits - 1):
        raise OverflowError("Lagrange intermediate exceeds its declared signed width")
    return value


def validate_query(words, phase, fmt):
    if not isinstance(fmt, LagrangeWordFormat):
        raise ValueError("Need explicit LagrangeWordFormat")
    if not isinstance(words, (tuple, list)) or len(words) != 4:
        raise ValueError("Need exactly four consecutive filter words")
    if type(phase) is not int or not 0 <= phase <= 1 << fmt.phase_bits:
        raise ValueError("Need bounded fractional phase word,including endpoint1")
    if any(isinstance(w, (bool, bool_)) for w in words):
        raise ValueError("Need words,not boolean flags")
    return tuple(unpack_filter_word(w) for w in words)


def aligned_components(triples, fmt):
    common = max((e for r, i, e in triples if r or i), default=0)
    components = tuple(
        tuple(
            rne_shift(t[k] << fmt.guard_bits, common - t[2])
            if fmt.alignment_guard_first
            else rne_shift(t[k], common - t[2]) << fmt.guard_bits
            for t in triples
        )
        for k in (0, 1)
    )
    erased = sum(
        t[k] != 0 and value == 0
        for k, values in enumerate(components)
        for t, value in zip(triples, values, strict=True)
    )
    return common, components, erased


def prepare_coefficients(values, bits):
    a, b, c, d = values
    return tuple(
        signed_word(x, bits)
        for x in (-a + 3 * b - 3 * c + d, 3 * a - 6 * b + 3 * c, -2 * a - 3 * b + 6 * c - d, 6 * b)
    )


def nearest_six(value):
    quotient, remainder = divmod(abs(value), 6)
    quotient += remainder > 3 or (remainder == 3 and quotient & 1)
    return -quotient if value < 0 else quotient


def pack_interpolated(real, imag, common, fmt):
    peak = max(abs(real), abs(imag))
    if not peak:
        return 0, dict(output_saturations=0, output_underflow=False, output_shift=0)
    shift = max(0, peak.bit_length() - (fmt.output_bits - 1))
    exponent = max(-128, min(127, common - fmt.guard_bits + shift))
    shift = exponent - common + fmt.guard_bits
    r, i = rne_shift(real, shift), rne_shift(imag, shift)
    if max(r, i) >= 2 ** (fmt.output_bits - 1) and exponent < 127:
        exponent += 1
        shift += 1
        r, i = rne_shift(real, shift), rne_shift(imag, shift)
    limit = 2 ** (fmt.output_bits - 1)
    saturations = int(not -limit <= r < limit) + int(not -limit <= i < limit)
    r, i = max(-limit, min(limit - 1, r)), max(-limit, min(limit - 1, i))
    return (
        (r & ((1 << fmt.output_bits) - 1))
        | ((i & ((1 << fmt.output_bits) - 1)) << fmt.output_bits)
        | ((exponent & 255) << (2 * fmt.output_bits))
    ), dict(
        output_saturations=saturations,
        output_underflow=common - fmt.guard_bits + max(0, peak.bit_length() - (fmt.output_bits - 1))
        < -128,
        output_shift=shift,
    )


def interpolate_filter_word(words, phase, fmt):
    """RNE alignment → denominator6 Horner → nearest-even /6 → normalized output.

    No floating evaluation,phase search,payload,CRC or timing truth is used.
    Inclusive phase1 uses the same four-point stencil;no hidden future sample.
    """
    triples = validate_query(words, phase, fmt)
    common, components, erased = aligned_components(triples, fmt)
    output = []
    for values in components:
        coefficients = prepare_coefficients(values, fmt.state_bits)
        value = coefficients[0]
        for coefficient in coefficients[1:]:
            product = signed_word(value * phase, fmt.state_bits + fmt.phase_bits + 1)
            value = signed_word(rne_shift(product, fmt.phase_bits) + coefficient, fmt.state_bits)
        output.append(nearest_six(value))
    word, info = pack_interpolated(*output, common, fmt)
    return word, dict(common_exponent=common, aligned_components_erased=erased, **info)


def unpack_interpolated_word(word, fmt):
    """Typed output layout;never parse a wider interpolant as an input46bit word."""
    if not isinstance(fmt, LagrangeWordFormat):
        raise ValueError("Need explicit compiled interpolation format")
    width = fmt.output_bits
    if type(word) is not int or not 0 <= word < 1 << (2 * width + 8):
        raise ValueError("Need bounded interpolant word for the compiled output width")

    def signed(value, bits):
        return value - (1 << bits) if value & (1 << (bits - 1)) else value

    mask = (1 << width) - 1
    return (
        signed(word & mask, width),
        signed((word >> width) & mask, width),
        signed(word >> (2 * width), 8),
    )
