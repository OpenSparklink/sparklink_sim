"""Independent110-digit Machin/Taylor and Fraction references, never receiver inputs."""

from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from fractions import Fraction
from functools import lru_cache


def raw_word(real, imag, exponent):
    return (real % (1 << 22)) | ((imag % (1 << 22)) << 22) | ((exponent % 256) << 44)


def fields(word, exponent_bits=8):
    values = []
    for shift, width in [(0, 22), (22, 22), (44, exponent_bits)]:
        value = (int(word) >> shift) & ((1 << width) - 1)
        values.append(value if value < 1 << (width - 1) else value - (1 << width))
    return values


def fraction_round(value):
    floor = value.numerator // value.denominator
    remainder = value - floor
    return floor + int(remainder > Fraction(1, 2) or (remainder == Fraction(1, 2) and floor % 2))


@lru_cache(maxsize=1)
def pi():
    with localcontext() as context:
        context.prec = 115

        def atan_inverse(n):
            x = Decimal(1) / n
            power = x
            result = Decimal(0)
            for k in range(400):
                term = power / (2 * k + 1)
                result += term if k % 2 == 0 else -term
                if abs(term) < Decimal("1e-112"):
                    return result
                power *= x * x
            raise AssertionError("Independent Machin series did not converge")

        return +(16 * atan_inverse(5) - 4 * atan_inverse(239))


@lru_cache(maxsize=8192)
def trig(phase40):
    with localcontext() as context:
        context.prec = 110
        angle = Decimal(phase40) * 2 * pi() / (1 << 40)
        if angle > pi():
            angle -= 2 * pi()
        sine = st = angle
        cosine = ct = Decimal(1)
        for k in range(1, 200):
            st *= -angle * angle / ((2 * k) * (2 * k + 1))
            ct *= -angle * angle / ((2 * k - 1) * (2 * k))
            sine += st
            cosine += ct
            if max(abs(st), abs(ct)) < Decimal("1e-105"):
                return +cosine, -sine
        raise AssertionError("Independent Taylor series did not converge")


def relative_error(word, output, fit, index=0):
    with localcontext() as context:
        context.prec = 105
        r, i, e = fields(word)
        nr, ni, ne = fields(output, 11)
        c, s = trig((fit["phase_turn40"] + fit["frequency_turn40"] * index) % (1 << 40))
        gain = Decimal(fit["gain"][0]) * Decimal(2) ** fit["gain"][1]
        scale = Decimal(2) ** e / gain
        x, y = (r * c - i * s) * scale, (r * s + i * c) * scale
        a, b = Decimal(nr) * Decimal(2) ** ne, Decimal(ni) * Decimal(2) ** ne
        if not (r or i):
            assert a == b == 0
            return Decimal(0)
        return (((a - x) ** 2 + (b - y) ** 2) / (x * x + y * y)).sqrt()


def rounded_coefficient(phase40):
    with localcontext() as context:
        context.prec = 105
        return tuple(int((x * 65536).to_integral_value(rounding=ROUND_HALF_EVEN))
                     for x in trig(phase40))
