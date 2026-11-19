"""Finite power/Gardner/PI/NCO on actual interpolation words (§6.2.1.2).

Three compiled experiments,not arbitrary waveform controls. RNE,explicit range
checks,power floor and rate/error clamps;no payload,CRC or decision inputs.
"""

import math
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.sync_fit_constants import GAIN_FLOOR
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, nearest_ratio, signed, unsigned
from nearlink_sdr.phy.timing import PSKTimingConfig

__all__ = ["TimingWordFormat", "WordTimingState"]

KP_Q48, KI_Q48 = 7476035428952, 29904141716
LOWER_Q32, UPPER_Q32 = 17171279249, 17188459119
ONE = 1 << 32


@dataclass(frozen=True)
class TimingWordFormat:
    """Compiled power30/error24,30/error32 or power38/error32;NCO64+32 in both."""

    power_bits: int = 30
    error_bits: int = 24

    def __post_init__(self):
        if (
            type(self.power_bits) is not int
            or type(self.error_bits) is not int
            or (self.power_bits, self.error_bits) not in [(30, 24), (30, 32), (38, 32)]
        ):
            raise ValueError("Need compiled timing power/error30/24,30/32 or38/32")

    @property
    def division_bits(self):
        return max(50, self.power_bits + self.error_bits + 2)


def positive(value, exponent, bits):
    if not value:
        return 0, 0
    shift = value.bit_length() - bits
    result = rne_shift(value, shift)
    if result >= 1 << bits:
        shift += 1
        result = rne_shift(value, shift)
    return unsigned(result, bits), signed(exponent + shift, 10)


def compare_scaled(a, ae, b, be):
    """Positive comparison without constructing an exponent-sized wide integer."""
    if not a or not b:
        return (a > 0) - (b > 0)
    ah, bh = a.bit_length() + ae, b.bit_length() + be
    if ah != bh:
        return (ah > bh) - (ah < bh)
    common = min(ae, be)  # Equal peaks bound alignment by operand widths.
    x, y = a << (ae - common), b << (be - common)
    unsigned(x, 50)
    unsigned(y, 50)
    return (x > y) - (x < y)


def power_update(old, energy, energy_exponent, fmt):
    """63P+energy with finite alignment,then/64 and normalized finite power."""
    m, e = old
    decay = unsigned((m << 6) - m, fmt.power_bits + 6)
    dm, de = positive(decay, e, 44)
    em, ee = positive(energy, energy_exponent, 44)
    common = max(exponent for value, exponent in [(dm, de), (em, ee)] if value)
    a = rne_shift(dm, common - de)
    b = rne_shift(em, common - ee) if em else 0
    value = unsigned(a + b, 45)
    result = positive(value, common - 6, fmt.power_bits)
    floor = positive(*GAIN_FLOOR, fmt.power_bits)
    floored = compare_scaled(*result, *floor) < 0
    return (
        (floor if floored else result),
        int(a == 0 and decay != 0) + int(b == 0 and energy != 0),
        floored,
    )


def ted_operands(previous, current, midpoint):
    pr, pi, pe = unpack_interpolated_word(previous, INPUT_FORMAT)
    cr, ci, ce = unpack_interpolated_word(current, INPUT_FORMAT)
    mr, mi, me = unpack_interpolated_word(midpoint, INPUT_FORMAT)
    peaks = [(max(abs(pr), abs(pi)), pe), (max(abs(cr), abs(ci)), ce)]
    common = max((peak.bit_length() + e for peak, e in peaks if peak), default=22) - 22
    aligned = [rne_shift(v, common - e) for v, e in [(pr, pe), (pi, pe), (cr, ce), (ci, ce)]]
    dr = signed(aligned[0] - aligned[2], 25)
    di = signed(aligned[1] - aligned[3], 25)
    erased = sum(v != 0 and a == 0 for v, a in zip([pr, pi, cr, ci], aligned, strict=True))
    return dr, di, mr, mi, signed(common + me, 10), erased


def division_job(numerator, exponent, power, fmt):
    """Exact clipping and bounded rounded-ratio job;huge shifts triaged first."""
    a, (p, pe) = abs(numerator), power
    negative = numerator < 0
    if compare_scaled(a, exponent, 2 * p, pe) > 0:
        return dict(bypass=(-1 if negative else 1) * (2 << fmt.error_bits), clipped=True)
    shift = fmt.error_bits + exponent - pe
    if not a or (shift < 0 and compare_scaled(2 * a, 0, p, -shift) <= 0):
        return dict(bypass=0, clipped=False)
    num = a << shift if shift >= 0 else a
    den = p if shift >= 0 else p << -shift
    return dict(
        numerator=unsigned(num, fmt.division_bits),
        denominator=unsigned(den, 49),
        negative=negative,
        clipped=False,
    )


def query(position, fraction_bits):
    cell, fraction = divmod(position, 1 << fraction_bits)
    return dict(cell=unsigned(cell, 64), phase=rne_shift(fraction, fraction_bits - 16))


class WordTimingState:
    """Packet-local finite state;arithmetic is functional,cycles supplied separately."""

    def __init__(self, power, start_tick, hold_symbols, fmt=None, config=None):
        self.fmt = TimingWordFormat() if fmt is None else fmt
        self.config = PSKTimingConfig() if config is None else config
        if not isinstance(self.fmt, TimingWordFormat) or self.config != PSKTimingConfig():
            raise ValueError("Experimental timing words support only frozen default loop gains")
        if type(start_tick) is not int or not 0 <= start_tick < 2**67:
            raise ValueError("Need nonnegative64-bit sample index plus eighth phase")
        if type(hold_symbols) is not int or hold_symbols not in (32, 62, 126):
            raise ValueError("Need actual32/62/126-symbol sync hold")
        if (
            not isinstance(power, (list, tuple))
            or len(power) != 2
            or any(type(x) is not int for x in power)
            or not 0 < power[0] < 2**54
            or not -512 <= power[1] < 512
        ):
            raise ValueError("Need positive unsigned54+signed10 initial power")
        self.power_word = positive(*power, self.fmt.power_bits)
        floor = positive(*GAIN_FLOOR, self.fmt.power_bits)
        if compare_scaled(*self.power_word, *floor) < 0:
            self.power_word = floor
        self.next_q32 = self.previous_q32 = start_tick << 29
        self.hold_symbols, self.count, self.previous_word = hold_symbols, 0, None
        self.omega_q32 = 4 * ONE
        self.error_word = 0
        self.rate_limit_events = self.error_clip_events = 0
        self.power_floor_events = self.power_alignment_erased = self.ted_alignment_erased = 0
        self.sps, self.damping, self.detector_gain = 4, 2**-0.5, 1.2
        self.kp, self.ki = KP_Q48 / 2**48, KI_Q48 / 2**48
        self.lower, self.upper = LOWER_Q32 / ONE, UPPER_Q32 / ONE

    @property
    def power(self):
        return math.ldexp(*self.power_word)

    @property
    def omega(self):
        return self.omega_q32 / ONE

    @property
    def queries(self):
        current = query(self.next_q32, 32)
        midpoint = (
            None if self.count < self.hold_symbols else query(self.previous_q32 + self.next_q32, 33)
        )
        return current, midpoint

    @property
    def required_end(self):
        return unsigned((self.next_q32 >> 32) + 3 + bool(self.next_q32 & (ONE - 1)), 64)

    def validate_words(self, current, midpoint):
        unpack_interpolated_word(current, INPUT_FORMAT)
        if (midpoint is None) != (self.count < self.hold_symbols):
            raise ValueError("Midpoint required exactly after actual sync hold")
        if midpoint is not None:
            unpack_interpolated_word(midpoint, INPUT_FORMAT)

    def commit(self, current, power, error, clipped, erased, ted_erased, floored, ki, kp):
        """One commit after finite products/division;no intermediate state mutation."""
        signed(ki, 21)
        signed(kp, 29)
        proposed = signed(self.omega_q32 + ki, 36)
        omega = max(LOWER_Q32, min(UPPER_Q32, proposed))
        step = unsigned(omega + kp, 35)
        if step == 0:
            raise OverflowError("Timing NCO cannot stop")
        next_position = unsigned(self.next_q32 + step, 96)
        counters = {
            name: unsigned(getattr(self, name) + amount, 64)
            for name, amount in [
                ("rate_limit_events", proposed != omega),
                ("error_clip_events", clipped),
                ("power_floor_events", floored),
                ("power_alignment_erased", erased),
                ("ted_alignment_erased", ted_erased),
                ("count", 1),
            ]
        }
        for name, value in counters.items():
            setattr(self, name, value)
        self.previous_q32, self.next_q32 = self.next_q32, next_position
        self.previous_word, self.power_word = int(current), power
        self.omega_q32, self.error_word = omega, signed(error, self.fmt.error_bits + 3)
        return self.snapshot()

    def advance_words(self, current, midpoint):
        self.validate_words(current, midpoint)
        r, i, e = unpack_interpolated_word(current, INPUT_FORMAT)
        energy = unsigned(r * r + i * i, 44)
        power, erased, floored = power_update(self.power_word, energy, 2 * e, self.fmt)
        error = ted_erased = 0
        clipped = False
        if midpoint is not None:
            dr, di, mr, mi, exponent, ted_erased = ted_operands(
                self.previous_word, current, midpoint
            )
            numerator = signed(dr * mr + di * mi, 48)
            job = division_job(numerator, exponent, power, self.fmt)
            clipped = job["clipped"]
            error = job.get("bypass")
            if error is None:
                error = nearest_ratio(job["numerator"], job["denominator"])
                if job["negative"]:
                    error = -error
        shift = 48 + self.fmt.error_bits - 32
        ki, kp = rne_shift(KI_Q48 * error, shift), rne_shift(KP_Q48 * error, shift)
        return self.commit(current, power, error, clipped, erased, ted_erased, floored, ki, kp)

    def snapshot(self):
        return dict(
            count=self.count,
            power=list(self.power_word),
            error_word=self.error_word,
            omega_q32=self.omega_q32,
            previous_q32=self.previous_q32,
            next_q32=self.next_q32,
            rate_limit_events=self.rate_limit_events,
            error_clip_events=self.error_clip_events,
            power_floor_events=self.power_floor_events,
            power_alignment_erased=self.power_alignment_erased,
            ted_alignment_erased=self.ted_alignment_erased,
        )

    def diagnostics(self, hold_symbols, rates):
        return dict(
            method="finite power/Gardner/PI/NCO;actual52bit interpolants",
            bandwidth=self.config.bandwidth,
            max_rate_ppm=self.config.max_rate_ppm,
            damping=self.damping,
            detector_gain=self.detector_gain,
            kp_samples=self.kp,
            ki_samples=self.ki,
            hold_symbols=hold_symbols,
            rate_limit_events=self.rate_limit_events,
            error_clip_events=self.error_clip_events,
            estimated_rate_ppm=float(np.median(rates[max(0, len(rates) - 128) :])),
            rate_state_final_ppm=float(rates[-1]),
            lock_status="not_classified",
            format=dict(
                power=self.fmt.power_bits, error_fraction=self.fmt.error_bits, nco_fraction=32
            ),
        )
