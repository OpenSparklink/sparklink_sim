"""Finite known-sync phase/CFO,gain and power initialization (§6.2.1.2).

Actual52bit interpolation words;32 vectoring steps;turn40;bounded moments and
unsigned54+signed10 sums. This node does not search,retry or use payload/CRC.
"""

import math
from dataclasses import dataclass

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    rne_shift,
    unpack_interpolated_word,
)
from nearlink_sdr.phy.sync_fit_constants import (
    ATAN_TURN40,
    GAIN_FLOOR,
    GAIN_OVER_N_Q40,
    POWER_OVER_N_Q40,
)

__all__ = ["SyncFitFormat", "WordSyncFit", "compile_sync_phases", "fit_as_floats"]

TURN = 1 << 40
HALF = TURN // 2
INPUT_FORMAT = LagrangeWordFormat(16, 2, 22)


def signed(value, bits):
    if not -(1 << (bits - 1)) <= value < 1 << (bits - 1):
        raise OverflowError("Sync-fit signed intermediate width exceeded")
    return value


def unsigned(value, bits):
    if not 0 <= value < 1 << bits:
        raise OverflowError("Sync-fit unsigned intermediate width exceeded")
    return value


def nearest_ratio(value, denominator):
    quotient, remainder = divmod(abs(value), denominator)
    quotient += 2 * remainder > denominator or (2 * remainder == denominator and quotient & 1)
    return -quotient if value < 0 else quotient


def wrapped(value):
    return (value + HALF) % TURN - HALF


@dataclass(frozen=True)
class SyncFitFormat:
    """Two compiled precision experiments;not arbitrary public width controls."""

    peak_bits: int = 34

    def __post_init__(self):
        if type(self.peak_bits) is not int or self.peak_bits not in (30, 34):
            raise ValueError("Need compiled CORDIC peak30 or34")

    @property
    def xy_bits(self):
        return self.peak_bits + 3


def compile_sync_phases(reference):
    """Host configuration only:actual unit PSK template → bounded4bit phase ROM."""
    if len(reference) not in (32, 62, 126):
        raise ValueError("Need actual32/62/126-symbol sync template")
    codes = []
    for z in reference:
        z = complex(z)
        if not math.isfinite(z.real) or not math.isfinite(z.imag):
            raise ValueError("Need finite actual sync template")
        code = round(math.atan2(z.imag, z.real) * 16 / math.tau) % 16
        ideal = complex(math.cos(code * math.tau / 16), math.sin(code * math.tau / 16))
        if abs(z - ideal) > 1e-12:
            raise ValueError("Sync template must be unit PSK on actual16-phase alphabet")
        codes.append(code)
    return tuple(codes)


def normalize_sum(value, exponent):
    """Nearest-even unsigned54,signed10 exponent;zero has canonical exponent0."""
    if not value:
        return 0, 0
    shift = value.bit_length() - 54
    result = rne_shift(value, shift)
    if result >= 1 << 54:
        shift += 1
        result = rne_shift(value, shift)
    return unsigned(result, 54), signed(exponent + shift, 10)


class PositiveWordSum:
    """Causal finite common-scale sum;alignment loss counted,never hidden."""

    def __init__(self):
        self.mantissa = self.exponent = self.erased = 0

    def advance(self, value, exponent):
        if value < 0:
            raise ValueError("Positive accumulator cannot accept negative energy")
        if not value:
            return
        mantissa, exponent = normalize_sum(value, exponent)
        if not self.mantissa:
            self.mantissa, self.exponent = mantissa, exponent
            return
        common = max(exponent, self.exponent)
        a, b = (
            rne_shift(self.mantissa, common - self.exponent),
            rne_shift(mantissa, common - exponent),
        )
        self.erased += int(a == 0) + int(b == 0)
        if self.erased >= 1 << 64:
            raise OverflowError("Positive accumulator erasure count exhausted")
        self.mantissa, self.exponent = normalize_sum(unsigned(a + b, 55), common)

    def scale(self, coefficient):
        unsigned(coefficient, 36)  # Compiled1/N and K/N are at most2**35.
        return normalize_sum(unsigned(self.mantissa * coefficient, 90), self.exponent - 40)


def prepare_vector(word, fmt):
    real, imag, exponent = unpack_interpolated_word(word, INPUT_FORMAT)
    peak = max(abs(real), abs(imag))
    shift = fmt.peak_bits - peak.bit_length() if peak else 0
    x, y = signed(real << shift, fmt.xy_bits), signed(imag << shift, fmt.xy_bits)
    angle = 0
    if x < 0:
        x, y, angle = -x, -y, -HALF
    return dict(real=real, imag=imag, exponent=exponent, shift=shift, x=x, y=y, angle=angle)


def vector_step(x, y, angle, index, fmt):
    direction = 1 if y >= 0 else -1
    return (
        signed(x + direction * (y >> index), fmt.xy_bits),
        signed(y - direction * (x >> index), fmt.xy_bits),
        signed(angle + direction * ATAN_TURN40[index], 42),
    )


def finish_vector(job):
    real, imag = job["real"], job["imag"]
    angle = wrapped(job["angle"])
    # Exact digital axes avoid inventing an angular error at quadrant boundaries.
    if not imag:
        angle = 0 if real >= 0 else -HALF
    elif not real:
        angle = TURN // 4 if imag > 0 else -TURN // 4
    return angle, unsigned(job["x"], 38), job["exponent"] - job["shift"]


class WordSyncFit:
    """One packet,one ordered pass,no array of observations or hidden replay.

    Template codes are actual configured sync phases. A zero vector has no
    phase:fit is explicitly invalid,not rescued using another sync/CFO answer.
    """

    def __init__(self, phase_codes, coarse_turn40, fmt=None):
        if not isinstance(phase_codes, (tuple, list)) or len(phase_codes) not in (32, 62, 126):
            raise ValueError("Need bounded actual sync phase ROM")
        if any(type(v) is not int or not 0 <= v < 16 for v in phase_codes):
            raise ValueError("Need4bit template phase codes")
        if type(coarse_turn40) is not int or not -HALF <= coarse_turn40 <= HALF:
            raise ValueError("Need bounded coarse turn40 seed,including positivehalf")
        self.fmt = SyncFitFormat() if fmt is None else fmt
        if not isinstance(self.fmt, SyncFitFormat):
            raise ValueError("Need compiled SyncFitFormat")
        self.codes, self.coarse = tuple(phase_codes), coarse_turn40
        self.used = self.sum_phase = self.sum_prefix = 0
        self.previous = self.unwrapped_phase = self.coarse_phase = 0
        self.magnitude, self.power = PositiveWordSum(), PositiveWordSum()
        self.stats = dict(zero_vectors=0, axis_vectors=0, unwrap_events=0, gain_floor_applied=0)
        self.result = None
        self.aborted = False

    def _count(self, key):
        if self.stats[key] == 2**64 - 1:
            raise OverflowError("Sync-fit statistic exhausted")
        self.stats[key] += 1

    def accumulate(self, job, raw_power):
        """Commit one already computed vector;shared by value and cycle models."""
        angle, magnitude, exponent = finish_vector(job)
        zero = not (job["real"] or job["imag"])
        if zero:
            self._count("zero_vectors")
        if not job["real"] or not job["imag"]:
            self._count("axis_vectors")
        phase = 0 if zero else wrapped(angle - (self.codes[self.used] << 36) - self.coarse_phase)
        difference = phase - self.previous
        if abs(difference) > HALF:
            difference -= TURN if difference > HALF else -TURN
            self._count("unwrap_events")
        self.unwrapped_phase = signed(self.unwrapped_phase + difference if self.used else phase, 48)
        self.sum_prefix = signed(self.sum_prefix + self.sum_phase, 64)
        self.sum_phase = signed(self.sum_phase + self.unwrapped_phase, 56)
        self.previous = phase
        self.coarse_phase = (self.coarse_phase + self.coarse) & (TURN - 1)
        self.magnitude.advance(magnitude, exponent)
        self.power.advance(unsigned(raw_power, 45), 2 * job["exponent"])
        self.used += 1

    def feed_word(self, word):
        if self.aborted or self.result is not None or self.used == len(self.codes):
            raise RuntimeError("Sync-fit packet closed or full")
        job = prepare_vector(word, self.fmt)
        for k in range(32):
            job["x"], job["y"], job["angle"] = vector_step(
                job["x"], job["y"], job["angle"], k, self.fmt
            )
        self.accumulate(job, job["real"] ** 2 + job["imag"] ** 2)

    def finish(self):
        if self.aborted or self.used != len(self.codes):
            raise RuntimeError("Incomplete or aborted sync-fit;no manufactured samples")
        if self.result is not None:
            return self.result.copy()
        n = self.used
        weighted = signed((n - 1) * self.sum_phase - 2 * self.sum_prefix, 64)
        slope = signed(nearest_ratio(weighted, n * (n * n - 1) // 6), 42)
        numerator = signed(2 * self.sum_phase - slope * (n - 1) * n, 64)
        phase = nearest_ratio(numerator, 2 * n) & (TURN - 1)
        gain = self.magnitude.scale(GAIN_OVER_N_Q40[n])
        power = self.power.scale(POWER_OVER_N_Q40[n])
        return self.complete(slope, phase, gain, power)

    def complete(self, slope, phase, gain, power):
        if not gain[0] or (gain[1], gain[0]) < (GAIN_FLOOR[1], GAIN_FLOOR[0]):
            gain = GAIN_FLOOR
            self._count("gain_floor_applied")
        self.result = dict(
            valid=self.stats["zero_vectors"] == 0,
            reason="zero_sync_phase" if self.stats["zero_vectors"] else None,
            residual_slope_turn40=slope,
            frequency_turn40=signed(slope + self.coarse, 42),
            phase_turn40=unsigned(phase, 40),
            gain=gain,
            power=power,
            symbols=self.used,
            stats=self.stats.copy(),
            magnitude_align_erased=self.magnitude.erased,
            power_align_erased=self.power.erased,
        )
        return self.result.copy()

    def abort(self):
        self.aborted = True
        self.result = None


def fit_as_floats(result):
    """Explicit host bridge for the remaining floating receiver,not word hardware."""
    if not result["valid"]:
        raise ValueError("Invalid sync-fit cannot drive the receiver")
    return dict(
        phase=result["phase_turn40"] * math.tau / TURN,
        slope=result["frequency_turn40"] * math.tau / TURN,
        gain=math.ldexp(*result["gain"]),
        power=math.ldexp(*result["power"]),
    )
