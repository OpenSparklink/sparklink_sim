"""Bounded known-sync residual energy and an integer soft-demapper gain."""

from dataclasses import dataclass
from math import isfinite

__all__ = ["PSKSoftGain", "PSKSyncGain", "prepare_psk_sync_gain"]

_CLOCKS = (100_000_000, 160_000_000, 200_000_000)
_RESIDUAL_DELAY = 4
_DIVISION_EDGES = 19


@dataclass(frozen=True)
class PSKSoftGain:
    """Normalized Q17 mantissa times a binary exponent, available at ready_edge."""

    mantissa: int
    exponent: int
    ready_edge: int

    def __post_init__(self):
        if (any(type(v) is not int for v in (self.mantissa, self.exponent, self.ready_edge))
                or not 65536 <= self.mantissa <= 131071
                or not -128 <= self.exponent <= 41 or self.ready_edge < 0):
            raise ValueError("Invalid registered PSK gain words")


def _q14(part):
    if not isfinite(part):
        raise ValueError("Need finite normalized I/Q")
    value = round(max(-8., min(8., part)) * 16384)
    clip = min(131071, max(-131072, value))
    return clip, clip != value or not -8.0 <= part <= 8.0


def _residual_square(residual):
    # One 17-by-17 multiplication; the possible 18th magnitude bit only
    # selects shift/add corrections. Both input words are signed18 Q14.
    low = residual & 0x1FFFF
    if residual >> 17:
        return low * low + (low << 18) + (1 << 34)
    return low * low


def _normalize_ratio(numerator, denominator):
    order = numerator.bit_length() - denominator.bit_length()
    top_num = numerator << max(0, -order)
    top_den = denominator << max(0, order)
    above = top_num >= top_den
    exponent = order + int(above)
    if exponent >= 0:
        denominator <<= exponent
    else:
        numerator <<= -exponent
    assert numerator < denominator <= 2 * numerator < 2**46
    return numerator, denominator, exponent


class PSKSyncGain:
    """Accept one Q14 observed/reference pair, with one sample of live state.

    Four registered sample edges precede the energy update. After the last
    sample, one normalization edge, 17 restoring quotient edges and one
    nearest-even rounding edge produce the gain. Observer events are diagnostic.
    """

    def __init__(self, count, *, observer=None):
        if type(count) is not int or not 1 <= count <= 256:
            raise ValueError("Need 1 to 256 known sync samples")
        self.expected = count
        self.samples = self.energy = 0
        self.squares = self.division_steps = 0
        self.edge = -1
        self.next_edge = self.result = None
        self._mode = "idle"
        self._pending = None
        self._residual = None
        self._square_i = self._square_q = 0
        self._remainder = self._divisor = self._exponent = self._fraction = 0
        self.observer = observer

    def _report(self, kind, edge, **fields):
        if self.observer is not None:
            self.observer(dict(kind=kind, edge=edge, **fields))

    def accept(self, edge, sample, reference):
        """Capture two signed Q14 pairs when the single sample slot is free."""
        self.advance(edge)
        if self._mode != "idle":
            raise ValueError("Sync gain sample slot is busy or already complete")
        words = (*sample, *reference)
        if (len(sample) != 2 or len(reference) != 2
                or any(type(x) is not int or not -(1 << 17) <= x < 1 << 17 for x in words)):
            raise ValueError("Need two signed18 Q14 I/Q pairs")
        self._pending = words
        self.next_edge = edge + 1
        self._mode = "subtract"
        self._report("accept", edge)

    def _step(self, edge):
        self.next_edge = edge + 1
        if self._mode == "subtract":
            i, q, ti, tq = self._pending
            self._residual = abs(i-ti), abs(q-tq)
            self._pending = None
            self._mode = "square_i"
        elif self._mode == "square_i":
            self._square_i = _residual_square(self._residual[0])
            self.squares += 1
            self._mode = "square_q"
        elif self._mode == "square_q":
            self._square_q = _residual_square(self._residual[1])
            self.squares += 1
            self._mode = "accumulate"
        elif self._mode == "accumulate":
            self.energy += self._square_i + self._square_q
            self.samples += 1
            assert 0 <= self.energy < 1 << 45
            self._report("accumulate", edge, samples=self.samples, energy=self.energy)
            if self.samples < self.expected:
                self.next_edge = None
                self._mode = "idle"
            else:
                self._mode = "normalize"
        elif self._mode == "normalize":
            # The smallest positive Q14 variance here is 1/(256*2**28),
            # greater than the 1e-12 floor. Only an exactly zero sum floors.
            num, den = ((2*self.expected) << 28, self.energy) if self.energy else (
                2_000_000_000_000, 1,
            )
            self._remainder, self._divisor, self._exponent = _normalize_ratio(num, den)
            self._mode = "divide"
            self._report("normalize", edge, exponent=self._exponent)
        elif self._mode == "divide":
            self._remainder <<= 1
            bit = self._remainder >= self._divisor
            if bit:
                self._remainder -= self._divisor
            self._fraction = (self._fraction << 1) | bit
            self.division_steps += 1
            self._report("quotient", edge, step=self.division_steps)
            if self.division_steps == 17:
                self._mode = "round"
        else:
            assert self._mode == "round"
            twice = self._remainder << 1
            m, e = self._fraction, self._exponent
            m += int(twice > self._divisor or (twice == self._divisor and m & 1))
            if m == 131072:
                m, e = 65536, e+1
            self.result = PSKSoftGain(m, e, edge)
            self.next_edge = None
            self._mode = "done"
            self._report("ready", edge, mantissa=m, exponent=e)

    def advance(self, edge):
        """Retire due clock edges; gain stays unavailable before its final edge."""
        if type(edge) is not int or edge < 0 or edge < self.edge:
            raise ValueError("Sync gain clock must be monotonic and nonnegative")
        while self.next_edge is not None and self.next_edge <= edge:
            self._step(self.next_edge)
        self.edge = edge
        return self.result


def prepare_psk_sync_gain(symbols, reference, *, first_edge, clock_hz, observer=None):
    """Offline quantization adapter for supplied known-sync symbols and indices.

    The paired reference has a known length. Symbol normalization and template
    selection are upstream services; this function charges gain readiness from
    the supplied input edges.
    """
    if type(clock_hz) is not int or clock_hz not in _CLOCKS:
        raise ValueError("Need 100, 160 or 200 MHz PSK clock")
    if type(first_edge) is not int or first_edge < 0:
        raise ValueError("Need a nonnegative first sync edge")
    estimator = PSKSyncGain(len(reference), observer=observer)
    period = clock_hz // 4_000_000
    clipped = 0
    for k, (sample, template) in enumerate(zip(symbols, reference, strict=True)):
        a, b = complex(sample), complex(template)
        fields = tuple(_q14(v) for v in (a.real, a.imag, b.real, b.imag))
        clipped += sum(flag for _, flag in fields)
        values = tuple(x for x, _ in fields)
        estimator.accept(first_edge + k*period, values[:2], values[2:])
    last = first_edge + (len(reference)-1)*period
    gain = estimator.advance(last + _RESIDUAL_DELAY + _DIVISION_EDGES)
    return gain, dict(energy=estimator.energy, samples=estimator.samples, squares=estimator.squares,
                      division_steps=estimator.division_steps, clipped_components=clipped,
                      last_input_edge=last, result_delay_edges=gain.ready_edge-last)
