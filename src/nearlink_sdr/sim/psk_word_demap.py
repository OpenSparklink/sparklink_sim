"""Finite PSK soft demapping from normalized wire symbols (§6.2.1.2, §6.10.4)."""

from collections import Counter, deque
from dataclasses import dataclass, replace
from functools import lru_cache
from math import cos, frexp, isfinite, pi, sin

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.scrambler import _lfsr_step
from nearlink_sdr.phy.tx_pipeline import TxConfig, _head_len
from nearlink_sdr.sim.b1_payload_service import B1PayloadFrame, replay_b1_payload_frames
from nearlink_sdr.sim.payload_polar_ingress import TimedLLR

__all__ = ["B1SymbolFrame", "PSKWordDemapper", "replay_b1_symbol_frames"]

_DELAY = 24
_CLOCK_RATES = (100_000_000, 160_000_000, 200_000_000)
_GRAY = {2: (0, 1, 3, 2), 3: (0, 1, 3, 2, 6, 7, 5, 4)}


@lru_cache(maxsize=4)
def _projection_rom(bps, parity):
    """Bake alternate rotations into coefficients of the transmitted points."""
    size = 1 << bps
    angles = [(1 + 2 * i - parity) * pi / size for i in range(size // 2)]
    return tuple((round(cos(a) * 65536), round(sin(a) * 65536)) for a in angles)


@lru_cache(maxsize=3)
def _jump_columns(length):
    states = [1 << b for b in range(7)]
    for _ in range(length):
        states = [_lfsr_step(value)[1] for value in states]
    return tuple(states)


def _jump_header(seed, length):
    state = 0
    for k, column in enumerate(_jump_columns(length)):
        if seed & (1 << k):
            state ^= column
    return state


def _signed(value, width):
    if not -(1 << (width - 1)) <= value < 1 << (width - 1):
        raise RuntimeError("PSK word exceeds declared arithmetic width")
    return value


def _variance_gain(variance):
    if not isfinite(variance) or variance <= 0:
        raise ValueError("Need finite positive normalized variance")
    ratio = 2.0 / max(variance, 1e-12)
    frac, exp = frexp(ratio)
    if exp < -128:
        return 0, -128
    mantissa = round(frac * (1 << 17))
    if mantissa == 1 << 17:
        mantissa >>= 1
        exp += 1
    return _signed(mantissa, 18), _signed(exp, 9)


def _rne(value, shift):
    sign = -1 if value < 0 else 1
    divide = 1 << shift
    quotient, residual = divmod(abs(value), divide)
    if 2 * residual > divide or (2 * residual == divide and quotient & 1):
        quotient += 1
    return sign * quotient


@dataclass(frozen=True)
class B1SymbolFrame:
    """Verified B1 context at the first normalized wire-symbol edge."""

    payload: B1PayloadFrame
    symbol_index: int
    whitening_seed: int
    variance: float
    head_coded_bits: int

    @classmethod
    def from_config(cls, first_edge: int, length_bytes: int, cfg: TxConfig, *,
                    symbol_index: int, noise_variance: float):
        frame = B1PayloadFrame.from_config(first_edge, length_bytes, cfg)
        frame = replace(frame, arrival=replace(frame.arrival, pilot_interval=cfg.pilot_interval))
        return cls(frame, symbol_index, cfg.whitening_seed, noise_variance,
                   _head_len(cfg.frame_type))

    @property
    def coded_bits(self):
        a = self.payload.arrival
        plan = b1_code_block_plan(a.payload_with_crc_bits, a.rate, standard=True)
        return sum(n for block in plan for n, _ in block.codewords)

    def for_llrs(self):
        a = self.payload.arrival
        return replace(self.payload, arrival=replace(a, first_edge=a.first_edge + _DELAY))


class PSKWordDemapper:
    """One wire symbol per nominal period, at most three registered soft bits.

    The coefficient and frame context ports are setup inputs. A whole-symbol
    reservation covers projection, maximum updates, scale and sign whitening.
    """

    def __init__(self, *, bits_per_symbol, n_bits, pilot_interval, symbol_index,
                 first_edge, clock_hz, noise_variance, whitening_seed=0,
                 head_coded_bits=0, fraction_bits=0, observer=None):
        fields = (bits_per_symbol, n_bits, pilot_interval, symbol_index, first_edge,
                  clock_hz, whitening_seed, head_coded_bits, fraction_bits)
        if any(type(x) is not int for x in fields):
            raise ValueError("Need integer PSK word service context")
        if (bits_per_symbol not in _GRAY or not 0 < n_bits < 2**17
                or pilot_interval not in (0, 1, 4, 8, 16)
                or symbol_index < 0 or first_edge < 0 or clock_hz not in _CLOCK_RATES
                or not 0 <= whitening_seed < 128 or head_coded_bits not in (0, 64, 256)
                or fraction_bits not in (0, 1)):
            raise ValueError("Invalid PSK word service context")
        self.bps, self.remaining = bits_per_symbol, n_bits
        self.interval, self.symbol_index = pilot_interval, symbol_index
        self.edge, self.period = first_edge, clock_hz // 4_000_000
        self.state = _jump_header(whitening_seed, head_coded_bits)
        self.frac = fraction_bits
        self.mantissa, exp = _variance_gain(noise_variance)
        self.shift = 47 - exp - fraction_bits
        self.observer = observer
        self.since_pilot = 0
        self.counts = Counter()

    @property
    def done(self):
        return self.remaining == 0

    def _sample(self, part):
        value = round(max(-8., min(8., part)) * 16384)
        clip = min(131071, max(-131072, value))
        self.counts["clipped_components"] += int(clip != value or not -8.0 <= part <= 8.0)
        return clip

    def _maxima(self, x, y, parity):
        # Eight multiplier issues at most. Each combination adds the two
        # signed products, then uses antipodal symmetry for the other half.
        half = tuple(_signed(x*c + y*s, 37) for c, s in _projection_rom(self.bps, parity))
        scores = (*half, *(-v for v in half))
        maxima = [[None, None] for _ in range(self.bps)]
        for label, score in zip(_GRAY[self.bps], scores, strict=True):
            for i, values in enumerate(maxima):
                bit = (label >> (self.bps - 1 - i)) & 1
                if values[bit] is None or score > values[bit]:
                    values[bit] = score
        self.counts["projection_products"] += 2 * len(half)
        self.counts["max_updates"] += len(scores) * self.bps
        return tuple(_signed(a - b, 38) for a, b in maxima)

    def _scale(self, difference):
        # 38x18 product as two signed-18-compatible DSP multiplicands, plus
        # alignment. The low slice is unsigned19 represented by signed20.
        low = difference & ((1 << 19) - 1)
        high = difference >> 19
        product = _signed(((high * self.mantissa) << 19) + low * self.mantissa, 56)
        value = _rne(product, self.shift) if self.shift < 56 else 0
        bound = 64 << self.frac
        self.counts["scale_partial_products"] += 2
        self.counts["saturated_llrs"] += int(abs(value) > bound)
        return min(bound, max(-bound, value))

    def feed(self, symbol):
        """Return tagged soft bits, or an empty tuple for a pilot."""
        if self.done:
            raise ValueError("Extra PSK wire symbol after field")
        z = complex(symbol)
        if not isfinite(z.real) or not isfinite(z.imag):
            raise ValueError("Need finite normalized I/Q")
        edge, index = self.edge, self.symbol_index
        self.edge += self.period
        self.symbol_index += 1
        if self.interval and self.since_pilot == self.interval:
            self.since_pilot = 0
            self.counts["pilots"] += 1
            if self.observer is not None:
                self.observer(dict(kind="pilot", edge=edge, symbol_index=index))
            return ()
        if self.interval:
            self.since_pilot += 1
        values = self._maxima(self._sample(z.real), self._sample(z.imag), index & 1)
        count = min(self.bps, self.remaining)
        result = []
        for diff in values[:count]:
            soft = self._scale(diff)
            mask, self.state = _lfsr_step(self.state)
            result.append(TimedLLR(edge + _DELAY, (-soft if mask else soft) / (1 << self.frac)))
        self.remaining -= count
        self.counts["data_symbols"] += 1
        self.counts["soft_bits"] += count
        if self.observer is not None:
            self.observer(dict(kind="data", edge=edge, symbol_index=index,
                               projection_ready=edge+10, maxima_ready=edge+18,
                               llr_ready=edge+_DELAY, soft_bits=count))
        return tuple(result)


def replay_b1_symbol_frames(
    frames, symbols, *, clock_hz, word_cycles, fraction_bits=0,
    demap_observer=None, **decoder_options,
):
    """Compose fixed demap edges with the bounded Polar and B1 MAC services."""
    waiting = deque()
    source = iter(symbols)
    ended = False
    end = object()

    def descriptions():
        for item in frames:
            if not isinstance(item, B1SymbolFrame):
                raise ValueError("Need validated B1 symbol frame descriptions")
            if waiting:
                raise RuntimeError("B1 symbol descriptor outran its single-entry handoff")
            waiting.append(item)
            yield item.for_llrs()

    def bits():
        nonlocal ended
        while waiting:
            item = waiting.popleft()
            a = item.payload.arrival
            model = PSKWordDemapper(
                bits_per_symbol=a.bits_per_symbol, n_bits=item.coded_bits,
                pilot_interval=a.pilot_interval, symbol_index=item.symbol_index,
                first_edge=a.first_edge, clock_hz=clock_hz, noise_variance=item.variance,
                whitening_seed=item.whitening_seed, head_coded_bits=item.head_coded_bits,
                fraction_bits=fraction_bits, observer=demap_observer,
            )
            while not model.done:
                value = next(source, end)
                if value is end:
                    ended = True
                    return
                yield from model.feed(value)

    result = replay_b1_payload_frames(
        descriptions(), bits(), word_cycles=word_cycles, clock_hz=clock_hz,
        fraction_bits=fraction_bits, **decoder_options,
    )
    if not ended and next(source, end) is not end:
        raise ValueError("Trailing PSK wire symbols do not belong to any frame")
    return result
