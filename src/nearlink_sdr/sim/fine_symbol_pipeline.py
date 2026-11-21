"""Exact, finite symbol C/E pipeline for configured PSK sync (§6.2.3).

Native25x18 products alone multiply. Every preadd, reconstruction, contribution
and accumulator crosses an edge. This is a node schedule, not RTL/Fmax or a
complete RX. Input is the already selected interpolation request, never truth.
"""

from dataclasses import dataclass

from nearlink_sdr.phy.acquisition_flow import signed_tile_terms

__all__ = ["FineSymbolPipeline", "FineSymbolReply", "FineSymbolRequest"]


def _bound(value, bits, signed=True):
    if type(value) is not int or not -(1 << (bits - 1)) * signed <= value < 1 << (
        bits - signed
    ):
        raise OverflowError("Exact fine-symbol format exceeded")
    return value


def _pair(value, bits):
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("Need a complex integer pair")
    for item in value:
        _bound(item, bits)


@dataclass(frozen=True)
class FineSymbolRequest:
    """361-bit source ticket; previous state belongs to this same phase."""

    token: int
    phase: int
    word: tuple
    previous: tuple | None
    previous_power: int
    real: int
    imag: int
    energy: int
    sign: tuple

    def __post_init__(self):
        _bound(self.token, 16, False)
        if type(self.phase) is not int or not 0 <= self.phase < 9:
            raise ValueError("Need one of nine compiled phases")
        _pair(self.word, 22)
        if self.previous is not None:
            _pair(self.previous, 22)
        _bound(self.previous_power, 44, False)
        _bound(self.real, 53)
        _bound(self.imag, 53)
        _bound(self.energy, 98, False)
        if (not isinstance(self.sign, tuple) or len(self.sign) != 2
                or any(type(v) is not int or v not in (-1, 0, 1) for v in self.sign)):
            raise ValueError("Need the configured differential template sign")
        if self.previous is None and any(
            (self.previous_power, self.real, self.imag, self.energy, *self.sign)
        ):
            raise ValueError("First symbol has zero previous C/E and no differential sign")


@dataclass(frozen=True)
class FineSymbolReply:
    token: int
    phase: int
    state: tuple


def _terms(a, b, left, right, count):
    terms = signed_tile_terms(a, b, left, right)
    if len(terms) != count:
        raise RuntimeError("Compiled native partial-product shape changed")
    return tuple((x, y) for x, y, _ in terms)


def _native(pair):
    a, b = pair
    _bound(a, 25)
    _bound(b, 18)
    return _bound(a * b, 43)


class FineSymbolPipeline:
    """16 slots, II1, registered output; a blocked output freezes every slot.

    No unbounded queue or future answer trace. The caller must hold/retry a
    rejected input and must account its own source/history deadline. The ADC
    is not backpressured by this internal node. Cancel flushes all old tickets.
    """

    def __init__(self):
        self.cells = [None] * 16
        self.next_token = 0
        self.cycle = -1
        self.stats = dict(accepted=0, completed=0, blocked=0, cancelled=0,
                          native_products=0, preadds=0, adds=0, sign_wires=0)

    def _count(self, key, count=1):
        self.stats[key] = _bound(self.stats[key] + count, 64, False)

    def _add(self, a, b, bits, signed=True):
        self._count("adds")
        return _bound(a + b, bits, signed)

    def _stage(self, stage, cell):
        req, values = cell
        if stage == 1:
            a, b = req.word
            c, d = req.previous or (0, 0)
            cd, ab, ba = [_bound(v, 23) for v in (c - d, a + b, b - a)]
            self._count("preadds", 3)
            operands = (*_terms(a, cd, 22, 23, 2), *_terms(d, ab, 22, 23, 2),
                        *_terms(c, ba, 22, 23, 2), *_terms(a, a, 22, 22, 2),
                        *_terms(b, b, 22, 22, 2))
            return req, operands
        if stage in (2, 4, 8, 10, 15):
            return cell  # Distinct registered input/result or output wires.
        if stage == 3:
            self._count("native_products", 10)
            return req, tuple(_native(pair) for pair in values)
        if stage == 5:
            return req, tuple(self._add(values[k], values[k + 1] << 17, 46)
                              for k in range(0, 10, 2))
        if stage == 6:
            p, q, r, rr, ii = values
            return req, (self._add(p, q, 45), self._add(p, r, 45),
                         self._add(rr, ii, 44, False))
        if stage == 7:
            dr, di, power = values
            r, i = req.sign
            self._count("sign_wires", 4)
            # Only compile-time {-1,0,1} selection/negation, not DSP products.
            tr = self._add(dr * r, di * i, 46)
            ti = self._add(di * r, -dr * i, 46)
            return req, (tr, ti, power, _terms(power, req.previous_power, 45, 45, 6))
        if stage == 9:
            tr, ti, power, operands = values
            self._count("native_products", 6)
            return req, (tr, ti, power, tuple(_native(pair) for pair in operands))
        if stage == 11:
            tr, ti, power, terms = values
            # Constant shifts [0,17,34,24,41,58], not variable shifters.
            shifts = (0, 17, 34, 24, 41, 58)
            pairs = tuple(self._add(terms[k] << shifts[k], terms[k + 1] << shifts[k + 1],
                                   90) for k in (0, 2, 4))
            return req, (tr, ti, power, pairs)
        if stage == 12:
            tr, ti, power, (a, b, c) = values
            return req, (tr, ti, power, (self._add(a, b, 90), c))
        if stage == 13:
            tr, ti, power, (a, b) = values
            return req, (tr, ti, power, self._add(a, b, 90))
        if stage == 14:
            tr, ti, power, product = values
            state = (req.word, self._add(req.real, tr, 53),
                     self._add(req.imag, ti, 53),
                     self._add(req.energy, product, 98, False), power)
            return req, FineSymbolReply(req.token, req.phase, state)
        raise RuntimeError("Unknown registered fine-symbol stage")

    def step(self, incoming=None, *, output_ready=True):
        """Advance one edge;return(accepted,old registered output or None)."""
        if incoming is not None and not isinstance(incoming, FineSymbolRequest):
            raise ValueError("Need one typed fine-symbol ticket or no input")
        if type(output_ready) is not bool:
            raise ValueError("Need explicit registered sink readiness")
        self.cycle = _bound(self.cycle + 1, 64, False)
        if self.cells[-1] is not None and not output_ready:
            self._count("blocked")
            return False, None
        if incoming is not None and incoming.token != self.next_token:
            raise ValueError("Input token must be the next16-bit ticket;clear at exhaustion")
        reply = None if self.cells[-1] is None else self.cells[-1][1]
        if reply is not None:
            self._count("completed")
        for index in range(14, -1, -1):
            self.cells[index + 1] = (
                None if self.cells[index] is None else self._stage(index + 1, self.cells[index])
            )
        self.cells[0] = None if incoming is None else (incoming, None)
        if incoming is not None:
            self.next_token += 1  # 17-bit exhausted value65536 is explicit.
            self._count("accepted")
        return incoming is not None, reply

    @property
    def idle(self):
        return not any(cell is not None for cell in self.cells)

    def clear(self):
        """One edge cancellation;no old ticket survives into the next epoch."""
        count = sum(cell is not None for cell in self.cells)
        self._count("cancelled", count)
        self.cells = [None] * 16
        self.next_token = 0
        self.cycle = _bound(self.cycle + 1, 64, False)
        return count

    def estimate(self):
        """Conservative finite state/port cost; no old register reuse credit."""
        extras = (0, 430, 430, 430, 430, 230, 134, 394,
                  394, 394, 394, 406, 316, 226, 312, 312)
        metadata = 361 * 16
        registers = metadata + sum(extras) + 16 + 17 + 64 + 8 * 64
        return dict(
            stages=16, II_edges=1, latency_edges=16,
            output_credits=16, finite_register_bits_upper=registers,
            source_ticket_bits=361, per_stage_extra_bits=extras,
            metadata_register_bits=metadata, no_old_state_subtracted=True,
            dedicated_DSP25x18=16, DSP_allocation="difference6,power4,energy6",
            DSP_internal_pipeline_bit_capacity=16 * (25 + 18 + 43 + 48),
            new_RAM_or_ROM_bits=0, new_sample_read_ports=0, new_sample_write_ports=0,
            source_ticket_ports=1, registered_reply_ports=1,
            largest_registered_add_bits=98, full_width_add_latency_assumption_edges=1,
            template_sign_mux_negate_width=46, no_general_sign_multiplier=True,
            rounding="none", saturation="none;format violation raises",
            overflow_policy="abort current experiment;no float fallback",
            backpressure="all16slots clock-enabled together;caller owns bounded retention",
            statistics_bits=8 * 64, source_RAM_and_interpolator_included=False,
            continuous16MSps_verified=False, system_resources_verified=False,
            actual_synthesis_or_Fmax=False,
        )
