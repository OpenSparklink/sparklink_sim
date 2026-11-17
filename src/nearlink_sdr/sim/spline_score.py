"""Bounded eighth-query reuse and full-growth differential score (§6.2.3).

The same prepared banks pass from the existing integer controller to this
serial micro-operation model. It is NOT coherent confirmation, a complete
receiver, a synthesized design or a throughput claim. Wide operations execute
with finite word checks and explicit serial latencies, including tiled products.
"""

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from nearlink_sdr.phy.spline_fixed import SplineFixedFormat, _rne_shift
from nearlink_sdr.phy.uniform_spline import _DIGITS
from nearlink_sdr.sim.spline_flow import SplineFlowAssumptions, _Run, spline_flow_cost

__all__ = ["SplineScoreAssumptions", "SplineScoreFlow", "SyncSignTemplate", "spline_score_cost"]


@dataclass(frozen=True)
class SyncSignTemplate:
    """Offline exact differential signs for a configured PSK sync, not payload."""

    words: tuple

    def __post_init__(self):
        if (
            not isinstance(self.words, tuple)
            or not 1 <= len(self.words) <= 127
            or any(
                not isinstance(w, tuple)
                or len(w) != 2
                or any(type(v) is not int or v not in (-1, 0, 1) for v in w)
                for w in self.words
            )
            or len({r * r + i * i for r, i in self.words}) != 1
            or self.words[0] == (0, 0)
        ):
            raise ValueError("Need1..127 same-energy differential sign pairs")

    @property
    def energy(self):
        return sum(r * r + i * i for r, i in self.words)

    @classmethod
    def from_symbols(cls, symbols):
        """Validate actual profile rotations; do not approximate arbitrary templates."""
        s = np.asarray(symbols)
        if (
            s.ndim != 1
            or not 2 <= len(s) <= 128
            or not np.isfinite(s).all()
            or not np.allclose(abs(s), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("Need configured unit PSK sync symbols")
        d = s[1:] * s[:-1].conj()
        for scale in (1.0, math.sqrt(2)):
            r, i = np.rint(d.real * scale), np.rint(d.imag * scale)
            if (
                np.all(abs(r) <= 1)
                and np.all(abs(i) <= 1)
                and np.allclose(d * scale, r + 1j * i, rtol=0, atol=1e-12)
            ):
                return cls(tuple((int(a), int(b)) for a, b in zip(r, i, strict=True)))
        raise ValueError("Template is not a uniform axis/diagonal differential sign alphabet")


@dataclass(frozen=True)
class SplineScoreAssumptions:
    """Compiled capacity, SPO read latency and serial operator assumptions."""

    capacity: int = 640
    reuse_stencil: bool = True
    read_latency: int = 1
    multiplier_latency: int = 3
    alu_segment_bits: int = 32

    def __post_init__(self):
        if (
            type(self.capacity) is not int
            or not 4 <= self.capacity <= 640
            or type(self.reuse_stencil) is not bool
            or type(self.read_latency) is not int
            or self.read_latency not in (1, 2)
            or self.multiplier_latency != 3
            or type(self.multiplier_latency) is not int
            or self.alu_segment_bits != 32
            or type(self.alu_segment_bits) is not int
        ):
            raise ValueError("Need explicit capacity4..640,read1/2,DSP latency3,ALU32")


def _operation(code, a, b, bits, signed, tag):
    return dict(code=code, a=a, b=b, bits=bits, signed=signed, tag=tag)


def _alu(code, a, b, bits, tag, signed=True):
    return (yield _operation(code, a, b, bits, signed, tag))


def _mul(a, b, wa, wb, tag):
    """Literal positive24x17 products fit signed25x18; never a wide host product."""
    left = yield from _alu("abs", a, 0, wa, tag + ".abs_a", False)
    right = yield from _alu("abs", b, 0, wb, tag + ".abs_b", False)
    total = 0
    for i in range(math.ceil(wa / 24)):
        for j in range(math.ceil(wb / 17)):
            part = yield _operation(
                "tile",
                (left >> (24 * i)) & ((1 << 24) - 1),
                (right >> (17 * j)) & ((1 << 17) - 1),
                43,
                True,
                tag + ".tile",
            )
            total = yield from _alu(
                "add", total, part << (24 * i + 17 * j), wa + wb, tag + ".partial", False
            )
    return (
        yield from _alu(
            "neg" if (a < 0) != (b < 0) else "mov", total, 0, wa + wb + 1, tag + ".sign"
        )
    )


def _query(cache, tick):
    j, fraction = divmod(tick, 8)
    if not fraction:
        return cache[("y", j)]
    values = (cache[("y", j)], cache[("y", j + 1)], cache[("m", j)], cache[("m", j + 1)])
    output = []
    for component in (0, 1):
        total = None
        for value, digits in zip(values, _DIGITS[fraction], strict=True):
            for sign, power in digits:
                part = value[component] << power
                total = yield from _alu(
                    "mov" if total is None else "add" if sign == 1 else "sub",
                    part if total is None else total,
                    0 if total is None else part,
                    33,
                    "phase_accumulator",
                )
        output.append((yield from _alu("rne", total, 10, 22, "phase")))
    return tuple(output)


def _needed(ticks):
    keys = set()
    for tick in ticks:
        j, fraction = divmod(tick, 8)
        keys.add(("y", j))
        if fraction:
            keys.update((("y", j + 1), ("m", j), ("m", j + 1)))
    return sorted(keys)


def _contribution(dr, di, r, i, bits):
    def signed(value, sign):
        if sign == 0:
            return 0
        if sign == -1:
            return (yield from _alu("neg", value, 0, bits, "template_sign"))
        return value

    a = yield from signed(dr, r)
    b = yield from signed(di, i)
    c = yield from signed(di, r)
    d = yield from signed(dr, i)
    real = yield from _alu("add", a, b, bits, "template_real")
    imag = yield from _alu("sub", c, d, bits, "template_imag")
    return real, imag


def _constant_product(value, constant, bits, tag):
    total = 0
    for k in range(constant.bit_length()):
        if constant & (1 << k):
            total = yield from _alu("add", total, value << k, bits, tag, False)
    return total


def _program(peaks, template, n, minimum, lookahead, reuse, threshold, emit):
    """Nine contexts, no retained18x128 symbol vectors; replay the chosen phase."""
    count = len(template.words) + 1
    guard = math.ceil(math.log2(count - 1))
    cb, eb = 46 + guard, 91 + guard
    best = None
    for peak in peaks:
        starts = [
            8 * peak + d
            for d in range(-4, 5)
            if 8 * peak + d >= minimum and 8 * peak + d + 32 * (count - 1) < 8 * (n - 1)
        ]
        states = [[None, 0, 0, 0] for _ in starts]
        for k in range(count):
            ticks = [s + 32 * k for s in starts]
            shared = (yield dict(code="read", keys=_needed(ticks))) if reuse and ticks else None
            for h, tick in enumerate(ticks):
                cache = shared if reuse else (yield dict(code="read", keys=_needed([tick])))
                word = yield from _query(cache, tick)
                previous, cr, ci, energy = states[h]
                if previous is not None:
                    rr = yield from _mul(word[0], previous[0], 22, 22, "difference_rr")
                    ii = yield from _mul(word[1], previous[1], 22, 22, "difference_ii")
                    ir = yield from _mul(word[1], previous[0], 22, 22, "difference_ir")
                    ri = yield from _mul(word[0], previous[1], 22, 22, "difference_ri")
                    dr = yield from _alu("add", rr, ii, 45, "difference_real")
                    di = yield from _alu("sub", ir, ri, 45, "difference_imag")
                    tr, ti = yield from _contribution(dr, di, *template.words[k - 1], 46)
                    cr = yield from _alu("add", cr, tr, cb, "correlation_real")
                    ci = yield from _alu("add", ci, ti, cb, "correlation_imag")
                    er = yield from _mul(dr, dr, 45, 45, "energy_real")
                    ei = yield from _mul(di, di, 45, 45, "energy_imag")
                    part = yield from _alu("add", er, ei, 91, "difference_power", False)
                    energy = yield from _alu("add", energy, part, eb, "energy_sum", False)
                states[h] = word, cr, ci, energy
                emit("sample", start_tick=starts[h], symbol=k, word=word)
        for start, (_, cr, ci, energy) in zip(starts, states, strict=True):
            rr = yield from _mul(cr, cr, cb, cb, "numerator_real")
            ii = yield from _mul(ci, ci, cb, cb, "numerator_imag")
            numerator = yield from _alu("add", rr, ii, 2 * cb + 1, "numerator", False)
            row = dict(
                start_tick=start,
                correlation=(cr, ci),
                numerator=numerator,
                energy=energy,
                denominator=energy * template.energy,
                zero_energy=not energy,
            )
            # Template energy is compile-time common to all candidates. Zero
            # score keeps a positive comparison denominator without division.
            wins = best is None
            if best is not None:
                left = yield from _mul(
                    numerator, max(best["energy"], 1), 2 * cb + 1, eb, "rank_left"
                )
                right = yield from _mul(
                    best["numerator"], max(energy, 1), 2 * cb + 1, eb, "rank_right"
                )
                cmp = yield from _alu("cmp", left, right, 2 * cb + eb + 2, "normalized_rank")
                key = start % 32, start // 32
                old_key = best["start_tick"] % 32, best["start_tick"] // 32
                wins = cmp > 0 or (cmp == 0 and key < old_key)
            if wins:
                best = row
            emit("score", **row)
    if best is not None:
        pn, pd = threshold
        left = yield from _constant_product(
            best["numerator"], pd * pd, 2 * cb + 1 + 26, "threshold_left"
        )
        right = yield from _constant_product(
            best["energy"], template.energy * pn * pn, eb + 34, "threshold_right"
        )
        cmp = yield from _alu("cmp", left, right, max(2 * cb + 27, eb + 34), "threshold_compare")
        best["threshold_pass"] = cmp >= 0 and (bool(best["energy"]) or pn == 0)
        for k in range(count):
            tick = best["start_tick"] + 32 * k
            cache = yield dict(code="read", keys=_needed([tick]))
            word = yield from _query(cache, tick)
            emit("selected_sample", symbol=k, word=word)
        best["disposition"] = (
            "maximum_needs_future"
            if best["start_tick"] + 32 * (count - 1) + lookahead > 8 * (n - 1)
            else "ready"
        )
    return best


class SplineScoreFlow:
    """Prepare then score using the same finite RAM and a single serial DSP.

    Diagnostic sink borrows output events; they do not form an internal vector
    buffer. No payload, CRC, expected start, timing or carrier truth is accepted.
    Peak positions are supplied by the already bounded coarse stage, never truth.
    """

    def __init__(self, assumptions=None):
        self.a = SplineScoreAssumptions() if assumptions is None else assumptions
        if not isinstance(self.a, SplineScoreAssumptions):
            raise ValueError("Need explicit SplineScoreAssumptions")

    def run(
        self,
        words,
        template,
        peaks,
        *,
        exponent=None,
        minimum_tick=0,
        lookahead_ticks=0,
        threshold=(17, 20),
        output_stalls=(),
        max_cycles=2_000_000,
        sink=None,
    ):
        widths = (19, 19, 8) if exponent is None else (18, 18)
        if (
            not isinstance(template, SyncSignTemplate)
            or not isinstance(words, (tuple, list))
            or not 4 <= len(words) <= self.a.capacity
            or any(
                not isinstance(w, (tuple, list))
                or len(w) != len(widths)
                or any(
                    type(v) is not int or not -(2 ** (bits - 1)) <= v < 2 ** (bits - 1)
                    for v, bits in zip(w, widths, strict=True)
                )
                for w in words
            )
            or not isinstance(peaks, (tuple, list))
            or len(peaks) > 2
            or any(type(p) is not int or not 0 <= p < self.a.capacity for p in peaks)
            or any(abs(x - y) < 4 for j, x in enumerate(peaks) for y in peaks[j + 1 :])
            or type(minimum_tick) is not int
            or not 0 <= minimum_tick <= 8 * self.a.capacity
            or type(lookahead_ticks) is not int
            or not 0 <= lookahead_ticks <= 32768
            or not isinstance(threshold, tuple)
            or len(threshold) != 2
            or any(type(v) is not int for v in threshold)
            or not 0 <= threshold[0] <= threshold[1] <= 4096
            or threshold[1] == 0
            or (exponent is not None and (type(exponent) is not int or not -512 <= exponent < 512))
            or not isinstance(output_stalls, (tuple, list))
            or len(output_stalls) > 16
            or any(
                not isinstance(v, tuple)
                or len(v) != 2
                or any(type(x) is not int for x in v)
                or not 0 <= v[0] < v[1] <= 2_000_000
                for v in output_stalls
            )
            or any(a[1] > b[0] for a, b in pairwise(output_stalls))
            or type(max_cycles) is not int
            or not 1 <= max_cycles <= 2_000_000
        ):
            raise ValueError(
                "Need bounded integer RRC words,template,0..2 separated integer peaks and finite horizon"
            )
        cycle, output_blocked = -1, 0

        def emit(event, **fields):
            nonlocal cycle, output_blocked
            if event == "selected_sample":
                while any(lo <= cycle < hi for lo, hi in output_stalls):
                    if cycle + 1 >= max_cycles:
                        raise TimeoutError
                    cycle += 1
                    output_blocked += 1
                    context.ram.advance(cycle, lambda *args: None)
            if sink is not None:
                sink(dict(cycle=cycle, event=event, **fields))

        context = _Run(
            SplineFixedFormat(18, 14),
            SplineFlowAssumptions(self.a.capacity, "inplace", self.a.read_latency, 2),
            list(map(tuple, words)),
            exponent,
            None,
        )
        while context.ready_cycle is None and context.cycle + 1 < max_cycles:
            context.tick()
        if context.ready_cycle is None:
            return dict(
                aborted=True,
                reason="preparation_cycle_limit",
                cycles=context.cycle + 1,
                best=None,
                whole_rx_verified=False,
            )
        cycle = context.cycle
        prep_cycles, prep_reads = cycle + 1, context.ram.reads
        context.ram.emit = emit
        emit("bank_handoff", n=len(words), exponent=context.exponent)
        program = _program(
            peaks,
            template,
            len(words),
            minimum_tick,
            lookahead_ticks,
            self.a.reuse_stencil,
            threshold,
            emit,
        )
        counts, peaks_bits, observed_bits = {}, {}, {}
        reads = 0
        reply = None
        best, aborted = None, False
        cache_peak = pending_peak = 0
        try:
            command = next(program)
            while True:
                if command["code"] == "read":
                    keys = command["keys"]
                    cache, pending = {}, set()
                    waiting = list(keys)

                    def received(tag, word):
                        key = tuple(tag)
                        pending.remove(key)
                        cache[key] = word

                    while waiting or pending:
                        if cycle + 1 >= max_cycles:
                            raise TimeoutError
                        cycle += 1
                        context.ram.advance(cycle, received)
                        after = []
                        for key in waiting:
                            if len(pending) == 4:
                                after.append(key)
                                continue
                            prefix, j = key
                            address = context.address("y" if prefix == "y" else "w", j)
                            if context.ram.issue(*address, "read", key):
                                pending.add(key)
                                reads += 1
                            else:
                                after.append(key)
                        waiting = after
                        cache_peak = max(cache_peak, len(cache))
                        pending_peak = max(pending_peak, len(pending))
                    reply = cache
                else:
                    code = command["code"]
                    counts[code] = counts.get(code, 0) + 1
                    if command["tag"] == "phase_accumulator":
                        counts["constant_shifted_operands"] = (
                            counts.get("constant_shifted_operands", 0) + 1
                        )
                    width = command["bits"]
                    peaks_bits[command["tag"]] = max(width, peaks_bits.get(command["tag"], 0))
                    latency = (
                        self.a.multiplier_latency
                        if code == "tile"
                        else max(1, math.ceil(width / self.a.alu_segment_bits))
                    )
                    if code == "rne":
                        latency += 1
                    emit("op_issue", **command, latency=latency)
                    for _ in range(latency):
                        if cycle + 1 >= max_cycles:
                            raise TimeoutError
                        cycle += 1
                        context.ram.advance(cycle, lambda *args: None)
                    a, b = command["a"], command["b"]
                    if code == "tile":
                        if not 0 <= a < 2**24 or not 0 <= b < 2**17:
                            raise OverflowError("DSP25x18 tile input range")
                        reply = a * b
                    elif code == "add":
                        reply = a + b
                    elif code == "sub":
                        reply = a - b
                    elif code == "rne":
                        reply = _rne_shift(a, b)
                    elif code == "abs":
                        reply = abs(a)
                    elif code == "neg":
                        reply = -a
                    elif code == "mov":
                        reply = a
                    else:
                        if code != "cmp":
                            raise RuntimeError("Unknown micro-operation")
                        reply = (a > b) - (a < b)
                    lo, hi = (
                        (-(2 ** (width - 1)), 2 ** (width - 1))
                        if command["signed"]
                        else (0, 2**width)
                    )
                    if not lo <= reply < hi:
                        emit("overflow", **command, result=reply)
                        raise OverflowError("Explicit finite arithmetic bound exceeded")
                    observed_bits[command["tag"]] = max(
                        observed_bits.get(command["tag"], 0),
                        abs(reply).bit_length() + int(command["signed"]),
                    )
                    emit("op_complete", tag=command["tag"], result=reply)
                command = program.send(reply)
        except StopIteration as done:
            best = done.value
        except TimeoutError:
            aborted = True
            emit("resource_drop", reason="cycle_limit")
        return dict(
            aborted=aborted,
            reason="cycle_limit" if aborted else None,
            cycles=cycle + 1,
            preparation_cycles=prep_cycles,
            scoring_cycles=cycle + 1 - prep_cycles,
            best=best,
            common_exponent=context.exponent,
            operation_counts=counts,
            declared_node_bits=peaks_bits,
            observed_node_bits=observed_bits,
            scoring_reads=reads,
            preparation_reads=prep_reads,
            output_blocked_cycles=output_blocked,
            cache_peak=cache_peak,
            pending_read_peak=pending_peak,
            prepared_source_words=tuple(
                context.ram.cells["y" + str(j % 2)][j // 2] for j in range(len(words))
            ),
            prepared_second_words=tuple(
                context.ram.cells["w" + str(j % 2)][j // 2] for j in range(len(words))
            ),
            selected_vectors_stored=0,
            adc_backpressure=False,
            whole_rx_verified=False,
            synthesis_or_timing_verified=False,
        )


def spline_score_cost(assumptions=None):
    """Finite registers/ports and explicit operator reservations, not synthesis."""
    a = SplineScoreAssumptions() if assumptions is None else assumptions
    if not isinstance(a, SplineScoreAssumptions):
        raise ValueError("Need explicit SplineScoreAssumptions")
    banks = spline_flow_cost(
        SplineFixedFormat(18, 14), SplineFlowAssumptions(a.capacity, read_latency=a.read_latency)
    )
    fields = dict(
        nine_previous=9 * 44,
        nine_correlations=9 * 2 * 53,
        nine_energies=9 * 98,
        stencil=(6 if a.reuse_stencil else 4) * 44,
        template_rom=127 * 4,
        best_numerator_energy_correlation=107 + 98 + 2 * 53 + 13,
        serial_alu_operand_result=3 * 208,
        tile_pipeline=3 * (43 + 9 + 1),
        multiplier_magnitude_partial_registers=2 * 208 + 208,
        cache_read_tags_flags=6 * 14,
        held_output_word_valid=45,
        control_loop_descriptors=512,
        operation_counters=10 * 64,
        peak_bit_monitors=40 * 16,
    )
    return dict(
        assumptions=vars(a),
        shared_preparation_banks=banks,
        additional_query_state_upper_bits=sum(fields.values()),
        state_components=fields,
        preparation_state_not_subtracted_as_free_sharing=True,
        shared_DSP25x18_count=1,
        serial_alu_segment_bits=32,
        wide_ALU_latency="ceil(result bits/32);macro value/schedule,not register-level carry RTL",
        DSP_latency=3,
        score_component_bits=45,
        correlation_bits_upper=53,
        energy_bits_upper=98,
        numerator_bits_upper=107,
        rank_cross_product_bits_upper=206,
        runtime_division=0,
        runtime_sqrt=0,
        runtime_trig=0,
        sign_template_multiplication="0/+1 wiring;negative signs charged as ALU negations;counted real/imag adds",
        threshold_products="compile-time constants;literal shift/add operations counted;denominator output is diagnostic only",
        full_window_copies_added=0,
        selected_vectors_buffered=0,
        chosen_vector_replay_charged=True,
        no_claim_on_float_angle_coherent_bridge=True,
        complete16msps_verified=False,
        original16kib_failure_preserved=True,
        synthesis_or_placement_evidence=False,
    )
