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
from nearlink_sdr.sim.score_word_runtime import ScoreWordRuntime
from nearlink_sdr.sim.spline_flow import SplineFlowAssumptions, _Run, spline_flow_cost
from nearlink_sdr.sim.word_arithmetic import (
    WordFabric,
    WordFabricAssumptions,
    word_fabric_cost,
)

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
    reuse_symbol_power: bool = False
    defer_all_future: bool = False

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
            or type(self.reuse_symbol_power) is not bool
            or type(self.defer_all_future) is not bool
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


def _symbol_power(word):
    """Full-growth norm of signed22-bit components;no rounding or saturation."""
    real = yield from _mul(word[0], word[0], 22, 22, "symbol_power_real")
    imag = yield from _mul(word[1], word[1], 22, 22, "symbol_power_imag")
    return (yield from _alu("add", real, imag, 44, "symbol_power", False))


def _monitor_tags(reuse_power, defer_future=False, gauss=False):
    """Compiled register slots for every score-node peak-width diagnostic."""
    products = (
        *(
            ("gauss_a", "gauss_d", "gauss_c")
            if gauss
            else ("difference_rr", "difference_ii", "difference_ir", "difference_ri")
        ),
        "numerator_real",
        "numerator_imag",
        "rank_left",
        "rank_right",
        *(
            ("symbol_power_real", "symbol_power_imag", "energy_product")
            if reuse_power
            else ("energy_real", "energy_imag")
        ),
    )
    tags = {
        p + suffix
        for p in products
        for suffix in (".abs_a", ".abs_b", ".tile", ".partial", ".sign")
    }
    tags.update(
        (
            "phase_accumulator",
            "phase",
            "difference_real",
            "difference_imag",
            "template_real",
            "template_imag",
            "correlation_real",
            "correlation_imag",
            "symbol_power" if reuse_power else "difference_power",
            "energy_sum",
            "template_sign",
            "numerator",
            "normalized_rank",
            "threshold_left",
            "threshold_right",
            "threshold_compare",
        )
    )
    if defer_future:
        tags.update(
            "geometry_" + suffix for suffix in ("add", "sub", "cmp", "move", "count", "flags")
        )
    if gauss:
        tags.update(("gauss_cd", "gauss_ab", "gauss_ba"))
    return frozenset(tags)


@dataclass(frozen=True)
class _GeometryDecision:
    """Host view of finite result metadata;no retained sample or score vectors."""

    best: dict | None
    positions: int
    deferred: bool


def _geometry(peaks, n, count, minimum, lookahead, emit):
    """At most two inclusive intervals,using original strict end/ready rules."""
    end = yield from _alu("sub", n, 1, 17, "geometry_sub")
    command = _operation("sub", end << 3, 32 * (count - 1), 17, True, "geometry_sub")
    command["constant_shifted_operand"] = True
    horizon = yield command
    upper = yield from _alu("sub", horizon, 1, 17, "geometry_sub")
    ready_upper = yield from _alu("sub", horizon, lookahead, 17, "geometry_sub")
    positions, ready = 0, False
    for peak in peaks:
        command = _operation("sub", peak << 3, 4, 17, True, "geometry_sub")
        command["constant_shifted_operand"] = True
        lo = yield command
        command = _operation("add", peak << 3, 4, 17, True, "geometry_add")
        command["constant_shifted_operand"] = True
        hi = yield command
        cmp = yield from _alu("cmp", lo, minimum, 17, "geometry_cmp")
        lo = yield from _alu("mov", lo if cmp >= 0 else minimum, 0, 17, "geometry_move")
        cmp = yield from _alu("cmp", hi, upper, 17, "geometry_cmp")
        hi = yield from _alu("mov", hi if cmp <= 0 else upper, 0, 17, "geometry_move")
        valid = (yield from _alu("cmp", lo, hi, 17, "geometry_cmp")) <= 0
        if valid:
            span = yield from _alu("sub", hi, lo, 17, "geometry_sub")
            span = yield from _alu("add", span, 1, 5, "geometry_count", False)
            positions = yield from _alu("add", positions, span, 5, "geometry_count", False)
            cmp = yield from _alu("cmp", lo, ready_upper, 17, "geometry_cmp")
            ready = bool(
                (yield from _alu("mov", int(ready or cmp <= 0), 0, 1, "geometry_flags", False))
            )
        emit(
            "geometry_interval", peak=peak, lower=lo, upper=hi, ready_upper=ready_upper, valid=valid
        )
    nonempty = (yield from _alu("cmp", positions, 0, 5, "geometry_cmp")) > 0
    deferred = bool(
        (yield from _alu("mov", int(nonempty and not ready), 0, 1, "geometry_flags", False))
    )
    emit("geometry_complete", positions=positions, any_ready=ready, deferred=deferred)
    return positions, deferred


def _guarded_program(
    peaks, template, n, minimum, lookahead, reuse, threshold, emit, reuse_power,
    concurrent=False, gauss=False
):
    positions, deferred = yield from _geometry(
        peaks, n, len(template.words) + 1, minimum, lookahead, emit
    )
    if deferred:
        emit("geometry_wait", positions=positions, scoring_evaluated=False)
        return _GeometryDecision(None, positions, True)
    best = yield from _program(
        peaks, template, n, minimum, lookahead, reuse, threshold, emit, reuse_power,
        concurrent, gauss
    )
    return _GeometryDecision(best, positions, False)


def _gauss_difference(word, previous):
    """Exact signed22 conjugate product;no input negation,rounding or saturation.

    Each sum/difference is signed23;each real22x23 product fits signed45
    (the generic multiplier conservatively returns46). The final differences
    retain the existing signed45 format. Even -2**21 inputs need no abs22
    signed restoration or negated-input overflow shortcut.
    """
    a, b = word
    c, d = previous
    cd = yield from _alu("sub", c, d, 23, "gauss_cd")
    ab = yield from _alu("add", a, b, 23, "gauss_ab")
    ba = yield from _alu("sub", b, a, 23, "gauss_ba")
    p = yield from _mul(a, cd, 22, 23, "gauss_a")
    q = yield from _mul(d, ab, 22, 23, "gauss_d")
    r = yield from _mul(c, ba, 22, 23, "gauss_c")
    real = yield from _alu("add", p, q, 45, "difference_real")
    imag = yield from _alu("add", p, r, 45, "difference_imag")
    return real, imag


def _phase_symbol(cache, tick, state, template, k, cb, eb, reuse_power, gauss=False):
    word = yield from _query(cache, tick)
    previous, cr, ci, energy, previous_power = state
    power = (yield from _symbol_power(word)) if reuse_power else 0
    if previous is not None:
        if gauss:
            dr, di = yield from _gauss_difference(word, previous)
        else:
            rr = yield from _mul(word[0], previous[0], 22, 22, "difference_rr")
            ii = yield from _mul(word[1], previous[1], 22, 22, "difference_ii")
            ir = yield from _mul(word[1], previous[0], 22, 22, "difference_ir")
            ri = yield from _mul(word[0], previous[1], 22, 22, "difference_ri")
            dr = yield from _alu("add", rr, ii, 45, "difference_real")
            di = yield from _alu("sub", ir, ri, 45, "difference_imag")
        tr, ti = yield from _contribution(dr, di, *template.words[k - 1], 46)
        cr = yield from _alu("add", cr, tr, cb, "correlation_real")
        ci = yield from _alu("add", ci, ti, cb, "correlation_imag")
        if reuse_power:
            part = yield from _mul(power, previous_power, 44, 44, "energy_product")
        else:
            er = yield from _mul(dr, dr, 45, 45, "energy_real")
            ei = yield from _mul(di, di, 45, 45, "energy_imag")
            part = yield from _alu("add", er, ei, 91, "difference_power", False)
        energy = yield from _alu("add", energy, part, eb, "energy_sum", False)
    return word, cr, ci, energy, power


def _program(
    peaks,
    template,
    n,
    minimum,
    lookahead,
    reuse,
    threshold,
    emit,
    reuse_power=False,
    concurrent=False,
    gauss=False,
):
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
        states = [[None, 0, 0, 0, 0] for _ in starts]
        for k in range(count):
            ticks = [s + 32 * k for s in starts]
            shared = (yield dict(code="read", keys=_needed(ticks))) if reuse and ticks else None
            if concurrent and ticks:
                jobs = tuple(
                    _phase_symbol(shared, tick, state, template, k, cb, eb, reuse_power, gauss)
                    for tick, state in zip(ticks, states, strict=True)
                )
                states = yield dict(code="fork", jobs=jobs)
                for h, state in enumerate(states):
                    emit("sample", start_tick=starts[h], symbol=k, word=state[0])
                continue
            for h, tick in enumerate(ticks):
                cache = shared if reuse else (yield dict(code="read", keys=_needed([tick])))
                states[h] = yield from _phase_symbol(
                    cache, tick, states[h], template, k, cb, eb, reuse_power, gauss
                )
                emit("sample", start_tick=starts[h], symbol=k, word=states[h][0])
        for start, (_, cr, ci, energy, _) in zip(starts, states, strict=True):
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

    def __init__(self, assumptions=None, *, word_fabric=None, gauss_difference=False):
        self.a = SplineScoreAssumptions() if assumptions is None else assumptions
        if not isinstance(self.a, SplineScoreAssumptions):
            raise ValueError("Need explicit SplineScoreAssumptions")
        if word_fabric is not None and (
            not isinstance(word_fabric, WordFabricAssumptions)
            or word_fabric.contexts != 9
            or not self.a.reuse_stencil
        ):
            raise ValueError("Fine concurrency needs9 contexts and shared stencil")
        self.word_fabric = word_fabric
        if type(gauss_difference) is not bool:
            raise ValueError("Need explicit compiled Gauss difference mode")
        self.gauss_difference = gauss_difference
        self.monitor_tags = _monitor_tags(
            self.a.reuse_symbol_power, self.a.defer_all_future, gauss_difference
        )

    def estimate(self):
        """Charge experimental contexts/fabric in addition to legacy state."""
        cost = spline_score_cost(self.a)
        if self.word_fabric is not None:
            fabric = word_fabric_cost(self.word_fabric)
            phase = dict(
                current_word=44,
                four_products=180,
                differential=90,
                contribution=92,
                power=44,
                magnitudes_partial=180,
                native_product=43,
                sign_cursors=10,
                query_accumulator=33,
                query_cursor=9,
                program_indices_flags=57,
                call_control=48,
                pending_command=480,
                entry_work_state=292,
                contribution_negative_temporaries=90,
                allocation_spare=100,
            )
            if self.gauss_difference:
                phase["gauss_intermediates_no_old_slot_credit"] = 3 * 23 + 3 * 46
            extra = fabric["state_bits"] + 9 * sum(phase.values())
            cost.update(
                word_fabric=fabric,
                phase_context_register_bits_each=phase,
                additional_phase_context_bits=9 * sum(phase.values()),
                wide_ALU_latency=(
                    'actual32bit steps over max(input/output widths)+2;RNE33/10to22 four '
                    'steps;registered feedback'
                ),
                peak_monitor_update=(
                    'one taken result per cycle;private mailboxes handle simultaneous unit '
                    'completions;monitor decode/LUT timing unsynthesized'
                ),
                additional_word_and_context_bits=extra,
                additional_query_state_upper_bits=cost["additional_query_state_upper_bits"] + extra,
                legacy_scratch_not_subtracted=True,
                phase_contexts=9,
                word_fabric_enabled=True,
            )
        if self.gauss_difference:
            contexts = 9 if self.word_fabric is not None else 1
            extra = contexts * (3 * 23 + 3 * 46)
            if self.word_fabric is None:
                cost["state_components"]["gauss_intermediates_no_old_slot_credit"] = extra
                cost["additional_query_state_upper_bits"] += extra
            cost.update(
                gauss_difference=dict(
                    conjugate=True,
                    real_products=3,
                    product_input_shapes=[(22, 23)] * 3,
                    signed_preadds=3,
                    preadd_bits=23,
                    product_declared_bits=46,
                    final_difference_bits=45,
                    native24x17_tiles_per_difference=6,
                    original_tiles_per_difference=8,
                    incremental_register_bits=extra,
                    incremental_research_cap_bits=2048,
                    no_old_register_slots_subtracted=True,
                    extra_DSP_RAM_or_sample_ports=0,
                    control_LUT_and_timing_qualification=False,
                ),
                peak_bit_monitor_slots=len(self.monitor_tags),
                peak_bit_monitor_tags=sorted(self.monitor_tags),
            )
        return cost

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

                    'Need bounded integer RRC words,template,0..2 separated integer peaks '
                    'and finite horizon'

            )
        cycle, output_blocked = -1, 0
        geometry_cycle = None
        fabric = None

        def emit(event, **fields):
            nonlocal cycle, output_blocked, geometry_cycle
            if event == "geometry_complete":
                geometry_cycle = cycle
            if event == "selected_sample":
                while any(lo <= cycle < hi for lo, hi in output_stalls):
                    if cycle + 1 >= max_cycles:
                        raise TimeoutError
                    cycle += 1
                    output_blocked += 1
                    context.ram.advance(cycle, lambda *args: None)
                    if fabric is not None:
                        fabric.tick()
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
                numerical_pass=None,
                numerical_rejection=False,
                resource_drop=True,
                preparation_arithmetic=context.ops.stats,
                cycles=context.cycle + 1,
                best=None,
                whole_rx_verified=False,
            )
        cycle = context.cycle
        prep_cycles, prep_reads = cycle + 1, context.ram.reads
        context.ram.emit = emit
        # Existing finite preparation already counts every saturating node.
        # Clipped words are evidence, never a valid input to candidate ranking.
        if context.ops.stats["saturations"]:
            emit("numerical_reject", reason="preparation_saturated", arithmetic=context.ops.stats)
            return dict(
                aborted=True,
                reason="preparation_saturated",
                numerical_pass=False,
                numerical_rejection=True,
                resource_drop=False,
                preparation_arithmetic=context.ops.stats,
                cycles=prep_cycles,
                preparation_cycles=prep_cycles,
                scoring_cycles=0,
                scoring_reads=0,
                scores_evaluated=0,
                best=None,
                whole_rx_verified=False,
            )
        emit("bank_handoff", n=len(words), exponent=context.exponent)
        program = (_guarded_program if self.a.defer_all_future else _program)(
            peaks,
            template,
            len(words),
            minimum_tick,
            lookahead_ticks,
            self.a.reuse_stencil,
            threshold,
            emit,
            self.a.reuse_symbol_power,
            self.word_fabric is not None,
            self.gauss_difference,
        )
        counts, peaks_bits, observed_bits = {}, {}, {}
        reads = 0
        reply = None
        best, aborted, geometry = None, False, None
        cache_peak = pending_peak = 0

        def word_advance():
            nonlocal cycle
            if cycle + 1 >= max_cycles:
                raise TimeoutError
            cycle += 1
            context.ram.advance(cycle, lambda *args: None)
            fabric.tick()

        def word_issue(command, latency):
            code, tag = command["code"], command["tag"]
            if tag not in self.monitor_tags or len(self.monitor_tags) > 80:
                raise RuntimeError("Score node lacks a reserved monitor register")
            counts[code] = counts.get(code, 0) + 1
            if tag == "phase_accumulator" or command.get("constant_shifted_operand"):
                key = "constant_shifted_operands"
                counts[key] = counts.get(key, 0) + 1
            peaks_bits[tag] = max(command["bits"], peaks_bits.get(tag, 0))
            emit("op_issue", **command, latency=latency)

        def word_complete(command, value, overflow):
            if overflow:
                emit("overflow", **command, result=value)
                raise OverflowError("Explicit finite arithmetic bound exceeded")
            width = abs(value).bit_length() + int(command["signed"])
            if width >= 256:
                raise OverflowError("Peak-width monitor exceeds reserved8-bit slot")
            tag = command["tag"]
            observed_bits[tag] = max(observed_bits.get(tag, 0), width)
            emit("op_complete", tag=tag, result=value)

        fabric = (
            None if self.word_fabric is None else WordFabric(self.word_fabric, initial_cycle=cycle)
        )
        runtime = (
            None
            if fabric is None
            else ScoreWordRuntime(
                fabric, word_advance, word_issue, word_complete, self.monitor_tags
            )
        )
        try:
            command = next(program)
            while True:
                if command["code"] == "fork":
                    word_advance()  # Registered phase dispatch; no same-edge RAM forwarding.
                    reply = runtime.fork(command["jobs"])
                    word_advance()  # Registered join before consuming next shared stencil.
                elif runtime is not None and command["code"] != "read":
                    reply = runtime.execute(command)
                elif command["code"] == "read":
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
                        if fabric is not None:
                            fabric.tick()
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
                    if command["tag"] not in self.monitor_tags or len(self.monitor_tags) > 80:
                        raise RuntimeError("Score node lacks a reserved monitor register")
                    counts[code] = counts.get(code, 0) + 1
                    if command["tag"] == "phase_accumulator" or command.get(
                        "constant_shifted_operand"
                    ):
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
                    if fabric is not None:
                        fabric.tick()
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
                    observed_width = abs(reply).bit_length() + int(command["signed"])
                    if observed_width >= 256:
                        raise OverflowError("Peak-width monitor exceeds reserved8-bit slot")
                    observed_bits[command["tag"]] = max(
                        observed_bits.get(command["tag"], 0), observed_width
                    )
                    emit("op_complete", tag=command["tag"], result=reply)
                command = program.send(reply)
        except StopIteration as done:
            best = done.value
            if isinstance(best, _GeometryDecision):
                geometry, best = best, best.best
        except TimeoutError:
            aborted = True
            emit("resource_drop", reason="cycle_limit")
        result = dict(
            aborted=aborted,
            reason="cycle_limit" if aborted else None,
            numerical_pass=None if aborted else True,
            numerical_rejection=False,
            resource_drop=aborted,
            preparation_arithmetic=context.ops.stats,
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
        if fabric is not None:
            result.update(
                word_fabric_statistics=dict(fabric.stats),
                word_fabric_state=word_fabric_cost(self.word_fabric),
                word_context_peak=runtime.peak_live,
                word_fabric_idle=fabric.idle,
            )
        if self.a.defer_all_future:
            deferred = geometry is not None and geometry.deferred
            result.update(
                geometry_wait=deferred,
                geometry_positions=None if geometry is None else geometry.positions,
                geometry_cycles=None
                if geometry_cycle is None
                else geometry_cycle + 1 - prep_cycles,
                scoring_evaluated=best is not None and not aborted,
                scores_evaluated=None
                if geometry is None
                else 0
                if deferred
                else geometry.positions,
            )
            if deferred:
                result.update(reason="all_valid_positions_need_future", numerical_pass=None)
            elif best is None and not aborted:
                result["numerical_pass"] = None
        return result


def spline_score_cost(assumptions=None):
    """Finite registers/ports and explicit operator reservations, not synthesis."""
    a = SplineScoreAssumptions() if assumptions is None else assumptions
    if not isinstance(a, SplineScoreAssumptions):
        raise ValueError("Need explicit SplineScoreAssumptions")
    banks = spline_flow_cost(
        SplineFixedFormat(18, 14), SplineFlowAssumptions(a.capacity, read_latency=a.read_latency)
    )
    fields = dict(
        preparation_saturation_guard=1,
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
        peak_bit_monitors=80 * 8,
    )
    if a.reuse_symbol_power:
        fields.update(nine_previous_powers=9 * 44, current_symbol_power=44, power_context_index=4)
    if a.defer_all_future:
        fields.update(
            geometry_horizons=3 * 17,
            geometry_interval_work=3 * 17,
            geometry_positions_flags_index=5 + 1 + 2 + 2,
            geometry_result_count_and_status=6,
        )
    return dict(
        assumptions=vars(a),
        shared_preparation_banks=banks,
        additional_query_state_upper_bits=sum(fields.values()),
        state_components=fields,
        peak_bit_monitor_slots=len(_monitor_tags(a.reuse_symbol_power, a.defer_all_future)),
        peak_bit_monitor_reserved_slots=80,
        peak_bit_monitor_slot_bits=8,
        peak_bit_monitor_tags=sorted(_monitor_tags(a.reuse_symbol_power, a.defer_all_future)),
        peak_monitor_update=(
            'one completed op per cycle;reserved80x8bit registers hold bit '
            'counts,not node values;LZ/decode/max pipeline and LUT/FF not '
            'synthesized'
        ),
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
        sign_template_multiplication=(
            '0/+1 wiring;negative signs charged as ALU negations;counted real/imag '
            'adds'
        ),
        threshold_products=(
            'compile-time constants;literal shift/add operations '
            'counted;denominator output is diagnostic only'
        ),
        full_window_copies_added=0,
        selected_vectors_buffered=0,
        chosen_vector_replay_charged=True,
        no_claim_on_float_angle_coherent_bridge=True,
        complete16msps_verified=False,
        original16kib_failure_preserved=True,
        synthesis_or_placement_evidence=False,
    )
