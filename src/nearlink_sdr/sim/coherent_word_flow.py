"""Chosen fine words→finite CFO/coherent gate (§6.2.3),one bounded transaction.

Reuse the existing CORDIC steps,NCO ROM and native word fabric. Wide products
execute as24x17 tiles;wide additions/comparison execute ALU32 carry steps.
This node does not search,decode,consume CRC or retain another IQ window.
"""

import math
from dataclasses import dataclass

from nearlink_sdr.phy.interpolation_word import rne_shift
from nearlink_sdr.phy.sync_coherent import phase_rotation_words
from nearlink_sdr.phy.sync_fit_word import (
    HALF,
    SyncFitFormat,
    signed,
    unsigned,
    vector_step,
    wrapped,
)
from nearlink_sdr.sim.fine_selection import FineSelection
from nearlink_sdr.sim.word_arithmetic import (
    WordCommand,
    WordFabric,
    WordFabricAssumptions,
    word_fabric_cost,
)

__all__ = ["CoherentWordReceipt", "CoherentWordService"]


@dataclass(frozen=True)
class CoherentWordReceipt:
    """Actual computed sums;acceptance is derived,never a caller-supplied bool.

    The source descriptor/cache are borrowed. Exponent cancels in Q. Floating
    values below are diagnostics only,not the hardware threshold comparison.
    """

    selection: FineSelection
    phase_codes: tuple
    angle_turn40: int
    step_word20: int
    correlation: tuple
    signal_energy: int
    numerator: int
    denominator: int
    comparison: int
    available_cycle: int

    def __post_init__(self):
        if not isinstance(self.selection, FineSelection):
            raise ValueError("Need the actual selected descriptor")
        if (
            not isinstance(self.phase_codes, tuple)
            or len(self.phase_codes) != len(self.selection.words)
            or any(type(v) is not int or not 0 <= v < 16 for v in self.phase_codes)
            or not isinstance(self.correlation, tuple)
            or len(self.correlation) != 2
            or any(type(v) is not int for v in self.correlation)
            or any(type(v) is not int for v in (
                self.angle_turn40, self.step_word20, self.signal_energy, self.numerator,
                self.denominator, self.comparison, self.available_cycle,
            ))
        ):
            raise ValueError("Need finite compiled coherent words")
        signed(self.angle_turn40, 40)
        unsigned(self.step_word20, 20)
        for value in self.correlation:
            signed(value, 48)
        unsigned(self.signal_energy, 52)
        unsigned(self.numerator, 97)
        unsigned(self.denominator, 91)
        unsigned(self.available_cycle, 63)
        # Host receipt consistency checks,not duplicate executable multipliers.
        if (
            self.numerator != sum(v * v for v in self.correlation)
            or self.denominator != self.signal_energy * len(self.phase_codes) * 2**32
            or self.comparison != (
                (20 * self.numerator > 9 * self.denominator)
                - (20 * self.numerator < 9 * self.denominator)
            )
            or self.available_cycle <= self.selection.available_cycle
        ):
            raise ValueError("Inconsistent actual coherent receipt")

    @property
    def accepted(self):
        return bool(
            self.signal_energy > 0
            and self.selection.numerator > 0
            and self.comparison >= 0
        )

    @property
    def quality(self):
        return self.numerator / self.denominator if self.denominator else 0.0

    def reference_confirmation(self):
        """Explicit float reporting bridge;the integer decision is unchanged."""
        exponent = 2 * self.selection.exponent - 32
        numerator = math.ldexp(self.numerator, exponent)
        denominator = math.ldexp(self.denominator, exponent)
        if not math.isfinite(numerator) or not math.isfinite(denominator):
            raise OverflowError("Coherent receipt exceeds floating diagnostic range")
        return dict(
            statistic="known-sync-CFO-compensated-coherent-Q",
            quality=self.quality,
            numerator=numerator,
            denominator=denominator,
            threshold=0.45,
            accepted=self.accepted,
            zero_energy=self.signal_energy == 0,
            zero_selected_correlation=self.selection.numerator == 0,
            omega_rad_per_symbol=self.angle_turn40 * math.tau / 2**40,
            cycles_per_symbol=self.angle_turn40 / 2**40,
            nco=dict(
                step_word=self.step_word20, phase_bits=20,
                coefficient_signed_bits=18, coefficient_fractional_bits=16,
                template_phase_folded=True, finite_sums_and_squared_gate=True,
            ),
            source="actual selected differential words;no frequency/phase or CRC retry",
        )


class CoherentWordService:
    """One borrowed cache,one ALU32/25x18 fabric credit and one held receipt.

    Each step advances real finite values one clock. CORDIC33/42-bit shift/add,
    normalization53-bit RNE and existing4-edge coefficient ROM are explicit
    macro assumptions,not synthesized timing. No future observation is used.
    """

    def __init__(self, selection, phase_codes, *, initial_cycle=None, sink=None):
        if (
            not isinstance(selection, FineSelection)
            or not isinstance(phase_codes, tuple)
            or len(phase_codes) != len(selection.words)
            or any(type(v) is not int or not 0 <= v < 16 for v in phase_codes)
            or (sink is not None and not callable(sink))
        ):
            raise ValueError("Need actual selected words and compiled profile phase codes")
        initial_cycle = selection.available_cycle if initial_cycle is None else initial_cycle
        if (
            type(initial_cycle) is not int
            or not selection.available_cycle <= initial_cycle < 2**63 - 32768
        ):
            raise ValueError("Need actual selection availability and bounded service clock")
        self.selection, self.codes, self.sink = selection, phase_codes, sink
        self.fabric = WordFabric(WordFabricAssumptions(1, 1), initial_cycle=initial_cycle, sink=sink)
        self.started = initial_cycle
        self.ready_cycle = None
        self.mode, self.result = "run", None
        self.stats = dict(cache_reads=0, native_products=0, cordic_steps=0,
                          coefficient_outputs=0, output_hold_edges=0, cancelled=0)
        self._program_owner = self._program()

    @property
    def cycle(self):
        return self.fabric.cycle

    @property
    def ready(self):
        return self.mode == "ready"

    def _count(self, name):
        self.stats[name] = unsigned(self.stats[name] + 1, 64)

    def _emit(self, event, **fields):
        if self.sink is not None:
            self.sink(dict(event=event, cycle=self.cycle, **fields))

    def _command(self, code, a, b=0, *, ab, bb=1, asigned=False, bsigned=False,
                 bits, signed_result=False):
        command = WordCommand(code, a, b, ab, bb, asigned, bsigned, bits, signed_result)
        token = self.fabric.next_token[0]
        while not self.fabric.issue(command, 0, token):
            yield
        if code == "tile":
            self._count("native_products")
        while True:
            yield
            reply = self.fabric.take(0)
            if reply is not None:
                if reply.overflow:
                    raise OverflowError("Coherent word intermediate exceeded declared format")
                return reply.value

    def _abs(self, value, bits):
        return (yield from self._command("abs", value, ab=bits, asigned=True, bits=bits))

    def _tile(self, a, b):
        return (yield from self._command("tile", a, b, ab=24, bb=17,
                                        bits=43, signed_result=True))

    def _signed_product(self, a, b, aa, ba):
        product = yield from self._tile(aa, ba)
        if (a < 0) != (b < 0):
            product = yield from self._command("neg", product, ab=43, asigned=True,
                                               bits=43, signed_result=True)
        return product

    def _sum(self, a, b, bits, *, signed_result=False, subtract=False):
        return (yield from self._command(
            "sub" if subtract else "add", a, b, ab=bits, bb=bits,
            asigned=signed_result, bsigned=signed_result, bits=bits,
            signed_result=signed_result,
        ))

    def _square22(self, magnitude):
        lo = yield from self._tile(magnitude, magnitude & 131071)
        hi = yield from self._tile(magnitude, magnitude >> 17)
        return (yield from self._sum(lo, hi << 17, 52))  # Fixed wire alignment.

    def _square48(self, value):
        value = yield from self._abs(value, 48)
        total = 0
        for i in range(2):
            a = (value >> (24 * i)) & (2**24 - 1)
            for j in range(3):
                b = (value >> (17 * j)) & 131071
                product = yield from self._tile(a, b)
                total = yield from self._sum(total, product << (24 * i + 17 * j), 97)
        return total

    def _constant(self, value, factor, bits):
        total = 0
        for j in range(factor.bit_length()):
            if (factor >> j) & 1:
                total = yield from self._sum(total, value << j, bits)
        return total

    def _program(self):
        source = self.selection
        real, imag = source.correlation  # Borrow existing signed53 descriptor fields.
        peak = max(abs(real), abs(imag))
        shift = peak.bit_length() - 30 if peak else 0
        yield  # Bounded53-bit LZC and signed alignment control.
        x, y = rne_shift(real, shift), rne_shift(imag, shift)
        angle = 0
        if x < 0:
            x, y, angle = -x, -y, -HALF
        signed(x, 33)
        signed(y, 33)
        yield  # RNE/alignment and quadrant register.
        fmt = SyncFitFormat(30)
        for j in range(32):
            x, y, angle = vector_step(x, y, angle, j, fmt)
            self._count("cordic_steps")
            yield
        angle = wrapped(angle)
        if not imag:
            angle = 0 if real >= 0 else -HALF
        elif not real:
            angle = 2**38 if imag > 0 else -(2**38)
        step = rne_shift(angle, 20) & (2**20 - 1)
        yield  # Axis bypass,turn40→20 RNE and modulo register.
        cr, ci, energy = yield from self._accumulate(source, step)
        rr, ii = (yield from self._square48(cr)), (yield from self._square48(ci))
        numerator = yield from self._sum(rr, ii, 97)
        reference = yield from self._constant(energy, len(self.codes), 59)
        denominator = reference << 32  # Known Q16 coefficient scale,not a divider.
        lhs = yield from self._constant(numerator, 20, 110)
        rhs = yield from self._constant(denominator, 9, 110)
        decision = yield from self._command("cmp", lhs, rhs, ab=110, bb=110,
                                            bits=2, signed_result=True)
        return CoherentWordReceipt(source, self.codes, angle, step, (cr, ci), energy,
                                   numerator, denominator, decision, self.cycle + 1)

    def _accumulate(self, source, step):
        """Original serial value/clock program;subclasses may schedule exact words."""
        phase, cr, ci, energy = 0, 0, 0, 0
        for j in range(len(self.codes)):
            self._emit("coherent_cache_read", id=source.ident, index=j)
            self._count("cache_reads")
            yield  # Original selected cache has one synchronous read port.
            yr, yi = source.words[j]
            combined = (phase + (self.codes[j] << 16)) & (2**20 - 1)
            for _ in range(4):
                yield  # Existing address/reflect/ROM/sign pipeline,one outstanding word.
            c, s = phase_rotation_words(combined, phase_bits=20)
            self._count("coefficient_outputs")
            phase = (phase + step) & (2**20 - 1)
            ar, ai = (yield from self._abs(yr, 22)), (yield from self._abs(yi, 22))
            ac, az = (yield from self._abs(c, 18)), (yield from self._abs(s, 18))
            rc = yield from self._signed_product(yr, c, ar, ac)
            iz = yield from self._signed_product(yi, s, ai, az)
            rz = yield from self._signed_product(yr, s, ar, az)
            ic = yield from self._signed_product(yi, c, ai, ac)
            r = yield from self._sum(rc, iz, 48, signed_result=True, subtract=True)
            i = yield from self._sum(rz, ic, 48, signed_result=True)
            cr = yield from self._sum(cr, r, 48, signed_result=True)
            ci = yield from self._sum(ci, i, 48, signed_result=True)
            er, ei = (yield from self._square22(ar)), (yield from self._square22(ai))
            energy = yield from self._sum(energy, er, 52)
            energy = yield from self._sum(energy, ei, 52)
        return cr, ci, energy

    def step(self):
        if self.mode in ("aborted", "taken"):
            raise RuntimeError("Retired coherent transaction cannot resume")
        if self.mode == "run" and self.cycle - self.started >= 32768:
            raise RuntimeError("Finite coherent node exceeded its compiled work bound")
        if self.ready:
            self._count("output_hold_edges")
        else:
            try:
                next(self._program_owner)
            except StopIteration as result:
                self.result, self.mode = result.value, "ready"
                self.ready_cycle = result.value.available_cycle
        self.fabric.tick()

    def take(self):
        if not self.ready:
            return None
        if not self.fabric.idle:
            raise RuntimeError("Coherent output preceded actual finite operation completion")
        result, self.result, self.mode = self.result, None, "taken"
        return result

    def abort(self):
        if self.mode in ("taken", "aborted"):
            raise RuntimeError("Retired coherent transaction cannot cancel again")
        self._program_owner.close()
        self.fabric.flush()  # One actual registered cancellation edge.
        self.result, self.mode = None, "aborted"
        self._count("cancelled")

    @staticmethod
    def estimate():
        fabric = word_fabric_cost(WordFabricAssumptions(1, 1))
        fields = dict(
            program_and_cache_control=16 + 7 + 8 + 20 + 20 + 2 * 63 + 8,
            normalization_CORDIC=2 * 33 + 42 + 7,
            input_and_magnitudes=44 + 44 + 36 + 36,
            coefficient_address_ROM_pipeline=20 + 12 + 20 + 34 + 8,
            native_product_staging=8 * 43,
            accumulated_complex_energy=2 * 48 + 52,
            wide_workspaces_upper=12 * 110,
            receipt_upper=2 * 48 + 52 + 97 + 91 + 40 + 20 + 2 + 63,
            diagnostic_counters=6 * 64,
        )
        total = fabric["state_bits"] + sum(fields.values())
        return dict(
            scope="one chosen cache→finite coherent receipt;not full receiver",
            fabric=fabric, register_fields=fields, register_bits_upper=total,
            register_research_cap_bits=8192,
            source_cache="borrow existing128x44 fine cache;no duplicate IQ",
            cache_read_ports=1, cache_read_latency=1,
            coefficient_ROM_bits=1024 * 17, coefficient_RAM18=1,
            coefficient_BRAM36_sites_upper=1, coefficient_ROM_read_ports=2,
            template_ROM_bits=128 * 4, template_LUT6_ROM_upper=8,
            template_MUXF7_upper=4,
            template_ROM_ports=1, new_IQ_RAM=0,
            CORDIC=dict(iterations=32, xy_bits=33, angle_bits=42,
                        variable_shifters33=2, adders33=2, angle_adder42=1,
                        own_angle_ROM_bits=32 * 40, own_angle_ROM_read_ports=1,
                        own_angle_LUT6_ROM_upper=40, no_packing_or_sharing_credit=True),
            normalization=dict(LZC53=1, absolute53=2, unsigned_compare53=1,
                               signed53_RNE_shifters=2, quadrant_negate33=2),
            phase_bits=20, coefficient_signed_bits=18, coefficient_fractional_bits=16,
            compiled_threshold=[9, 20], comparison_bits=110,
            wide_products="48x48 decomposed into six24x17 native tiles each",
            divide_sqrt_or_runtime_trig_in_acceptance=0,
            quality_division_is_reporting_only=True,
            DSP25x18=fabric["DSP25x18"], ALU32=fabric["ALU32"],
            concurrent_packets=1, finite_work_bound_edges=32768,
            finite_work_bound_excludes_completed_output_hold=True,
            changing_template_configuration_cycles=None,
            macro_clock_assumptions="existing4-edge ROM;one LZC/RNE stage;one CORDIC iteration/edge",
            sharing_with_other_modules_not_credited=True,
            actual_synthesis=False, LUT_FF_or_achieved_Fmax=None,
            full_RX_latency=None, full_fixed_RX=False, continuous16MSps=False,
        )
