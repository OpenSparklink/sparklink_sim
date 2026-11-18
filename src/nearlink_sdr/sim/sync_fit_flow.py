"""Value-carrying finite initializer:CORDIC,DSP tiles and restoring division.

One input context,one packet result;no trace of future observations or answers.
Macro timing assumptions are explicit,not synthesis or whole-RX throughput.
"""

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.sync_fit_constants import GAIN_OVER_N_Q40, POWER_OVER_N_Q40
from nearlink_sdr.phy.sync_fit_word import (
    HALF,
    INPUT_FORMAT,
    TURN,
    WordSyncFit,
    finish_vector,
    normalize_sum,
    signed,
    unsigned,
    vector_step,
    wrapped,
)

__all__ = ["SyncFitWordFlow"]


def const_program(factor):
    return {
        31: (("shift", 5), ("sub", 0)),
        61: (("shift", 6), ("sub", 1), ("sub", 0)),
        125: (("shift", 7), ("sub", 1), ("sub", 0)),
        32: (("shift", 5),),
        62: (("shift", 6), ("sub", 1)),
        126: (("shift", 7), ("sub", 1)),
    }[factor]


class SyncFitWordFlow:
    """Single word credit,iterative32-step vectoring and one serial25x18 DSP.

    Fixed40 clocks per accepted word,then bounded finalization. Word and final
    workspaces are mutually exclusive. A held output never admits a new packet.
    """

    def __init__(self, phase_codes, coarse_turn40, fmt=None):
        self._fit = WordSyncFit(phase_codes, coarse_turn40, fmt)
        self.fmt, self.symbols = self._fit.fmt, len(self._fit.codes)
        self.cycle = 0
        self.first_cycle = self.ready_cycle = None
        self.mode, self.job = "collect", None
        self.result = None
        self.output_taken = False
        self.stats = dict(
            accepted_words=0,
            input_backpressure=0,
            cordic_iterations=0,
            native_products=0,
            divider_steps=0,
            output_stall_cycles=0,
        )

    def _count(self, key):
        if self.stats[key] == 2**64 - 1:
            raise OverflowError("Sync-fit flow statistic exhausted")
        self.stats[key] += 1

    @property
    def input_ready(self):
        return self.mode == "collect" and self._fit.used < self.symbols

    @property
    def ready(self):
        return self.mode == "ready" and not self.output_taken

    def issue(self, word):
        unpack_interpolated_word(word, INPUT_FORMAT)
        if self.mode == "aborted":
            raise RuntimeError("Aborted fit cannot resume")
        if not self.input_ready:
            self._count("input_backpressure")
            return False
        self._count("accepted_words")
        self.job = dict(word=word, age=0)
        self.mode = "word"
        if self.first_cycle is None:
            self.first_cycle = self.cycle
        return True

    def _word(self):
        j, f = self.job, self._fit
        age = j["age"]
        if age == 0:
            r, i, e = unpack_interpolated_word(j.pop("word"), INPUT_FORMAT)
            peak = max(abs(r), abs(i))
            j.update(
                real=r,
                imag=i,
                exponent=e,
                shift=self.fmt.peak_bits - peak.bit_length() if peak else 0,
            )
        elif age == 1:
            x, y = j["real"] << j["shift"], j["imag"] << j["shift"]
            j.update(
                x=signed(-x if x < 0 else x, self.fmt.xy_bits),
                y=signed(-y if x < 0 else y, self.fmt.xy_bits),
                angle=-HALF if x < 0 else 0,
                products=[],
            )
        if 2 <= age <= 33:
            j["x"], j["y"], j["angle"] = vector_step(j["x"], j["y"], j["angle"], age - 2, self.fmt)
            self._count("cordic_iterations")
        if 3 <= age <= 6:
            # Four issues at ages1..4,three registered DSP clocks per product.
            k = age - 3
            value = j["real"] if k < 2 else j["imag"]
            operand = (value & 131071) if k % 2 == 0 else (value >> 17)
            j["products"].append(signed(value * operand, 43))
            self._count("native_products")
        if age == 7:
            j["power_r"] = signed(j["products"][0] + (j["products"][1] << 17), 45)
        elif age == 8:
            j["power_i"] = signed(j["products"][2] + (j["products"][3] << 17), 45)
        elif age == 9:
            j["raw_power"] = unsigned(j.pop("power_r") + j.pop("power_i"), 45)
            j.pop("products")
        if age == 34:
            angle, magnitude, exponent = finish_vector(j)
            zero = not (j["real"] or j["imag"])
            if zero:
                f._count("zero_vectors")
            if not j["real"] or not j["imag"]:
                f._count("axis_vectors")
            j["zero"] = zero
            j["phase"] = wrapped(angle - (f.codes[f.used] << 36))
            j["sums"] = []
            for acc, value, exp in (
                (f.magnitude, magnitude, exponent),
                (f.power, j["raw_power"], 2 * j["exponent"]),
            ):
                m, e = normalize_sum(value, exp)
                common = max(e, acc.exponent) if acc.mantissa else e
                j["sums"].append(dict(skip=value == 0, m=m, e=e, common=common))
        elif age == 35:
            j["phase"] = 0 if j["zero"] else wrapped(j["phase"] - f.coarse_phase)
            for acc, item in zip((f.magnitude, f.power), j["sums"], strict=True):
                if item["skip"]:
                    continue
                a = rne_shift(acc.mantissa, item["common"] - acc.exponent)
                b = rne_shift(item["m"], item["common"] - item["e"])
                item.update(a=a, b=b, lost=int(acc.mantissa != 0 and a == 0) + int(b == 0))
        elif age == 36:
            j["difference"] = signed(j["phase"] - f.previous, 41)
            d = j["difference"]
            j["wrap"] = 1 if d > HALF else -1 if d < -HALF else 0
            for item in j["sums"]:
                if not item["skip"]:
                    item["sum"] = unsigned(item.pop("a") + item.pop("b"), 55)
        elif age == 37:
            j["difference"] = signed(j["difference"] - j["wrap"] * TURN, 41)
            if j["wrap"]:
                f._count("unwrap_events")
            for acc, item in zip((f.magnitude, f.power), j["sums"], strict=True):
                if item["skip"]:
                    continue
                acc.mantissa, acc.exponent = normalize_sum(item["sum"], item["common"])
                acc.erased = unsigned(acc.erased + item["lost"], 64)
        elif age == 38:
            j["unwrapped"] = signed(
                f.unwrapped_phase + j["difference"] if f.used else j["phase"], 48
            )
        elif age == 39:
            f.unwrapped_phase = j["unwrapped"]
            f.sum_prefix = signed(f.sum_prefix + f.sum_phase, 64)
            f.sum_phase = signed(f.sum_phase + f.unwrapped_phase, 56)
            f.previous = j["phase"]
            f.coarse_phase = (f.coarse_phase + f.coarse) & (TURN - 1)
            f.used += 1
            self.job = None
            if f.used == self.symbols:
                self._start_const(f.sum_phase, self.symbols - 1, "weighted")
            else:
                self.mode = "collect"
            return
        j["age"] += 1

    def _start_const(self, value, factor, after):
        self.base, self.acc, self.pc = value, 0, 0
        self.program, self.after = const_program(factor), after
        self.mode = "constant"

    def _constant(self):
        op, shift = self.program[self.pc]
        self.acc = signed(
            self.base << shift if op == "shift" else self.acc - (self.base << shift), 64
        )
        self.pc += 1
        if self.pc == len(self.program):
            self.mode = self.after

    def _start_divide(self, numerator, denominator, target):
        self.div_numerator = signed(numerator, 64)
        self.den = unsigned(denominator, 19)
        self.target = target
        self.mode = "div_load"

    def _divide(self):
        if self.mode == "div_load":
            self.negative = self.div_numerator < 0
            self.num = unsigned(abs(self.div_numerator), 64)
            self.q = self.remainder = self.div_index = 0
            self.mode = "div_step"
        elif self.mode == "div_step":
            r = unsigned((self.remainder << 1) | ((self.num >> (63 - self.div_index)) & 1), 20)
            bit = r >= self.den
            self.remainder = r - self.den if bit else r
            self.q = unsigned((self.q << 1) | int(bit), 64)
            self.div_index += 1
            self._count("divider_steps")
            if self.div_index == 64:
                self.mode = "div_round"
        elif self.mode == "div_round":
            self.q = unsigned(
                self.q
                + int(
                    2 * self.remainder > self.den or (2 * self.remainder == self.den and self.q & 1)
                ),
                64,
            )
            self.mode = "div_sign"
        else:
            value = signed(-self.q if self.negative else self.q, 64)
            if self.target == "slope":
                self.slope = signed(value, 42)
                self._start_const(self.slope, self.symbols - 1, "intercept_N")
            else:
                self.phase = value & (TURN - 1)
                self._start_scale("gain")

    def _start_scale(self, role):
        self.role = role
        acc = self._fit.magnitude if role == "gain" else self._fit.power
        self.a, self.exponent = acc.mantissa, acc.exponent
        self.b = (GAIN_OVER_N_Q40 if role == "gain" else POWER_OVER_N_Q40)[self.symbols]
        self.scale_acc = self.tile = self.tile_age = 0
        self.mode = "scale"

    def _scale(self):
        if self.tile_age == 2:
            a = (self.a >> (24 * (self.tile // 3))) & ((1 << 24) - 1)
            b = (self.b >> (17 * (self.tile % 3))) & 131071
            self.product = unsigned(a * b, 43)
            self.carry = 0
            self._count("native_products")
        elif 3 <= self.tile_age <= 5:
            segment = self.tile_age - 3
            bits = 32 if segment < 2 else 26
            shift = 32 * segment
            mask = (1 << bits) - 1
            product = unsigned(self.product << (24 * (self.tile // 3) + 17 * (self.tile % 3)), 90)
            value = ((self.scale_acc >> shift) & mask) + ((product >> shift) & mask) + self.carry
            self.scale_acc = (self.scale_acc & ~(mask << shift)) | ((value & mask) << shift)
            self.carry = value >> bits
            if segment == 2 and self.carry:
                raise OverflowError("Sync-fit tiled product growth")
        self.tile_age += 1
        if self.tile_age == 6:
            self.tile += 1
            self.tile_age = 0
            if self.tile == 9:
                self.mode = "scale_peak"

    def _pack_scale(self):
        if self.mode == "scale_peak":
            self.scale_shift = max(0, self.scale_acc.bit_length() - 54)
            self.mode = "scale_round"
        elif self.mode == "scale_round":
            self.scale_m = rne_shift(self.scale_acc, self.scale_shift)
            self.mode = "scale_pack"
        else:
            if not self.scale_acc:
                value = (0, 0)
            else:
                if self.scale_m >= 1 << 54:
                    self.scale_shift += 1
                    self.scale_m = rne_shift(self.scale_acc, self.scale_shift)
                value = (
                    unsigned(self.scale_m, 54),
                    signed(self.exponent - 40 + self.scale_shift, 10),
                )
            if self.role == "gain":
                self.gain = value
                self._start_scale("power")
            else:
                self.power = value
                self.mode = "complete"

    def advance(self):
        if self.cycle >= 2**63 - 1:
            raise OverflowError("Sync-fit clock exhausted")
        if self.mode == "aborted":
            raise RuntimeError("Aborted fit cannot advance")
        if self.mode == "word":
            self._word()
        elif self.mode == "constant":
            self._constant()
        elif self.mode == "weighted":
            self._start_divide(
                signed(self.acc - 2 * self._fit.sum_prefix, 64),
                self.symbols * (self.symbols * self.symbols - 1) // 6,
                "slope",
            )
        elif self.mode == "intercept_N":
            self._start_const(self.acc, self.symbols, "intercept")
        elif self.mode == "intercept":
            self._start_divide(
                signed(2 * self._fit.sum_phase - self.acc, 64), 2 * self.symbols, "phase"
            )
        elif self.mode.startswith("div_"):
            self._divide()
        elif self.mode == "scale":
            self._scale()
        elif self.mode.startswith("scale_"):
            self._pack_scale()
        elif self.mode == "complete":
            self.result = self._fit.complete(self.slope, self.phase, self.gain, self.power)
            self.ready_cycle = self.cycle + 1
            self.mode = "ready"
        elif self.ready:
            self._count("output_stall_cycles")
        self.cycle += 1

    def take(self):
        if not self.ready:
            return None
        self.output_taken = True
        return self.result.copy()

    def abort(self):
        if self.mode == "aborted":
            return None
        owned = int(self.job is not None)
        self.job = self.result = None
        self._fit.abort()
        self.mode = "aborted"
        return dict(owned_input=owned, packet_context=1)

    def estimate(self):
        xy = self.fmt.xy_bits
        packet = 56 + 64 + 40 + 48 + 41 + 40 + 7 + 7 + 1 + 1
        sums = 2 * (54 + 10)
        word = 52 + 6 + 2 * xy + 42 + 172 + 135 + 40 + 40 + 41 + 41 + 48 + 2 + 602 + 6 + 7
        final = (
            64
            + 64
            + 64
            + 64
            + 20
            + 19
            + 7
            + 1
            + 1
            + 5
            + 2
            + 54
            + 40
            + 90
            + 43
            + 3
            + 4
            + 1
            + 6
            + 55
            + 10
            + 1
        )
        result = 40 + 42 + 42 + 2 * 64 + 7 + 1 + 2
        stats = (6 + 4 + 2) * 64
        clock = 3 * 64
        DSP_internal = 3 * (25 + 18 + 43)
        control = 34
        return dict(
            input_bits=52,
            CORDIC_steps=32,
            CORDIC_xy_bits=xy,
            angle_fraction_bits=40,
            sum_mantissa_bits=54,
            sum_exponent_bits=10,
            moment_S_bits=56,
            moment_T_bits=64,
            additional_DSP25x18=1,
            DSP_latency_assumed=3,
            wide_add_RNE_cycles_assumed=1,
            operations_per_packet=dict(
                native_products=4 * self.symbols + 18,
                CORDIC_real_adds=64 * self.symbols,
                CORDIC_angle_adds=32 * self.symbols,
                CORDIC_variable_shifts=64 * self.symbols,
                atan_ROM_reads=32 * self.symbols,
                prefix_moment_adds=2 * self.symbols,
                restoring_division_bit_steps=128,
                source_interpolant_reads=self.symbols,
                sqrt_runtime=0,
            ),
            packet_context_bits=packet,
            positive_sum_bits=sums,
            word_workspace_bits=word,
            final_workspace_bits=final,
            workspaces_mutually_exclusive=True,
            shared_workspace_bits=max(word, final),
            packet_result_bits=result,
            statistics_bits=stats,
            clock_timestamp_bits=clock,
            DSP_internal_state_bits_upper=DSP_internal,
            control_metadata_bits=control,
            register_state_upper_bits=packet
            + sums
            + max(word, final)
            + result
            + stats
            + clock
            + DSP_internal
            + control,
            template_bits=512,
            atan_ROM_bits=1280,
            coefficient_context_bits=240 + 64 + 57,
            constant_microcode_bits_upper=128,
            logical_template_and_constant_bits=512 + 1280 + 240 + 64 + 57 + 128,
            template_read_ports=1,
            atan_read_ports=1,
            additional_source_read_ports=0,
            template_mapping_estimate="128x4 ROM:8LUT6+4MUXF7;separate from32x40 atanROM",
            other_LUTs_and_timing=None,
            per_word_cycles=40,
            output_credits=1,
            input_credits=1,
            full_ADC_or_RX16MSps=False,
            known_sync_hold_PI_head_decoder_cycles=None,
            synthesis_or_timing=False,
        )
