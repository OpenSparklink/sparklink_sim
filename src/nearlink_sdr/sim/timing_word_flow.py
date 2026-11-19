"""Value-carrying finite timing service:four25x18 DSPs and restoring division.

One packet context,one input credit,held output. Macro latency assumptions are
explicit and do not establish synthesis,timing or a full16MS/s receiver.
"""

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.sync_fit_constants import GAIN_FLOOR
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, signed, unsigned
from nearlink_sdr.phy.timing_word import (
    KI_Q48,
    KP_Q48,
    LOWER_Q32,
    UPPER_Q32,
    WordTimingState,
    compare_scaled,
    division_job,
    positive,
    ted_operands,
)

__all__ = ["TimingWordService"]


def two_real_products(a, b, c, d):
    """Each signed25x22 uses two real DSP tiles;unsigned low17 fits signed18."""
    return [(a, b & 131071, 0), (a, b >> 17, 17), (c, d & 131071, 0), (c, d >> 17, 17)]


def coefficient_products(coefficient, error):
    """Positive43x signed35 needs four25x18 tiles,not one arbitrary multiplier."""
    return [
        (a, b, sa + sb)
        for a, sa in [(coefficient & (2**24 - 1), 0), (coefficient >> 24, 24)]
        for b, sb in [(error & 131071, 0), (error >> 17, 17)]
    ]


class TimingWordService:
    """Four parallel DSP tiles(latency3),wide add/RNE1,ratio preparation3.

    Power and TED share the bank with constant PI products. A fixed-width
    restoring divider performs exactly one finite remainder step per edge.
    These are unsynthesized scheduling assumptions,not free combinational work.
    """

    def __init__(self, state):
        if not isinstance(state, WordTimingState):
            raise ValueError("Need one explicit finite timing packet state")
        self.state = state
        self.cycle = 0
        self.mode, self.job, self.result = "idle", None, None
        self.first_cycle = self.ready_cycle = None
        self.taken_this_cycle = self.issued_this_cycle = False
        self.stats = dict(
            accepted=0,
            completed=0,
            native_products=0,
            divider_steps=0,
            backpressure=0,
            output_stall_edges=0,
            aborted=0,
        )

    def _count(self, key, amount=1):
        self.stats[key] = unsigned(self.stats[key] + amount, 64)

    @property
    def input_ready(self):
        return self.mode == "idle" and not self.taken_this_cycle

    @property
    def ready(self):
        return self.mode == "ready"

    def issue(self, current, midpoint):
        self.state.validate_words(current, midpoint)
        if self.mode in ("aborted", "released"):
            raise RuntimeError("Closed packet cannot resume")
        if not self.input_ready or self.issued_this_cycle:
            self._count("backpressure")
            return False
        self._count("accepted")
        self.issued_this_cycle = True
        self.first_cycle, self.ready_cycle = self.cycle, None
        self.job = dict(
            current=int(current),
            midpoint=None if midpoint is None else int(midpoint),
            clipped=False,
            error=0,
            ted_erased=0,
            ki=0,
            kp=0,
        )
        r, i, self.job["exponent"] = unpack_interpolated_word(current, INPUT_FORMAT)
        self._start_products(two_real_products(r, r, i, i), "power")
        return True

    def _start_products(self, terms, role):
        for a, b, _ in terms:
            signed(a, 25)
            signed(b, 18)
        self.terms, self.role, self.age = terms, role, 0
        self.mode = "products"

    def _products(self):
        if self.age == 1:
            self.products = [signed(a * b, 43) for a, b, _ in self.terms]
            self._count("native_products", 4)
        elif self.age == 3:
            values = [signed(p << t[2], 80) for p, t in zip(self.products, self.terms, strict=True)]
            self.pairs = [signed(values[0] + values[1], 80), signed(values[2] + values[3], 80)]
        elif self.age == 4:
            result = signed(sum(self.pairs), 80)
            j = self.job
            if self.role == "power":
                j["energy"] = unsigned(result, 44)
                self.mode, self.age = "power", 0
            elif self.role == "ted":
                j["numerator"] = signed(result, 48)
                self.mode, self.age = "ratio_prepare", 0
            elif self.role == "ki":
                j["ki_product"] = result
                self.mode, self.age = "omega", 0
            else:
                j["kp_product"] = result
                self.mode, self.age = "tail", 0
            return
        self.age += 1

    def _power(self):
        j, f = self.job, self.state.fmt
        m, e = self.state.power_word
        if self.age == 0:
            j["decay"] = unsigned((m << 6) - m, f.power_bits + 6)
        elif self.age == 1:
            j["dm"], j["de"] = positive(j["decay"], e, 44)
            j["em"], j["ee"] = positive(j["energy"], 2 * j["exponent"], 44)
        elif self.age == 2:
            j["common"] = max(
                exp for value, exp in [(j["dm"], j["de"]), (j["em"], j["ee"])] if value
            )
            j["a"] = rne_shift(j["dm"], j["common"] - j["de"])
            j["b"] = rne_shift(j["em"], j["common"] - j["ee"]) if j["em"] else 0
            j["erased"] = int(j["a"] == 0 and j["decay"] != 0) + int(
                j["b"] == 0 and j["energy"] != 0
            )
        elif self.age == 3:
            j["power_sum"] = unsigned(j["a"] + j["b"], 45)
        elif self.age == 4:
            j["power"] = positive(j["power_sum"], j["common"] - 6, f.power_bits)
        elif self.age == 5:
            floor = positive(*GAIN_FLOOR, f.power_bits)
            j["floored"] = compare_scaled(*j["power"], *floor) < 0
            if j["floored"]:
                j["power"] = floor
            self.mode, self.age = ("omega" if j["midpoint"] is None else "ted_prepare"), 0
            return
        self.age += 1

    def _ratio(self):
        if self.age < 2:
            self.age += 1  # Explicit leading/compare/shift macro pipeline,latency3.
            return
        j = self.job
        division = division_job(j.pop("numerator"), j["ted_exponent"], j["power"], self.state.fmt)
        j["clipped"] = division["clipped"]
        if "bypass" in division:
            j["error"] = division["bypass"]
            self._after_error()
        else:
            self.num, self.den, self.negative = (
                division["numerator"],
                division["denominator"],
                division["negative"],
            )
            self.quotient = self.remainder = self.div_index = 0
            self.mode = "divide"

    def _after_error(self):
        if self.job["error"]:
            self._start_products(coefficient_products(KI_Q48, self.job["error"]), "ki")
        else:
            self.mode, self.age = "omega", 0

    def _divide(self):
        f = self.state.fmt
        index = f.division_bits - 1 - self.div_index
        value = unsigned((self.remainder << 1) | ((self.num >> index) & 1), 50)
        bit = value >= self.den
        self.remainder = unsigned(value - self.den if bit else value, 49)
        self.quotient = unsigned((self.quotient << 1) | bit, f.error_bits + 2)
        self.div_index += 1
        self._count("divider_steps")
        if self.div_index == f.division_bits:
            self.mode = "round_ratio"

    def _omega(self):
        j = self.job
        if self.age == 0:
            j["ki"] = rne_shift(j.get("ki_product", 0), 48 + self.state.fmt.error_bits - 32)
        elif self.age == 1:
            j["proposed"] = signed(self.state.omega_q32 + j["ki"], 36)
        elif self.age == 2:
            j["omega"] = max(LOWER_Q32, min(UPPER_Q32, j["proposed"]))
            if j["error"]:
                self._start_products(coefficient_products(KP_Q48, j["error"]), "kp")
            else:
                self.mode, self.age = "tail", 0
            return
        self.age += 1

    def _tail(self):
        j = self.job
        if self.age == 0:
            j["kp"] = rne_shift(j.get("kp_product", 0), 48 + self.state.fmt.error_bits - 32)
        elif self.age == 1:
            j["step"] = unsigned(j["omega"] + j["kp"], 35)
        elif self.age == 2:
            j["next"] = unsigned(self.state.next_q32 + j["step"], 96)
        elif self.age == 3:
            self.result = self.state.commit(
                j["current"],
                j["power"],
                j["error"],
                j["clipped"],
                j["erased"],
                j["ted_erased"],
                j["floored"],
                j["ki"],
                j["kp"],
            )
            assert self.result["next_q32"] == j["next"]
            assert self.result["omega_q32"] == j["omega"]
            self.job, self.mode, self.ready_cycle = None, "ready", self.cycle + 1
            self._count("completed")
            return
        self.age += 1

    def step(self):
        if self.cycle == 2**64 - 1:
            raise OverflowError("Timing service clock exhausted")
        if self.mode == "products":
            self._products()
        elif self.mode == "power":
            self._power()
        elif self.mode == "ted_prepare":
            if self.age == 2:
                dr, di, mr, mi, e, lost = ted_operands(
                    self.state.previous_word, self.job["current"], self.job["midpoint"]
                )
                self.job.update(ted_exponent=e, ted_erased=lost)
                self._start_products(two_real_products(dr, mr, di, mi), "ted")
            else:
                self.age += 1  # Peak/scale,aligned components,difference:three clock macro.
        elif self.mode == "ratio_prepare":
            self._ratio()
        elif self.mode == "divide":
            self._divide()
        elif self.mode == "round_ratio":
            doubled = unsigned(2 * self.remainder, 50)
            q = self.quotient + (doubled > self.den or (doubled == self.den and self.quotient & 1))
            self.job["error"] = signed(-q if self.negative else q, self.state.fmt.error_bits + 3)
            self._after_error()
        elif self.mode == "omega":
            self._omega()
        elif self.mode == "tail":
            self._tail()
        elif self.mode == "ready":
            self._count("output_stall_edges")
        self.cycle = unsigned(self.cycle + 1, 64)
        self.taken_this_cycle = self.issued_this_cycle = False

    def take(self):
        if not self.ready:
            return None
        result, self.result, self.mode = self.result, None, "idle"
        self.taken_this_cycle = True
        return result

    def abort(self):
        if self.mode in ("aborted", "released"):
            return
        self._count("aborted")
        self.job = self.result = None
        self.mode = "aborted"

    def release(self):
        """Retire a drained packet context without counting a work cancellation."""
        if self.mode != "idle" or self.job is not None or self.result is not None:
            raise RuntimeError("Timing context must drain before normal release")
        self.mode = "released"

    def estimate(self):
        f = self.state.fmt
        persistent = 192 + 52 + f.power_bits + 10 + 35 + f.error_bits + 3 + 6 * 64
        registers = persistent + 104 + 160 + 516 + 320 + 224 + 7 * 64
        return dict(
            DSP25x18=4,
            DSP_latency=3,
            DSP_II=1,
            extra_raw_RAM_bits=0,
            state_register_bits_upper=registers,
            constant_bits=43 + 35 + 2 * 35 + 64,
            multiplier_bits=dict(
                power="2x22by22=4tiles",
                TED="2x25by22=4tiles",
                PI="positive43or35bySigned27or35=4tiles",
            ),
            divider=dict(
                numerator_bits=f.division_bits,
                denominator_bits=49,
                remainder_bits=50,
                quotient_bits=f.error_bits + 2,
                steps=f.division_bits,
            ),
            NCO_bits=96,
            midpoint_add_bits=97,
            PI_add_bits=36,
            wide_product_tree_bits=80,
            power_add_bits=max(f.power_bits + 6, 44) + 1,
            input_credits=1,
            held_outputs=1,
            input_word_bits=52,
            additional_source_RAM_ports=0,
            power_peak_normalizers="two44bit;zero excluded;two aligned44bit sums",
            TED_peak_normalizers="two22bit peaks;four24bit components;two25bit differences",
            PI_increment_register_bits=[21, 29],
            intermodule_DSP_sharing_credited=False,
            assumptions=(
                "DSP3;wide add/compare/RNE1;power6;TEDpeak/shift/difference3;ratio3;no synthesis"
            ),
            complete_logic_LUTs_or_timing=None,
            synthesized=False,
            full_receiver16MSps=False,
        )
