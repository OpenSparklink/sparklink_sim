"""Parallel raw-angle word calendar; timing assumptions remain unsynthesized."""

from nearlink_sdr.phy.carrier_fixed import _ATAN_WORDS, _FREQUENCY_BOUNDS, _rne
from nearlink_sdr.phy.carrier_parallel import ParallelAngleCarrierLoop
from nearlink_sdr.phy.sync_coherent import phase_rotation_words
from nearlink_sdr.sim.carrier_fixed_flow import (
    CarrierWordEngine,
    _simulate_carrier_words,
    carrier_word_cost,
)

__all__ = ["ParallelCarrierEngine", "parallel_carrier_cost", "simulate_parallel_carrier"]


class ParallelCarrierEngine(CarrierWordEngine):
    """One word,32 clocks; current phase is snapshotted for both parallel paths."""

    def __init__(self, modulation, start):
        self.node = ParallelAngleCarrierLoop(modulation, start)
        self.active = None
        self.done = False

    def tick(self):
        if self.active is None or self.done:
            raise RuntimeError("Need an active unfinished carrier word")
        s = self.active
        s["cycle"] += 1
        c = s["cycle"]
        if c == 1:
            peak = max(abs(s["a"]), abs(s["b"]))
            s["zero"] = peak == 0
            s["shift"] = 0 if peak == 0 else 21 - peak.bit_length()
        elif c == 2:
            s["x"], s["y"] = s["a"] << s["shift"], s["b"] << s["shift"]
        elif c == 3:
            s["angle"] = 0
            if s["x"] < 0:
                s["x"], s["y"], s["angle"] = -s["x"], -s["y"], -(2**31)
            s["power_products"] = (s["a"] ** 2, s["b"] ** 2)
        elif c == 4:
            s["raw_power"] = sum(s["power_products"])
            s["cos"], s["sin"] = phase_rotation_words(s["phase"] & (2**32 - 1))
        elif c == 5:
            s["power"] = _rne(s["raw_power"], 4)
            self.node.stats["power_erased"] += int(s["raw_power"] > 0 and s["power"] == 0)
        elif c == 6:
            previous = self.node.power.power
            s["low"], s["high"] = s["power"] * 16 < previous, s["power"] > 4 * previous
            self.node.stats["low_power_holds"] += int(s["low"])
            self.node.stats["high_power_holds"] += int(s["high"])
        elif c == 7:
            a, b, co, si = s["a"], s["b"], s["cos"], s["sin"]
            s["products"] = (a * co, b * si, a * si, b * co)
        elif c == 8:
            a, b, d, e = s["products"]
            s["sums"] = (a - b, d + e)
        elif c == 9:
            raw = [_rne(v, 16) for v in s["sums"]]
            self.node.stats["output_saturations"] += sum(not -(2**17) <= v < 2**17 for v in raw)
            s["real"], s["imag"] = (min(2**17 - 1, max(-(2**17), v)) for v in raw)
        if c == 13:
            self.node.power.advance_words([s["power"]])
        # Angle is independent of the rotation and gate. Compute all18 slots;
        # the power decision only controls whether its result enters the PI.
        if 4 <= c <= 21 and not s["zero"] and s["y"] != 0:
            k = c - 4
            x, y = s["x"], s["y"]
            direction = 1 if y > 0 else -1
            s["x"], s["y"] = x + direction * (y >> k), y - direction * (x >> k)
            s["angle"] += direction * _ATAN_WORDS[k]
            if (
                not -(2**23) <= s["x"] < 2**23
                or not -(2**23) <= s["y"] < 2**23
                or not -(2**33) <= s["angle"] < 2**33
            ):
                self.node.stats["cordic_overflows"] += 1
                raise ArithmeticError("Parallel calendar CORDIC scratch overflow")
        elif c == 22:
            s["angle"] = (s["angle"] + 2**31) % 2**32 - 2**31 - s["phase"]
        elif c == 23:
            if (self.node.start + self.node.used) & 1:
                s["angle"] += 2**31 // s["m"]
        elif c == 24:
            sector = 2**32 // s["m"]
            s["limit"] = sector // 4
            s["error"] = (
                0 if s["zero"] or s["low"] or s["high"] else s["angle"] % sector - sector // 2
            )
            if s["zero"] and not (s["low"] or s["high"]):
                self.node.stats["zero_vector_holds"] += 1
        elif c == 25:
            self.node.stats["error_clamps"] += int(abs(s["error"]) > s["limit"])
            s["error"] = min(s["limit"], max(-s["limit"], s["error"]))
        elif c == 26:
            s["fi"], s["pi"] = _rne(s["error"], 11), _rne(s["error"], 5)
        elif c == 27:
            s["next_f"] = s["frequency"] + s["fi"]
        elif c == 28:
            bound = _FREQUENCY_BOUNDS[s["m"]]
            self.node.stats["frequency_clamps"] += int(abs(s["next_f"]) > bound)
            s["next_f"] = min(bound, max(-bound, s["next_f"]))
        elif c == 29:
            s["next_phase"] = s["phase"] + s["next_f"]
        elif c == 30:
            s["next_phase"] += s["pi"]
        elif c == 31:
            self.node.stats["phase_wraps"] += int(not -(2**31) <= s["next_phase"] < 2**31)
            self.node.phase = (s["next_phase"] + 2**31) % 2**32 - 2**31
            self.node.frequency = s["next_f"]
        elif c == 32:
            self.node.used += 1
            self.done = True
        return dict(cycle=c, x=s.get("x"), y=s.get("y"), angle=s.get("angle"), done=self.done)


def parallel_carrier_cost(input_depth=4):
    """Allocation bound,not a LUT/FF/synthesis result; ADC bridge stays outside."""
    cost = carrier_word_cost(input_depth)
    cost.update(
        service_clocks=32,
        CORDIC_source="raw signed18-bitIQ;phase subtraction after18iterations",
        angle_subtract_ALU_bits=34,
        angle_subtract_reuse="CORDIC angle ALU4..21 then subtraction22/rotation23;PI starts26",
        parallel_PI_RNE_units=2,
        additional_angle_ALU_steering_bits=68,
        additional_PI_RNE_LUT_FF=None,
        additional_angle_steering_LUT_FF=None,
        logical_bits=cost["logical_bits"],
        new_state_reason="same owned raw word,phase snapshot,CORDIC and PI scratch;extra steering is combinational,not zero-cost hardware",
        operations_per_word=dict(
            real_18x18_multiplications=6,
            CORDIC_additions_worst=54,
            CORDIC_variable_shifts_worst=36,
            CORDIC_direction_comparisons_worst=18,
            angle_phase_subtractions=1,
            simultaneous_PI_RNE=2,
            power_macro="existing8cycle bounded fixed node;see carrier_weight_flow",
            runtime_division=0,
            runtime_square_root=0,
            zero_or_held_vectors_still_scheduled=True,
        ),
        assumptions="NCO4,DSP3,wideadd/RNE1,parallel3ALU CORDIC1iteration/cycle,power8,2parallelPI-RNE,state forwarding",
        actual_synthesis_or_timing=False,
    )
    return cost


def simulate_parallel_carrier(real, imag, **kwargs):
    """Same finite source/FIFOs/stalls/drop policy as B2,with32-cycle service."""
    return _simulate_carrier_words(
        real, imag, engine_type=ParallelCarrierEngine, service_clocks=32, **kwargs
    )
