"""Exact B2 integer carrier with a shorter, explicit PI feedback calendar."""

from nearlink_sdr.phy.carrier_fixed import _FREQUENCY_BOUNDS, _rne
from nearlink_sdr.phy.sync_fit_word import signed
from nearlink_sdr.sim.carrier_fixed_flow import (
    CarrierWordEngine,
    _simulate_carrier_words,
    carrier_word_cost,
)

__all__ = ["ParallelPICarrierEngine", "parallel_pi_carrier_cost", "simulate_parallel_pi_carrier"]

_SERVICE_EDGES = 38


class ParallelPICarrierEngine(CarrierWordEngine):
    """Use the serial B2 front and retire the same full-width feedback after 38 edges."""

    def tick(self):
        state = self.active
        if state is None or self.done or state["cycle"] < 33:
            return super().tick()

        state["cycle"] += 1
        edge = state["cycle"]
        if edge == 34:
            state["fi"] = _rne(state["error"], 11)
            state["pi"] = _rne(state["error"], 5)
        elif edge == 35:
            state["next_f"] = state["frequency"] + state["fi"]
            state["next_phase"] = signed(state["phase"] + state["pi"], 33)
        elif edge == 36:
            bound = _FREQUENCY_BOUNDS[state["m"]]
            self.node.stats["frequency_clamps"] += int(abs(state["next_f"]) > bound)
            state["next_f"] = min(bound, max(-bound, state["next_f"]))
        elif edge == 37:
            state["next_phase"] = signed(state["next_phase"] + state["next_f"], 34)
        elif edge == _SERVICE_EDGES:
            phase = state["next_phase"]
            self.node.stats["phase_wraps"] += int(not -(2**31) <= phase < 2**31)
            self.node.phase = (phase + 2**31) % 2**32 - 2**31
            self.node.frequency = state["next_f"]
            self.node.used += 1
            self.done = True

        return dict(
            cycle=edge,
            phase=state["phase"],
            x=state["x"],
            y=state["y"],
            angle=state["angle"],
            error=state["error"],
            frequency=state.get("next_f"),
            done=self.done,
        )


def parallel_pi_carrier_cost(input_depth=4):
    """Count the additional carry path and reserve one partial sum register."""
    cost = carrier_word_cost(input_depth)
    state = cost["state"]
    state["parallel_PI_phase_partial_reserve"] = 34
    return cost | dict(
        service_clocks=_SERVICE_EDGES,
        selected_output_latency_cycles=_SERVICE_EDGES + 1,
        PI_parallel_RNE_units=2,
        PI_extra_adder_bits=34,
        PI_phase_partial_required_bits=33,
        PI_new_registered_bits_upper=34,
        logical_bits=sum(state.values()),
        additional_DSP=0,
        additional_RAMB18=0,
        additional_memory_ports=0,
        assumptions=(cost["assumptions"] + ";two RNE and two adders concurrently; "
                     "phase/frequency/index commit at edge 38"),
    )


def simulate_parallel_pi_carrier(real, imag, **kwargs):
    """Use the original bounded FIFOs, source calendar and drop semantics."""
    return _simulate_carrier_words(
        real, imag, engine_type=ParallelPICarrierEngine, service_clocks=_SERVICE_EDGES, **kwargs
    )
