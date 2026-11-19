"""Finite NCO→registered interpolation descriptors (§6.2.1.2 implementation model).

Actual96/97bit coordinates,phase16 RNE and causal history/future bounds.
Three macro edges,one credit,held output;wide add/compare1 is not synthesis.
"""

from nearlink_sdr.phy.interpolation_word import rne_shift
from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.phy.timing_word import ONE, WordTimingState

__all__ = ["TimingQueryService"]


class TimingQueryService:
    """Snapshot one stable timing state;no floating geometry or timing-state mutation."""

    def __init__(self):
        self.cycle = 0
        self.mode = "idle"
        self.result = None
        self.inputs = self.partial = self.decoded = None
        self.age = 0
        self.first_cycle = self.ready_cycle = None
        self.taken_this_cycle = self.issued_this_cycle = False
        self.stats = dict.fromkeys(
            ["accepted", "completed", "backpressure", "output_stall_edges", "aborted"], 0
        )

    def _count(self, key):
        self.stats[key] = unsigned(self.stats[key] + 1, 64)

    @property
    def input_ready(self):
        return self.mode == "idle" and not self.taken_this_cycle

    @property
    def ready(self):
        return self.mode == "ready"

    def issue(self, state):
        if not isinstance(state, WordTimingState):
            raise ValueError("Need an explicit stable finite timing state")
        if self.mode in ("released", "aborted"):
            raise RuntimeError("Closed query context cannot resume")
        values = dict(
            current=unsigned(state.next_q32, 96),
            previous=unsigned(state.previous_q32, 96),
            count=unsigned(state.count, 64),
            tracked=state.count >= state.hold_symbols,
        )
        if not self.input_ready or self.issued_this_cycle:
            self._count("backpressure")
            return False
        self._count("accepted")
        self.inputs = values
        self.partial = self.decoded = None
        self.first_cycle, self.ready_cycle = self.cycle, None
        self.mode, self.age = "running", 0
        self.issued_this_cycle = True
        return True

    def step(self):
        if self.cycle == 2**64 - 1:
            raise OverflowError("Query service clock exhausted")
        if self.mode == "running":
            if self.age == 0:
                x = self.inputs
                self.partial = dict(
                    midpoint=unsigned(x["current"] + x["previous"], 97),
                    cell=x["current"] >> 32,
                    fraction=x["current"] & (ONE - 1),
                    count=x["count"],
                    tracked=x["tracked"],
                )
            elif self.age == 1:
                x = self.partial
                current = dict(cell=unsigned(x["cell"], 64), phase=rne_shift(x["fraction"], 16))
                midpoint = (
                    dict(
                        cell=unsigned(x["midpoint"] >> 33, 64),
                        phase=rne_shift(x["midpoint"] & (2**33 - 1), 17),
                    )
                    if x["tracked"] else None
                )
                self.decoded = dict(
                    queries=(current, midpoint),
                    future_end=unsigned(x["cell"] + 3 + bool(x["fraction"]), 64),
                    count=x["count"],
                )
            else:
                x = self.decoded
                first = min(q["cell"] for q in x["queries"] if q is not None)
                self._count("completed")
                self.result = x | dict(retain_before=unsigned(max(0, first - 1), 64))
                self.mode = "ready"
                self.ready_cycle = self.cycle + 1
            self.age += 1
        elif self.mode == "ready":
            self._count("output_stall_edges")
        self.cycle += 1
        self.taken_this_cycle = self.issued_this_cycle = False

    def take(self):
        if not self.ready:
            return None
        result, self.result = self.result, None
        self.inputs = self.partial = self.decoded = None
        self.mode = "idle"
        self.taken_this_cycle = True
        return result

    def abort(self):
        if self.mode in ("released", "aborted"):
            return
        self._count("aborted")
        self.inputs = self.partial = self.decoded = self.result = None
        self.mode = "aborted"

    def release(self):
        if self.mode != "idle":
            raise RuntimeError("Query context must drain before normal release")
        self.mode = "released"

    def estimate(self):
        stages = dict(input=257, midpoint_and_split=258, rounded_queries=291, held_output=355)
        control = 64 + 128 + 16 + len(self.stats) * 64
        return dict(
            pipeline_register_bits=stages,
            control_clock_counter_bits=control,
            logical_register_bits_upper=sum(stages.values()) + control,
            extra_DSP=0,
            new_raw_RAM_bits=0,
            added_source_ports=0,
            latency_edges=3,
            input_credits=1,
            held_outputs=1,
            minimum_II_with_immediate_take=4,
            wide_add_bits=97,
            future_retire_compare_add_bits=64,
            RNE_input_bits=[32, 33],
            phase_bits=17,
            state_count_compare_bits=64,
            assumption="three stages;97bit add and64bit compare/add1;not mapped or timed",
            complete_LUT_FF_or_timing=None,
            synthesis=False,
        )
