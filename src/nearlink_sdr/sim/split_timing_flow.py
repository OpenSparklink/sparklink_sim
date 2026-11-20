"""Separate finite detector and PI/NCO owners (§6.2.1.2 implementation experiment).

Actual DSP/divider edges,one credit and one held result per owner. The two banks
are dedicated:8DSP,not an unproved shared4DSP implementation. A packet scheduler,
error FIFO,source lifetime and complete input throughput are separate work.
"""

from nearlink_sdr.phy.interpolation_word import unpack_interpolated_word
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, signed, unsigned
from nearlink_sdr.phy.timing_word import WordTimingState
from nearlink_sdr.sim.timing_overlap_flow import ParallelTEDTimingWordService
from nearlink_sdr.sim.timing_query_flow import TimingQueryService
from nearlink_sdr.sim.timing_word_flow import TimingWordService

__all__ = [
    "SplitTimingQueryService", "TimingDetectorService", "TimingPIService", "split_timing_states"
]


class _DetectorState:
    def __init__(self, seed):
        self.fmt, self.hold_symbols = seed.fmt, seed.hold_symbols
        self.power_word, self.previous_word = seed.power_word, None
        self.count = self.error_word = 0
        self.error_clip_events = self.power_floor_events = 0
        self.power_alignment_erased = self.ted_alignment_erased = 0

    def validate_words(self, current, midpoint):
        unpack_interpolated_word(current, INPUT_FORMAT)
        if (midpoint is None) != (self.count < self.hold_symbols):
            raise ValueError("Midpoint required exactly after actual sync hold")
        if midpoint is not None:
            unpack_interpolated_word(midpoint, INPUT_FORMAT)

    def commit(self, j):
        current = j["current"]
        unpack_interpolated_word(current, INPUT_FORMAT)
        power = j["power"]
        if not unsigned(power[0], self.fmt.power_bits):
            raise ValueError("Detector power cannot be zero")
        signed(power[1], 10)
        error = signed(j["error"], self.fmt.error_bits + 3)
        counters = {
            k: unsigned(getattr(self, k) + amount, 64)
            for k, amount in [
                ("count", 1), ("error_clip_events", j["clipped"]),
                ("power_floor_events", j["floored"]),
                ("power_alignment_erased", j["erased"]),
                ("ted_alignment_erased", j["ted_erased"]),
            ]
        }
        for k, value in counters.items():
            setattr(self, k, value)
        self.power_word, self.previous_word, self.error_word = power, int(current), error
        return self.snapshot()

    def snapshot(self):
        return dict(
            count=self.count, power=list(self.power_word), previous_word=self.previous_word,
            error_word=self.error_word, error_clip_events=self.error_clip_events,
            power_floor_events=self.power_floor_events,
            power_alignment_erased=self.power_alignment_erased,
            ted_alignment_erased=self.ted_alignment_erased,
        )

    def persistent_bits(self):
        # Include7-bit compiled hold selector and a previous-word valid bit.
        return self.fmt.power_bits + 10 + 52 + 1 + self.fmt.error_bits + 3 + 5 * 64 + 7


class _PIState:
    def __init__(self, seed):
        self.fmt, self.hold_symbols = seed.fmt, seed.hold_symbols
        self.previous_q32, self.next_q32 = seed.previous_q32, seed.next_q32
        self.omega_q32 = seed.omega_q32
        self.count = self.error_word = self.rate_limit_events = 0

    def commit(self, j):
        signed(j["ki"], 21)
        signed(j["kp"], 29)
        previous = unsigned(self.next_q32, 96)
        position = unsigned(j["next"], 96)
        if not j["step"] or position != previous + j["step"]:
            raise OverflowError("PI/NCO cannot stop or change the registered step")
        omega = unsigned(j["omega"], 35)
        error = signed(j["error"], self.fmt.error_bits + 3)
        limits = unsigned(self.rate_limit_events + (j["proposed"] != omega), 64)
        count = unsigned(self.count + 1, 64)
        self.count, self.rate_limit_events = count, limits
        self.previous_q32, self.next_q32, self.omega_q32 = previous, position, omega
        self.error_word = error
        return self.snapshot()

    def snapshot(self):
        return dict(
            count=self.count, previous_q32=self.previous_q32, next_q32=self.next_q32,
            omega_q32=self.omega_q32, applied_error_word=self.error_word,
            rate_limit_events=self.rate_limit_events,
        )

    def persistent_bits(self):
        return 2 * 96 + 35 + self.fmt.error_bits + 3 + 2 * 64 + 7


def split_timing_states(power, start_tick, hold_symbols, fmt=None, config=None):
    """Validate one canonical seed,then retain only each owner's actual words.

    The temporary host seed is discarded;no duplicate detector power or NCO is
    silently retained. Each owner has its own count,so future queries cannot
    advance the detector's observed input count.
    """
    seed = WordTimingState(power, start_tick, hold_symbols, fmt, config)
    return _DetectorState(seed), _PIState(seed)


def _rebase_register_upper(result, state):
    f = state.fmt
    old = 192 + 52 + f.power_bits + 10 + 35 + f.error_bits + 3 + 6 * 64
    result["state_register_bits_upper"] += state.persistent_bits() - old
    result.update(
        actual_persistent_bits=state.persistent_bits(),
        estimate_basis="replace original persistent owner;keep all inherited temporary allocations",
        removed_unused_temporary_register_savings_credited=False,
        complete_logic_LUTs_or_timing=None, complete_latency_cycles=None,
        shared_bank_credit=False, packet_FIFO_context_and_arbitration_bits=None,
        full_receiver16MSps=False, synthesized=False,
    )
    return result


class TimingDetectorService(ParallelTEDTimingWordService):
    """Existing power/TED/divider,then one registered detector-only commit.

    No PI/NCO update is executed here. The output index refers to an observed
    symbol,not an ahead-of-input query index. The result is held until taken.
    """

    def _valid_state(self, state):
        return type(state) is _DetectorState

    def _power(self):
        super()._power()
        if self.mode == "omega":  # Known-sync power still takes all six actual edges.
            self.mode = "detector_commit"

    def _after_error(self):
        self.mode = "detector_commit"

    def step(self):
        if self.mode != "detector_commit":
            return super().step()
        cycle = unsigned(self.cycle + 1, 64)
        completed = unsigned(self.stats["completed"] + 1, 64)
        result = self.state.commit(self.job)  # All state bounds preflight before any write.
        self.result, self.job, self.mode = result, None, "ready"
        self.ready_cycle = self.cycle = cycle
        self.stats["completed"] = completed
        self.taken_this_cycle = self.issued_this_cycle = False

    def estimate(self):
        result = _rebase_register_upper(super().estimate(), self.state)
        result.update(
            role="power/TED/error only", NCO_bits=0, PI_add_bits=0,
            multiplier_bits=dict(power="2x22by22=4tiles", TED="2x25by22=4tiles"),
            detector_commit_edges=1, query_or_PI_count_mutation=False,
        )
        return result


class TimingPIService(TimingWordService):
    """One supplied finite clipped error→actual KI/KP DSP work→registered NCO.

    This port does not pick an error age or fabricate a detector result. A
    future bounded packet scheduler must deliver the correct tagged aged error.
    """

    def __init__(self, state):
        super().__init__(state)  # No unused packet-time divider choice on this port.

    def _valid_state(self, state):
        return type(state) is _PIState

    def issue(self, error):
        if type(error) is not int or abs(error) > 2 << self.state.fmt.error_bits:
            raise ValueError("Need an actual finite clipped error word")
        if self.state.count < self.state.hold_symbols and error:
            raise ValueError("Known sync cannot apply a timing error")
        if self.mode in ("aborted", "released"):
            raise RuntimeError("Closed packet cannot resume")
        if not self.input_ready or self.issued_this_cycle:
            self._count("backpressure")
            return False
        self._count("accepted")
        self.first_cycle, self.ready_cycle = self.cycle, None
        self.issued_this_cycle = True
        self.job = dict(error=error, ki=0, kp=0)
        self._after_error()
        return True

    def _tail(self):
        if self.age != 3:
            return super()._tail()
        completed = unsigned(self.stats["completed"] + 1, 64)
        result = self.state.commit(self.job)
        self.result, self.job, self.mode = result, None, "ready"
        self.ready_cycle = self.cycle + 1
        self.stats["completed"] = completed

    def estimate(self):
        result = _rebase_register_upper(super().estimate(), self.state)
        result.update(
            role="PI/NCO only", divider=None, NCO_bits=96,
            multiplier_bits=dict(PI="positive43or35bySigned27or35=4tiles"),
            detector_power_or_count_mutation=False,
        )
        return result


class SplitTimingQueryService(TimingQueryService):
    """Original three registered query edges,explicit independent NCO owner only."""

    def _valid_state(self, state):
        return type(state) is _PIState
