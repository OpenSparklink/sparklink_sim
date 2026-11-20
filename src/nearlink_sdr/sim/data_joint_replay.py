"""Data timing/normalization on existing independent banks; one ordered join.

Known-sync and real head services are inherited unchanged. No speculative next
timing query, additional source history, or whole continuous-RX claim.
"""

from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay

__all__ = ["DataJointFrameReplay"]


class DataJointFrameReplay(HeadWordsFrameReplay):
    """Same current52bit input, one pair credit and existing held result registers."""

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.stats.update(dict.fromkeys(["data_joint_requests", "data_joint_completions"], 0))

    def _issue_pair(self):
        timing, normalization = self.timing_service, self.normalization_service
        self.consumer.validate_symbol_words(
            *self.words, after_samples=self.history.end, future_end=self.future_end
        )
        if (not normalization.serial_mode
                or normalization.state.count != self.consumer._active["count"]):
            raise RuntimeError("Data normalization must drain known-sync and match current index")
        if timing.normal is None:
            available = timing.hold.drained and not timing.hold.taken_this_cycle
        else:
            available = timing.normal.input_ready and not timing.normal.issued_this_cycle
        if not available or not normalization.can_issue:
            raise RuntimeError("Both data service credits must pass preflight before either issue")
        # Counts and words are checked before either bank takes ownership. No
        # ready mask or queue silently drops the other half of an input pair.
        unsigned(normalization.stats["accepted"] + 1, 64)
        if timing.normal is not None:
            unsigned(timing.normal.stats["accepted"] + 1, 64)
        for key in ("timing_requests", "normalization_requests", "data_joint_requests"):
            unsigned(self.stats[key] + 1, 64)
        if not timing.issue(*self.words) or not normalization.issue(self.words[0]):
            raise RuntimeError("A preflighted paired service admission changed at the same edge")
        for key in ("timing_requests", "normalization_requests", "data_joint_requests"):
            self._count(key)
        self.mode = "timing_run"  # Reuse the existing code;no new FSM state or next-query flag.

    def _take_pair(self):
        timing, normalization = self.timing_service, self.normalization_service
        normal = timing.normal
        if normal is None:
            raise RuntimeError("Data pair has no actual timing owner")
        result = normalization.peek()  # Host observation aliases an existing held register.
        if not normal.ready or result is None:
            return
        n = self.consumer._active["count"]
        if (normal.result["count"] != n + 1 or result["index"] != n
                or result["count"] != n + 1 or result["raw_word"] != self.words[0]
                or timing.state.previous_word != self.words[0]):
            raise RuntimeError("Timing/normalization pair changed absolute index or source word")
        if not normalization.validate_take():
            raise RuntimeError("Normalization owner must preflight its held output")
        counters = ("timing_completions", "normalization_completions", "normalization_operations",
                    "symbol_operations", "data_joint_completions")
        for key in counters:
            unsigned(self.stats[key] + 1, 64)
        reply, result = timing.take(), normalization.take()
        self.consumer.accept_normalized_timing_result(result, after_samples=self.history.end)
        for key in counters:
            self._count(key)
        self._observe_service("timing", timing, reply)
        self._observe_service("normalization", normalization, result)
        self._plan()  # Both owners have transferred before any next timing query or head gate.

    def step(self):
        if self.done:
            return
        if self.mode not in ("timing_issue", "timing_run"):
            return super().step()
        local = self.cycle - self.service_epoch
        services = (self.timing_service, self.query_service,
                    self.normalization_service, self.head_owner)
        if any(service.cycle != local for service in services):
            raise RuntimeError("Data pair and head must share the actual transport edge")
        if self.mode == "timing_issue":
            self._issue_pair()
        else:
            self._count("timing_edges")
            self._count("normalization_edges")
            self._take_pair()
        self.kernel.advance()
        for service in services:
            service.step()

    def _close(self, event):
        pair = dict(
            active=self.mode in ("timing_issue", "timing_run"),
            stored_symbols=len(event.get("symbols", [])),
            timing_commits=self.consumer.word_timing.count,
            normalization_commits=self.consumer.normalization.count,
            next_query_waits_for_both=True,
        )
        super()._close(event)
        event["data_joint_service"] = pair

    def estimate(self):
        return super().estimate() | dict(
            data_joint_controller_increment_bits=128,
            data_joint_counters_already_in_controller=True,
            data_joint_reuses_timing_issue_and_run_modes=True,
            data_joint_pair_credits=1,
            data_joint_new_state_registers_except_counters=0,
            data_joint_additional_DSP=0,
            data_joint_additional_source_RAM_bits=0,
            data_joint_additional_source_or_cache_ports=0,
            data_joint_result_storage="existing timing and normalization held registers",
            data_joint_current_word_broadcast_bits=52,
            data_joint_guard_comparison_bits_upper=dict(
                source_equality=[52, 52], ordered_index=[64, 18, 18],
                inherited_source_future_bounds=True, counter_guards_are_existing64bit_checks=True,
            ),
            data_joint_guard_latency_assumption="same registered admission/join edge;"
                "wide equality/count/future guards not synthesized or timed",
            data_joint_DSP_bank_sharing_credited=False,
            data_joint_logic_or_Fmax=None,
            complete_latency_cycles=None,
            whole_receiver16MSps=False,
        )
