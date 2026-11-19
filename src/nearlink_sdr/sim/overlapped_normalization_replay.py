"""Registered cache broadcast and atomic known-sync timing/normalization join.

Normal data retains the original serial query/Gardner/PI/normalization schedule.
Only known-sync scheduling changes. Head/residual carrier/LLR/decoder remain
explicit functional barriers, never zero-cycle work or a complete16MS/s claim.
"""

from copy import deepcopy

from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.hold_timing_flow import PipelinedPacketTimingService
from nearlink_sdr.sim.normalization_pipeline import PipelinedNormalizationService
from nearlink_sdr.sim.normalized_word_replay import NormalizedWordFrameReplay

__all__ = ["OverlappedNormalizationFrameReplay"]


class OverlappedNormalizationFrameReplay(NormalizedWordFrameReplay):
    """Four joint credits,one cache read/word;no extra original-sample history."""

    def _make_timing_service(self, state):
        self.normalization_service = PipelinedNormalizationService(
            self.consumer.claim_normalization_state()
        )
        return PipelinedPacketTimingService(
            state, self.cache, self.cycle, word_sink=self.normalization_service
        )

    def _join(self, item, normalized):
        hold, service = self.timing_service.hold, self.normalization_service
        n = self.consumer._active["count"]
        if (item["index"] != n or normalized["index"] != n
                or item["word"] != normalized["raw_word"]
                or self.consumer.word_timing.count != n
                or self.consumer.normalization.count != n):
            raise RuntimeError("Known-sync join changed source word or ordered owner identity")
        cache = self.cache.cache
        cell, phase = cache.first_cell + 4 * n, cache.phase
        if self.consumer.word_timing.next_q32 != (cell << 32) + (phase << 16):
            raise RuntimeError("Known-sync join changed the fixed sample progression")
        future_end = cell + 3 + int(phase > 0)
        self.consumer.validate_symbol_words(
            item["word"], None, after_samples=self.history.end, future_end=future_end
        )
        if not hold.validate_take() or not service.validate_take():
            raise RuntimeError("Both registered results must pass atomic transfer preflight")
        counters = ["timing_requests", "timing_completions", "cache_reads",
                    "normalization_completions", "normalization_operations", "symbol_operations"]
        for key in counters:
            unsigned(self.stats[key] + 1, 64)
        self.words = [item["word"], None]
        self.queries = [dict(role=0, cell=cell, phase=phase)]
        self.future_end = future_end
        reply = self.timing_service.take()
        result = service.take()
        self._observe_word(0, item["word"])
        self.consumer.accept_normalized_timing_result(result, after_samples=self.history.end)
        for key in counters:
            self._count(key)
        if self.consumer.word_timing.count == self.consumer.word_timing.hold_symbols:
            service.enter_serial()
        # Host diagnostics copy already committed values;no second computation or live state.
        self._observe_service("timing", self.timing_service, reply)
        self._observe_service("normalization", service, result)
        self._plan()

    def step(self):
        if self.done:
            return
        if self.mode != "hold_pipeline":
            return super().step()
        service = self.normalization_service
        local = self.cycle - self.service_epoch
        if not all(s.cycle == local for s in
                   (service, self.timing_service, self.query_service)):
            raise RuntimeError("Broadcast/join services must use the same actual edge")
        self._count("timing_edges")
        self._count("normalization_edges")
        item, normalized = self.timing_service.hold.peek(), service.peek()
        if item is not None and normalized is not None:
            self._join(item, normalized)
        accepted = service.stats["accepted"]
        self.kernel.advance()
        self.timing_service.step()  # Exactly one registered cache reply fans out here.
        self.query_service.step()
        service.step()
        self._count("normalization_requests", service.stats["accepted"] - accepted)

    def _close(self, event):
        service = self.normalization_service
        detail = dict(owned=service.owned, peak_owned=service.peak_owned,
                      serial_data_mode=service.serial_mode,
                      pending_slots=deepcopy(service.slots),
                      private_phase=service.private_phase, private_index=service.private_index,
                      pending_snapshot_is_host_only=True)
        super()._close(event)
        detail["statistics"] = service.pipeline_stats.copy()
        event["normalization_pipeline"] = detail

    def estimate(self):
        return super().estimate() | dict(
            known_sync_registered_cache_broadcast=True,
            known_sync_atomic_ordered_timing_normalization_join=True,
            known_sync_joint_credit_limit=4, extra_original_sample_RAM_bits=0,
            extra_source_or_cache_read_ports=0,
            join_extra_state_bits=0,
            join_state_basis="reuse existing hold mode,received-word/query registers and counters;"
                "peek observations alias actual output registers,not hardware copies",
            join_combinational_comparisons="52bit source equality;18bit ordered counts;"
                "existing40bit normalization and96bit timing progression;64bit counter checks",
            data_timing_or_PI_schedule_changed=False,
            normalization_and_timing_DSP_shared=False,
            full_receiver_cycles=None,
        )
