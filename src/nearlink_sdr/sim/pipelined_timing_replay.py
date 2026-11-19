"""Exact known-sync lookahead→existing strict frame and ordinary timing service.

Only fixed known-sync queries are preplanned;actual data-symbol queries and
Gardner/PI remain unchanged. Unknown normalization/header/decoder still block.
"""

from copy import deepcopy

from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.finite_timing_replay import FiniteTimingFrameReplay
from nearlink_sdr.sim.hold_timing_flow import PipelinedPacketTimingService

__all__ = ["PipelinedTimingFrameReplay"]


class PipelinedTimingFrameReplay(FiniteTimingFrameReplay):
    """Opt-in bounded hold prefetch;no new search,phase grid or answer retry."""

    def __init__(self, profile, prepared, *, word_timing_format, **options):
        if not isinstance(
            word_timing_format, TimingWordFormat
        ) or word_timing_format != TimingWordFormat(30, 32):
            raise ValueError("Need explicit compiled30/32 before claiming prepared ownership")
        super().__init__(profile, prepared, word_timing_format=word_timing_format, **options)

    def _make_timing_service(self, state):
        return PipelinedPacketTimingService(state, self.cache, self.cycle)

    def _plan(self):
        if self.timing_service is None:
            return super()._plan()
        if self.consumer.word_timing.count < self.consumer.word_timing.hold_symbols:
            self.mode = "hold_pipeline"
        else:
            super()._plan()

    def step(self):
        if self.done:
            return
        if self.mode != "hold_pipeline":
            return super().step()
        local = self.cycle - self.service_epoch
        if self.timing_service.cycle != local or self.query_service.cycle != local:
            raise RuntimeError("Shared hold/query clocks lost their single edge")
        hold = self.timing_service.hold
        item = hold.peek()
        self._count("timing_edges")
        if item is not None:
            n = item["index"]
            cache = self.cache.cache
            cell, phase = cache.first_cell + 4 * n, cache.phase
            expected = self.consumer.word_timing.next_q32
            if expected != (cell << 32) + (phase << 16):
                raise RuntimeError("Fixed known-sync query changed NCO identity")
            self.words = [item["word"], None]
            self.queries = [dict(role=0, cell=cell, phase=phase)]
            self.future_end = cell + 3 + int(phase > 0)
            self.consumer.validate_symbol_words(
                *self.words, after_samples=self.history.end, future_end=self.future_end
            )
            reply = self.timing_service.take()
            if reply is None or reply["count"] != n + 1:
                raise RuntimeError("One hold acceptance must commit one actual timing result")
            self._count("timing_requests")
            self._count("timing_completions")
            self._count("cache_reads")
            self._observe_word(0, item["word"])
            self._observe_service("timing", self.timing_service, reply)
            # Reuse the original unknown normalization barrier and later real CRC/MAC gates.
            self._barrier("symbol_normalization")
        self.kernel.advance()
        self.timing_service.step()
        self.query_service.step()

    def _close(self, event):
        hold = self.timing_service.hold
        detail = dict(
            statistics=hold.stats.copy(),
            requested=hold.requested,
            computed=hold.computed,
            accepted=hold.stats["accepted"],
            queued=len(hold.outputs),
            energy_inflight=hold.energy is not None,
            power_inflight=hold.power_job is not None,
            old4096_node_state_budget_passed=self.timing_service.estimate()[
                "old4096_state_budget_passed"
            ],
            pending_contexts=deepcopy(
                dict(
                    outputs=list(hold.outputs),
                    energy=hold.energy,
                    power=hold.power_job,
                    cache_reply=hold.cache.pending,
                )
            ),
            pending_context_snapshot_is_host_only=True,
        )
        super()._close(event)
        event["known_sync_pipeline"] = detail

    def estimate(self):
        result = super().estimate()
        return result | dict(
            known_sync_cache_lookahead=True,
            query_phase_or_grid_changed=False,
            known_sync_query_generation="fixed exact progression,one cache pass",
            extra_hold_controller_bits=0,
            hold_control_basis="one extra mode code in existing4bit field;"
            "reuse existing received-word/query descriptor registers",
            extra_identity_compare_bits=96,
            unknown_services_still_explicit=True,
        )
