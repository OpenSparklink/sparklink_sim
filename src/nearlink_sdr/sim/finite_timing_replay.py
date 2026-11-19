"""Actual bounded source/cache/kernel→timing/query calendars→strict frame port.

One owner and one commit per symbol. Remaining carrier/normalization/header/
decoder arithmetic still stops at explicit off-clock barriers,not zero cycles.
"""

from copy import deepcopy

from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay
from nearlink_sdr.sim.timing_query_flow import TimingQueryService
from nearlink_sdr.sim.timing_word_flow import TimingWordService

__all__ = ["FiniteTimingFrameReplay"]


class FiniteTimingFrameReplay(SelectedFrameReplay):
    """Explicit experiment,not a default receiver or complete continuous ADC schedule."""

    def __init__(self, profile, prepared, *, word_timing_format, service_observer=None, **options):
        if not isinstance(word_timing_format, TimingWordFormat):
            raise ValueError("Need an explicit compiled finite timing format")
        if service_observer is not None and not callable(service_observer):
            raise ValueError("Need an optional host-only service observer")
        self.service_observer = service_observer
        self.timing_service = self.query_service = None
        super().__init__(profile, prepared, word_timing_format=word_timing_format, **options)
        self.timing_service = self._make_timing_service(self.consumer.claim_timing_state())
        self.query_service = TimingQueryService()
        self.service_epoch = self.cycle
        self.stats.update(dict.fromkeys([
            "timing_requests", "timing_completions", "query_requests", "query_replies",
            "normalization_operations", "timing_edges", "query_edges",
            "blocked_normalization_edges", "stale_query_rejections",
        ], 0))
        self._plan()

    def _make_timing_service(self, state):
        return TimingWordService(state)

    def _plan(self):
        if self.timing_service is None:
            self.mode = "unclaimed_services"  # Constructor has not transferred the timing owner.
            return
        boundary = self.consumer.boundary_kind
        if boundary is not None:
            super()._barrier(boundary)
            return
        if not self.query_service.issue(self.consumer.word_timing):
            raise RuntimeError("Query credit not retired before next timing state")
        self._count("query_requests")
        self.mode = "nco_query"

    def _barrier(self, kind):
        if kind == "symbol":
            self.mode = "timing_issue"
        else:
            super()._barrier(kind)

    def _observe_word(self, role, word):
        if self.observer is not None:
            query = next(q for q in self.queries if q["role"] == role)
            self.observer(dict(
                symbol=self.consumer._active["count"], role=role,
                cell=query["cell"], phase=query["phase"], word=word,
                transport_edge=self.cycle, clock_evidence_valid=self.cycle_evidence_valid,
            ))

    def _observe_service(self, kind, service, values):
        if self.service_observer is not None:
            self.service_observer(dict(
                kind=kind, issued_edge=self.service_epoch + service.first_cycle,
                ready_edge=self.service_epoch + service.ready_cycle,
                taken_edge=self.cycle, latency=service.ready_cycle - service.first_cycle,
                global_epoch_cycle_evidence_valid=self.cycle_evidence_valid,
                values=deepcopy(values),  # Host-only copy;callback cannot steer the live query.
            ))

    def step(self):
        if self.done:
            return
        local = self.cycle - self.service_epoch
        if local != self.timing_service.cycle or local != self.query_service.cycle:
            raise RuntimeError("Finite services must advance exactly once per transport edge")
        handled = True
        if self.mode == "nco_query":
            self._count("query_edges")
            reply = self.query_service.take()
            if reply is not None:
                if reply["count"] != self.consumer.word_timing.count:
                    self._count("stale_query_rejections")
                    raise RuntimeError("Query describes an expired timing state")
                self._count("query_replies")
                self._observe_service("query", self.query_service, reply)
                queries = [dict(role=role, **q) for role, q in enumerate(reply["queries"])
                           if q is not None]
                self._retire(reply["retain_before"])
                self._plan_samples(queries, reply["future_end"])
        elif self.mode == "timing_issue":
            self.consumer.validate_symbol_words(
                *self.words, after_samples=self.history.end, future_end=self.future_end
            )
            if not self.timing_service.issue(*self.words):
                raise RuntimeError("Timing credit not retired before next symbol")
            self._count("timing_requests")
            self.mode = "timing_run"
        elif self.mode == "timing_run":
            self._count("timing_edges")
            reply = self.timing_service.take()
            if reply is not None:
                if reply["count"] != self.consumer._active["count"] + 1:
                    raise RuntimeError("Timing service skipped or repeated a symbol")
                self._count("timing_completions")
                self._observe_service("timing", self.timing_service, reply)
                super()._barrier("symbol_normalization")
        else:
            if self.mode == "cache":
                q = self.queries[0]
                if (
                    q["cell"] != self.cache.cache.first_cell + 4 * self.consumer._active["count"]
                    or q["phase"] != self.cache.cache.phase
                ):
                    raise RuntimeError("Registered NCO query disagrees with sealed sync cache")
            if self.pending_service is not None and (
                self.pending_service["kind"] == "symbol_normalization"
            ):
                self._count("blocked_normalization_edges")
            super().step()
            handled = False
        if handled:
            self.kernel.advance()
        self.timing_service.step()
        self.query_service.step()

    def resolve_functional(self):
        if self.done or self.pending_service is None:
            raise RuntimeError("No unresolved normalization/header/payload service")
        if self.pending_service["kind"] != "symbol_normalization":
            return super().resolve_functional()
        # This is the old carrier rotation/gain/storage only;the actual service
        # has already committed the finite timing state. Never advance it here.
        self.consumer.accept_timing_result(self.words[0], after_samples=self.history.end)
        self.cycle_evidence_valid = False
        self._count("functional_resolutions")
        self._count("normalization_operations")
        self._count("symbol_operations")
        self.pending_service = None
        self._plan()
        return None

    def _close(self, event):
        timing, query = self.timing_service, self.query_service
        cancelled = dict(timing=timing.mode != "idle", query=query.mode != "idle")
        for service in (timing, query):
            if service.mode == "idle":
                service.release()
            else:
                service.abort()
        super()._close(event)
        event["finite_timing_services"] = dict(
            actual_timing_and_query_values_executed=True,
            timing_statistics=timing.stats.copy(), query_statistics=query.stats.copy(),
            timing_state_commits=self.consumer.word_timing.count,
            stored_symbols=len(event.get("symbols", [])),
            cancelled_inflight=cancelled, controller_statistics=self.stats.copy(),
            service_epoch=self.service_epoch, last_transport_edge=self.cycle,
            remaining_carrier_normalization_header_decoder_cycles=None,
            complete_latency_cycles=None, whole_receiver16MSps=False,
        )

    def estimate(self):
        result = super().estimate()
        # Nine added64bit counters are already included by the shared estimate.
        result["new_controller_state_bits"] += 64 + 4
        result.update(
            controller_increment_over_previous_bits=9 * 64 + 64 + 4,
            timing_service=self.timing_service.estimate(),
            query_service=self.query_service.estimate(),
            service_epoch_and_control_bits=68,
            extra_query_source_ports=0,
            cache_request_identity_control=dict(
                compare64=1, compare17=1, shift_by2=1, add64=1,
                assumption="same registered cache-request edge;combinational timing unverified",
            ),
            timing_state_owners=1,
            remaining_float_work_has_explicit_barrier=True,
            combined_service_and_interpolator_LUT_or_timing=None,
            service_observer_is_host_only_not_trace_RAM=True,
        )
        return result
