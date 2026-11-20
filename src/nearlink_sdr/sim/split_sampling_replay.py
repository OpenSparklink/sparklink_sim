"""Tagged split timing→actual source/interpolation→real word-head (§6.2.1.2/6.3/6.9).

One selected answer and bounded queues. Dedicated detector/PI/normalization DSP
banks;no unproved resource sharing. Payload carrier/LLR/decoder remain explicit
off-clock work,so this is not a complete continuous16MS/s receiver.
"""

from copy import deepcopy
from math import ceil, log2

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.phy.timing import PSKTimingConfig
from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.phy.word_head_rx import WordHeadPSKReceiver
from nearlink_sdr.sim.normalization_pipeline import PipelinedNormalizationService
from nearlink_sdr.sim.psk_hard_flow import HeaderFromWordsService
from nearlink_sdr.sim.retained_stencil_replay import _overlap_words
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay
from nearlink_sdr.sim.split_timing_flow import (
    SplitTimingQueryService,
    TimingDetectorService,
    TimingPIService,
    split_timing_states,
)

__all__ = ["SplitSamplingFrameReplay"]


class _TimingReceipt:
    """Read-only compatibility view of one joined pair,never the live ahead NCO.

    The pair's words remain in one charged receipt slot after queue removal.
    Shared display helpers do not execute detector,PI or query arithmetic.
    """

    snapshot = WordTimingState.snapshot
    diagnostics = WordTimingState.diagnostics
    validate_words = WordTimingState.validate_words
    queries = WordTimingState.queries
    required_end = WordTimingState.required_end
    omega = WordTimingState.omega
    power = WordTimingState.power

    def __init__(self, detector, pi, fmt, hold):
        if detector["count"] != pi["count"]:
            raise RuntimeError("A receipt cannot join different symbol indices")
        self.detector, self.pi, self.fmt, self.hold_symbols = detector, pi, fmt, hold
        self.config, self.sps = PSKTimingConfig(), 4
        self.damping, self.detector_gain = 2**-.5, 1.2
        self.kp, self.ki = 7476035428952 / 2**48, 29904141716 / 2**48

    def __getattr__(self, key):
        if key == "power_word":
            return self.detector["power"]
        if key in self.detector:
            return self.detector[key]
        if key in self.pi:
            return self.pi[key]
        raise AttributeError(key)


class _SplitReceiver(WordHeadPSKReceiver):
    def _make_timing_state(self, *args):
        self.detector, self.pi = split_timing_states(*args)
        return self.receipt(self.detector.snapshot(), self.pi.snapshot())

    def receipt(self, detector, pi):
        return _TimingReceipt(detector, pi, self.detector.fmt, self.detector.hold_symbols)


class SplitSamplingFrameReplay(SelectedFrameReplay):
    """One query/transport credit,finite tagged joins and a real strict head gate.

    NCO query n is captured before PI update n. The latter consumes only the
    registered error n-age (zero during startup),not a future detector result.
    Query/sample frontier and delivered-symbol frontier are separate counters.
    """

    _retain_completed_stencil = True

    def __init__(self, profile, prepared, *, error_age, normalization_format,
                 word_timing_format, head_layout="dense", service_observer=None, **options):
        if type(error_age) is not int or error_age not in (0, 1, 2, 3):
            raise ValueError("Need compiled error age0,1,2,3")
        if word_timing_format != TimingWordFormat(30, 32):
            raise ValueError("Need explicit compiled timing30/32")
        if not isinstance(normalization_format, NormalizationFormat):
            raise ValueError("Need explicit normalization format")
        if profile.config.sps != 4 or profile.config.symbol_rate_mhz != 4:
            raise ValueError("Split sampling is compiled for4Msym/s SPS4")
        if options.get("reuse_stencil") is not True:
            raise ValueError("Need explicit ordered stencil reuse")
        if service_observer is not None and not callable(service_observer):
            raise ValueError("Need a read-only optional service observer")
        self.error_age, self.capacity = error_age, error_age + 2
        self.normalization_format, self.service_observer = normalization_format, service_observer
        self.head_owner = self._make_head_owner(profile, head_layout)
        self.head_owner.begin()
        self.detector_service = None
        super().__init__(profile, prepared, word_timing_format=word_timing_format, **options)
        self.consumer.claim_timing_state()  # One ownership transfer;receipt cannot run a service.
        self.detector_service = TimingDetectorService(
            self.consumer.detector, divider="radix4-prefix"
        )
        self.pi_service = TimingPIService(self.consumer.pi)
        self.query_service = SplitTimingQueryService()
        self.normalization_service = PipelinedNormalizationService(
            self.consumer.claim_normalization_state()
        )
        self.service_epoch = self.cycle
        self.sample_index = self.query_count = 0
        self.detector_messages, self.pi_messages = {}, {}
        self.error_messages, self.sample_messages = {}, {}
        self.peak_messages = [0] * 4
        self.transport_eof = False
        self.current_position = 0
        self.stats.update(dict.fromkeys([
            "query_requests", "query_replies", "detector_requests", "detector_completions",
            "PI_requests", "PI_completions", "joined_symbols", "header_completions",
            "cross_symbol_words_reused", "admission_stall_edges", "max_PI_lead",
        ], 0))
        self.mode = "waiting"
        self._plan()

    def _make_head_owner(self, profile, layout):
        return HeaderFromWordsService(profile, layout=layout)

    def _make_consumer(self, profile, initial_fit, word_timing_format, selection):
        return _SplitReceiver(
            profile, initial_fit=initial_fit, word_timing_format=word_timing_format,
            normalization_format=self.normalization_format, head_owner=self.head_owner, **selection
        )

    def _query_limit(self):
        if not self.consumer.head_receipt_taken:
            return self.consumer._head_required
        return self.consumer.frame_end_symbol

    def _plan(self):
        if self.detector_service is None:
            self.mode = "unclaimed"
            return
        if self.mode != "waiting" or self.transport_eof:
            return
        limit = self._query_limit()
        if (limit is not None and self.sample_index < limit
                and len(self.sample_messages) < self.capacity
                and self.query_service.input_ready and self.consumer.pi.count == self.sample_index):
            position = self.consumer.pi.next_q32
            count = unsigned(self.query_count + 1, 64)
            unsigned(self.stats["query_requests"] + 1, 64)
            if not self.query_service.issue(self.consumer.pi):
                raise RuntimeError("Preflighted query lost its one credit")
            self.current_position = position
            self.query_count = count
            self._count("query_requests")
            self.mode = "nco_query"

    def _preflight_samples(self, queries, future_end):
        unsigned(future_end, 64)
        for query in queries:
            unsigned(query["cell"] - 1, 64)
            unsigned(query["cell"] + 3, 64)
            unsigned(query["phase"], 17)
        count = 0
        if self.sample_index >= len(self.consumer.reference) and len(self.staged) == 4:
            count = _overlap_words(self.stage_start, queries[-1]["cell"] - 1)
        unsigned(self.stats["cross_symbol_words_reused"] + count, 64)
        return count

    def _plan_samples(self, queries, future_end, *, compaction):
        previous = self.staged
        self.future_end, self.words, self.queries = future_end, [None, None], queries
        if self.sample_index < len(self.consumer.reference):
            self.mode, self.cache_requested = "cache", False
            return
        self.queries.reverse()
        self.query_index = 0
        self.stage_start = self.queries[0]["cell"] - 1
        self.staged = previous[4 - compaction:] if compaction else []
        self._count("cross_symbol_words_reused", compaction)
        self.mode = "queries"

    def _observe_word(self, role, word):
        if self.observer is not None:
            q = next(q for q in self.queries if q["role"] == role)
            self.observer(dict(symbol=self.sample_index, role=role, cell=q["cell"],
                               phase=q["phase"], word=word, transport_edge=self.cycle,
                               clock_evidence_valid=self.cycle_evidence_valid))

    def _barrier(self, kind):
        if kind == "symbol":
            self.mode = "admit"
        else:
            super()._barrier(kind)

    def _observe_service(self, kind, value):
        if self.service_observer is not None:
            self.service_observer(dict(kind=kind, cycle=self.cycle, values=deepcopy(value)))

    def _take_results(self):
        detector, pi = self.detector_service, self.pi_service
        if detector.ready and len(self.detector_messages) < self.capacity:
            row, index = detector.result, detector.result["count"] - 1
            if (index < self.consumer.detector.hold_symbols
                    or len(self.error_messages) < self.capacity):
                if index in self.detector_messages or index in self.error_messages:
                    raise RuntimeError("Detector result repeated a tagged index")
                due = unsigned(self.cycle + 1, 64)
                unsigned(self.stats["detector_completions"] + 1, 64)
                self.detector_messages[index] = detector.take()
                if index >= self.consumer.detector.hold_symbols:
                    self.error_messages[index] = due, row["error_word"]
                self._count("detector_completions")
                self._observe_service("detector", row)
        if pi.ready and len(self.pi_messages) < self.capacity:
            row = pi.result
            index = row["count"] - 1
            if index in self.pi_messages:
                raise RuntimeError("PI receipt repeated a tagged index")
            unsigned(self.stats["PI_completions"] + 1, 64)
            self.pi_messages[index] = pi.take()
            self._count("PI_completions")
            self._observe_service("PI", row)

    def _join(self):
        n, normal = self.consumer._active["count"], self.normalization_service
        value = normal.peek()
        if n not in self.detector_messages or n not in self.pi_messages or value is None:
            return
        d, p = self.detector_messages[n], self.pi_messages[n]
        position, future = self.sample_messages[n]
        if (d["count"] != n + 1 or p["count"] != n + 1 or value["index"] != n
                or value["raw_word"] != d["previous_word"] or p["previous_q32"] != position
                or self.history.end < future or not normal.validate_take()):
            raise RuntimeError("Tagged join changed index,source word,center or actual future")
        receipt = self.consumer.receipt(d, p)
        unsigned(self.stats["joined_symbols"] + 1, 64)
        unsigned(self.stats["symbol_operations"] + 1, 64)
        result = normal.take()
        self.consumer.word_timing = self.consumer._active["loop"] = receipt
        self.consumer.accept_normalized_timing_result(result, after_samples=self.history.end)
        del self.detector_messages[n], self.pi_messages[n], self.sample_messages[n]
        self._count("joined_symbols")
        self._count("symbol_operations")
        self._observe_service("joined", dict(detector=d, PI=p, normalization=result))

    def _admit(self):
        d, normal, n = self.detector_service, self.normalization_service, self.sample_index
        if not d.input_ready or not normal.can_issue or len(self.sample_messages) >= self.capacity:
            self._count("admission_stall_edges")
            return
        if d.state.count != n or self.history.end < self.future_end:
            raise RuntimeError("Actual symbol input skipped a detector or its source future")
        d.state.validate_words(*self.words)
        for key in ("detector_requests",):
            unsigned(self.stats[key] + 1, 64)
        unsigned(normal.stats["accepted"] + 1, 64)
        unsigned(d.stats["accepted"] + 1, 64)
        if not d.issue(*self.words) or not normal.issue(self.words[0]):
            raise RuntimeError("Preflighted detector/normalization pair changed credit")
        self.sample_messages[n] = self.current_position, self.future_end
        self.sample_index = unsigned(n + 1, 64)
        self._count("detector_requests")
        self.mode = "waiting"

    def _issue_pi(self):
        n, hold = self.consumer.pi.count, self.consumer.pi.hold_symbols
        if (not self.pi_service.input_ready or n >= self.query_count
                or n > self.consumer.detector.count + self.error_age):
            return
        source = n - self.error_age
        if n < hold + self.error_age:
            error = 0
        elif source in self.error_messages and self.error_messages[source][0] <= self.cycle:
            error = self.error_messages[source][1]
        else:
            return
        unsigned(self.stats["PI_requests"] + 1, 64)
        if not self.pi_service.issue(error):
            raise RuntimeError("Preflighted PI credit changed")
        if n >= hold + self.error_age:
            del self.error_messages[source]
        self._count("PI_requests")

    def step(self):
        if self.done:
            return
        services = (self.detector_service, self.pi_service, self.query_service,
                    self.normalization_service, self.head_owner)
        if any(s.cycle != self.cycle - self.service_epoch for s in services):
            raise RuntimeError("Every actual split service must share the transport edge")
        self._take_results()
        self._join()
        boundary = self.consumer.boundary_kind
        if boundary == "header" and self.head_owner.ready:
            unsigned(self.stats["header_completions"] + 1, 64)
            unsigned(self.stats["header_operations"] + 1, 64)
            event = self.consumer.accept_header_result(after_samples=self.history.end)
            self._count("header_completions")
            self._count("header_operations")
            if event is not None:
                self._close(event)
                return
        elif boundary == "payload" and self.pending_service is None:
            self._barrier("payload")
        if self.mode == "nco_query" and self.query_service.ready:
            reply = self.query_service.result
            if reply["count"] != self.sample_index:
                raise RuntimeError("Query belongs to a different input frontier")
            queries = [dict(role=role, **q) for role, q in enumerate(reply["queries"])
                       if q is not None]
            compaction = self._preflight_samples(queries, reply["future_end"])
            unsigned(self.stats["query_replies"] + 1, 64)
            self.query_service.take()
            self._count("query_replies")
            self._retire(reply["retain_before"])
            self._plan_samples(queries, reply["future_end"], compaction=compaction)
        elif self.mode == "cache":
            n = self.sample_index
            if not self.cache_requested:
                q = self.queries[0]
                self.cache_requested = self.cache.request(q["cell"], q["phase"], self.cycle)
            reply = self.cache.take(self.cycle)
            if reply is not None:
                if reply[0] != n:
                    raise RuntimeError("Known-sync cache changed ordered sample index")
                self.words[0] = reply[1]
                self._count("cache_reads")
                self._observe_word(0, reply[1])
                self.mode = "admit"
        elif self.mode == "queries" and not self.transport_eof:
            self._step_queries()
        elif self.mode == "admit":
            self._admit()
        elif self.mode == "service":
            self._count("blocked_service_edges")
        if (self.transport_eof and not self.sample_messages
                and self.consumer.boundary_kind != "header"):
            super()._finish_eof()
            return
        self._plan()
        self._issue_pi()
        for i, q in enumerate((self.detector_messages, self.pi_messages,
                               self.error_messages, self.sample_messages)):
            self.peak_messages[i] = max(self.peak_messages[i], len(q))
            if len(q) > self.capacity:
                raise RuntimeError("A finite tagged queue exceeded its compiled capacity")
        lead = self.consumer.pi.count - self.consumer.detector.count
        self.stats["max_PI_lead"] = max(self.stats["max_PI_lead"], lead)
        if lead > self.error_age + 1:
            raise RuntimeError("PI exceeded the bounded age-dependent detector lead")
        self.kernel.advance()
        for service in services:
            service.step()

    def end_input(self):
        needs_future = self.wants_replay_input
        self.eof = True  # Drain already admitted symbols and the actual head before reporting EOF.
        if needs_future:
            self.transport_eof = True

    def _finish_eof(self):
        self.transport_eof = True

    def _close(self, event):
        detail = dict(
            detector=self.consumer.detector.snapshot(), PI=self.consumer.pi.snapshot(),
            peak_queues=self.peak_messages.copy(), capacity_each=self.capacity,
            discarded_error_indices=list(self.error_messages),
            discarded_PI_indices=list(self.pi_messages),
            query_frontier=self.query_count, sample_frontier=self.sample_index,
            delivered_frontier=self.consumer.word_timing.count,
            max_PI_lead=self.stats["max_PI_lead"],
            actual_head=self.head_owner.snapshot(), error_age=self.error_age,
            head_queries_before_valid_CRC_bounded_by_actual_header=True,
            complete_latency_cycles=None, whole_receiver16MSps=False,
        )
        for service in (self.detector_service, self.pi_service, self.query_service,
                        self.normalization_service, self.head_owner):
            service.abort()
        for q in (self.detector_messages, self.pi_messages, self.error_messages,
                  self.sample_messages):
            q.clear()
        super()._close(event)
        event["split_sampling"] = detail

    def estimate(self):
        result = super().estimate()
        d, p = self.consumer.detector, self.consumer.pi
        # Conservative registers:no RAM sharing,implicit dictionary or observer storage credit.
        slots = self.capacity * (d.persistent_bits() + p.persistent_bits()
                                 + 2 * 64 + 35 + 64 + 96 + 64 + 4)
        pointers = 4 * (2 * ceil(log2(self.capacity)) + ceil(log2(self.capacity + 1)))
        control = (slots + d.persistent_bits() + p.persistent_bits()
                   + 3 * 64 + 4 * 64 + 16 + 96 + pointers)
        result["new_controller_state_bits"] += control
        return result | dict(
            split_detector=self.detector_service.estimate(), PI=self.pi_service.estimate(),
            query=self.query_service.estimate(),
            normalization=self.normalization_service.estimate(self.normalization_format.mode),
            head=self.head_owner.cost(), tagged_queue_and_receipt_bits_upper=control,
            dedicated_DSP_banks=dict(interpolation=6, detector=4, PI=4, normalization=4),
            DSP_shared=False, per_kind_queue_capacity=self.capacity,
            queue_register_ports="one push/one pop per kind;no inferred shared RAM credit",
            queue_pointer_count_bits_upper=pointers,
            queue_index_match_comparisons64_upper=4 * self.capacity,
            queue_RAM_inference_or_physical_port_proof=False,
            old4096_timing_node_budget_passed=False, old2048_controller_budget_passed=False,
            payload_cycles=None, complete_latency_cycles=None, whole_receiver16MSps=False,
            synthesis=False,
        )
