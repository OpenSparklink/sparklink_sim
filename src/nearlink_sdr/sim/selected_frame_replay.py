"""Prepared source/cache/fit→real interpolants→strict selected frame (§6.3/6.9).

Known transport stages carry real values. Unmodeled arithmetic stops at a
service barrier. Explicit functional resolution supplies no cycle evidence.
No CRC answer retry,raw tail copy or automatic input throttling is hidden here.
"""

import math

import numpy as np

from nearlink_sdr.phy.interpolant_rx import InterpolantPSKReceiver
from nearlink_sdr.phy.sync_fit_word import TURN, compile_sync_phases
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService

__all__ = ["SelectedFrameReplay"]


class SelectedFrameReplay:
    """One source bank,one existing interpolator,one cache and one pending operation.

    `step` never resolves unknown PI/normalization/header/payload work. Calling
    `resolve_functional` is an explicit off-clock reference operation;all later
    absolute timing is unverified. A live source can instead continue while a
    barrier is blocked,with explicit bounded overflow. The two modes are not
    combined into a complete16MS/s claim.
    """

    _retain_completed_stencil = False  # Compiled subclass choice;not a packet-time fallback.

    def __init__(self, profile, prepared, *, reuse_stencil=False, observer=None,
                 word_timing_format=None, **selection):
        if type(reuse_stencil) is not bool:
            raise ValueError("Need explicit within-symbol stencil reuse choice")
        if observer is not None and not callable(observer):
            raise ValueError("Need an optional host-only read-only observer")
        if not isinstance(prepared, InitialSyncFitService) or not prepared.ready:
            raise ValueError("Need a completed unclaimed source/cache/fit service")
        stencil = prepared.stencil
        history, cache = stencil.history, stencil.output
        consumer = self._make_consumer(profile, prepared.fit.result, word_timing_format, selection)
        first = consumer.selected_start
        cell = math.floor(first)
        phase = round((first - cell) * 65536)
        if (
            history.capacity != 512
            or history.pending is not None
            or stencil.interpolator.live
            or cache.pending is not None
            or cache.cache.state != "sealed"
            or cache.cache.read != 0
            or cache.cache.first_cell != cell
            or cache.cache.phase != phase
            or prepared.fit._fit.codes != compile_sync_phases(consumer.reference)
            or prepared.fit._fit.coarse != round(consumer.selected_delta * TURN / math.tau)
            or history.end > consumer.sample_limit
        ):
            raise ValueError("Prepared geometry/profile/seed/ownership does not match selection")
        if history.head > cell + 4 * len(consumer.reference) - 3:
            raise ValueError("Last-sync raw words expired before first data midpoint")
        self.initial_fit = prepared.fit.take()  # Exactly one ownership transfer,after all guards.
        self.consumer = consumer
        self.consumer._observe(history.end)
        self.prepared = prepared
        self.history, self.cache = history, cache
        self.kernel = stencil.interpolator  # Borrow the same6DSP pipeline,not a second instance.
        self.reuse_stencil = reuse_stencil
        self.observer = observer
        self.cycle_evidence_valid = True
        self.eof = False
        self.mode = "idle"
        self.pending_service = self.event = None
        self._mac_taken = self._source_returned = False
        self.staged = []
        self.words = [None, None]
        self.queries = []
        self.query_index = 0
        self.cache_requested = False
        self.required_end = cell + 4 * (len(consumer.reference) - 1) + 3
        self.stats = dict.fromkeys(
            [
                "cache_reads",
                "data_queries",
                "raw_reads",
                "overlap_words_reused",
                "symbol_operations",
                "header_operations",
                "payload_operations",
                "blocked_service_edges",
                "input_words",
                "input_overflow",
                "functional_resolutions",
            ],
            0,
        )
        self._plan()

    def _make_consumer(self, profile, initial_fit, word_timing_format, selection):
        if word_timing_format is None:
            return InterpolantPSKReceiver(profile, initial_fit=initial_fit, **selection)
        from nearlink_sdr.phy.timing_word_rx import WordTimingPSKReceiver

        return WordTimingPSKReceiver(profile, initial_fit=initial_fit,
                                     word_timing_format=word_timing_format, **selection)

    @property
    def cycle(self):
        return self.kernel.cycle

    @property
    def done(self):
        return self.event is not None

    def _count(self, key, value=1):
        if not 0 <= self.stats[key] + value < 2**64:
            raise OverflowError("Replay controller counter exhausted")
        self.stats[key] += value

    def _barrier(self, kind):
        self.mode = "service"
        self.pending_service = dict(
            kind=kind, issued_transport_edge=self.cycle, arithmetic_cycles=None
        )

    def _observe_word(self, role, word):
        if self.observer is not None:
            if hasattr(self.consumer, 'word_queries'):
                query = self.consumer.word_queries[role]
                cell, phase = query['cell'], query['phase']
            else:
                position = self.consumer.sample_positions[role]
                cell = math.floor(position)
                phase = round((position - cell) * 65536)
            self.observer(
                dict(
                    symbol=self.consumer._active["count"],
                    role=role,
                    cell=cell,
                    phase=phase,
                    word=word,
                    transport_edge=self.cycle,
                    clock_evidence_valid=self.cycle_evidence_valid,
                )
            )

    def _plan(self):
        kind = self.consumer.boundary_kind
        if kind is not None:
            self._barrier(kind)
            return
        current, midpoint = self.consumer.sample_positions
        future_end = (self.consumer.future_sample_end
                      if hasattr(self.consumer, 'word_queries') else math.ceil(current + 3))
        positions = [(0, current), (1, midpoint)]
        if hasattr(self.consumer, 'word_queries'):
            queries = [dict(role=role, **self.consumer.word_queries[role])
                       for role, t in positions if t is not None]
        else:
            queries = [
                dict(role=role, cell=math.floor(t), phase=round((t - math.floor(t)) * 65536))
                for role, t in positions if t is not None
            ]
        self._plan_samples(queries, future_end)

    def _plan_samples(self, queries, future_end):
        """Shared transport only;finite controllers supply actual registered descriptors."""
        self.future_end = future_end
        self.words = [None, None]
        self.queries = queries
        if self.consumer._active["count"] < len(self.consumer.reference):
            self.mode = "cache"
            self.cache_requested = False
            return
        if self.reuse_stencil:
            self.queries.reverse()
        self.query_index = 0
        self.staged = []
        self.stage_start = self.queries[0]["cell"] - 1
        self.mode = "queries"

    @property
    def wants_replay_input(self):
        if self.done or self.eof or self.mode != "queries":
            return False
        if self.query_index < len(self.queries) and len(self.staged) < 4:
            return self.stage_start + len(self.staged) >= self.history.end
        return self.history.end < self.future_end

    def feed_replay_words(self, words):
        """Pausable functional source;accept only the requested prefix,never ADC evidence."""
        if (
            not isinstance(words, (list, tuple, np.ndarray))
            or not 1 <= len(words) <= 32
            or (isinstance(words, np.ndarray) and words.ndim != 1)
        ):
            raise ValueError("Need1..32 replay filter words")
        used = 0
        for word in words:
            if not self.wants_replay_input:
                break
            if not self.write_input(word):
                break
            used += 1
            # A real source port accepts at most one word at this edge.
            break
        return used

    def write_input(self, word):
        """One actual external edge;ADC caller must handle every false/terminal return."""
        if self.done:
            return False
        if self.eof:
            raise RuntimeError("EOF source cannot resume")
        if self.history.last_write_cycle == self.cycle:
            raise ValueError("At most one source word per transport edge")
        if self.history.end == self.consumer.sample_limit:
            self.abort("retention_limit")
            return False
        if not self.history.write(self.history.end, word, self.cycle):
            self._count("input_overflow")
            self.abort("history_overflow")
            return False
        self._count("input_words")
        self.consumer._observe(self.history.end)
        return True

    def _retire(self, index):
        self.history.retire_before(max(self.history.head, min(index, self.history.end)), self.cycle)

    def _step_queries(self):
        reply = self.history.take(self.cycle)
        if reply is not None:
            if reply[0] != self.stage_start + len(self.staged):
                raise RuntimeError("Registered raw reply changed stencil identity")
            self.staged.append(reply[1])
        if self.query_index < len(self.queries) and len(self.staged) == 4:
            query = self.queries[self.query_index]
            if self.kernel.issue(tuple(self.staged), query["phase"], tag=query["role"]):
                self._count("data_queries")
                self.required_end = max(self.required_end, self.stage_start + 4)
                self.query_index += 1
                if self.query_index < len(self.queries):
                    following = self.queries[self.query_index]["cell"] - 1
                    difference = following - self.stage_start
                    if self.reuse_stencil:
                        if not 0 <= difference <= 4:
                            raise RuntimeError("Ordered within-symbol stencil geometry changed")
                        self.staged = self.staged[difference:]
                        self._count("overlap_words_reused", len(self.staged))
                    else:
                        self.staged = []
                    self.stage_start = following
                else:
                    if not self._retain_completed_stencil:
                        self.staged = []
                    # Both kernels own their raw copies. Future midpoint cannot precede
                    # the current center because the existing loop step stays positive.
                    self._retire(self.queries[0 if not self.reuse_stencil else 1]["cell"] - 1)
        if self.query_index < len(self.queries) and len(self.staged) < 4:
            index = self.stage_start + len(self.staged)
            if index < self.history.head:
                self.abort("expired_history")
                return
            if self.history.pending is None and index < self.history.end:
                if self.history.request(index, self.cycle):
                    self._count("raw_reads")
            elif self.eof and index >= self.history.end and self.history.pending is None:
                self._finish_eof()
                return
        result = self.kernel.take()
        if result is not None:
            role = result["tag"]
            if role not in (0, 1) or self.words[role] is not None:
                raise RuntimeError("Interpolant reply lost role or repeated ownership")
            self.words[role] = result["word"]
            self._observe_word(role, result["word"])
        if all(w is not None for w in self.words):
            if self.history.end >= self.future_end:
                self._barrier("symbol")
            elif self.eof:
                self._finish_eof()

    def step(self):
        if self.done:
            return
        if self.mode == "service":
            self._count("blocked_service_edges")  # No unknown arithmetic is run here.
        elif self.mode == "cache":
            n = self.consumer._active["count"]
            if not self.cache_requested:
                cell = self.cache.cache.first_cell + 4 * n
                self.cache_requested = self.cache.request(cell, self.cache.cache.phase, self.cycle)
            reply = self.cache.take(self.cycle)
            if reply is not None:
                if reply[0] != n:
                    raise RuntimeError("Sync replay lost absolute symbol order")
                self._count("cache_reads")
                self.words[0] = reply[1]
                self._observe_word(0, reply[1])
                self._barrier("symbol")
        elif self.mode == "queries":
            self._step_queries()
        self.kernel.advance()

    def resolve_functional(self):
        """Explicit off-clock floating work,never a zero-cycle hardware completion."""
        if self.done or self.pending_service is None:
            raise RuntimeError("No unresolved symbol/header/payload service")
        kind = self.pending_service["kind"]
        self.cycle_evidence_valid = False
        self._count("functional_resolutions")
        self.pending_service = None
        if kind == "symbol":
            self.consumer.push_symbol(*self.words, after_samples=self.history.end)
            self._count("symbol_operations")
            if self.consumer.required_history_start is not None:
                self._retire(self.consumer.required_history_start)
        else:
            self._count(kind + "_operations")
            event = self.consumer.parse_boundary(after_samples=self.history.end)
            if event is not None:
                self._close(event)
                return event
        self._plan()
        return None

    def end_input(self):
        needs_future = self.wants_replay_input
        self.eof = True
        # Registered replies/kernels and known-sync replay may still drain.
        if needs_future:
            self._finish_eof()

    def take_mac_result(self):
        """Deliver the real strict MAC result once;retain all failures as diagnostics."""
        from nearlink_sdr.phy.mac_interface import mac_result_from_events

        if not self.done:
            raise RuntimeError("Frame has not reached a terminal gate")
        if self._mac_taken:
            return mac_result_from_events([])
        self._mac_taken = True
        return mac_result_from_events([self.event])

    def release_source(self):
        """Return the borrowed bank with its unconsumed suffix;no copy or full-tail clear."""
        if not self.done:
            raise RuntimeError("Active frame still owns the source bank")
        if self._source_returned:
            return None
        self._source_returned = True
        return self.history

    def _finish_eof(self):
        (event,) = self.consumer.finish()
        self._close(event)

    def abort(self, reason="cancelled"):
        if self.done:
            return None
        if reason not in ["cancelled", "history_overflow", "retention_limit", "expired_history"]:
            raise ValueError("Need explicit replay resource/ownership failure")
        finish_reason = "retention_limit" if reason == "retention_limit" else "history_overflow"
        (event,) = self.consumer.finish(reason=finish_reason)
        event["frame"].update(status=reason, resource_reason=reason, payload=None)
        self._close(event)
        return event

    def _close(self, event):
        self.history.pending = None
        self.kernel.stages = [None] * 7
        self.cache.abort(self.cycle)
        self.staged = []
        self.words = [None, None]
        self.pending_service = None
        self._retire(self.required_end)
        self.mode = "done"
        event.update(
            replay_transport=dict(
                cycle_evidence_valid=self.cycle_evidence_valid,
                arithmetic_service_cycles=None,
                complete_latency_cycles=None,
                source_retired_before=self.history.head,
                unconsumed_source_suffix=self.history.end - self.history.head,
                statistics=self.stats.copy(),
                reuse_stencil=self.reuse_stencil,
            ),
            all_sync_power_updates_preserved=True,
        )
        self.event = event

    def estimate(self):
        # 64pending-edge,64required-end,64future-end,64stage-start,2x(64cell+17phase+1role),
        # 4x46 staging,2x52 replies+valids,query/count/control flags and11 counters.
        control = 4 * 64 + 2 * (64 + 17 + 1) + 4 * 46 + 2 * (52 + 1) + 32 + len(self.stats) * 64
        return dict(
            new_controller_state_bits=control,
            source_staging_words=4,
            pending_interpolants=2,
            new_raw_RAM_bits=0,
            extra_DSP=0,
            borrowed_history=self.history.estimate(),
            borrowed_cache=self.cache.estimate(),
            borrowed_interpolator=self.kernel.estimate(),
            source_read_ports=1,
            source_write_ports=1,
            within_symbol_compaction_mux2_upper=368 if self.reuse_stencil else 0,
            compaction_index_comparisons64=2 if self.reuse_stencil else 0,
            complete_controller_LUT_or_timing=None,
            remaining_float_state=self.consumer.reference_memory(),
            arithmetic_service_cycles=None,
            full_receiver16MSps=False,
            synthesis=False,
            host_observer_is_not_a_hardware_trace_buffer=True,
        )
