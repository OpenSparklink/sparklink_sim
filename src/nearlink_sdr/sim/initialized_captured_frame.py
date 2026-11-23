"""Actual chosen cache→finite initialization/timing/head,with one bounded raw bridge.

Existing §6.2.1.2/6.3/6.9 nodes are reused. Private1024-word1R1W storage and
all source reads are charged;no ADC pause or CRC-selected answer retry. Float
metadata conversion and residual payload work remain explicit unmodeled work.
"""

import numpy as np

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.sync_fit_word import compile_sync_phases, unsigned
from nearlink_sdr.sim.capture_fine import unpack_filter_word
from nearlink_sdr.sim.capture_frame import CapturedFrameReplay
from nearlink_sdr.sim.coherent_interleaved import (
    InterleavedCoherentWordService,
    ScheduledCoherentCapturedFrameReplay,
)
from nearlink_sdr.sim.fine_selection import FineSelection
from nearlink_sdr.sim.initialized_frame_replay import InitializedFrameReplay
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

__all__ = ["InitializedCapturedFrameReplay"]


class InitializedCapturedFrameReplay(ScheduledCoherentCapturedFrameReplay):
    """One actual selection,one private1024-word ring,compiled8-word prefetch.

    Only the chosen window is admitted. Queries/CRC/length and real frame-end
    consumption belong to existing finite owners. `functional` explicitly
    resolves remaining payload work;`stop` terminates at that unknown service.
    Neither mode establishes complete hardware cycles or a full fixed receiver.
    """

    def __init__(self, profile, *, coherent_threshold, payload_mode, **options):
        if payload_mode not in ("functional", "stop") or type(payload_mode) is not str:
            raise ValueError("Need explicit functional or stop payload mode")
        if not isinstance(profile, PSKReceiveProfile) or coherent_threshold != .45:
            raise ValueError("Need actual PSK profile and compiled9/20 coherent gate")
        if profile.head_backend not in (
                "word-dense", "word-depth-parity"):
            raise ValueError("Need compiled9/20 gate and explicit actual word head")
        # Do not construct the old floating consumer or its latency-replay head calendar.
        CapturedFrameReplay.__init__(self, profile, coherent_threshold=.45, **options)
        self.payload_mode, self.structure = payload_mode, "interleaved-signed22x18"
        self._service_class = InterleavedCoherentWordService
        self.codes = compile_sync_phases(self.profile.sync_symbols())
        self.clocked = True
        self.word_service = self.word_receipt = None
        self._word_waiting = False
        self.word_stats = dict(issued=0, completed=0, cancelled=0)
        self.history = self.flow = self.fields = None
        self.prefill_end = 0
        self.phase = "idle"
        self.bridge_stats = dict.fromkeys([
            "copied_words", "initialized", "functional_payloads", "unmodeled_payload_stops",
            "private_full_wait_edges", "retired_words",
        ], 0)

    def _bridge_count(self, key, n=1):
        self.bridge_stats[key] = unsigned(self.bridge_stats[key] + n, 64)

    def begin(self, selection, cycle):
        if isinstance(selection, FineSelection):
            future = (selection.start_tick // 8 + 4 * (len(selection.words) - 1) + 3
                      + bool(selection.start_tick % 8))
            if future > min(selection.window, self.sample_limit):
                raise ValueError("Chosen sync future exceeds the real bounded input")
        if not self._begin_selection(selection, cycle):
            return
        self.word_service = self._new_word_service(selection, cycle)
        self._count_word("issued")
        self._word_waiting, self.phase = True, "confirmation"

    def _complete_word_gate(self):
        receipt = self.word_service.result
        if not self._word_waiting or not self.word_service.ready or receipt is None:
            raise RuntimeError("Actual confirmation must complete before initialization")
        fields = self.selection.reference_selection() if receipt.accepted else None
        first = self.selection.start_tick // 8 - 1
        future = (self.selection.start_tick // 8 + 4 * (len(self.codes) - 1) + 3
                  + bool(self.selection.start_tick % 8))
        if receipt.accepted and not 0 <= first < future <= self.selection.window:
            raise ValueError("Chosen sync future must belong to the actual copied window")
        history = ReceiveHistoryPort(1024, first_index=first) if receipt.accepted else None
        self.word_receipt = self.word_service.take()
        self._word_waiting = False
        self._count_word("completed")
        if not receipt.accepted:
            self._terminal(self._failure("sync_rejected"))
            return
        self._count("coherent_accepted")
        self.fields, self.history, self.prefill_end = fields, history, future
        self.index, self.phase = first, "prefill"
        self.retired_through = max(self.retired_through, self.origin + first)
        if self.eof:
            self._terminal(self._failure("truncated_header"))

    def _copy_limit(self):
        if self.flow is None:
            return self.prefill_end
        if self.flow.done or self.flow.pending_service is not None or self.flow.frame is None:
            return self.history.end
        frame = self.flow.frame
        if not hasattr(frame, "future_end"):
            return self.history.end
        limit = frame._query_limit()  # Real head boundary until CRC/geometry actually pass.
        remaining = None if limit is None else limit - frame.sample_index
        extra = 8 if remaining is not None and remaining > 3 else 0
        # Three minimum legal NCO steps exceed8+one rounding sample. The last
        # three symbols therefore never prefetch beyond their actually needed future.
        return min(self.sample_limit, frame.future_end + extra)

    @property
    def needs_word(self):
        return (self.active and not self.done and not self.eof and not self._word_waiting
                and self.pending is None and self.history is not None
                and self.history.live < 1024 and self.history.end < self._copy_limit())

    def issue_read(self, index, word, cycle):
        self._edge(cycle)
        if not self.needs_word or index != self.source_index:
            raise ValueError("Need exactly the next bounded private-copy read")
        unpack_filter_word(word)  # Literal finite46-bit validation;no float raw conversion.
        self._count("registered_reads")
        self.pending = index, int(word), cycle + 1

    def _accept_copy(self, cycle):
        index, word, _ = self.pending
        if index != self.source_index or self.index != self.history.end:
            raise RuntimeError("Copy reply changed its selected source index")
        unsigned(self.bridge_stats["copied_words"] + 1, 64)
        unsigned(self.stats["read_returns"] + 1, 64)
        if self.flow is not None and self.flow.cycle != cycle:
            raise RuntimeError("Copy reply must arrive at the initialized owner edge")
        accepted = (self.history.write(self.index, word, cycle) if self.flow is None
                    else self.flow.write_input(word))
        if not accepted:
            raise RuntimeError("A reserved raw reply lost its one private write credit")
        self.pending = None
        self.index += 1
        self._bridge_count("copied_words")
        self._count("read_returns")
        if self.raw_observer is not None:
            self.raw_observer(dict(id=self.selection.ident, index=index, word=word,
                                   returned_cycle=cycle, functional_reference=False))
        if self.flow is None and self.history.end == self.prefill_end:
            unsigned(self.bridge_stats["initialized"] + 1, 64)
            self.flow = self.rx = InitializedFrameReplay(
                self.profile, self.history, start_cycle=cycle, **self.fields,
                sample_limit=self.sample_limit, coherent_threshold=None,
                confirmed_receipt=self.word_receipt,
            )
            self.phase = "initialized_frame"
            self._bridge_count("initialized")

    def _observe_head(self):
        if self.header is not None or self.flow is None or self.flow.frame is None:
            return
        row = self.flow.frame.consumer.header_result
        if row is not None:
            self.header = row
            self._count("header_valid" if row.get("header_crc") is True
                        and row.get("length_ok") is True else "header_failed")

    def tick(self, cycle, *, blocked=False):
        if type(blocked) is not bool:
            raise ValueError("Need explicit downstream readiness")
        if not self.active or self.done:
            self._edge(cycle)
            return
        if cycle != self.cycle + 1:
            raise ValueError("Finite frame bridge requires every actual clock edge")
        self._edge(cycle)
        if self._word_waiting:
            while self.word_service.cycle < cycle:
                self.word_service.step()
            if self.word_service.ready and not blocked:
                self._complete_word_gate()
            return
        if self.pending is not None and self.pending[2] <= cycle and not blocked:
            self._accept_copy(cycle)
        if self.flow is None:
            return
        if self.flow.cycle != cycle:
            raise RuntimeError("Private initialized owner must share the actual source edge")
        if self.flow.pending_service is not None and not blocked:
            if self.payload_mode == "stop":
                self._bridge_count("unmodeled_payload_stops")
                event = self._failure("payload_service_unmodeled")
                event["frame"].update(header_crc=self.header.get("header_crc"),
                                      length_ok=self.header.get("length_ok"),
                                      mcs=self.header.get("mcs"))
                self.flow.abort("cancelled")
                self._terminal(event)
                return
            self._bridge_count("functional_payloads")
            self._count("functional_resolutions")
            self.flow.resolve_functional()  # Explicit non-cycle payload reference,never free HW.
        if not self.flow.done:
            self.flow.step()
        old = self.retired_through
        self.retired_through = max(old, self.origin + self.flow.consumed_history_end)
        self._bridge_count("retired_words", self.retired_through - old)
        if not blocked:
            self._observe_head()
            if self.flow.done:
                self._terminal(self.flow.event)
        if self.history.live == 1024:
            self._bridge_count("private_full_wait_edges")

    @staticmethod
    def _failure(status):
        return dict(frame=dict(status=status, header_crc=None, data_crc=None,
                               length_ok=None, payload=None), symbols=np.empty(0, complex))

    def end_input(self, cycle):
        self._edge(cycle)
        if self.done or self.eof:
            return
        if self.pending is not None:
            raise RuntimeError("Drain actual registered copy before EOF")
        self.eof = True
        if self._word_waiting:
            return  # Confirmation really drains;no fabricated raw sync or header.
        if self.flow is None:
            self._terminal(self._failure("truncated_header"))
        else:
            self.flow.end_input()

    def abort(self, reason, cycle):
        if reason not in ("tail_history_expired", "source_epoch_aborted", "cancelled"):
            raise ValueError("Need explicit capture source/resource cancellation")
        self._edge(cycle)
        if not self.active or self.done:
            return
        if self._word_waiting:
            if cycle <= self.word_service.cycle:
                raise ValueError("Cancellation needs the next coherent registered edge")
            while self.word_service.cycle < cycle - 1:
                self.word_service.step()
            self.word_service.abort()
            self._word_waiting = False
            self._count_word("cancelled")
        inner = None
        if self.flow is not None:
            inner = self.flow.event if self.flow.done else self.flow.abort("cancelled")
        self.pending = None
        self._count("resource_aborts")
        # The real owner already stopped above. Keep its checked head and
        # partial diagnostics;do not turn a resource failure into delivery.
        event = self._failure(reason) if inner is None else dict(inner)
        event["frame"] = dict(event["frame"], status=reason,
                              resource_reason=reason, payload=None)
        self._terminal(event)

    def _terminal(self, event):
        event["coherent_word_service"] = self._coherent_evidence()
        event["initialized_capture_bridge"] = dict(
            phase=self.phase, payload_mode=self.payload_mode,
            actual_terminal_cycle=self.cycle,
            statistics=self.bridge_stats.copy(), private_history_words=1024,
            private_history=None if self.history is None else self.history.estimate(),
            peak_private_words=0 if self.history is None else self.history.stats["peak_live"],
            initialized_cost=None if self.flow is None else self.flow.estimate(),
            descriptor_score_angle_reference_cycles=None, payload_cycles=None,
            complete_fixed_or_cycles=False, new_global_source_read_ports=0,
        )
        CapturedFrameReplay._terminal(self, event)
        event["copied_source_end"] = self.source_index
        event["consumed_source_end"] = self.retired_through
        event["arithmetic_execution"] = (
            "finite init/timing/head;explicit remaining float metadata/payload"
        )
        self.phase = "terminal"

    def release(self):
        if not self.done or not self._taken:
            raise RuntimeError("Terminal must be taken before private source release")
        if self.flow is not None:
            self.flow.release_source()
        CapturedFrameReplay.release(self)
        self.history = self.flow = self.fields = None
        self.word_service = self.word_receipt = None
        self._word_waiting, self.phase = False, "idle"

    def estimate(self):
        base = CapturedFrameReplay.estimate(self)
        fields = dict(local_bounds=2 * 21, flags_and_phase_and_receipt=9, bridge_statistics=6 * 64,
                      coherent_statistics_and_flag=3 * 64 + 1)
        extra = sum(fields.values())
        port = ReceiveHistoryPort(1024).estimate()
        known = (base["register_bits_upper"] + extra
                 + self._word_service_cost()["register_bits_upper"]
                 + port["address_timestamp_and_reply_bits"] + port["statistic_counter_bits"])
        return base | dict(
            actual_peak_reference_bytes=None, reference_peak_not_measured=True,
            functional_raw_history_samples=0, functional_host_mirror_copies=0,
            old_float_raw_consumer_constructed=False,
            register_bits_upper=None, known_register_bits_lower=None,
            known_register_reservation_subtotal=known,
            register_subtotal_is_component_reservation_not_physical_lower_bound=True,
            complete_initialized_registers_or_profile_descriptor=None,
            new_bridge_control_fields=fields, new_bridge_control_bits=extra,
            new_bridge_control_research_cap_bits=2048,
            private_history=ReceiveHistoryPort(1024).estimate(),
            new_private_source_BRAM36=2, original_global_source_still_reserved=True,
            initial_abort_frontier=dict(
                reconstruct_from_existing_discard_count=True,
                extra_state_bits=0, extra_RAM_DSP_or_ports=0,
                subtract_bits_upper=64, subtracts_per_initial_abort=1,
                existing_registered_frontier_reused=True,
                combinational_select_and_subtract_mapping=None, Fmax_verified=False,
            ),
            actual_source_read_ports_added=0, private_global_read_copy_and_write_are_charged=True,
            coherent=self._word_service_cost(),
            initialized_static_DSP=19, coherent_and_initial_fit_DSP_sharing_credit=False,
            initialized_source_sync_head_BRAM36=5 if self.profile.config.frame_type == 2 else 6,
            initialized_runtime=None if self.flow is None else self.flow.estimate(),
            selected_fine_cache_remains_separate=True,
            source_clock_and_metadata_register_costs_not_zero=True,
            descriptor_score_angle_reference_cycles=None, payload_cycles=None,
            full_fixed_or16MSps=False, complete_latency_cycles=None, actual_synthesis=False,
        )
