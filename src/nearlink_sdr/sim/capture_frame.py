"""Borrowed captured window/shared tail→existing real header/frame owner (§6.3/6.9).

Every feed/confirmation/EOF is an explicit off-clock functional resolution.
Registered raw transport is modeled;arithmetic service and whole latency remain
unknown. No private whole-tail banks or CRC-selected answer retry are added.
"""

from dataclasses import replace

import numpy as np

from nearlink_sdr.phy.mac_interface import mac_result_from_events
from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.fine_selection import FineSelection

__all__ = ["CapturedFrameReplay"]


class CapturedFrameReplay:
    """One active candidate,one registered word reply,existing bounded float RX.

    Prefix and shared source RAM are borrowed by the composing transport. This
    object never retains a whole tail or original capture. An actual consumed
    prefix retires future overlapping candidates regardless of CRC outcome.
    That ownership policy is explicit;it is not a new peak/CFO retry strategy.
    """

    def __init__(
        self,
        profile,
        *,
        coherent_threshold,
        sample_limit=2**20,
        sink=None,
        raw_observer=None,
    ):
        if (
            not isinstance(profile, PSKReceiveProfile)
            or profile.config.sps != 4
            or profile.config.symbol_rate_mhz != 4
            or profile.control_format not in ("A2", "B1")
            or profile.mcs_from_lqi
            or profile.length_contract != "standard"
            or profile.config.mcs_index not in (8, 12)
            or not set(profile.allowed_mcs) <= {8, 12}
            or type(sample_limit) is not int
            or not 4 <= sample_limit <= 2**20
            or isinstance(coherent_threshold, (bool, np.bool_))
            or not isinstance(coherent_threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(coherent_threshold)
            or not 0 < coherent_threshold <= 1
        ):
            raise ValueError("Need ordinary standard A2/B1,MCS8/12,SPS4/4Msym and bounded input")
        if (sink is not None and not callable(sink)) or (
            raw_observer is not None and not callable(raw_observer)
        ):
            raise ValueError("Need optional caller-owned read-only evidence callbacks")
        self.profile, self.sink, self.raw_observer = replace(profile), sink, raw_observer
        self.threshold, self.sample_limit = float(coherent_threshold), sample_limit
        self.rx = self.selection = self.pending = self.event = None
        self.origin = self.index = self.retired_through = 0
        self.cycle, self.eof = -1, False
        self._taken = False
        self.stats = dict.fromkeys(
            (
                "accepted",
                "coherent_accepted",
                "overlap_rejected",
                "registered_reads",
                "read_returns",
                "functional_resolutions",
                "header_valid",
                "header_failed",
                "terminal",
                "resource_aborts",
            ),
            0,
        )
        self.peak_reference_bytes = 0
        self.header = None

    def _count(self, name):
        if self.stats[name] == 2**64 - 1:
            raise OverflowError("Frame ownership statistic exhausted")
        self.stats[name] += 1

    def _edge(self, cycle):
        if type(cycle) is not int or not self.cycle <= cycle < 2**63 - 1:
            raise ValueError("Need monotone bounded frame clock")
        self.cycle = cycle

    @property
    def done(self):
        return self.event is not None

    @property
    def active(self):
        return self.selection is not None

    @property
    def needs_word(self):
        return self.active and not self.done and self.pending is None

    @property
    def source_index(self):
        return self.origin + self.index

    def begin(self, selection, cycle):
        if self.active or not isinstance(selection, FineSelection):
            raise ValueError("Need one unowned actual fine selection")
        self._edge(cycle)
        if cycle < selection.available_cycle or len(selection.words) != len(
            self.profile.sync_symbols()
        ):
            raise ValueError("Fine selection is not available or belongs to another profile")
        self.selection, self.origin = selection, selection.source_end - selection.window
        self.index, self.pending, self.event, self.eof, self._taken = 0, None, None, False, False
        self.header = None
        if selection.source_start_tick < 8 * self.retired_through:
            self._count("overlap_rejected")
            self._terminal(
                dict(
                    frame=dict(
                        status="source_already_consumed",
                        header_crc=None,
                        data_crc=None,
                        length_ok=None,
                        payload=None,
                    )
                )
            )
            return
        self._count("accepted")
        self._count("functional_resolutions")
        self.rx = SelectedPSKReceiver(
            self.profile,
            **selection.reference_selection(),
            sample_limit=self.sample_limit,
            coherent_threshold=self.threshold,
            timing_backend="python",
            history_capacity=512,
        )
        memory = self.rx.reference_memory()["persistent_numpy_storage_bytes"]
        self.peak_reference_bytes = max(memory, self.peak_reference_bytes)
        # This does not read a word or invent a cycle for the confirmation arithmetic.
        rejected = self.rx.feed_filtered(np.empty(0, complex))
        if rejected:
            self._terminal(rejected[0])
        else:
            self._count("coherent_accepted")

    def issue_read(self, index, word, cycle):
        self._edge(cycle)
        if not self.needs_word or index != self.source_index:
            raise ValueError("Need exactly the next owned raw word and one read credit")
        physical_filter_word(word)  # Validate the literal46bit port before state mutation.
        self._count("registered_reads")
        self.pending = index, int(word), cycle + 1

    def tick(self, cycle, *, blocked=False):
        self._edge(cycle)
        if type(blocked) is not bool:
            raise ValueError("Need explicit downstream readiness")
        if self.pending is None or self.pending[2] > cycle or blocked or self.done:
            return
        index, word, _ = self.pending
        if index != self.source_index:
            raise RuntimeError("Registered frame reply lost its original source identity")
        self.pending = None
        self._count("read_returns")
        if self.raw_observer is not None:
            self.raw_observer(
                dict(
                    id=self.selection.ident,
                    index=index,
                    word=word,
                    returned_cycle=cycle,
                    functional_reference=True,
                )
            )
        self._count("functional_resolutions")
        events = self.rx.feed_filtered(np.asarray([physical_filter_word(word)]))
        used = self.rx.last_input_consumed
        if used not in (0, 1):
            raise RuntimeError("One raw port word produced an invalid consumed prefix")
        self.index += used
        self.retired_through = max(self.retired_through, self.source_index)
        self._accept_reference_results(events)
        if not events and used != 1:
            raise RuntimeError("Active strict consumer neither consumed nor terminated")

    def _accept_reference_results(self, events):
        """Functional handoff hook; a composing calendar may defer availability."""
        if self.header is None and self.rx.header_result is not None:
            self.header = self.rx.header_result
            self._count(
                "header_valid"
                if self.header.get("header_crc") is True and self.header.get("length_ok") is True
                else "header_failed"
            )
        if events:
            self._terminal(events[0])

    def end_input(self, cycle):
        self._edge(cycle)
        if self.eof or self.done:
            return
        if self.pending is not None:
            raise RuntimeError("Drain actual registered raw reply before EOF")
        self.eof = True
        self._count("functional_resolutions")
        events = self.rx.finish()
        if len(events) != 1:
            raise RuntimeError("Selected EOF must return one actual terminal result")
        self._terminal(events[0])

    def abort(self, reason, cycle):
        self._edge(cycle)
        if not self.active or self.done:
            return
        if reason not in ("tail_history_expired", "source_epoch_aborted", "cancelled"):
            raise ValueError("Need explicit frame source/resource failure")
        self.pending = None
        self._count("resource_aborts")
        # Resource failure cannot deliver a partial or CRC-passing prefix.
        self._terminal(
            dict(
                frame=dict(
                    status=reason,
                    resource_reason=reason,
                    header_crc=None,
                    data_crc=None,
                    length_ok=None,
                    payload=None,
                )
            )
        )

    def _terminal(self, event):
        self._count("terminal")
        event.update(
            selection_id=self.selection.ident,
            source_window_origin=self.origin,
            consumed_source_end=self.source_index,
            retired_through_source=self.retired_through,
            header_decision=self.header,
            complete_latency_cycles=None,
            arithmetic_execution="explicit off-clock functional resolution",
            complete_cycle_evidence_valid=False,
            hardware_cycles_verified=False,
            no_private_full_tail=True,
        )
        self.event = event

    def take(self):
        if not self.done or self._taken:
            return None
        self._taken = True
        event = self.event
        mac = mac_result_from_events([event])
        if self.sink is not None:
            self.sink(event, mac)
        return event, mac

    def release(self):
        if not self.done or not self._taken:
            raise RuntimeError("Actual terminal outcome must be taken before raw ownership release")
        self.rx = self.selection = self.pending = self.event = None
        self.header = None

    def estimate(self):
        fields = dict(
            clock=63,
            origin=63,
            next_local_index=21,
            retired_through=63,
            pending_index_word_due_valid=63 + 46 + 63 + 1,
            EOF_taken_terminal_flags=3,
            statistics=10 * 64,
            reference_peak_counter=64,
            header_metadata_borrowed=True,
        )
        register_bits = sum(v for v in fields.values() if type(v) is int)
        return dict(
            register_fields=fields,
            register_bits_upper=register_bits,
            private_full_tail_banks=0,
            prefix_is_borrowed_owned_slot=True,
            source_tail_is_existing_shared_ring=True,
            one_raw_reply_credit=True,
            raw_read_latency=1,
            functional_raw_history_samples=512,
            functional_host_mirror_copies=2,
            actual_peak_reference_bytes=self.peak_reference_bytes,
            float_consumer_hardware_LUT_FF_DSP_RAM_ports=None,
            confirmation_feed_header_carrier_decode_cycles=None,
            complete_latency_cycles=None,
            full_fixed_or16MSps=False,
            actual_synthesis=False,
            default_changed=False,
        )
