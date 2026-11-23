"""One captured selection→actual initialization→bounded frame owner (§6.3/6.9).

Explicit1024-word trial. The capture producer and payload service are not given
zero cycles. Default512-word replay and all sync/loop/CRC choices remain unchanged.
"""

import math
import weakref
from dataclasses import replace

import numpy as np

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.selected_rx import _validate_selected_geometry
from nearlink_sdr.phy.sync_coherent import coherent_sync_confirmation
from nearlink_sdr.phy.sync_fit_word import (
    GAIN_FLOOR,
    TURN,
    SyncFitFormat,
    compile_sync_phases,
    unsigned,
)
from nearlink_sdr.phy.timing_word import TimingWordFormat, compare_scaled, positive
from nearlink_sdr.sim.coherent_word_flow import CoherentWordReceipt
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.packed_head_replay import PackedHeadSamplingReplay, _PackedWords
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from nearlink_sdr.sim.split_sampling_replay import _SplitReceiver
from nearlink_sdr.sim.split_timing_flow import split_timing_states

__all__ = ["InitializedFrameReplay", "TimingContextInstallService"]


class TimingContextInstallService:
    """Three registered stages,one upstream-held fit and one held output credit.

    Stage1 normalizes54→30bit power with RNE;stage2 clamps its finite floor;
    stage3 installs the unchanged detector/PI reset words. The canonical state
    constructor is a host consistency check,not another uncharged datapath.
    """

    def __init__(self, *, start_cycle=0):
        if type(start_cycle) is not int:
            raise ValueError("Need integer service clock")
        unsigned(start_cycle, 63)
        self.cycle, self.mode = start_cycle, "idle"
        self.fit = self.power = self.result = None
        self.tick, self.hold = 0, 0
        self.stats = dict(accepted=0, completed=0, held_edges=0)

    @property
    def ready(self):
        return self.mode == "ready"

    def issue(self, fit, start_tick, hold_symbols):
        if self.mode != "idle":
            return False
        if (
            not isinstance(fit, dict)
            or fit.get("valid") is not True
            or fit.get("symbols") != hold_symbols
            or hold_symbols not in (32, 62, 126)
            or type(start_tick) is not int
            or not 0 <= start_tick < 2**67
        ):
            raise ValueError("Need one actual valid fit and bounded timing descriptor")
        power = fit.get("power")
        if (
            not isinstance(power, (list, tuple))
            or len(power) != 2
            or any(type(v) is not int for v in power)
            or not 0 < power[0] < 2**54
            or not -512 <= power[1] < 512
        ):
            raise ValueError("Need positive54+signed10 upstream-held power")
        self.fit, self.tick, self.hold = fit, start_tick, hold_symbols
        self.mode = "normalize"
        self.stats["accepted"] = unsigned(self.stats["accepted"] + 1, 64)
        return True

    def step(self):
        edge = unsigned(self.cycle + 1, 63)
        if self.mode == "normalize":
            self.power, self.mode = positive(*self.fit["power"], 30), "floor"
        elif self.mode == "floor":
            floor = positive(*GAIN_FLOOR, 30)  # Compiled constant,wired rather than runtime RAM.
            if compare_scaled(*self.power, *floor) < 0:
                self.power = floor
            self.mode = "install"
        elif self.mode == "install":
            owners = split_timing_states(self.power, self.tick, self.hold, TimingWordFormat(30, 32))
            if owners[0].power_word != self.power:
                raise RuntimeError("Registered initial power disagrees with canonical reset")
            self.result, self.fit = owners, None
            self.mode = "ready"
            self.stats["completed"] = unsigned(self.stats["completed"] + 1, 64)
        elif self.ready:
            self.stats["held_edges"] = unsigned(self.stats["held_edges"] + 1, 64)
        elif self.mode == "aborted":
            raise RuntimeError("Aborted context cannot emit a stale receipt")
        self.cycle = edge

    def take(self):
        if not self.ready:
            return None
        result, self.result = self.result, None
        self.mode = "taken"
        return result

    def abort(self):
        self.fit = self.result = self.power = None
        self.mode = "aborted"

    def estimate(self):
        return dict(
            stages=3,
            held_fit_is_borrowed=True,
            extra_fit_copy_bits=0,
            register_bits_upper=67 + 7 + 40 + 3 + 64 + 3 * 64,
            output_detector_PI_words="existing owners,moved once;not duplicated",
            LZC54=1,
            variable_shift54_RNE=1,
            add54=1,
            scaled_compare50_upper=1,
            DSP=0,
            new_RAM=0,
            width_stage_assumption="one registered stage each;Fmax unverified",
            LUT_FF_or_Fmax=None,
            synthesis=False,
            full_receiver16MSps=False,
        )


class _InstalledReceiver(_SplitReceiver):
    def __init__(self, *args, timing_owners, confirmed_receipt=None, **kwargs):
        self._installed_owners = timing_owners
        self._upstream_word_receipt = confirmed_receipt
        super().__init__(*args, **kwargs)

    def _confirm_selected_context(self, *args):
        receipt = self._upstream_word_receipt
        if receipt is None:
            return super()._confirm_selected_context(*args)
        return receipt.reference_confirmation(), receipt.accepted

    def _make_timing_state(self, power, start_tick, hold_symbols, fmt, config):
        self.detector, self.pi = self._installed_owners
        del self._installed_owners
        if (
            self.detector.count
            or self.pi.count
            or self.detector.hold_symbols != hold_symbols
            or self.pi.next_q32 != start_tick << 29
            or self.detector.fmt != fmt
        ):
            raise RuntimeError("Installed timing words belong to another packet")
        return self.receipt(self.detector.snapshot(), self.pi.snapshot())


class _Frame(PackedHeadSamplingReplay):
    _history_capacity = 1024

    def __init__(self, *args, timing_owners, head_owner, confirmed_receipt=None, **kwargs):
        self._installed_owners, self._compiled_head = timing_owners, head_owner
        self._upstream_word_receipt = confirmed_receipt
        super().__init__(*args, **kwargs)
        del self._installed_owners, self._compiled_head

    def _make_head_owner(self, profile, layout):
        return self._compiled_head

    def _make_consumer(self, profile, initial_fit, word_timing_format, selection):
        return _InstalledReceiver(
            profile,
            initial_fit=initial_fit,
            word_timing_format=word_timing_format,
            normalization_format=self.normalization_format,
            head_owner=self.head_owner,
            timing_owners=self._installed_owners,
            confirmed_receipt=self._upstream_word_receipt,
            **selection,
        )


class InitializedFrameReplay:
    """Borrow one already captured1024-word bank,then never pause a supplied ADC.

    The caller must give its actual selected-window end and clock. Full sync
    future samples must already be present;capture delay is not erased. There is
    no bulk IQ buffer or automatic retry. Inputs use one word/edge at most.
    A payload barrier can be explicitly resolved off-clock for function only.
    """

    def __init__(
        self,
        profile,
        history,
        *,
        start_cycle,
        start_tick,
        delta,
        differential_score,
        selected_sync,
        sample_limit,
        coherent_threshold=None,
        confirmed_receipt=None,
    ):
        sync = _validate_selected_geometry(
            profile, start_tick, delta, differential_score, selected_sync, sample_limit
        )
        # Latch this packet before delayed init/head/payload ownership transfers.
        # PSKReceiveProfile is frozen,but its negotiated TxConfig is mutable.
        profile = replace(profile)
        accepted = differential_score >= 0.85
        if confirmed_receipt is not None:
            self.checked_word_confirmation(confirmed_receipt, profile, start_cycle=start_cycle,
                start_tick=start_tick, delta=delta, differential_score=differential_score,
                selected_sync=sync, threshold=coherent_threshold)
            accepted = confirmed_receipt.accepted
        elif coherent_threshold is not None:
            accepted = coherent_sync_confirmation(
                sync, profile.sync_symbols(), delta, coherent_threshold
            )["accepted"]  # Host precondition check of the same already selected answer.
        if not accepted or start_tick < 8:
            raise ValueError("Need one already accepted selection;no new confirmation or retry")
        if (
            not isinstance(history, ReceiveHistoryPort)
            or history.capacity != 1024
            or history.pending is not None
            or getattr(history, "_initialized_owner", None) is not None
            or type(start_cycle) is not int
            or not max(0, history.cycle) <= start_cycle < 2**63 - 1
        ):
            raise ValueError("Need exclusive compiled1024-word history and its actual clock")
        cell, part = divmod(start_tick, 8)
        symbols = len(sync)
        future_end = cell + 4 * (symbols - 1) + 3 + bool(part)
        if history.head > cell - 1 or history.end < future_end or history.end > sample_limit:
            raise ValueError("Selected source must retain actual past and all sync future words")
        # This is the one physical head owner,allocated throughout init,not a validation clone.
        self.head_owner = _PackedWords(
            profile, layout="dense" if profile.config.frame_type == 2 else "depth-parity"
        )
        self.profile, self.history = profile, history
        # Borrow the same held receipt,not another cache copy.
        self._upstream_word_receipt = confirmed_receipt
        self.selection = dict(
            start_tick=start_tick,
            delta=delta,
            differential_score=differential_score,
            selected_sync=sync.copy(),
            sample_limit=sample_limit,
            coherent_threshold=coherent_threshold,
        )
        self.prepared = InitialSyncFitService(
            history,
            compile_sync_phases(profile.sync_symbols()),
            round(delta * TURN / math.tau),
            cell,
            part * 8192,
            fmt=SyncFitFormat(30),
            start_cycle=start_cycle,
        )
        self.start_cycle, self.fit_ready_cycle, self.transfer_cycle = start_cycle, None, None
        self.phase, self.install, self.frame = "initial_fit", None, None
        self.eof, self._mac_taken, self._source_returned = False, False, False
        self._event = None
        self.stats = dict(input_words=0, input_overflow=0, initial_discarded_words=0)
        history._initialized_owner = weakref.ref(self)  # Host identity;one charged ownership bit.

    @property
    def cycle(self):
        return self.frame.cycle if self.frame is not None else self.prepared.cycle

    @property
    def done(self):
        return self._event is not None or (self.frame is not None and self.frame.done)

    @property
    def event(self):
        return self._event if self.frame is None else self.frame.event

    @property
    def pending_service(self):
        return None if self.frame is None else self.frame.pending_service

    @staticmethod
    def checked_word_confirmation(receipt, profile, *, start_cycle, start_tick, delta,
                                  differential_score, selected_sync, threshold=None):
        """Host wiring checks for the already computed9/20 decision,not a second search.

        No caller success flag. The actual immutable receipt,profile/template,
        source words/coordinates and availability must describe this transaction.
        """
        if (not isinstance(receipt, CoherentWordReceipt) or threshold is not None
                or not receipt.accepted or type(start_cycle) is not int
                or start_cycle < receipt.available_cycle
                or receipt.phase_codes != compile_sync_phases(profile.sync_symbols())):
            raise ValueError("Need this available accepted word confirmation and no second gate")
        fields = receipt.selection.reference_selection()
        if (start_tick != fields["start_tick"] or delta != fields["delta"]
                or differential_score != fields["differential_score"]
                or not np.array_equal(selected_sync, fields["selected_sync"])):
            raise ValueError("Word confirmation belongs to another selected context")
        return receipt.reference_confirmation()

    def write_input(self, word):
        if self.done:
            return False
        if self.eof:
            raise RuntimeError("EOF source cannot resume")
        if self.frame is not None:
            return self.frame.write_input(word)
        if self.history.last_write_cycle == self.cycle:
            raise ValueError("At most one source word per edge")
        if self.history.end == self.selection["sample_limit"]:
            self.abort("retention_limit")
            return False
        if not self.history.write(self.history.end, word, self.cycle):
            self.stats["input_overflow"] = unsigned(self.stats["input_overflow"] + 1, 64)
            self.abort("history_overflow")
            return False
        self.stats["input_words"] = unsigned(self.stats["input_words"] + 1, 64)
        return True

    def step(self):
        if self.done:
            return
        if self.frame is not None:
            self.frame.step()
            return
        if self.phase == "initial_fit":
            self.prepared.step()
            if self.prepared.ready:
                if not self.prepared.fit.result["valid"]:
                    self.abort("invalid_initial_fit")
                    return
                self.fit_ready_cycle = self.cycle
                self.install = TimingContextInstallService(start_cycle=self.cycle)
                self.install.issue(
                    self.prepared.fit.result,
                    self.selection["start_tick"],
                    len(self.profile.sync_symbols()),
                )
                self.phase = "install"
        else:
            self.install.step()
            self.prepared.step(output_ready=False)  # Ready fit/cache held;the same clock advances.
            if self.install.ready:
                self.transfer_cycle = self.cycle
                self.frame = _Frame(
                    self.profile,
                    self.prepared,
                    timing_owners=self.install.take(),
                    head_owner=self.head_owner,
                    confirmed_receipt=self._upstream_word_receipt,
                    error_age=1,
                    normalization_format=NormalizationFormat(),
                    word_timing_format=TimingWordFormat(30, 32),
                    reuse_stencil=True,
                    **self.selection,
                )
                self.phase = "frame"
                if self.eof:
                    self.frame.end_input()

    def end_input(self):
        if self.done or self.eof:
            return
        self.eof = True
        if self.frame is not None:
            self.frame.end_input()
        # Init's complete sync is already present;all admitted work drains before frame EOF.

    def resolve_functional(self):
        if self.frame is None:
            raise RuntimeError("Only an actual frame payload barrier can resolve off-clock")
        return self.frame.resolve_functional()

    def abort(self, reason="cancelled"):
        if self.done:
            return None
        allowed = ("cancelled", "history_overflow", "retention_limit", "invalid_initial_fit")
        if reason not in allowed:
            raise ValueError("Need explicit selected-window failure")
        if self.frame is not None:
            return self.frame.abort(reason)
        self.stats["initial_discarded_words"] = self.history.live
        self.prepared.abort()
        if self.install is not None:
            self.install.abort()
        self.head_owner.abort()
        self._event = dict(
            frame=dict(
                status=reason,
                header_crc=None,
                data_crc=None,
                length_ok=None,
                payload=None,
                resource_reason=reason,
            ),
            symbols=np.empty(0, complex),
            input_domain="matched-filtered",
            sync_start_filtered=self.selection["start_tick"] / 8,
            initialized_window_statistics=self.stats.copy(),
            complete_latency_cycles=None,
            hardware_cycles_verified=False,
        )
        self.phase = "done"
        return self._event

    def take_mac_result(self):
        from nearlink_sdr.phy.mac_interface import mac_result_from_events

        if not self.done:
            raise RuntimeError("Frame has not reached a terminal gate")
        if self._mac_taken:
            return mac_result_from_events([])
        self._mac_taken = True
        return mac_result_from_events([self.event])

    def release_source(self):
        if not self.done:
            raise RuntimeError("Active selected window still owns its history")
        if self._source_returned:
            return None
        if self.history._initialized_owner() is not self:
            raise RuntimeError("Selected-window ownership token changed")
        self.history._initialized_owner = None
        self._source_returned = True
        return self.history

    def estimate(self):
        history = self.history.estimate() | dict(
            physical_banks=2,
            native_bits=73728,
            global_reads_per_clock=1,
            global_writes_per_clock=1,
            new_bank_decode=True,
            read_mux46=True,
            mapping="two RAMB36E1 SDP72x512;address-high-bank;one global1R1W",
            mux_LUT_FF_and_Fmax=None,
        )
        return dict(
            borrowed_history=history,
            initialization=self.prepared.estimate(),
            installer=(self.install or TimingContextInstallService()).estimate(),
            head_reserved_through_init=self.head_owner.header.cost(),
            wrapper_register_upper=3 * 64 + 3 * 64 + 2 + 4 + 2,
            unmapped_reference_and_descriptor_bits=None,
            upstream_confirmation_cycles=None,
            coherent_recheck="host precondition;selected-clock includes upstream work",
            selected_descriptor_and_reference_float_state="existing reference;not full fixed",
            frame=None if self.frame is None else self.frame.estimate(),
            duplicate_source_cache_head_or_kernel=False,
            inter_packet_kernel_or_head_sharing_credit=False,
            multi_profile_hardware_reservations_not_summed=True,
            static_DSP_no_unproved_sharing=19,
            DSP_parts=dict(
                shared_init_frame_interpolator=6,
                dedicated_initial_fit=1,
                dedicated_detector_PI_normalization=12,
            ),
            old18DSP_budget_passed=False,
            source_cache_head_BRAM36_minimum=(5 if self.profile.config.frame_type == 2 else 6),
            RRC_capture_and_payload_cycles=None,
            complete_latency_cycles=None,
            full_receiver16MSps=False,
            synthesis=False,
        )
