"""Actual selected cache→finite coherent gate→existing word-head/raw owner.

Only coherent confirmation is replaced. Timing,bootstrap,carrier and payload
remain explicit floating/off-clock work. Functional-word mode is a diagnostic
comparison,not zero-cycle hardware. No gate uses CRC or expected payload.
"""

import math

from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.phy.sync_fit_word import compile_sync_phases
from nearlink_sdr.sim.captured_header_flow import ClockedHeaderCapturedFrameReplay
from nearlink_sdr.sim.coherent_word_flow import CoherentWordService

__all__ = ["WordConfirmedCapturedFrameReplay"]


class _ConfirmedReceiver(SelectedPSKReceiver):
    """Install this transaction's actual finite receipt before any sample feed."""

    def __init__(self, *args, receipt, **options):
        peak = max(abs(v) for pair in receipt.selection.words for v in pair)
        if math.ldexp(peak, receipt.selection.exponent) > 1e70:
            raise OverflowError("Selected word bridge exceeds original float consumer range")
        super().__init__(*args, coherent_threshold=None, **options)
        self._confirmation = receipt.reference_confirmation()
        self._selection_accepted = receipt.accepted


class WordConfirmedCapturedFrameReplay(ClockedHeaderCapturedFrameReplay):
    """One selected-word service followed by the original strictly parsed frame.

    `clocked=False` resolves actual finite math off clock only to separate
    quantization/decision differences from the positive service latency.
    Default receivers are unchanged;this compiled experiment is explicit.
    """

    def __init__(self, profile, *, coherent_threshold, clocked=True, **options):
        if type(clocked) is not bool or coherent_threshold != 0.45:
            raise ValueError("Need compiled9/20 coherent threshold and explicit clock mode")
        super().__init__(profile, coherent_threshold=coherent_threshold, **options)
        self.codes = compile_sync_phases(self.profile.sync_symbols())
        self.clocked = clocked
        self.word_service = self.word_receipt = None
        self._word_waiting = False
        self.word_stats = dict(issued=0, completed=0, cancelled=0)

    def _count_word(self, name):
        if self.word_stats[name] == 2**64 - 1:
            raise OverflowError("Word confirmation statistic exhausted")
        self.word_stats[name] += 1

    @property
    def needs_word(self):
        return super().needs_word and not self._word_waiting

    def begin(self, selection, cycle):
        if not self._begin_selection(selection, cycle):
            return
        self.word_service = self._new_word_service(selection, cycle)
        self._count_word("issued")
        self._word_waiting = True
        if not self.clocked:
            while not self.word_service.ready:
                self.word_service.step()
            self._complete_word_gate()

    def _new_word_service(self, selection, cycle):
        return CoherentWordService(selection, self.codes, initial_cycle=cycle)

    def _word_service_cost(self):
        return CoherentWordService.estimate()

    def _complete_word_gate(self):
        if not self._word_waiting or not self.word_service.ready:
            raise RuntimeError("Word confirmation preceded actual finite completion")
        self.word_receipt = self.word_service.take()
        self._word_waiting = False
        self._count_word("completed")
        self._start_reference_consumer()

    def _make_receiver(self):
        if (
            self.word_receipt is None
            or self.word_receipt.selection is not self.selection
            or self.word_receipt.phase_codes != self.codes
        ):
            raise RuntimeError("Finite receipt belongs to another selected window/profile")
        return _ConfirmedReceiver(
            self.profile, **self.selection.reference_selection(),
            receipt=self.word_receipt, sample_limit=self.sample_limit,
            timing_backend="python", history_capacity=512,
        )

    def tick(self, cycle, *, blocked=False):
        if not self._word_waiting:
            return super().tick(cycle, blocked=blocked)
        self._edge(cycle)
        if type(blocked) is not bool:
            raise ValueError("Need explicit downstream readiness")
        while self.word_service.cycle < cycle:
            self.word_service.step()
        if self.word_service.ready and not blocked:
            self._complete_word_gate()

    def end_input(self, cycle):
        if self._word_waiting:
            raise RuntimeError("Drain actual word confirmation before selected EOF")
        return super().end_input(cycle)

    def abort(self, reason, cycle):
        if reason not in ("tail_history_expired", "source_epoch_aborted", "cancelled"):
            raise ValueError("Need explicit source/resource cancellation")
        if self.active and not self.done and self._word_waiting:
            if type(cycle) is not int or cycle <= self.word_service.cycle:
                raise ValueError("Word cancellation requires its next registered edge")
            while self.word_service.cycle < cycle - 1:
                self.word_service.step()
            self.word_service.abort()
            self._word_waiting = False
            self._count_word("cancelled")
        return super().abort(reason, cycle)

    def _terminal(self, event):
        service, receipt = self.word_service, self.word_receipt
        event["coherent_word_service"] = dict(
            actual_finite_values=receipt is not None, service_started=service is not None,
            finite_computation_completed=service is not None and service.ready_cycle is not None,
            clocked=self.clocked,
            computed_edges=None if service is None or service.ready_cycle is None
            else service.ready_cycle - service.started,
            elapsed_owned_edges=None if service is None else service.cycle - service.started,
            issued_cycle=None if service is None else service.started,
            ready_cycle=None if receipt is None else receipt.available_cycle,
            service_state=None if service is None else service.mode,
            statistics=None if service is None else dict(service.stats),
            fabric_statistics=None if service is None else dict(service.fabric.stats),
            word_quality=None if receipt is None else receipt.quality,
            word_decision=None if receipt is None else receipt.accepted,
            word_correlation=None if receipt is None else receipt.correlation,
            signal_energy=None if receipt is None else receipt.signal_energy,
            numerator=None if receipt is None else receipt.numerator,
            denominator=None if receipt is None else receipt.denominator,
            angle_turn40=None if receipt is None else receipt.angle_turn40,
            step_word20=None if receipt is None else receipt.step_word20,
            arithmetic_bridge_bootstrap_carrier_and_payload_cycles=None,
            functional_mode_is_not_zero_cycle_hardware=not self.clocked,
            complete_RX_cycle_evidence=False,
        )
        return super()._terminal(event)

    def release(self):
        super().release()
        self.word_service = self.word_receipt = None
        self._word_waiting = False

    def estimate(self):
        original = super().estimate()
        service = self._word_service_cost()
        controller = 3 * 64 + 2
        total = original["register_bits_upper"] + service["register_bits_upper"] + controller
        return original | dict(
            coherent_word_service=service,
            coherent_word_controller_bits=controller,
            register_fields=original["register_fields"] | dict(
                coherent_word_service=service["register_bits_upper"],
                coherent_word_controller=controller,
            ),
            combined_owner_register_bits=total,
            register_bits_upper=total,
            word_gate_control_node_bits=service["register_bits_upper"] + controller,
            word_gate_control_research_cap_bits=8192,
            no_new_IQ_tail_banks=True,
            finite_coherent_values_executed=True,
            finite_coherent_latency_charged=self.clocked,
            remaining_bootstrap_carrier_demap_payload_cycles=None,
            complete_latency_cycles=None,
            complete_cycle_evidence_valid=False,
        )
