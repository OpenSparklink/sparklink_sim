"""Actual serial finite normalization after actual timing, then shared strict frame/MAC.

No unmodeled symbol exp/division is functionally resolved. Header/residual
carrier/LLR/decoder remain explicit barriers; full latency remains unknown.
"""

from copy import deepcopy

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.normalized_word_rx import NormalizedWordPSKReceiver
from nearlink_sdr.sim.normalization_word_flow import SymbolNormalizationService
from nearlink_sdr.sim.pipelined_timing_replay import PipelinedTimingFrameReplay

__all__ = ["NormalizedWordFrameReplay"]


class NormalizedWordFrameReplay(PipelinedTimingFrameReplay):
    """One extra4DSP bank; serial baseline, not an unproved shared/overlapped schedule."""

    def __init__(self, profile, prepared, *, normalization_format, **options):
        if not isinstance(normalization_format, NormalizationFormat):
            raise ValueError("Need compiled normalization before claiming prepared ownership")
        self.normalization_format = normalization_format
        self.normalization_service = None
        super().__init__(profile, prepared, **options)
        self.stats.update(dict.fromkeys(
            ["normalization_requests", "normalization_completions", "normalization_edges"], 0
        ))

    def _make_consumer(self, profile, initial_fit, word_timing_format, selection):
        return NormalizedWordPSKReceiver(
            profile, initial_fit=initial_fit, normalization_format=self.normalization_format,
            word_timing_format=word_timing_format, **selection,
        )

    def _make_timing_service(self, state):
        self.normalization_service = SymbolNormalizationService(
            self.consumer.claim_normalization_state()
        )
        return super()._make_timing_service(state)

    def step(self):
        if self.done:
            return
        service = self.normalization_service
        if service.cycle != self.cycle - self.service_epoch:
            raise RuntimeError("Normalization must advance on the same actual transport edges")
        if self.mode == "word_normalize":
            self._count("normalization_edges")
            result = service.take()
            if result is not None:
                self.consumer.accept_normalized_timing_result(
                    result, after_samples=self.history.end
                )
                self._observe_service("normalization", service, result)
                self._count("normalization_completions")
                self._count("normalization_operations")
                self._count("symbol_operations")
                self._plan()
            self.kernel.advance()
            self.timing_service.step()
            self.query_service.step()
        else:
            super().step()
            if (not self.done and self.pending_service is not None
                    and self.pending_service["kind"] == "symbol_normalization"):
                if not service.issue(self.words[0]):
                    raise RuntimeError("One actual normalization credit was not retired")
                self._count("normalization_requests")
                self.pending_service = None
                self.mode = "word_normalize"
        service.step()

    def resolve_functional(self):
        if (self.pending_service is not None
                and self.pending_service["kind"] == "symbol_normalization"):
            raise RuntimeError("Finite symbol normalization must execute actual service edges")
        return super().resolve_functional()

    def _close(self, event):
        service = self.normalization_service
        detail = dict(
            state=self.consumer.normalization.snapshot(),
            preparation_ready_cycle=service.preparation_ready_cycle,
            pending=deepcopy(service.job), pending_snapshot_is_host_only=True,
            cancelled_inflight=service.mode != "idle", actual_symbol_words_executed=True,
            residual_carrier_header_LLR_decoder_cycles=None, full_receiver16MSps=False,
        )
        if service.mode == "idle":
            service.release()
        else:
            service.abort()
        detail["statistics"] = service.stats.copy()
        super()._close(event)
        event["normalization_service"] = detail

    def estimate(self):
        result = super().estimate()
        # The three new64bit controller counters are included by the common estimator.
        return result | dict(
            normalization_service=self.normalization_service.estimate(self.normalization_format.mode),
            normalization_controller_increment_bits=3 * 64,
            normalization_uses_dedicated_DSP_bank=True,
            normalization_sharing_credit_with_other_modules=False,
            symbol_normalization_cycles_are_actual=True,
            residual_carrier_header_LLR_decoder_cycles=None,
            whole_receiver16MSps=False,
        )
