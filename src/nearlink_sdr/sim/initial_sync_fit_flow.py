"""One interpolant broadcast to cache and finite fit;exclusive borrowed1R1W history.

This closes initial fitting only. Known-sync replay power updates,PI,carrier,
header and decoder service still have no cycle evidence and are not zero.
"""

from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT
from nearlink_sdr.sim.initial_interpolant_flow import InitialStencilService
from nearlink_sdr.sim.sync_fit_flow import SyncFitWordFlow

__all__ = ["InitialSyncFitService"]


class InitialSyncFitService:
    """Bounded source/interpolator/cache/fit credits;no second raw or cache pass."""

    def __init__(
        self, history, phase_codes, coarse_turn40, first_cell, phase, *, fmt=None, start_cycle=0
    ):
        self.fit = SyncFitWordFlow(phase_codes, coarse_turn40, fmt)
        self.stencil = InitialStencilService(
            history, INPUT_FORMAT, len(phase_codes), first_cell, phase, start_cycle=start_cycle
        )
        self.fit.cycle = start_cycle
        self.eof = False

    @property
    def cycle(self):
        return self.stencil.cycle

    @property
    def ready(self):
        return self.stencil.ready and self.fit.ready

    def end_input(self):
        """EOF cannot manufacture missing stencil samples;available work may drain."""
        if self.fit.output_taken:
            raise RuntimeError("Initial service ownership has transferred")
        self.eof = True
        if self.stencil.history.end < self.stencil.activation_end:
            return self.abort(reason="truncated_sync_future")
        return None

    def step(self, *, output_ready=True):
        if type(output_ready) is not bool:
            raise ValueError("Need explicit downstream fit readiness")
        if self.fit.cycle != self.cycle:
            raise RuntimeError("Fit and stencil clocks must remain on the same edge")
        if self.fit.output_taken:
            raise RuntimeError("Initial service ownership has transferred")
        committed = self.stencil.step(output_ready=output_ready and self.fit.input_ready)
        if committed is not None and not self.fit.issue(committed["word"]):
            raise RuntimeError("Broadcast cache/fit credit was not atomic")
        self.fit.advance()
        return committed

    def abort(self, *, reason="resource_overflow"):
        if self.fit.output_taken:
            raise RuntimeError("Initial service ownership has transferred")
        detail = self.stencil.abort()
        if detail is None:
            return None
        return dict(reason=reason, stencil=detail, fit=self.fit.abort())

    def estimate(self):
        stencil, fit = self.stencil.estimate(), self.fit.estimate()
        return dict(
            stencil=stencil,
            fit=fit,
            extra_wrapper_state_bits=1,  # EOF flag;clocks/control borrowed above.
            register_state_bits=stencil["incremental_control_bits"]
            + fit["register_state_upper_bits"]
            + 1,
            source_read_ports=1,
            source_write_ports=1,
            second_cache_or_source_pass=False,
            broadcast_word_bits=52,
            initial_fit_cycles="executable,actual integers;40/word plus bounded finalization",
            known_sync_replay_PI_carrier_head_decoder_cycles=None,
            full_receiver16MSps=False,
            synthesis=False,
        )
