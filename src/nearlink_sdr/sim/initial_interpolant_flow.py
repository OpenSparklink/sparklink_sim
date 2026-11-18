"""Registered sync-cache and actual1R1W source→interpolator transfer (§6.2.1.2).

This node stops at cache-ready. Initial fit,PI,carrier,header and decoder service
remain unknown. A continuous selected-window input is not a whole ADC/RX model.
"""

from nearlink_sdr.phy.initial_interpolant import InitialInterpolantCache
from nearlink_sdr.sim.interpolation_flow import LagrangeInterpolationFlow
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

__all__ = ["InitialInterpolantPort", "InitialStencilService"]


class InitialInterpolantPort:
    """One cache,registered1 read,one held reply;serialized fill/replay phases."""

    def __init__(self, fmt, symbols):
        self.cache = InitialInterpolantCache(fmt, symbols)
        self.cycle = self.last_write = self.last_read = -1
        self.pending = None
        self.aborted = False

    def _edge(self, cycle):
        if type(cycle) is not int or not 0 <= cycle < 2**63 - 1 or cycle < self.cycle:
            raise ValueError("Need monotonic signed64 cache clock")
        self.cycle = cycle

    def write(self, cell, phase, word, cycle):
        self._edge(cycle)
        if self.last_write == cycle:
            return False
        self.cache.store(cell, phase, word)
        self.last_write = cycle
        return True

    def request(self, cell, phase, cycle):
        self._edge(cycle)
        if self.pending is not None or cycle in (self.last_write, self.last_read):
            return False
        index = self.cache.read
        word = self.cache.reuse(cell, phase)
        self.pending = index, word, cycle + 1
        self.last_read = cycle
        return True

    def take(self, cycle):
        self._edge(cycle)
        if self.pending is None or cycle < self.pending[2]:
            return None
        index, word, _ = self.pending
        self.pending = None
        return index, word

    def abort(self, cycle):
        self._edge(cycle)
        self.pending = None
        self.cache.release()
        self.aborted = True

    def estimate(self):
        bits = 2 * self.cache.fmt.output_bits + 8
        # clock/last-write/last-request64 each;held index7/word/ready64/valid1;abort1.
        register_bits = 3 * 64 + 7 + bits + 64 + 1 + 1
        return self.cache.estimate() | dict(
            registered_read_latency=1,
            held_response_words=1,
            registered_response_and_clock_bits=register_bits,
            complete_controller_bits=self.cache.estimate()["controller_bits"] + register_bits,
            physical_registered_response_and_clock_state_included=True,
        )


class InitialStencilService:
    """Borrow one exclusive history bank;four-word staging,seven kernel contexts.

    Initialization waits for the same last-sync future-sample condition. Each
    accepted kernel input then owns its four raw words. Reuse permits retiring
    these source words,except the final two needed by the first data midpoint.
    No fit completion or packet acceptance is fabricated by this service.
    """

    def __init__(
        self, history, fmt, symbols, first_cell, phase, *, start_cycle=0, retire_after_staging=True
    ):
        if not isinstance(history, ReceiveHistoryPort) or history.pending is not None:
            raise ValueError("Need exclusive source history without another pending reply")
        self.output = InitialInterpolantPort(fmt, symbols)
        if type(first_cell) is not int or not history.head < first_cell < 2**63 - 4 * symbols:
            raise ValueError("Need available nonnegative stencil origin and bounded progression")
        self.output.cache._key(first_cell, phase, 0)
        if type(start_cycle) is not int or not 0 <= start_cycle < 2**63 - 1:
            raise ValueError("Need signed64 service start clock")
        if type(retire_after_staging) is not bool:
            raise ValueError("Need explicit retirement policy")
        self.history, self.first_cell, self.phase = history, first_cell, phase
        self.interpolator = LagrangeInterpolationFlow(fmt)
        self.interpolator.cycle = start_cycle
        self.symbols, self.retire_after_staging = symbols, retire_after_staging
        self.staged, self.offered = [], 0
        self.enabled = self.aborted = False
        self.ready_cycle = None
        self.aborted_inflight = 0
        self.future_wait_cycles = 0
        self.activation_end = first_cell + 4 * (symbols - 1) + 3 + int(phase > 0)

    @property
    def cycle(self):
        return self.interpolator.cycle

    @property
    def ready(self):
        return self.ready_cycle is not None and not self.aborted

    def step(self, *, output_ready=True):
        if type(output_ready) is not bool:
            raise ValueError("Need explicit cache write readiness")
        if self.aborted:
            raise RuntimeError("Aborted candidate cannot resume or emit stale words")
        committed = None
        cycle, source = self.cycle, self.history
        if not self.enabled and source.end >= self.activation_end:
            if source.head > self.first_cell - 1:
                raise RuntimeError("Initial stencil expired before service activation")
            source.retire_before(self.first_cell - 1, cycle)
            self.enabled = True
        if not self.enabled:
            if self.future_wait_cycles == 2**64 - 1:
                raise OverflowError("Initial future-wait counter exhausted")
            self.future_wait_cycles += 1
        if self.enabled:
            reply = source.take(cycle)
            if reply is not None:
                expected = self.first_cell - 1 + 4 * self.offered + len(self.staged)
                if reply[0] != expected or len(self.staged) == 4:
                    raise RuntimeError("Registered source reply changed index or staging credit")
                self.staged.append(reply[1])
            if len(self.staged) == 4 and self.interpolator.issue(
                tuple(self.staged), self.phase, tag=self.offered
            ):
                self.staged.clear()
                if self.retire_after_staging:
                    boundary = self.first_cell - 1 + 4 * (self.offered + 1)
                    if self.offered == self.symbols - 1:
                        boundary -= 2
                    source.retire_before(boundary, cycle)
                self.offered += 1
            if self.offered < self.symbols and len(self.staged) < 4:
                index = self.first_cell - 1 + 4 * self.offered + len(self.staged)
                if source.pending is None and index < source.end:
                    source.request(index, cycle)
        if output_ready:
            result = self.interpolator.take()
            if result is not None:
                index = self.output.cache.written
                if result["tag"] != index:
                    raise RuntimeError("Interpolation reply lost initial-sync order")
                if not self.output.write(
                    self.first_cell + 4 * index, self.phase, result["word"], cycle
                ):
                    raise RuntimeError("Cache write credit was consumed twice")
                if self.output.cache.written == self.symbols:
                    self.ready_cycle = cycle
                committed = result | {"cache_write_cycle": cycle}
        self.interpolator.advance()
        return committed

    def abort(self):
        """Explicit candidate invalidation,not silent ADC continuation or packet delivery."""
        if self.aborted:
            return None
        self.aborted_inflight = self.interpolator.live
        disposition = dict(
            interpolation_contexts=self.aborted_inflight,
            source_reply=int(self.history.pending is not None),
            staged_words=len(self.staged),
            cache_reply=int(self.output.pending is not None),
        )
        self.interpolator.stages = [None] * 7
        self.staged.clear()
        self.history.pending = None  # Exclusive borrowed port belongs to this candidate.
        self.history.retire_before(self.history.end, self.cycle)
        self.output.abort(self.cycle)
        self.aborted = True
        return disposition  # Host diagnostic;no unbounded hardware event buffer.

    def estimate(self):
        phase = self.output.cache.fmt.phase_bits + 1
        # firstcell64,phase,offered7,staging count3/payload184,enable/abort2,
        # ready timestamp64+valid1,aborted in-flight count4. Activation bound is combinational.
        # Include the service's own length7/config1 and future-wait diagnostic64.
        walker = 64 + phase + 7 + 3 + 184 + 2 + 64 + 1 + 4 + 7 + 1 + 64
        cache = self.output.estimate()
        return dict(
            borrowed_source=self.history.estimate(),
            interpolation=self.interpolator.estimate(),
            cache=cache,
            source_staging_words=4,
            source_staging_bits=184,
            walker_control_and_staging_bits=walker,
            incremental_control_bits=walker + cache["complete_controller_bits"],
            source_read_ports=1,
            source_write_ports=1,
            cache_ports_fill1_replay1=True,
            source_reads_per_known_sync=4 * self.symbols,
            retained_raw_tail_words_for_first_midpoint=2,
            source_address_add64_paths=2,
            source_index_add_sub10_paths=3,
            controller_comparison_paths_upper=12,
            controller_LUT_and_timing=None,
            initial_fit_cycles=None,
            PI_carrier_head_decoder_cycles=None,
            full_receiver16msps=False,
            selected_window_input_is_not_whole_ADC=True,
            synthesis=False,
        )
