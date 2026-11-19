"""Explicit experimental finite timing bridge;carrier/header/decoder still float."""

from nearlink_sdr.phy.interpolant_rx import InterpolantPSKReceiver, physical_interpolant
from nearlink_sdr.phy.timing_word import ONE, WordTimingState

__all__ = ["WordTimingPSKReceiver"]


class WordTimingPSKReceiver(InterpolantPSKReceiver):
    """Actual52bit words→finite timing→unchanged normalization and strict parser.

    Query descriptors use integer96/97bit coordinates,not binary64 positions.
    Float views remain diagnostic/compatibility bridges. No default promotion.
    """

    def __init__(self, profile, *, word_timing_format, **options):
        super().__init__(profile, **options)
        self.word_timing = WordTimingState(
            self.initial_fit["power"],
            options["start_tick"],
            len(self.reference),
            word_timing_format,
            self.timing_config,
        )
        self._active["loop"] = self.word_timing

    @property
    def word_queries(self):
        if self.done or self.boundary_kind is not None:
            return None
        return self.word_timing.queries

    @property
    def future_sample_end(self):
        return self.word_timing.required_end

    @property
    def sample_positions(self):
        if self.word_queries is None:
            return None
        state = self.word_timing
        return (
            state.next_q32 / ONE,
            None
            if state.count < state.hold_symbols
            else (state.previous_q32 + state.next_q32) / (2 * ONE),
        )

    @property
    def required_history_start(self):
        if self._active is None:
            return None
        return max(0, min(q["cell"] for q in self.word_timing.queries if q is not None) - 1)

    def push_symbol(self, current_word, midpoint_word, *, after_samples):
        if self.done or self.boundary_kind is not None:
            raise RuntimeError("Symbol input closed or awaiting strict header/payload gate")
        if after_samples < self.future_sample_end:
            raise ValueError("Actual interpolation future samples have not arrived")
        if self._active["count"] >= self.max_frame_symbols:
            raise RuntimeError("Frame exceeded configured legal bound")
        self.word_timing.validate_words(current_word, midpoint_word)
        # Validate before committing finite state or observation counters.
        if (
            type(after_samples) is not int
            or not self.input_samples <= after_samples <= self.sample_limit
        ):
            raise ValueError("Need monotonic actually observed source end")
        z = physical_interpolant(current_word)
        self.word_timing.advance_words(current_word, midpoint_word)
        self._observe(after_samples)
        self._store_selected_symbol(
            z,
            self.word_timing.next_q32 / ONE,
            self.word_timing.error_word / 2**self.word_timing.fmt.error_bits,
            (self.word_timing.omega / self.sps - 1) * 1e6,
        )

    def _event(self, row, *, end_padding=False):
        event = super()._event(row, end_padding=False)
        return self._tag(event)

    def _tag(self, event):
        event.update(
            word_timing_state=self.word_timing.snapshot(),
            timing_coordinates="unsigned64+fraction32;midpoint97bit sum",
            remaining_carrier_normalization_and_decoder_float=True,
            timing_node_cycle_model="TimingWordService;not executed by functional bridge",
        )
        return event

    def finish(self, *, reason="eof"):
        return [self._tag(e) for e in super().finish(reason=reason)]

    def reference_memory(self):
        result = super().reference_memory()
        result.pop("remaining_loop_carrier_and_decoder_float")
        return result | dict(
            timing_power_TED_PI_NCO_finite=True,
            remaining_carrier_normalization_and_decoder_float=True,
        )
