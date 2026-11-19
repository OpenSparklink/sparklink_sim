"""Finite symbol rotation/gain → unchanged real header/length/CRC/MAC gates."""

from nearlink_sdr.phy.interpolant_rx import physical_interpolant
from nearlink_sdr.phy.normalization_word import (
    NormalizationFormat,
    WordNormalizationState,
    physical_normalized_word,
)
from nearlink_sdr.phy.timing_word import ONE
from nearlink_sdr.phy.timing_word_rx import WordTimingPSKReceiver

__all__ = ["NormalizedWordPSKReceiver"]


class NormalizedWordPSKReceiver(WordTimingPSKReceiver):
    """Explicit experiment; residual carrier/header/LLR/decoder still use the shared reference."""

    def __init__(self, profile, *, normalization_format, **options):
        if not isinstance(normalization_format, NormalizationFormat):
            raise ValueError("Need an explicit compiled symbol normalization structure")
        super().__init__(profile, **options)
        self.normalization = WordNormalizationState(self.initial_fit, normalization_format)
        self._normalization_claimed = False

    def claim_normalization_state(self):
        if self._normalization_claimed or self.done or self._active["count"]:
            raise RuntimeError("Need one normalization owner before the first symbol")
        self._normalization_claimed = True
        return self.normalization

    def _store_timed_symbol(self, current_word, after_samples):
        if self._normalization_claimed:
            raise RuntimeError("Actual normalization owner must supply its registered result")
        result = self.normalization.advance_word(current_word)
        self._store_normalized_word(current_word, result["normalized_word"], after_samples)

    def accept_normalized_timing_result(self, result, *, after_samples):
        """Copy actual55bit result once; never run timing,exp,gain division or NCO here."""
        t, n = self.word_timing, self.normalization
        if (not self._timing_claimed or not self._normalization_claimed
                or self.done or self.boundary_kind is not None):
            raise RuntimeError("Need both actual owners and an open selected symbol port")
        index = self._active["count"]
        if (result["index"] != index or t.count != index + 1 or n.count != index + 1
                or result["raw_word"] != t.previous_word
                or result["raw_word"] != n.last_source_word
                or result["normalized_word"] != n.last_normalized_word):
            raise RuntimeError("Normalized result is missing,duplicated or belongs to another word")
        required = (t.previous_q32 >> 32) + 3 + bool(t.previous_q32 & (ONE - 1))
        if (type(after_samples) is not int
                or not max(self.input_samples, required) <= after_samples <= self.sample_limit):
            raise ValueError("Need actually observed interpolation future samples")
        self._store_normalized_word(result["raw_word"], result["normalized_word"], after_samples)

    def _store_normalized_word(self, source_word, output_word, after_samples):
        self._observe(after_samples)
        self._store_normalized_symbol(
            physical_normalized_word(output_word), physical_interpolant(source_word),
            self.word_timing.next_q32 / ONE,
            self.word_timing.error_word / 2**self.word_timing.fmt.error_bits,
            (self.word_timing.omega / self.sps - 1) * 1e6,
        )

    def _tag(self, event):
        event = super()._tag(event)
        event.update(
            word_normalization_state=self.normalization.snapshot(),
            selected_symbol_rotation_and_gain_finite=True,
            normalization_output_word_bits=55,
            remaining_header_carrier_refinement_LLR_decoder_float=True,
            normalization_exp_division_recomputed_at_acceptance=False,
        )
        return event

    def reference_memory(self):
        return super().reference_memory() | dict(
            selected_symbol_rotation_and_gain_finite=True,
            remaining_header_carrier_refinement_LLR_decoder_float=True,
        )
