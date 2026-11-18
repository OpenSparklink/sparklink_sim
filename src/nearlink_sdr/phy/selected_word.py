"""Opt-in real filter-word interpolation with existing float tracking/parser.

Only interpolation is fixed here. Initialization,Gardner/PI,carrier,head,LLR
and decoding remain float. This is not a complete word receiver or cycle proof.
"""

import math

import numpy as np

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    interpolate_filter_word,
    unpack_interpolated_word,
)
from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.sim.capture_fine import unpack_filter_word

__all__ = ["WordInterpolatedPSKReceiver"]


def physical_filter_word(word):
    r, i, exponent = unpack_filter_word(word)
    return complex(math.ldexp(r, exponent), math.ldexp(i, exponent))


class WordInterpolatedPSKReceiver(SelectedPSKReceiver):
    """One512-word stencil store;default and native float receiver stay unchanged.

    The inherited float history duplicates this word store for reference-control
    compatibility. Both allocations are reported;no such duplicate is claimed
    required or free on hardware. Causal input batches contain at most32 words.
    """

    def __init__(self, profile, *, word_format, **selection):
        if not isinstance(word_format, LagrangeWordFormat):
            raise ValueError("Need explicit interpolation word format")
        if selection.get("timing_backend", "python") != "python":
            raise ValueError("Word interpolation currently has Python timing only;no native ABI")
        if selection.get("history_capacity", 512) != 512:
            raise ValueError("Word interpolation currently validates512 history")
        selection.update(timing_backend="python", history_capacity=512)
        super().__init__(profile, **selection)
        self.word_format, self._words = word_format, [0] * 512
        self.interpolation_stats = dict(
            queries=0, aligned_components_erased=0, output_saturations=0, output_underflows=0
        )

    def feed_filtered(self, samples):
        raise RuntimeError("Use feed_words;do not silently quantize float input")

    def feed_words(self, words):
        """Accept actual unsigned46bit words and stop at the same strict frame gate."""
        if not isinstance(words, (list, tuple, np.ndarray)) or not 1 <= len(words) <= 32:
            raise ValueError("Need1..32 filter words")
        if isinstance(words, np.ndarray) and words.ndim != 1:
            raise ValueError("Need one-dimensional filter words")
        self.last_input_consumed = consumed = 0
        if not self.done and (not self._selection_accepted or self.selected_start < 1):
            return super().feed_filtered(np.empty(0, complex))
        for word in words:
            if self.done:
                break
            if self.input_samples == self.sample_limit:
                return self.finish(reason="retention_limit")
            history = self._sample_history
            if history.count == history.capacity:
                return self.finish(reason="history_overflow")
            if isinstance(word, (bool, np.bool_)):
                raise ValueError("Need filter words,not booleans")
            value = physical_filter_word(word)
            self._words[self.input_samples % 512] = int(word)
            events = super().feed_filtered(np.array([value]))
            consumed += self.last_input_consumed
            self.last_input_consumed = consumed
            if events:
                return events
        return []

    def _sample(self, position):
        cell = math.floor(position)
        history = self._sample_history
        if not history.origin <= cell - 1 or cell + 2 >= history.end:
            raise RuntimeError("Word stencil future/expired;no manufactured samples")
        phase = round((position - cell) * 2**self.word_format.phase_bits)
        words = tuple(self._words[k % 512] for k in range(cell - 1, cell + 3))
        result, info = interpolate_filter_word(words, phase, self.word_format)
        self.interpolation_stats["queries"] += 1
        for key in ("aligned_components_erased", "output_saturations"):
            self.interpolation_stats[key] += info[key]
        self.interpolation_stats["output_underflows"] += info["output_underflow"]
        if any(v >= 2**64 for v in self.interpolation_stats.values()):
            raise OverflowError("Word interpolation counters exhausted")
        r, i, exponent = unpack_interpolated_word(result, self.word_format)
        return complex(math.ldexp(r, exponent), math.ldexp(i, exponent))

    def _word_event(self, event):
        event["interpolation"] = "finite denominator6 Lagrange Horner;RNE phase and words"
        event["word_interpolation"] = dict(
            phase_bits=self.word_format.phase_bits,
            guard_bits=self.word_format.guard_bits,
            output_bits=self.word_format.output_bits,
            statistics=self.interpolation_stats.copy(),
            other_nodes_fixed=False,
            service_cycles_verified=False,
        )
        return event

    def _event(self, row, *, end_padding=False):
        return self._word_event(super()._event(row, end_padding=end_padding))

    def _bare_event(self, status, reason=None):
        return self._word_event(super()._bare_event(status, reason))

    def reference_memory(self):
        return super().reference_memory() | dict(
            additional_word_store_logical_bits=512 * 46,
            word_store_writes=self.input_samples,
            interpolation_quality_statistic_bits=4 * 64,
            Python_integer_object_overhead_included=False,
            word_interpolation_only=True,
            interpolation_stats=self.interpolation_stats.copy(),
            all_other_receiver_nodes_fixed=False,
            native_word_interpolation=False,
        )
