"""External actual interpolants→shared float tracking→strict frame gate (§6.3/6.9).

No raw store,interpolation,refit or answer search. Missing arithmetic service
cycles remain unknown. This is a causal value consumer,not a fixed PI receiver.
"""

import math

import numpy as np

from nearlink_sdr.phy.interpolation_word import unpack_interpolated_word
from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, fit_as_floats

__all__ = ["InterpolantPSKReceiver"]


def physical_interpolant(word):
    r, i, e = unpack_interpolated_word(word, INPUT_FORMAT)
    return complex(math.ldexp(r, e), math.ldexp(i, e))


class InterpolantPSKReceiver(SelectedPSKReceiver):
    """One confirmed selection and its finite fit;explicit symbol and parse operations."""

    def _allocate_input_history(self, history_capacity, start_tick):
        return None, np.empty(0, complex)  # Raw words stay in the external owned source bank.

    def __init__(self, profile, *, initial_fit, **selection):
        if selection.get("timing_backend", "python") != "python":
            raise ValueError("External interpolants currently use the shared Python float loop")
        selection["timing_backend"] = "python"
        super().__init__(profile, **selection)
        if not self._selection_accepted or self.selected_start < 1:
            raise ValueError("Need an already confirmed upstream selection with past history")
        if (
            not isinstance(initial_fit, dict)
            or initial_fit.get("valid") is not True
            or type(initial_fit.get("symbols")) is not int
            or initial_fit["symbols"] != len(self.reference)
        ):
            raise ValueError("Need a complete valid finite fit for the actual sync length")
        for key, lower, upper in [
            ("phase_turn40", 0, 2**40),
            ("frequency_turn40", -(2**41), 2**41),
        ]:
            if type(initial_fit.get(key)) is not int or not lower <= initial_fit[key] < upper:
                raise ValueError("Invalid finite phase/frequency result")
        for key in ["gain", "power"]:
            value = initial_fit.get(key)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or any(type(v) is not int for v in value)
                or not 0 < value[0] < 2**54
                or not -512 <= value[1] < 512
            ):
                raise ValueError("Need bounded positive54+signed10 initialization")
        values = fit_as_floats(initial_fit)
        if not np.isfinite(list(values.values())).all() or values["gain"] <= 0:
            raise ValueError("Invalid float bridge from finite initialization")
        self.initial_fit = initial_fit.copy()
        # Previous is unused while midpoint=None;symbol0 installs the actual value.
        self._start_selected_state(
            self.selected_start,
            values["phase"],
            values["slope"],
            values["gain"],
            values["power"],
            0j,
            self.selected_score,
            self._confirmation,
        )
        self._selection_started = True

    def feed_filtered(self, samples):
        raise RuntimeError("Use push_symbol/parse_boundary;no private raw history")

    @property
    def boundary_kind(self):
        if self.done:
            return None
        a = self._active
        if a["count"] == self._head_required and a["header"] is None:
            return "header"
        if a["end"] is not None and a["count"] == a["end"]:
            return "payload"
        return None

    @property
    def sample_positions(self):
        if self.done or self.boundary_kind is not None:
            return None
        a = self._active
        return (
            a["next"],
            None if a["count"] < len(self.reference) else (a["previous_t"] + a["next"]) / 2,
        )

    def _observe(self, after_samples):
        if (
            type(after_samples) is not int
            or not self.input_samples <= after_samples <= self.sample_limit
        ):
            raise ValueError("Need monotonic actually observed source end")
        self.input_samples = self._processed = after_samples

    def push_symbol(self, current_word, midpoint_word, *, after_samples):
        if self.done or self.boundary_kind is not None:
            raise RuntimeError("Symbol input is closed or awaiting the strict frame gate")
        current, midpoint = self.sample_positions
        if (midpoint is None) != (midpoint_word is None):
            raise ValueError("Midpoint belongs only to tracked data symbols")
        if self._active["count"] >= self.max_frame_symbols:
            raise RuntimeError("Frame exceeded configured legal bound")
        if after_samples < math.ceil(current + 3):
            raise ValueError("Interpolation future samples have not actually arrived")
        z = physical_interpolant(current_word)
        mid = None if midpoint_word is None else physical_interpolant(midpoint_word)
        self._observe(after_samples)
        self._accept_selected_symbol(z, mid)

    def parse_boundary(self, *, after_samples):
        if self.boundary_kind is None:
            raise RuntimeError("No complete header or payload to parse")
        self._observe(after_samples)
        if self.header_decision_after_filtered_samples is None:
            self.header_decision_after_filtered_samples = after_samples
        event = self._check_selected_boundary()
        if event is not None:
            event["header_decision_after_filtered_samples"] = (
                self.header_decision_after_filtered_samples
            )
        return event

    def _event(self, row, *, end_padding=False):
        event = super()._event(row, end_padding=False)
        event.update(
            input_domain="actual52bit interpolants",
            initialization=self.initial_fit.copy(),
            all_sync_power_updates_preserved=True,
            arithmetic_service_cycles=None,
            full_receiver16MSps=False,
        )
        return event

    def finish(self, *, reason="eof"):
        if reason not in ("eof", "retention_limit", "history_overflow"):
            raise ValueError("Need explicit EOF or resource termination")
        if not self.done and self._active is not None and self._active["count"] == 0:
            event = self._bare_event("truncated_sync" if reason == "eof" else reason)
            event.update(
                input_domain="actual52bit interpolants",
                arithmetic_service_cycles=None,
                initialization=self.initial_fit.copy(),
                sync_symbols_consumed=0,
            )
            return [event]
        return super().finish(reason=reason)

    def reference_memory(self):
        return super().reference_memory() | dict(
            private_raw_samples=0,
            interpolation_or_initial_fit_recomputed=False,
            remaining_loop_carrier_and_decoder_float=True,
        )
