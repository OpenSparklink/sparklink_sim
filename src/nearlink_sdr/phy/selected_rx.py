"""Bounded already-filtered PSK candidate → real header/frame boundary (§6.3/6.9).

One supplied selection only. No capture search, expected payload, CRC-based
frequency/phase retry, RRC refiltering or fabricated EOF samples. This is a
floating reference port, not a complete fixed/cycle/hardware receiver.
"""

import numpy as np

from nearlink_sdr.phy.retained_history import RetainedSampleHistory
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.sync_coherent import coherent_sync_confirmation

__all__ = ["SelectedPSKReceiver"]


class SelectedPSKReceiver(PSKStreamReceiver):
    """Consume one selected candidate; stop at the strictly parsed frame end.

    Coordinates are filtered-sample indices relative to the supplied window.
    `selected_sync` is the ONE sync vector already chosen by the upstream fine
    search, including its original interpolation. `delta` is its differential
    CFO estimate, not a CRC-selected estimate. Tracking reuses the existing
    four-point/Gardner implementation. Caller retains unconsumed trailing input.
    """

    def __init__(
        self,
        profile,
        *,
        start_tick,
        delta,
        differential_score,
        selected_sync,
        sample_limit,
        coherent_threshold,
        timing_backend="auto",
        timing_config=None,
        carrier_method="coherent",
        history_capacity=None,
    ):
        if not isinstance(profile, PSKReceiveProfile):
            raise ValueError("Need an explicit PSKReceiveProfile")
        if profile.config.sps != 4 or profile.config.symbol_rate_mhz != 4:
            raise ValueError("Selected reference port currently validates 4Msym/SPS4 only")
        if (
            type(sample_limit) is not int
            or not 4 <= sample_limit <= 2**20
            or type(start_tick) is not int
            or not 0 <= start_tick < 8 * 2**20
            or isinstance(delta, (bool, np.bool_))
            or not np.isscalar(delta)
            or not np.isfinite(delta)
            or abs(delta) > np.pi
            or isinstance(differential_score, (bool, np.bool_))
            or not np.isscalar(differential_score)
            or not np.isfinite(differential_score)
            or not 0 <= differential_score <= 1
        ):
            raise ValueError("Need bounded selected eighth-tick, finite CFO/score and sample limit")
        sync = np.asarray(selected_sync, dtype=complex)
        if (
            sync.ndim != 1
            or len(sync) != len(profile.sync_symbols())
            or not np.isfinite(sync).all()
        ):
            raise ValueError("Need exactly the supplied selected sync vector")
        super().__init__(
            profile,
            np.zeros(65, complex),
            timing_backend=timing_backend,
            timing_config=timing_config,
            carrier_method=carrier_method,
            quiet_batching=False,
            adaptive_quiet=False,
            defer_quiet_fir=False,
        )
        self.sample_limit = sample_limit
        self._sample_history, self._retained = self._allocate_input_history(
            history_capacity, start_tick
        )
        self._selected_sync = sync.copy()
        self.selected_start = start_tick / 8
        self.selected_delta = float(delta)
        self.selected_score = float(differential_score)
        self._confirmation = None
        if coherent_threshold is not None:
            self._confirmation = coherent_sync_confirmation(
                sync, self.reference, delta, coherent_threshold
            )
        self._selection_accepted = (
            self._confirmation["accepted"]
            if self._confirmation is not None
            else differential_score >= 0.85
        )
        self._selection_started = self._terminal = False
        self.last_input_consumed = 0
        self.last_event = None
        self.header_decision_after_filtered_samples = None
        self._origin = 0 if self._sample_history is None else self._sample_history.origin
        self._filtered_values = self._retained[:0]

    def _allocate_input_history(self, history_capacity, start_tick):
        """Reference-storage hook;external interpolants need no private raw history."""
        history = (
            None
            if history_capacity is None
            else RetainedSampleHistory(history_capacity, max(0, start_tick // 8 - 1))
        )
        retained = np.empty(self.sample_limit, complex) if history is None else history.storage
        return history, retained

    @property
    def done(self):
        return self._terminal

    @property
    def header_result(self):
        """Actual strict header metadata, or None until validation has happened."""
        if self._active is not None and self._active["header"] is not None:
            return dict(self._active["header"])
        if self.last_event is not None:
            return dict(self.last_event["frame"])
        return None

    @property
    def frame_end_symbol(self):
        row = self.header_result
        return None if row is None else row.get("frame_end_symbol")

    def feed(self, samples):
        raise RuntimeError("Use feed_filtered; raw IQ must first pass the configured frontend")

    def iter_feed(self, samples):
        raise RuntimeError("Use feed_filtered; one explicit upstream selection is required")

    def _release_sample_history(self, event):
        if self._sample_history is not None:
            event["raw_history_at_termination"] = self._sample_history.diagnostics()
            self._sample_history.retire_before(self._sample_history.end)
            self._refresh_history()

    def _bare_event(self, status, reason=None):
        row = dict(
            status=status,
            header_crc=None,
            data_crc=None,
            length_ok=None,
            payload=None,
            profile=self.profile.name,
        )
        if reason is not None:
            row["resource_reason"] = reason
        event = dict(
            frame=row,
            sync_start_filtered=self.selected_start,
            symbol_centers_filtered=np.empty(0),
            symbols=np.empty(0, complex),
            selection_source="one upstream differential maximum; no retry",
            delivered_after_filtered_samples=self.input_samples,
            hardware_cycles_verified=False,
            input_domain="matched-filtered",
        )
        self._release_sample_history(event)
        self.last_event, self._terminal, self._closed = event, True, True
        return event

    def _event(self, row, *, end_padding=False):
        event = super()._event(row, end_padding=False)
        event["sync_start_filtered"] = event.pop("sync_start_raw")
        event["symbol_centers_filtered"] = event.pop("symbol_centers_raw")
        event["delivered_after_filtered_samples"] = event.pop("emitted_after_samples")
        event.update(
            input_domain="matched-filtered",
            hardware_cycles_verified=False,
            selection_source="one upstream differential maximum; no retry",
        )
        event["interpolation"] = "existing four-point local cubic; frontend already filtered"
        self._release_sample_history(event)
        self.last_event, self._terminal, self._closed = event, True, True
        return event

    def feed_filtered(self, samples):
        """Consume canonically one word at a time, independent of caller blocks.

        `last_input_consumed` names this call's consumed prefix. Once terminal,
        remaining samples belong to the caller; they are never buffered here.
        """
        if (
            not isinstance(samples, np.ndarray)
            or samples.ndim != 1
            or samples.dtype.kind not in "fiuc"
        ):
            raise ValueError("Need a numeric 1D already-filtered NumPy array")
        values = samples  # Borrow caller storage; do not allocate/cast the entire block.
        self.last_input_consumed = 0
        if self.done:
            return []
        if not self._selection_accepted:
            return [self._bare_event("sync_confirmation_rejected")]
        if self.selected_start < 1:
            return [self._bare_event("insufficient_sync_history", "one past sample required")]
        for value in values:
            if self.input_samples == self.sample_limit:
                return self.finish(reason="retention_limit")
            if not np.isfinite(value):
                raise ValueError("Need finite consumed filtered samples")
            if self._sample_history is None:
                self._retained[self.input_samples] = value
            elif not self._sample_history.append(self.input_samples, value):
                return self.finish(reason="history_overflow")
            self.input_samples += 1
            self._processed = self.input_samples
            self.last_input_consumed += 1
            if self._sample_history is None:
                self._filtered_values = self._retained[: self.input_samples]
            else:
                self._refresh_history()
            last_sync = self.selected_start + (len(self.reference) - 1) * self.sps
            if not self._selection_started and last_sync <= self.input_samples - 3:
                self._activate_selected(
                    self.selected_start,
                    self.selected_delta,
                    self.selected_score,
                    self._confirmation,
                )
                self._selection_started = True
            if self._active is not None:
                events = self._track()
                count = len(events[0]["symbols"]) if events else self._active["count"]
                if (
                    self.header_decision_after_filtered_samples is None
                    and count >= self._head_required
                ):
                    self.header_decision_after_filtered_samples = self.input_samples
                if events:
                    events[0]["header_decision_after_filtered_samples"] = (
                        self.header_decision_after_filtered_samples
                    )
                    return events
                self._retire_sample_history()
        return []

    def _refresh_history(self):
        self._origin = self._sample_history.origin
        self._filtered_values = self._sample_history.view

    def _retire_sample_history(self):
        if self._sample_history is None or self._active is None:
            return
        self._sample_history.retire_before(self.required_history_start)
        self._refresh_history()

    @property
    def required_history_start(self):
        """Earliest raw dependency of the next current/midpoint interpolation."""
        if self._active is None:
            return None
        a = self._active
        earliest = a["next"]
        if a["count"] >= len(self.reference):
            earliest = min(earliest, (a["previous_t"] + a["next"]) / 2)
        # Current and (when enabled) midpoint each use floor(t)-1..floor(t)+2.
        return max(0, int(np.floor(earliest)) - 1)

    def finish(self, *, reason="eof"):
        """Seal observed data only; retain EOF and storage exhaustion separately."""
        if reason not in ("eof", "retention_limit", "history_overflow"):
            raise ValueError("Need explicit eof, retention_limit or history_overflow")
        if self.done:
            return []
        if not self._selection_accepted:
            return [self._bare_event("sync_confirmation_rejected")]
        if self._active is None:
            status = "truncated_sync" if reason == "eof" else reason
            return [self._bare_event(status, None if reason == "eof" else reason)]
        row = decode_psk_frame(
            self._values[: self._active["count"]],
            self.profile,
            carrier_method=self.carrier_method,
            **self._carrier_weight_options,
        )
        if reason != "eof":
            row.update(
                parser_status=row["status"],
                status=reason,
                resource_reason=reason,
                payload=None,
            )
        return [self._event(row)]

    def reference_memory(self):
        """Persistent reference arrays, not a proposed FPGA bit-width allocation."""
        roots = {}
        for name, value in vars(self).items():
            if not isinstance(value, np.ndarray):
                continue
            owner = value
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            roots.setdefault(id(owner), dict(names=[], bytes=owner.nbytes))["names"].append(name)
        event_arrays = (
            0
            if self.last_event is None
            else sum(
                value.nbytes for value in self.last_event.values() if isinstance(value, np.ndarray)
            )
        )
        return dict(
            persistent_numpy_storage_bytes=sum(item["bytes"] for item in roots.values()),
            persistent_numpy_allocations=list(roots.values()),
            emitted_event_arrays_bytes=event_arrays,
            caller_arrays_and_python_object_overhead_included=False,
            raw_history=(
                None if self._sample_history is None else self._sample_history.diagnostics()
            ),
            retained_filtered_bytes=self._retained.nbytes,
            selected_sync_bytes=self._selected_sync.nbytes,
            legal_symbol_array_bytes=sum(
                a.nbytes for a in (self._values, self._centers, self._rates, self._errors)
            ),
            legal_symbol_bound=self.max_frame_symbols,
            raw_frontend_base_arrays_present_but_unused=True,
            decoder_FFT_Polar_temporaries_included=False,
            hardware_resource_or_cycles_verified=False,
        )
