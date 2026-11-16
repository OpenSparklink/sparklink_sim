"""Bounded packet-incremental PSK receiver (§6.3/6.4).

FIR state survives blocks/packets. Each known-sync candidate starts fresh timing,
carrier and frame state. Local cubic interpolation needs two future filtered
samples; coherent payload carrier fitting/CRC wait for one bounded complete frame.
Caller chunk boundaries never determine internal processing quanta.
"""
from dataclasses import replace

import numpy as np

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_fields import psk_field_layout
from nearlink_sdr.phy.psk_sync import PSKCaptureConfig, _RustIntegerScreen, search_psk_sync
from nearlink_sdr.phy.rx_pipeline import payload_coded_length
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.timing import GardnerPIState, PSKTimingConfig

try:
    from nearlink_sdr_accel import StreamTiming as _RustStreamTiming
except ImportError:
    _RustStreamTiming = None

__all__ = ["PSKStreamReceiver"]


def _fir_block(taps, block, state):
    """One-dimensional FIR, a[0]=1, with transposed delay state.

    This is SciPy's FIR convolution/state update without generic axis, dtype or
    denominator handling on every short block. All callers own validated 1D
    float64/complex128 arrays and a nonempty block. Output/state retain the same
    arithmetic and finite memory, including blocks shorter than the filter.
    """
    full = np.convolve(taps, np.asarray(block, dtype=state.dtype))
    full[:len(state)] += state
    return full[:len(block)], full[len(block):]


def _cubic(values, origin, position):
    cell = int(np.floor(position - origin))
    u = position - origin - cell
    a, b, c, d = values[cell-1:cell+3]
    return (-u*(u-1)*(u-2)*a/6 + (u+1)*(u-1)*(u-2)*b/2
            - (u+1)*u*(u-2)*c/2 + (u+1)*u*(u-1)*d/6)


class PSKStreamReceiver:
    """Incremental strict PHY with fixed sample history and one bounded frame.

    ``feed`` emits candidate dictionaries, including failures and actual symbol
    trajectories; a valid frame contains a CRC-gated payload in ``event['frame']``.
    It emits as soon as a frame is complete, without waiting for EOF. ``finish``
    flushes centered-filter zero padding at the actual input end and reports an
    unfinished candidate as truncated. No recorded sample center is invented.
    """

    def __init__(self, profile: PSKReceiveProfile, quiet: np.ndarray, *,
                 timing_subdivisions: int = 8, energy_window_symbols: int = 8,
                 timing_config: PSKTimingConfig | None = None,
                 carrier_method: str = "coherent", quiet_batching: bool = True,
                 capture_config: PSKCaptureConfig | None = None,
                 timing_backend: str = "auto", defer_quiet_fir: bool = True,
                 integer_screen_backend: str = 'auto', adaptive_quiet: bool = True):
        if not isinstance(profile, PSKReceiveProfile):
            raise ValueError("Explicit PSKReceiveProfile required")
        self.profile = replace(profile, config=replace(profile.config))
        if timing_backend not in ('auto', 'python', 'rust'):
            raise ValueError("timing_backend must be auto, python or rust")
        if timing_backend == 'rust' and _RustStreamTiming is None:
            raise RuntimeError("Rust stream timing extension is not installed")
        self.timing_backend = ('rust' if timing_backend != 'python' and
                               _RustStreamTiming is not None else 'python')
        cfg = self.profile.config
        if (not isinstance(cfg.sps, int) or isinstance(cfg.sps, bool) or cfg.sps < 4
                or not isinstance(timing_subdivisions, int)
                or isinstance(timing_subdivisions, bool) or not 1 <= timing_subdivisions <= 32
                or not isinstance(energy_window_symbols, int)
                or not 1 <= energy_window_symbols <= 64):
            raise ValueError("Need integer SPS>=4, subdivisions1..32 and energy window1..64")
        if carrier_method not in ("coherent", "phase_regression"):
            raise ValueError("Unsupported carrier method")
        self.timing_config = timing_config or PSKTimingConfig()
        if not isinstance(self.timing_config, PSKTimingConfig):
            raise ValueError("Need PSKTimingConfig")
        self.sps, self.subdivisions = cfg.sps, timing_subdivisions
        if not isinstance(quiet_batching, bool):
            raise ValueError("quiet_batching must be boolean")
        self.quiet_batching = quiet_batching
        if not isinstance(adaptive_quiet, bool):
            raise ValueError('adaptive_quiet must be boolean')
        self.adaptive_quiet = adaptive_quiet
        if not isinstance(defer_quiet_fir, bool):
            raise ValueError("defer_quiet_fir must be boolean")
        self.defer_quiet_fir = defer_quiet_fir
        self.quiet_deferred_fir_samples = 0
        self.quiet_fast_path_samples = 0
        self.carrier_method = carrier_method
        self.capture_config = capture_config or PSKCaptureConfig()
        if not isinstance(self.capture_config, PSKCaptureConfig):
            raise ValueError("Need PSKCaptureConfig")
        if integer_screen_backend not in ('auto', 'python', 'rust'):
            raise ValueError('integer_screen_backend must be auto, python or rust')
        if integer_screen_backend == 'rust' and _RustIntegerScreen is None:
            raise RuntimeError('Rust integer sync screen is not installed')
        self.integer_screen_backend = ('rust' if integer_screen_backend != 'python' and
                                       _RustIntegerScreen is not None else 'python')
        if self.capture_config.integer_screen is None or self.subdivisions == 1:
            self.integer_screen_backend = 'disabled'
        self.reference = profile.sync_symbols()
        self.h = rrc_filter(.4, self.sps)
        self.delay = (len(self.h)-1)//2
        self.quantum = 8*self.sps
        self.quiet_batch_samples = 128*self.quantum
        self._quiet_probe_credit = self.quiet_batch_samples
        self._quiet_credit_limit = 2*self.quiet_batch_samples
        self._quiet_probe_after = 0
        self._energy_gate_open = False
        self.quiet_probe_calls = self.quiet_probe_samples = self.quiet_probe_state_samples = 0
        self.quiet_prefix_jumps = self.quiet_probe_vetoes = 0
        self.search_samples = max(64, len(self.reference)+32)*self.sps
        self.history_limit = self.search_samples + 4*self.quantum
        self.future_samples = self.delay+2
        self._fir_state = np.zeros(len(self.h)-1, dtype=complex)
        self._quiet_tail = None
        q = np.asarray(quiet)
        window = self.sps * energy_window_symbols
        if (q.ndim != 1 or not window <= len(q) <= 2**22 or len(q) < 65
                or not np.isfinite(q).all()):
            raise ValueError("Need finite quiet calibration of65..2**22 samples fitting energy window")
        self._box = np.ones(window)/window
        warmup = max(len(self.h)-1, window-1)
        retained = ((self.history_limit+warmup+self.quantum-1)//self.quantum)*self.quantum
        can_defer = (defer_quiet_fir and window <= self.quantum and
                     retained <= self.history_limit+len(self.h)-1+window-1)
        self._quiet_min_probe = retained+self.quantum*(1 if can_defer else 2)
        self.threshold = self.capture_config.energy_multiplier*float(np.quantile(
            np.convolve(abs(q)**2, self._box, "same"), .999))
        self._energy_state = np.zeros(window-1)
        self._energy = np.empty(0)
        self._filtered = np.empty(0, dtype=complex)
        self._origin = -self.delay
        self._pending = np.empty(0, dtype=complex)
        self.input_samples, self._processed = 0, 0
        self._skip_until = 0.
        self._active = None
        self._closed = False
        self.peak_history_samples = 0
        self.sync_searches = 0
        # Cumulative integers only; no per-window log or IQ accumulation.
        self.capture_stats = dict(energy_checks=0, energy_triggers=0,
                                  phase_hypotheses=0, scored_positions=0,
                                  integer_phase_hypotheses=0, integer_scored_positions=0,
                                  integer_screen_rejections=0, full_searches=0,
                                  computed_phase_hypotheses=0, reused_phase_hypotheses=0,
                                  computed_scored_positions=0, integer_evaluations=0,
                                  integer_evaluated_positions=0, integer_native_screens=0,
                                  integer_reference_fallbacks=0,
                                  correlation_qualifying_searches=0, maxima_needing_future=0,
                                  sync_candidates=0, header_crc_ok=0, header_crc_failed=0,
                                  data_crc_ok=0, data_crc_failed=0, truncated_header=0,
                                  truncated_data=0, invalid_metadata=0)
        self.max_frame_symbols = self._frame_bound()
        self._values = np.empty(self.max_frame_symbols, complex)
        self._centers = np.empty(self.max_frame_symbols)
        self._rates = np.empty(self.max_frame_symbols)
        self._errors = np.empty(self.max_frame_symbols)
        self._head_required = self._header_requirement()

    @property
    def _filtered(self):
        self._materialize_quiet()
        return self._filtered_values

    @_filtered.setter
    def _filtered(self, value):
        self._filtered_values = value

    def _materialize_quiet(self):
        """Rebuild one bounded proven-quiet tail in canonical arithmetic before use."""
        tail = self._quiet_tail
        if tail is None:
            return
        self._quiet_tail = None
        self._fir_state = np.zeros(len(self.h)-1, complex)
        self._energy_state = np.zeros(len(self._box)-1)
        filtered, energy = [], []
        for offset in range(0, len(tail), self.quantum):
            part = tail[offset:offset+self.quantum]
            y, self._fir_state = _fir_block(self.h, part, self._fir_state)
            e, self._energy_state = _fir_block(self._box, abs(part)**2, self._energy_state)
            filtered.append(y)
            energy.append(e)
        self._filtered_values = np.concatenate(filtered)[-self.history_limit:]
        self._energy = np.concatenate(energy)[-self.search_samples:]
        self._origin = self._processed-self.delay-len(self._filtered_values)
        self.peak_history_samples = max(self.peak_history_samples, self.buffered_samples)

    def _header_requirement(self):
        cfg = self.profile.config
        nbits = 64 if cfg.frame_type == 2 else 256
        bps = 1 if cfg.frame_type == 4 else 2
        interval = 16 if cfg.frame_type == 2 else 4
        positions, _ = psk_field_layout(nbits, bps, interval, cfg.pilot_interval == 0)
        return len(self.reference)+int(positions[-1])+1

    def _frame_bound(self):
        cfg = self.profile.config
        head_bits = 64 if cfg.frame_type == 2 else 256
        head_bps = 1 if cfg.frame_type == 4 else 2
        _, head_wire = psk_field_layout(head_bits, head_bps, 16 if cfg.frame_type == 2 else 4,
                                         cfg.pilot_interval == 0)
        largest = 0
        for mcs in self.profile.allowed_mcs:
            candidate = replace(cfg, mcs_index=mcs)
            nbits = payload_coded_length(self.profile.max_payload_bytes*8, candidate)
            bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3}[candidate.mod_str]
            _, size = psk_field_layout(nbits, bps, cfg.pilot_interval)
            largest = max(largest, size)
        return len(self.reference)+head_wire+largest

    @property
    def buffered_samples(self):
        if self._quiet_tail is not None:
            return len(self._quiet_tail)+len(self._pending)
        return len(self._filtered)+len(self._pending)+len(self._fir_state)+len(self._energy_state)

    @property
    def buffered_symbols(self):
        return 0 if self._active is None else self._active['count']

    def feed(self, samples: np.ndarray) -> list[dict]:
        """Convenience list of events; use iter_feed to bound caller-owned output as well."""
        return list(self.iter_feed(samples))

    def iter_feed(self, samples: np.ndarray):
        """Yield events while consuming IQ with bounded state/output; exhaust before next feed."""
        if self._closed:
            raise ValueError("Cannot feed after finish")
        samples = np.asarray(samples)
        if samples.ndim != 1:
            raise ValueError("Need one-dimensional IQ")
        self.input_samples += len(samples)
        pos = 0
        if len(self._pending):
            need = min(self.quantum-len(self._pending), len(samples))
            block = np.r_[self._pending, samples[:need]]
            self._pending = np.empty(0, complex)
            pos += need
            if len(block) < self.quantum:
                if not np.isfinite(block).all():
                    raise ValueError("Need finite IQ samples")
                self._pending = block
                return
            yield from self._process(block)
        while len(samples)-pos >= self.quantum:
            if self.quiet_batching and self._active is None:
                probe = 0
                if self.adaptive_quiet:
                    if (not self._energy_gate_open and self._processed >= self._quiet_probe_after
                            and self._quiet_probe_credit >= self._quiet_min_probe):
                        available = (len(samples)-pos)//self.quantum*self.quantum
                        credit = self._quiet_probe_credit//self.quantum*self.quantum
                        probe = min(self.quiet_batch_samples, available, credit)
                        if probe < self._quiet_min_probe:
                            probe = 0
                elif (self._processed % self.quiet_batch_samples == 0
                        and len(samples)-pos >= self.quiet_batch_samples):
                    probe = self.quiet_batch_samples
                if probe:
                    consumed = self._quiet_jump(samples[pos:pos+probe])
                    if consumed:
                        pos += consumed
                        continue
            block = samples[pos:pos+self.quantum]
            yield from self._process(block)
            pos += self.quantum
        if not np.isfinite(samples[pos:]).all():
            raise ValueError("Need finite IQ samples")
        self._pending = samples[pos:].astype(complex, copy=True)

    def finish(self) -> list[dict]:
        """Flush FIR lookahead once, then emit any unfinished frame as truncated."""
        if self._closed:
            return []
        events = []
        if len(self._pending):
            block, self._pending = self._pending, np.empty(0, complex)
            events.extend(self._process(block))
        remaining = self.future_samples
        while remaining:
            n = min(self.quantum, remaining)
            events.extend(self._process(np.zeros(n, complex)))
            remaining -= n
        if self._active is not None:
            a = self._active
            status = "truncated_header" if a['count'] < self._head_required else "truncated_data"
            row = decode_psk_frame(self._values[:a['count']], self.profile,
                                   carrier_method=self.carrier_method)
            row.update(status=status, payload=None)
            events.append(self._event(row, end_padding=True))
        self._closed = True
        return events

    def _quiet_jump(self, block):
        """Skip only provably inactive quanta; reconstruct their exact finite history.

        A vector energy check may veto batching, never synchronization. Its
        conservative margin excludes threshold-rounding ties. Replaying a tail
        longer than the FIR/energy memory in canonical quanta erases the zero
        initialization, preserving subsequent searches and NCO state independently
        of the caller's chunk size. The tail keeps each canonical convolution and
        state addition, but fuses history concatenation and known-inactive dispatch.
        No future input is awaited for this fast path.
        """
        if np.any(self._energy > self.threshold):
            self._energy_gate_open = True
            return 0
        if not np.isfinite(block).all():
            raise ValueError("Need finite IQ samples")
        self.quiet_probe_calls += 1
        self.quiet_probe_samples += len(block)
        self._quiet_probe_credit -= len(block)
        energy_state = self._energy_state
        if self._quiet_tail is not None:
            # For a box no longer than one quantum, its final delay state only
            # depends on that last quantum, regardless of earlier input/state.
            _, energy_state = _fir_block(self._box, abs(self._quiet_tail[-self.quantum:])**2,
                                          np.zeros(len(self._box)-1))
            self.quiet_probe_state_samples += self.quantum
        energy, _ = _fir_block(self._box, abs(block)**2, energy_state)
        real_dtype = block.real.dtype
        epsilon = np.finfo(real_dtype).eps if real_dtype.kind == 'f' else np.finfo(float).eps
        margin = max(1e-12, 64*len(self._box)*epsilon)
        uncertain = np.flatnonzero(energy > self.threshold*(1-margin))
        if len(uncertain):
            self.quiet_probe_vetoes += 1
            if not self.adaptive_quiet:
                return 0
            safe = int(uncertain[0])//self.quantum*self.quantum
            self._quiet_probe_after = self._processed+safe+self.quantum
            if safe < self._quiet_min_probe:
                return 0
            self.quiet_prefix_jumps += 1
            block = block[:safe]
        else:
            self._quiet_probe_after = self._processed+len(block)
        warmup = max(len(self.h)-1, len(self._box)-1)
        retained = ((self.history_limit+warmup+self.quantum-1)//self.quantum)*self.quantum
        cache_budget = self.history_limit+len(self.h)-1+len(self._box)-1
        if (self.defer_quiet_fir and len(self._box) <= self.quantum
                and retained <= cache_budget and retained < len(block)):
            # Replace existing history/state, rather than adding another cache.
            # Keep the original input dtype for exact abs()**2 energy arithmetic.
            self._quiet_tail = block[-retained:].copy()
            self._filtered_values = np.empty(0, complex)
            self._energy = np.empty(0)
            self._fir_state = np.empty(0, complex)
            self._energy_state = np.empty(0)
            self._processed += len(block)
            self._quiet_probe_credit = min(self._quiet_credit_limit,
                                            self._quiet_probe_credit+len(block))
            self._origin = self._processed-self.delay
            self.quiet_fast_path_samples += len(block)-retained
            self.quiet_deferred_fir_samples += len(block)
            self.capture_stats['energy_checks'] += 1
            self._energy_gate_open = False
            self.peak_history_samples = max(self.peak_history_samples, self.buffered_samples)
            return len(block)
        tail_size = ((self.history_limit+warmup+self.quantum-1)//self.quantum+1)*self.quantum
        if tail_size >= len(block):
            return 0
        skipped = len(block)-tail_size
        self._processed += skipped
        self.quiet_fast_path_samples += skipped
        self._fir_state.fill(0)
        self._energy_state.fill(0)
        filtered_parts, energy_parts = [], []
        for offset in range(skipped, len(block), self.quantum):
            part = block[offset:offset+self.quantum]
            filtered, self._fir_state = _fir_block(self.h, part, self._fir_state)
            local_energy, self._energy_state = _fir_block(
                self._box, abs(part)**2, self._energy_state)
            filtered_parts.append(filtered)
            energy_parts.append(local_energy)
        self._filtered = np.concatenate(filtered_parts)[-self.history_limit:]
        self._energy = np.concatenate(energy_parts)[-self.search_samples:]
        self._processed += tail_size
        self._quiet_probe_credit = min(self._quiet_credit_limit,
                                        self._quiet_probe_credit+len(block))
        self._origin = self._processed-self.delay-len(self._filtered)
        self.capture_stats['energy_checks'] += 1
        self._energy_gate_open = False
        if np.any(self._energy > self.threshold):
            raise RuntimeError("Quiet fast-path invariant failed")
        self.peak_history_samples = max(self.peak_history_samples, self.buffered_samples)
        return len(block)

    def _process(self, block):
        self._materialize_quiet()
        if not np.isfinite(block).all():
            raise ValueError("Need finite IQ samples")
        filtered, self._fir_state = _fir_block(self.h, block, self._fir_state)
        energy, self._energy_state = _fir_block(self._box, abs(block)**2, self._energy_state)
        self._filtered = np.concatenate((self._filtered, filtered))
        self._energy = np.concatenate((self._energy, energy))[-self.search_samples:]
        self._processed += len(block)
        if self._quiet_probe_credit < self._quiet_credit_limit:
            self._quiet_probe_credit = min(self._quiet_credit_limit,
                                            self._quiet_probe_credit+len(block))
        events = []
        if self._active is None:
            self._search()
        if self._active is not None:
            events.extend(self._track())
        if len(self._filtered) > self.history_limit:
            self._filtered = self._filtered[-self.history_limit:]
        self._origin = self._processed-self.delay-len(self._filtered)
        self.peak_history_samples = max(self.peak_history_samples, self.buffered_samples)
        return events

    def _search(self):
        if len(self._filtered) < self.search_samples:
            return
        self.capture_stats['energy_checks'] += 1
        self._energy_gate_open = bool(np.any(self._energy > self.threshold))
        if not self._energy_gate_open:
            return
        self.capture_stats['energy_triggers'] += 1
        z = self._filtered[-self.search_samples:]
        origin = self._origin+len(self._filtered)-len(z)
        self.sync_searches += 1
        diagnostic = {}
        found = search_psk_sync(z, self.sps, self.reference, self.subdivisions,
                                 min_start=max(1., self._skip_until-origin), lookahead=4*self.sps,
                                 diagnostics=diagnostic,
                                 integer_screen=self.capture_config.integer_screen,
                                 integer_backend=('python' if self.integer_screen_backend == 'disabled'
                                                  else self.integer_screen_backend))
        for key in ('phase_hypotheses', 'scored_positions', 'integer_phase_hypotheses',
                    'integer_scored_positions', 'computed_phase_hypotheses',
                    'reused_phase_hypotheses', 'computed_scored_positions', 'integer_evaluations',
                    'integer_evaluated_positions', 'integer_native_screens',
                    'integer_reference_fallbacks'):
            self.capture_stats[key] += diagnostic[key]
        self.capture_stats['integer_screen_rejections'] += int(
            diagnostic['disposition'] == 'integer_screen_rejected')
        self.capture_stats['full_searches'] += int(diagnostic['phase_hypotheses'] > 0)
        self.capture_stats['maxima_needing_future'] += int(
            diagnostic['disposition'] == 'maximum_needs_future')
        self.capture_stats['correlation_qualifying_searches'] += int(
            diagnostic['best_score'] is not None and diagnostic['best_score'] >= .85)
        if found is None or found['score'] < .85:
            return
        self.capture_stats['sync_candidates'] += 1
        start = origin+found['phase']+found['offset']*self.sps
        centers = start+np.arange(len(self.reference))*self.sps
        raw = np.array([_cubic(self._filtered, self._origin, t) for t in centers])
        t = np.arange(len(raw))
        corrected = raw*np.exp(-1j*found['delta']*t)
        slope, phase = np.polyfit(t, np.unwrap(np.angle(corrected*self.reference.conj())), 1)
        gain = max(float(np.mean(abs(corrected))), 1e-30)
        loop = GardnerPIState(self.sps, float(np.mean(abs(raw)**2)), self.timing_config)
        self._active = dict(start=start, next=start, previous=raw[0], previous_t=start,
                            count=0, end=None, header=None, loop=loop,
                            slope=slope+found['delta'], phase=phase, gain=gain, score=found['score'])
        self._active['native'] = None
        if self.timing_backend == 'rust':
            self._active['native'] = _RustStreamTiming(
                sps=float(self.sps), start=start, previous=(float(raw[0].real), float(raw[0].imag)),
                power=loop.power, kp=loop.kp, ki=loop.ki, lower=loop.lower, upper=loop.upper,
                phase=phase, slope=self._active['slope'], gain=gain, hold=len(self.reference))

    def _track_native_batch(self, a, latest):
        target = self._head_required if a['end'] is None else a['end']
        state = a['native'].advance(self._filtered, float(self._origin), float(latest), target,
                                    self._values, self._centers, self._rates, self._errors)
        count, next_t, real, imag, previous_t, power, omega, rate_events, clip_events = state
        a.update(count=count, next=next_t, previous=complex(real, imag), previous_t=previous_t)
        loop = a['loop']
        loop.power, loop.omega = power, omega
        loop.rate_limit_events, loop.error_clip_events = rate_events, clip_events

    def _track(self):
        a, events = self._active, []
        latest = min(self.input_samples-1, self._origin+len(self._filtered)-3)
        while a['next'] <= latest:
            count, t, loop = a['count'], a['next'], a['loop']
            if count >= self.max_frame_symbols:
                raise RuntimeError("Frame exceeded configured legal bound")
            if a['native'] is not None:
                self._track_native_batch(a, latest)
            else:
                current = _cubic(self._filtered, self._origin, t)
                mid = (_cubic(self._filtered, self._origin, (a['previous_t']+t)/2)
                       if count >= len(self.reference) else None)
                step, error = loop.advance(current, a['previous'], mid)
                self._values[count] = current*np.exp(-1j*(a['phase']+a['slope']*count))/a['gain']
                self._centers[count], self._rates[count], self._errors[count] = t, (loop.omega/self.sps-1)*1e6, error
                a.update(count=count+1, previous=current, previous_t=t, next=t+step)
            if a['count'] == self._head_required:
                row = decode_psk_frame(self._values[:a['count']], self.profile,
                                       carrier_method=self.carrier_method)
                self.capture_stats['header_crc_ok'] += int(row['header_crc'] is True)
                self.capture_stats['header_crc_failed'] += int(row['header_crc'] is False)
                if row['status'] != 'truncated_data':
                    events.append(self._event(row))
                    break
                a['end'], a['header'] = row['frame_end_symbol'], row
                if a['end'] > self.max_frame_symbols:
                    raise RuntimeError("Verified length exceeded frame bound")
            if a['end'] is not None and a['count'] == a['end']:
                row = decode_psk_frame(self._values[:a['count']], self.profile,
                                       carrier_method=self.carrier_method)
                events.append(self._event(row))
                break
        return events

    def _event(self, row, *, end_padding=False):
        a = self._active
        n = a['count']
        self.capture_stats['data_crc_ok'] += int(row.get('data_crc') is True)
        self.capture_stats['data_crc_failed'] += int(row.get('data_crc') is False)
        if row['status'] in ('truncated_header', 'truncated_data'):
            self.capture_stats[row['status']] += 1
        self.capture_stats['invalid_metadata'] += int(
            row['status'] in ('invalid_length', 'unsupported_mcs'))
        # Only metadata from the selected sync decides the next search interval.
        # CRC success never chooses a different phase, seed or sync answer.
        if a['end'] is None:
            self._skip_until = a['start']+len(self.reference)*self.sps
        else:
            self._skip_until = a['next']
        event = dict(frame=row, sync_start_raw=a['start'], sync_score=a['score'],
                     gain_codes=a['gain'],
                     sync_evm=float(np.sqrt(np.mean(abs(
                         self._values[:len(self.reference)]-self.reference)**2))),
                     interpolation="four-point local cubic Lagrange after causal RRC",
                     timing_backend=self.timing_backend,
                     integer_screen_backend=self.integer_screen_backend,
                     cfo_hz=a['slope']*self.profile.config.symbol_rate_mhz*1e6/(2*np.pi),
                     emitted_after_samples=min(self._processed, self.input_samples),
                     used_end_padding=end_padding or self._processed > self.input_samples,
                     symbols=self._values[:n].copy(), symbol_centers_raw=self._centers[:n].copy(),
                     timing_rate_ppm=self._rates[:n].copy(), timing_detector_error=self._errors[:n].copy(),
                     timing_loop=a['loop'].diagnostics(len(self.reference), self._rates[:n]))
        self._active = None
        return event
