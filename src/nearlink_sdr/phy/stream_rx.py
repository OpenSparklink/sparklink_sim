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
                 integer_screen_backend: str = 'auto', adaptive_quiet: bool = True,
                 correlation_backend: str = 'reference', correlation_input_full_scale: float = 1024.,
                 correlation_input_bits: int = 16, tone_notch=None,
                 correlation_mantissa_bits: int = 10,
                 sync_search_method: str = 'reference', normalized_coarse: bool = False,
                 rrc_backend: str = 'reference', carrier_weight_kappa: int | None = None,
                 carrier_weight_fixed: bool = False, sync_candidate_count: int = 1):
        if not isinstance(profile, PSKReceiveProfile):
            raise ValueError("Explicit PSKReceiveProfile required")
        self.profile = replace(profile, config=replace(profile.config))
        self.tone_notch = None
        if tone_notch is not None:
            from nearlink_sdr.phy.tone_notch import CausalToneNotch, ToneNotchConfig
            if not isinstance(tone_notch, ToneNotchConfig) or quiet_batching:
                raise ValueError("Tone notch requires config and quiet_batching=False")
            self.tone_notch = CausalToneNotch(tone_notch, profile.config.sps)
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
        if carrier_weight_kappa is not None:
            from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights
            BoundedCarrierWeights(carrier_weight_kappa, fixed=carrier_weight_fixed)
            if carrier_method != "coherent":
                raise ValueError("Experimental weights require coherent carrier search")
        elif type(carrier_weight_fixed) is not bool or carrier_weight_fixed:
            raise ValueError("Fixed carrier weights require explicit kappa")
        self._carrier_weight_options = ({} if carrier_weight_kappa is None else
                                       dict(carrier_weight_kappa=carrier_weight_kappa,
                                            carrier_weight_fixed=carrier_weight_fixed))
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
        if rrc_backend not in ('reference', 'word-full12-c18'):
            raise ValueError('rrc_backend must be reference or word-full12-c18')
        if rrc_backend != 'reference' and (
                quiet_batching or defer_quiet_fir or cfg.sps != 4 or cfg.symbol_rate_mhz != 4):
            raise ValueError('Word RRC requires SPS4/4Msym, quiet_batching=False and '
                             'defer_quiet_fir=False')
        self.rrc_backend = rrc_backend
        self._rrc_word_model = None
        if rrc_backend != 'reference':
            from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
            from nearlink_sdr.phy.rrc_word import FixedRRCFormat
            self._rrc_word_model = FullMantissaRRC(FixedRRCFormat(12, 18))
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
        if correlation_backend not in ('reference', 'structure', 'fixed', 'scaled'):
            raise ValueError('correlation_backend must be reference, structure, fixed or scaled')
        self.correlation_backend = correlation_backend
        if sync_search_method not in ('reference','local-parabolic','bounded-grid'):
            raise ValueError('Unsupported explicit sync search method')
        if (sync_search_method != 'reference' and
                (correlation_backend == 'reference' or self.sps != 4 or self.subdivisions != 8)):
            raise ValueError('Local sync needs an explicit SPS4/subdivision8 causal bank')
        if (type(normalized_coarse) is not bool or
                (normalized_coarse and (correlation_backend != 'scaled' or
                                       sync_search_method not in ('local-parabolic','bounded-grid')))):
            raise ValueError('Q12 ranking needs the scaled local sync path')
        if (type(sync_candidate_count) is not int or sync_candidate_count not in (1,2) or
                (sync_candidate_count != 1 and sync_search_method != 'bounded-grid')):
            raise ValueError('Candidate count1/2 requires explicit bounded-grid search')
        self.sync_search_method, self.normalized_coarse = sync_search_method,normalized_coarse
        self.sync_candidate_count = sync_candidate_count
        self.correlation_bank = None
        if correlation_backend != 'reference':
            from nearlink_sdr.phy.acquisition_bank import DifferentialBank
            if quiet_batching or self.subdivisions == 1 or not self.capture_config.integer_screen:
                raise ValueError('Structure bank requires quiet_batching=False and integer screen')
            bank_class, options = DifferentialBank, {}
            if correlation_backend == 'fixed':
                from nearlink_sdr.phy.acquisition_fixed import (
                    FixedBankFormat,
                    FixedDifferentialBank,
                )
                bank_class = FixedDifferentialBank
                options['format'] = FixedBankFormat(input_bits=correlation_input_bits)
            elif correlation_backend == 'scaled':
                from nearlink_sdr.phy.acquisition_scaled import (
                    ScaledBankFormat,
                    ScaledDifferentialBank,
                )
                bank_class = ScaledDifferentialBank
                options['scaled_format'] = ScaledBankFormat(correlation_mantissa_bits)
                if normalized_coarse:
                    from nearlink_sdr.phy.coarse_rank import RankedCoarseBank
                    bank_class = RankedCoarseBank
            self.correlation_bank = bank_class(
                self.reference, self.sps, self.search_samples, self.capture_config.integer_screen,
                origin=-self.delay, input_full_scale=correlation_input_full_scale, **options)
            self.integer_screen_backend = correlation_backend
            if normalized_coarse:
                self.integer_screen_backend += '-q12'
        self.tone_correlation_bank = None
        self._tone_filtered = np.empty(0, complex)
        if self.tone_notch is not None:
            if not self.capture_config.integer_screen or self.subdivisions == 1:
                raise ValueError('Tone fallback requires an enabled integer screen and fine search')
            from nearlink_sdr.phy.acquisition_bank import DifferentialBank
            tone_bank_class, tone_options = DifferentialBank, {}
            if correlation_backend == 'fixed':
                tone_bank_class = FixedDifferentialBank
                tone_options['format'] = FixedBankFormat(correlation_input_bits)
            elif correlation_backend == 'scaled':
                tone_bank_class = bank_class
                tone_options['scaled_format'] = ScaledBankFormat(correlation_mantissa_bits)
            self.tone_correlation_bank = tone_bank_class(
                self.reference, self.sps, self.search_samples, self.capture_config.integer_screen,
                origin=-self.delay, input_full_scale=correlation_input_full_scale, **tone_options)
        self.future_samples = self.delay+2
        self._fir_state = np.zeros(0 if self._rrc_word_model is not None else len(self.h)-1,
                                   dtype=complex)
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
        if self.tone_notch is not None:
            self.capture_stats.update(tone_fallback_searches=0, tone_candidates=0)
        if self.sync_search_method != 'reference':
            self.capture_stats.update(coarse_ranking_recomputed_positions=0)
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
        if self._rrc_word_model is not None:
            raise RuntimeError('Word RRC cannot reconstruct deferred floating FIR state')
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
    def fir_array_bytes(self):
        """Persistent NumPy FIR arrays only; Python objects/other RX caches separate."""
        total = self.h.nbytes + self._fir_state.nbytes
        word = self._rrc_word_model
        if word is not None:
            total += sum(a.nbytes for a in (word._r, word._i, word._e, word.coefficients))
        return total

    @property
    def buffered_samples(self):
        if self._quiet_tail is not None:
            return len(self._quiet_tail)+len(self._pending)
        fir_history = (len(self._fir_state) if self._rrc_word_model is None
                       else len(self._rrc_word_model._r))
        return (len(self._filtered)+len(self._pending)+fir_history+
                len(self._energy_state)+len(self._tone_filtered))

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
            if self._rrc_word_model is not None:
                self._rrc_word_model.stats['zero_flush_samples'] += n
            remaining -= n
        if self._active is not None:
            a = self._active
            status = "truncated_header" if a['count'] < self._head_required else "truncated_data"
            row = decode_psk_frame(self._values[:a['count']], self.profile,
                                   carrier_method=self.carrier_method,
                                   **self._carrier_weight_options)
            row.update(status=status, payload=None)
            events.append(self._event(row, end_padding=True))
        self._closed = True
        if self._rrc_word_model is not None:
            self._rrc_word_model.seal()
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
        if self._rrc_word_model is None:
            filtered, self._fir_state = _fir_block(self.h, block, self._fir_state)
        else:
            filtered = self._rrc_word_model.feed(block)
        if self.tone_notch is not None:
            tone = self.tone_notch.advance(filtered)
            self.tone_correlation_bank.advance(tone)
            self._tone_filtered = np.concatenate((self._tone_filtered, tone))
        if self.correlation_bank is not None:
            self.correlation_bank.advance(filtered)
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
            raw = self._filtered
            if self._active.get('tone_path'):
                self._filtered = self._tone_filtered
            events.extend(self._track())
            self._filtered = raw
        if len(self._filtered) > self.history_limit:
            self._filtered = self._filtered[-self.history_limit:]
        self._tone_filtered = self._tone_filtered[-self.history_limit:]
        self._origin = self._processed-self.delay-len(self._filtered)
        self.peak_history_samples = max(self.peak_history_samples, self.buffered_samples)
        return events

    def _search(self):
        rejected = self.capture_stats['integer_screen_rejections']
        self._search_path()
        if (self._active is not None or self.tone_notch is None or not self.tone_notch.active or
                self.capture_stats['integer_screen_rejections'] == rejected):
            return
        # Raw path has strict priority. A CRC result can never select a path or retry.
        raw, bank, backend = self._filtered, self.correlation_bank, self.integer_screen_backend
        self._filtered, self.correlation_bank = self._tone_filtered, self.tone_correlation_bank
        self.integer_screen_backend = ('tone-'+self.correlation_backend+
                                       ('-q12' if self.normalized_coarse else ''))
        self.capture_stats['tone_fallback_searches'] += 1
        try:
            self._search_path()
            if self._active is not None:
                self._active['tone_path'] = True
                self._active['screen_backend'] = self.integer_screen_backend
                self.capture_stats['tone_candidates'] += 1
        finally:
            self._filtered, self.correlation_bank = raw, bank
            self.integer_screen_backend = backend

    def _search_path(self):
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
        lower = max(1., self._skip_until-origin)
        enabled, queried, coarse_start, coarse_starts = True, 0, None, None
        if self.correlation_bank is not None:
            bounds = (origin+lower,origin+len(z)-1-(len(self.reference)-1)*self.sps)
            if self.normalized_coarse and self.sync_search_method == 'bounded-grid':
                enabled,queried,peaks = self.correlation_bank.query_peaks(
                    *bounds,count=self.sync_candidate_count,phase_origin=origin)
                coarse_starts = [peak-origin for peak in peaks]
                if enabled and not coarse_starts:
                    raise RuntimeError('Enabled normalized screen has no ranked peaks')
            elif self.normalized_coarse:
                enabled,queried,peak = self.correlation_bank.query_best(*bounds,phase_origin=origin)
                if enabled and peak is None:
                    raise RuntimeError('Enabled normalized screen has no ranked position')
                coarse_start = None if peak is None else peak-origin
            else:
                enabled,queried = self.correlation_bank.gate(*bounds)
        if enabled:
            search, options = search_psk_sync, {}
            if self.sync_search_method == 'local-parabolic':
                from nearlink_sdr.phy.sync_refine import refine_psk_sync
                search,options = refine_psk_sync,dict(coarse_start=coarse_start)
            elif self.sync_search_method == 'bounded-grid':
                from nearlink_sdr.phy.sync_candidates import search_candidate_grid
                search,options = search_candidate_grid,dict(
                    coarse_starts=coarse_starts,candidate_count=self.sync_candidate_count)
            found = search(z, self.sps, self.reference, self.subdivisions,
                                     min_start=lower, lookahead=4*self.sps,
                                     diagnostics=diagnostic,
                                     integer_screen=(self.capture_config.integer_screen
                                                     if self.correlation_bank is None else None),
                                     integer_backend=('python' if self.correlation_bank is not None or
                                                      self.integer_screen_backend == 'disabled'
                                                      else self.integer_screen_backend),**options)
        else:
            found = None
            diagnostic = dict.fromkeys(('phase_hypotheses', 'scored_positions',
                'computed_phase_hypotheses', 'reused_phase_hypotheses', 'computed_scored_positions',
                'integer_evaluations', 'integer_evaluated_positions', 'integer_native_screens',
                'integer_reference_fallbacks'), 0)
            diagnostic.update(best_score=None, disposition='integer_screen_rejected')
        if self.correlation_bank is not None:
            diagnostic.update(integer_phase_hypotheses=self.sps,
                              integer_scored_positions=queried, integer_evaluations=0,
                              integer_evaluated_positions=0, integer_native_screens=0,
                              integer_reference_fallbacks=0)
        for key in ('phase_hypotheses', 'scored_positions', 'integer_phase_hypotheses',
                    'integer_scored_positions', 'computed_phase_hypotheses',
                    'reused_phase_hypotheses', 'computed_scored_positions', 'integer_evaluations',
                    'integer_evaluated_positions', 'integer_native_screens',
                    'integer_reference_fallbacks'):
            self.capture_stats[key] += diagnostic[key]
        self.capture_stats['integer_screen_rejections'] += int(
            diagnostic['disposition'] == 'integer_screen_rejected')
        self.capture_stats['full_searches'] += int(diagnostic['phase_hypotheses'] > 0)
        if self.sync_search_method != 'reference':
            self.capture_stats['coarse_ranking_recomputed_positions'] += diagnostic.get(
                'coarse_ranking_positions',0)
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
                                       carrier_method=self.carrier_method,
                                       **self._carrier_weight_options)
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
                                       carrier_method=self.carrier_method,
                                       **self._carrier_weight_options)
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
                     integer_screen_backend=a.get('screen_backend', self.integer_screen_backend),
                     cfo_hz=a['slope']*self.profile.config.symbol_rate_mhz*1e6/(2*np.pi),
                     emitted_after_samples=min(self._processed, self.input_samples),
                     used_end_padding=end_padding or self._processed > self.input_samples,
                     symbols=self._values[:n].copy(), symbol_centers_raw=self._centers[:n].copy(),
                     timing_rate_ppm=self._rates[:n].copy(), timing_detector_error=self._errors[:n].copy(),
                     timing_loop=a['loop'].diagnostics(len(self.reference), self._rates[:n]))
        if self.tone_notch is not None:
            event['acquisition_path'] = 'tone' if a.get('tone_path') else 'raw'
        if self.sync_search_method != 'reference':
            event['sync_search_method'] = self.sync_search_method
            event['normalized_coarse'] = self.normalized_coarse
        self._active = None
        return event
