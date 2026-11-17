"""Hash-frozen regression for an explicitly selected acquisition model.

Independent quiet records may use a conservative closed-energy-gate certificate.
That is an offline proof shortcut only: all samples are charged continuous hardware
work, and numerical internal bank states at EOF are explicitly not asserted.
"""

import argparse
import json
import sys
import tarfile
from collections import Counter, OrderedDict
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rx_stream_frozen_replay as frozen_stream
from audit_capture_hardware import profile as ordinary_profile
from audit_capture_hardware import score as ordinary_score
from audit_capture_hardware import sha, write
from audit_rx_boundary import score as factory_score
from audit_rx_soft_capture import diagnostic_bits
from profile_rx_stream import number

from nearlink_sdr.phy.acquisition_bank import DifferentialBank, normalized_greater
from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, FixedDifferentialBank
from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, ScaledDifferentialBank
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig


class PointObserver:
    """Bounded verification harness only; its point cache is not proposed hardware state."""

    def __init__(self, primary):
        self.primary, self.points = primary, OrderedDict()

    def __getattr__(self, name):
        return getattr(self.primary, name)

    def advance(self, values):
        for point in self.primary.advance(values, inspect=True):
            self.points[point.start] = point
            if len(self.points) > self.primary.window_samples:
                self.points.popitem(last=False)


class DiagnosticReceiver(PSKStreamReceiver):
    """Observe original normalized coarse scores after actual selection, without feedback."""

    def __init__(self, *args, diagnostic_file, **kwargs):
        super().__init__(*args, **kwargs)
        self.correlation_bank = PointObserver(self.correlation_bank)
        if self.tone_correlation_bank is not None:
            self.tone_correlation_bank = PointObserver(self.tone_correlation_bank)
        self.diagnostic_file = diagnostic_file
        self.numeric = dict(eligible_score_observations=0, max_score_error=0.,
            max_squared_score_error=0., max_normalized_rank_loss=0.,
            unqualified_max_rank_loss=0., eligible_rank_queries=0,
            point_gate_disagreements=0, query_gate_disagreements=0, budget_violations=0,
            diagnostic_cache_capacity=self.search_samples,
            scope='offline observer, no feedback; energy-open raw and executed tone-fallback windows')

    def _search(self):
        context, tone_context = None, None
        previous_tone_queries = self.capture_stats.get('tone_fallback_searches', 0)
        if len(self._filtered) >= self.search_samples and np.any(self._energy > self.threshold):
            z = self._filtered[-self.search_samples:]
            origin = self._origin+len(self._filtered)-len(z)
            context = z, origin, max(1., self._skip_until-origin)
            if self.tone_correlation_bank is not None:
                tone_context = self._tone_filtered[-self.search_samples:], origin, context[2]
        super()._search()
        if context is not None:
            self._observe_context(self.correlation_bank, *context, path='raw')
        if (tone_context is not None and
                self.capture_stats['tone_fallback_searches'] > previous_tone_queries):
            self._observe_context(self.tone_correlation_bank, *tone_context, path='tone')

    def _observe_context(self, observer, z, origin, lower, *, path):
        bank = observer.primary
        dr = self.reference[1:]*self.reference[:-1].conj()
        chosen, chosen_reference, reference_best = None, 0., 0.
        ref_gate, model_gate = False, False
        reference_scores = {}
        for phase in range(self.sps):
            y = z[phase::self.sps]
            dy = y[1:]*y[:-1].conj()
            numerator = np.correlate(dy, dr, 'valid')
            norm = np.sqrt(np.convolve(abs(dy)**2, np.ones(len(dr)), 'valid')*sum(abs(dr)**2))
            scores = abs(numerator)/np.maximum(norm, 1e-30)
            for j, reference_score in enumerate(scores):
                local = phase+j*self.sps
                if local < lower:
                    continue
                point = observer.points[int(origin+local)]
                reference_scores[point.start] = float(reference_score)
                model_score = float(np.sqrt(point.score_squared))
                fixed = isinstance(bank, FixedDifferentialBank)
                passed = point.energy > 0 and ((point.numerator << bank.format.threshold_fractional_bits)
                    >= bank.widths['threshold_integer']*point.energy if fixed else
                    point.numerator >= bank.threshold_squared*point.coefficient_energy*point.energy)
                ref_gate |= reference_score >= self.capture_config.integer_screen
                model_gate |= passed
                different = bool(passed) != bool(reference_score >= self.capture_config.integer_screen)
                self.numeric['point_gate_disagreements'] += int(different)
                if chosen is None or normalized_greater(point, chosen):
                    chosen, chosen_reference = point, float(reference_score)
                reference_best = max(reference_best, float(reference_score))
                error, squared_error = abs(model_score-reference_score), abs(point.score_squared-reference_score**2)
                eligible = reference_score >= .55 and bank.stats.get('input_saturations', 0) == 0
                violation = False
                if eligible:
                    self.numeric['eligible_score_observations'] += 1
                    self.numeric['max_score_error'] = max(self.numeric['max_score_error'], float(error))
                    self.numeric['max_squared_score_error'] = max(self.numeric['max_squared_score_error'], float(squared_error))
                    violation = error > .005 if fixed else squared_error > 1e-10
                    self.numeric['budget_violations'] += int(violation)
                if violation or different:
                    with self.diagnostic_file.open('a') as log:
                        log.write(json.dumps(dict(path=path, query_after_samples=self._processed,
                            start=point.start, reference_score=float(reference_score), model_score=model_score,
                            point_gate_changed=different, numeric_budget_failed=bool(violation)))+'\n')
        self.numeric['query_gate_disagreements'] += int(bool(ref_gate) != bool(model_gate))
        query = getattr(bank,'last_rank_query',None)
        if (query is not None and query[0] == origin+lower and
                query[1] == origin+len(z)-1-(len(self.reference)-1)*self.sps and query[2] is not None):
            chosen_reference = reference_scores[query[2]]
        rank_loss = reference_best-chosen_reference
        self.numeric['unqualified_max_rank_loss'] = max(
            self.numeric['unqualified_max_rank_loss'], rank_loss)
        if reference_best >= .55 and bank.stats.get('input_saturations', 0) == 0:
            self.numeric['eligible_rank_queries'] += 1
            self.numeric['max_normalized_rank_loss'] = max(
                self.numeric['max_normalized_rank_loss'], rank_loss)
            if rank_loss > .005:
                with self.diagnostic_file.open('a') as log:
                    log.write(json.dumps(dict(path=path, query_after_samples=self._processed,
                        reference_best=reference_best, normalized_rank_loss=rank_loss,
                        numeric_rank_budget_failed=True))+'\n')


def closed_gate_certificate(x, q, sps=4):
    """Sufficient whole-record condition for no search from fresh state; no truth/CRC used."""
    box = np.ones(8*sps)/(8*sps)
    if not np.isfinite(x).all() or not np.isfinite(q).all():
        return None
    with np.errstate(over='ignore', invalid='ignore'):
        power, calibration = abs(x)**2, abs(q)**2
    if not np.isfinite(power).all() or not np.isfinite(calibration).all():
        return None
    threshold = 2*float(np.quantile(np.convolve(calibration, box, 'same'), .999))
    peak = float(np.max(np.convolve(power, box, 'full'), initial=0))
    # Positive length32 FIR sums have no feedback beyond one quantum. This margin
    # safely dominates the different floating summation orders of the scalar API.
    error_bound = 128*len(box)*np.finfo(float).eps*float(np.max(power, initial=0))
    if not ((peak == threshold == 0) or peak+error_bound < threshold):
        return None
    component_bound = float(max(np.max(abs(x.real), initial=0), np.max(abs(x.imag), initial=0))*
                            np.sum(abs(rrc_filter(.4, sps))))
    return dict(method='offline closed raw-energy-gate certificate, not numerical bank replay',
        threshold=threshold, max_causal_energy=peak, summation_error_bound=error_bound,
        rrc_component_upper_bound=component_bound,
        conclusion='Fresh receiver never starts synchronization; EOF zero padding cannot open gate',
        internal_bank_state_at_eof='not evaluated; never presented as an equal-state result')


def tone_closed_certificate(x, config):
    """Independent sufficient no-activation proof, with conservative dot/sum margins."""
    if config is None:
        return dict(method='tone suppression disabled')
    if (not np.isfinite(x).all() or np.any(abs(x.real) > 1e70) or
            np.any(abs(x.imag) > 1e70)):
        return None
    from scipy.signal import lfilter
    taps = rrc_filter(.4, 4)
    values = lfilter(taps, [1.], np.r_[x, np.zeros(42, complex)])
    delayed = np.r_[np.zeros(4, complex), values[:-4]]
    power, old_power = abs(values)**2, abs(delayed)**2
    indices = np.arange(0, len(values), config.block_samples)
    dot = np.add.reduceat(values*delayed.conj(), indices)
    a, b = np.add.reduceat(power, indices), np.add.reduceat(old_power, indices)
    bound = 1024*config.block_samples*np.finfo(float).eps
    product = a*b
    numerator = abs(dot)**2
    filter_error = (4*len(taps)*np.finfo(float).eps*sum(abs(taps))*
                    max(np.max(abs(x.real), initial=0), np.max(abs(x.imag), initial=0)))
    current_abs = np.add.reduceat(abs(values), indices)
    previous_abs = np.add.reduceat(abs(delayed), indices)
    e0_error = 2*filter_error*current_abs+config.block_samples*filter_error**2
    e4_error = 2*filter_error*previous_abs+config.block_samples*filter_error**2
    dot_error = filter_error*(current_abs+previous_abs)+config.block_samples*filter_error**2
    upper_n = (abs(dot)+dot_error)**2+bound*product
    lower_product = np.maximum(a-e0_error, 0)*np.maximum(b-e4_error, 0)
    # Cauchy bounds dominate complex dot summation roundoff; uncertain blocks
    # are conservatively treated as possibly qualifying, never proved inactive.
    possible = (upper_n > 0) & (upper_n >=
                                config.coherence_squared*lower_product*(1-bound))
    if np.any(possible[1:] & possible[:-1]):
        return None
    return dict(method='no two consecutive possibly coherent blocks, includes42 EOF zeros',
                blocks=len(indices), margin=bound, rrc_output_error_bound=filter_error,
                max_coherence_squared=float(np.max(np.divide(numerator, product,
                    out=np.zeros(len(product)), where=product > 0), initial=0)))


def checked_manifest(root, relative):
    manifest = json.loads((root/'manifest.json').read_text())
    expected = manifest[str(relative)]
    value = expected['sha256'] if isinstance(expected, dict) else expected
    path = root/relative
    assert sha(path) == value, path
    return path


class Replay:
    def __init__(self, args):
        if (getattr(args,'carrier_weight_kappa',None) is None and
                getattr(args,'carrier_weight_fixed',False)):
            raise ValueError('Fixed carrier weights require explicit kappa before replay/proofs')
        self.args, self.rows, self.stream_index = args, [], 0
        self.selection = json.loads(args.selection.read_text())
        from nearlink_sdr.phy.tone_notch import ToneNotchConfig
        radius = getattr(args, 'tone_radius', None)
        self.tone_config = None if radius is None else ToneNotchConfig(radius=radius)
        self.bits = (self.selection['mantissa_bits'] if args.backend == 'scaled' else
                     self.selection['hardware_input_bits'])
        self.full_scale = (1. if args.backend == 'scaled' else
                           self.selection['hardware_input_full_scale'])
        if self.bits is None or self.selection['held_out_seen']:
            raise ValueError('Need a frozen development-only format')
        self.baseline = args.evidence_root/'rx-capture-front/adaptive-final-reproduction'
        original = args.evidence_root/'rx-timing-tracking/final'
        self.original_plan = json.loads(checked_manifest(original, 'frozen-plan.json').read_text())
        self.factory = PSKReceiveProfile.ws63_ft2_a3(
            sync_word=number(self.original_plan['sync_word']),
            data_crc_seed=number(self.original_plan['crc_seed']), pilot_interval=0)
        self.out = args.output
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/'evidence').mkdir()
        self.driver_hash = sha(Path(__file__))
        self.sources = {str(p): sha(p) for p in Path('src').rglob('*.py')}
        write(self.out/'frozen-plan.json', dict(groups=args.group, backend=args.backend,
            selection=str(args.selection.resolve()), selection_sha256=sha(args.selection),
            input_bits=self.bits, full_scale=self.full_scale, chunks=4096,
            tone_radius=getattr(args, 'tone_radius', None),
            sync_search_method=getattr(args,'sync_method','reference'),
            normalized_coarse=getattr(args,'normalized_coarse',False),
            rrc_backend=getattr(args,'rrc_backend','reference'),
            carrier_weight_kappa=getattr(args,'carrier_weight_kappa',None),
            carrier_weight_fixed=getattr(args,'carrier_weight_fixed',False),
            quiet_certificate='only zero-target quiet controls; numeric bank work always charged',
            sources=self.sources, driver_sha256=sha(Path(__file__))))
        (self.out/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
        with tarfile.open(self.out/'source.tar', 'w') as archive:
            for p in [*self.sources, 'pyproject.toml', 'uv.lock']:
                archive.add(p, arcname=p)

    def receive(self, x, q, profile, *, quiet=False):
        self.stream_index += 1
        # Existing floating tone/RRC bounds do not prove finite-word internal
        # behavior. Process every quiet sample on the experimental word path.
        certificate = (closed_gate_certificate(x, q, profile.config.sps)
                       if quiet and getattr(self.args,'rrc_backend','reference') == 'reference'
                       else None)
        if certificate is not None:
            proof = tone_closed_certificate(x, self.tone_config)
            if proof is None:
                certificate = None
            else:
                certificate['tone_never_activates_proof'] = proof
        if certificate is not None:
            if self.args.backend == 'scaled':
                bank = ScaledDifferentialBank(profile.sync_symbols(), 4, 256,
                    scaled_format=ScaledBankFormat(self.bits))
            elif self.args.backend == 'fixed':
                bank = FixedDifferentialBank(profile.sync_symbols(), 4, 256,
                    input_full_scale=self.full_scale, format=FixedBankFormat(self.bits))
            else:
                bank = DifferentialBank(profile.sync_symbols(), 4, 256,
                                         input_full_scale=self.full_scale)
            certificate['no_input_saturation_proved'] = (
                certificate['rrc_component_upper_bound'] < self.full_scale-
                self.full_scale/(1 << (self.bits-1)))
            if self.args.backend == 'scaled':
                certificate['no_input_saturation_proved'] = None
                certificate['format_scope'] = (
                    'Closed search gate only; no scaled internal-state or range proof')
            return [], dict(certificate=certificate, numerically_processed_samples=0,
                rrc_backend=getattr(self.args,'rrc_backend','reference'),
                rrc_state_observed=False,
                continuous_work_charged_samples=len(x), eof_virtual_padding_samples=42,
                continuous_bank_instances=1 if self.tone_config is None else 2,
                cost_scope='per-bank estimate; detector/notch/RRC/fine/loop work is separate',
                operations_per_charged_sample=bank.worst_operations_per_sample())
        diagnostic_file = self.out/'evidence'/f'stream{self.stream_index:04}-numeric-changes.jsonl'
        rx = DiagnosticReceiver(profile, q, quiet_batching=False, diagnostic_file=diagnostic_file,
            defer_quiet_fir=False, rrc_backend=getattr(self.args,'rrc_backend','reference'),
            carrier_weight_kappa=getattr(self.args,'carrier_weight_kappa',None),
            carrier_weight_fixed=getattr(self.args,'carrier_weight_fixed',False),
            tone_notch=self.tone_config,
            sync_search_method=getattr(self.args,'sync_method','reference'),
            normalized_coarse=getattr(self.args,'normalized_coarse',False),
            correlation_backend=self.args.backend, correlation_input_bits=self.bits,
            correlation_mantissa_bits=self.bits if self.args.backend == 'scaled' else 10,
            correlation_input_full_scale=self.full_scale)
        events = []
        for start in range(0, len(x), 4096):
            events.extend(rx.iter_feed(x[start:start+4096]))
        events.extend(rx.finish())
        bank = rx.correlation_bank
        return events, dict(certificate=None, numerically_processed_samples=len(x),
            rrc_backend=rx.rrc_backend, fir_array_bytes=rx.fir_array_bytes,
            rrc_state_observed=rx._rrc_word_model is not None,
            rrc_stats=None if rx._rrc_word_model is None else dict(rx._rrc_word_model.stats),
            continuous_work_charged_samples=len(x), eof_virtual_padding_samples=rx.future_samples,
            capture_stats=rx.capture_stats, bank_stats=bank.stats,
            numeric_diagnostics=rx.numeric,
            tone_stats=None if rx.tone_notch is None else rx.tone_notch.stats,
            continuous_bank_instances=1 if self.tone_config is None else 2,
            tone_bank_stats=(None if rx.tone_correlation_bank is None else
                             rx.tone_correlation_bank.stats),
            tone_bank_operations=(None if rx.tone_correlation_bank is None else
                                  rx.tone_correlation_bank.worst_operations_per_sample()),
            tone_operations=None if rx.tone_notch is None else rx.tone_notch.operation_estimate(),
            cost_scope=('bank estimates include raw energy separately per bank '
                        '(conservative double charge); RRC/fine/loops excluded'),
            numeric_change_log=str(diagnostic_file) if diagnostic_file.exists() else None,
            operations_per_charged_sample=bank.worst_operations_per_sample(),
            emulator_state_array_bytes=bank.state_array_bytes)

    def record(self, group, ident, meta, events, frames, cost, old_frames,
               old_file=None, old_prefix='', old_correct='correct', extra=None):
        key = group+'-'+ident
        arrays = {f'{i}-{f}': e[f] for i, e in enumerate(events) for f in
                  ('symbols', 'symbol_centers_raw', 'timing_rate_ppm', 'timing_detector_error')}
        np.savez_compressed(self.out/'evidence'/(key+'.npz'), **arrays)
        def good(f):
            return bool(f.get(old_correct))

        def start(f):
            return f.get('sync_start_raw', f.get('sync_start', f.get('sync')))
        lost = [start(f) for f in old_frames if good(f) and not any(
            g.get('whole_packet_ok', g.get('correct', False)) and
            abs(start(g)-start(f)) <= .125 for g in frames)]
        maximum = 0.
        if old_file is not None:
            with np.load(old_file) as previous:
                for i, f in enumerate(old_frames):
                    matches = [j for j, g in enumerate(frames) if abs(start(g)-start(f)) <= 4]
                    if len(matches) != 1:
                        continue
                    j = matches[0]
                    field = f'{old_prefix}{i}-symbol_centers_raw'
                    actual = events[j]['symbol_centers_raw']
                    if field in previous:
                        old = previous[field]
                        n = min(len(old), len(actual))
                        maximum = max(maximum, float(np.max(abs(old[:n]-actual[:n]), initial=0)))
        row = dict(group=group, id=ident, input=meta, frames=frames, cost=cost,
            baseline_correct=sum(good(f) for f in old_frames),
            correct=sum(f.get('whole_packet_ok', f.get('correct', False)) for f in frames),
            lost_success_positions=lost, max_matched_center_delta=maximum, extra=extra)
        diagnostics = cost.get('numeric_diagnostics', {})
        row['numeric_budget_passed'] = (not diagnostics.get('budget_violations', 0) and
            diagnostics.get('max_normalized_rank_loss', 0) <= .005 and
            not cost.get('bank_stats', {}).get('intermediate_overflows', 0))
        row['baseline_captures'], row['captures'] = len(old_frames), len(frames)
        row['candidate_loss_positions'] = [start(f) for f in old_frames if not any(
            abs(start(g)-start(f)) <= 4 for g in frames)]
        row['unexpected_zero_target_candidates'] = (len(frames) if
            meta.get('source') == 'quiet' or group == 'negatives' else 0)
        row['known_packets'] = (0 if meta.get('source') == 'quiet' or group == 'negatives' else
            1 if meta.get('source') == 'simulation' or group in
            ('uncoded', 'held-old', 'coded-hard', 'coded-maxlog') else None)
        self.rows.append(row)
        with (self.out/'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        write(self.out/'regressions.json', [r for r in self.rows if r['lost_success_positions'] or
                                            r['max_matched_center_delta'] > .25])
        if len(self.rows) % 16 == 0:
            print('replayed', len(self.rows), 'inputs', flush=True)

    def original(self):
        a = self.args
        archived = a.evidence_root/'rx-chain-hardening/parser-replay'
        original = a.evidence_root/'rx-timing-tracking/final'
        frozen_stream.frozen.verify_manifest(archived)
        frozen_stream.frozen.verify_manifest(original)
        rows = [json.loads(s) for s in (archived/'results.jsonl').read_text().splitlines()]
        before = {r['id']: r for r in [json.loads(s) for s in checked_manifest(
                  self.baseline/'original-frozen', 'results.jsonl').read_text().splitlines()]}
        old_quiet = {r['sample_id']: r for r in self.original_plan['quiet_inputs']}
        new_quiet = {r['input']['sample_id']: r['input'] for r in rows if r['source'] == 'quiet'}
        for p in rows:
            x, q = frozen_stream.load_input(p, archived, original, a.dataset, a.quiet_dataset,
                                            old_quiet, new_quiet)
            events, cost = self.receive(x, q, self.factory, quiet=p['source'] == 'quiet')
            frames = frozen_stream.score(p, events)
            self.record('original', p['id'], p, events, frames, cost, before[p['id']]['frames'],
                old_file=checked_manifest(self.baseline/'original-frozen', 'evidence/'+p['id']+'.npz'),
                old_prefix='whole-', old_correct='whole_packet_ok')

    def uncoded(self):
        folder = self.baseline/'uncoded-boundary'
        for r in json.loads(checked_manifest(folder, 'results.json').read_text()):
            path = folder/r['input']['path']
            assert sha(path) == r['input']['sha256']
            old_file = checked_manifest(folder, r['evidence'])
            with np.load(path) as data:
                x, q = data['iq'], data['quiet']
            events, cost = self.receive(x, q, self.factory)
            step = 4*(1+r['ppm']*1e-6)
            frames = factory_score(events, r['length'],
                dict(sync_start=512+r['offset']+40*step, step=step), r['kind'])
            with np.load(old_file) as data:
                negative, nc = self.receive(data['quiet_negative'], q, self.factory, quiet=True)
            self.record('uncoded', r['id'], r['input'], events, frames, cost, r['frames'],
                old_file=old_file, extra=dict(negative_candidates=len(negative), negative_cost=nc))

    def held(self):
        folder = self.baseline/'held-out'
        for r in json.loads(checked_manifest(folder, 'results.json').read_text()):
            path = Path(r['path'])
            assert sha(path) == r['sha256']
            with np.load(path) as data:
                p = ordinary_profile(r['mcs'])
                events, cost = self.receive(data['iq'], data['quiet'], p)
                meta = dict(mcs=r['mcs'], packets=[dict(sync_start=r['truth'], step=r['step'],
                    length=255, mcs=r['mcs'], expected_hex=data['expected'].tobytes().hex(),
                    ppm=100, cfo_hz=30000)])
                frames = ordinary_score(events, meta)['frames']
            self.record('held-old', r['id'], r, events, frames, cost, r['arms']['selected']['frames'],
                old_file=checked_manifest(folder, 'evidence/'+r['id']+'.npz'), old_prefix='selected-')

    def coded(self):
        folder = self.baseline/'coded-replay'
        for r in json.loads(checked_manifest(folder, 'results.json').read_text()):
            path = Path(r['file'])
            assert sha(path) == r['sha256']
            p = ordinary_profile(r['mcs'])
            with np.load(path) as data:
                events, cost = self.receive(data['iq'], data['quiet'], p)
                expected = data['expected'].tobytes()
            for arm, soft in [('hard', False), ('maxlog', True)]:
                frames, selected_events = [], []
                selected = replace(p, soft_payload=soft)
                for e in events:
                    kappa = getattr(self.args,'carrier_weight_kappa',None)
                    options = ({} if kappa is None else dict(carrier_weight_kappa=kappa,
                               carrier_weight_fixed=getattr(self.args,'carrier_weight_fixed',False)))
                    event = dict(e, frame=decode_psk_frame(e['symbols'], selected, **options))
                    f = event['frame']
                    bits = diagnostic_bits(event, selected)
                    frames.append(dict(status=f['status'], correct=f['status'] == 'valid' and f['payload'] == expected and
                        f.get('header', {}).get('data_length') == 255 and f.get('mcs') == r['mcs'],
                        header_crc=f['header_crc'], data_crc=f['data_crc'], length_ok=f['length_ok'],
                        carrier=f.get('carrier'), soft_information=f.get('soft_information'),
                        mcs=f.get('mcs'), sync_start=e['sync_start_raw'],
                        bit_errors=None if bits is None else int(np.sum(bits != np.unpackbits(
                            np.frombuffer(expected, np.uint8))))))
                    selected_events.append(event)
                old_file = checked_manifest(folder, 'evidence/'+r['id']+'-results.npz')
                with np.load(old_file) as old:
                    previous = [dict(f, sync_start=float(old[f'{arm}-{j}-symbol_centers_raw'][0]))
                                for j, f in enumerate(r['arms'][arm]['frames'])]
                # One acquisition, two independent payload decodes; no CRC feeds sync/state.
                self.record('coded-'+arm, r['id'], r, selected_events, frames, cost, previous,
                            old_file=old_file, old_prefix=arm+'-')

    def weak(self):
        folder = self.args.evidence_root/'rx-capture-front/adaptive-weak-hardware-replay'
        for r in json.loads(checked_manifest(folder, 'results.json').read_text()):
            path = Path(r['input_path'])
            assert sha(path) == r['input_sha256']
            for source in [r['source'], *r['calibration_sources']]:
                assert sha(self.args.dataset/source['path']) == source['sha256']
            with np.load(path) as data:
                events, cost = self.receive(data['iq'], data['quiet'], self.factory)
                frames = factory_score(events, 255, kind=r['kind'])
                negative, nc = self.receive(data['negative'], data['quiet'], self.factory, quiet=True)
            self.record('weak-hardware', r['id'], r, events, frames, cost,
                        r['arms']['selected']['frames'],
                        old_file=checked_manifest(folder, 'evidence/'+r['id']+'.npz'),
                        old_prefix='selected-', extra=dict(negative_candidates=len(negative), negative_cost=nc))

    def negatives(self):
        folder = self.args.evidence_root/'rx-capture-front/independent-negatives'
        plan = json.loads(checked_manifest(folder, 'plan.json').read_text())
        p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
        for r in plan['inputs']:
            path = Path(r['path'])
            assert sha(path) == r['sha256']
            with np.load(path) as data:
                events, cost = self.receive(data['iq'], data['quiet'], p,
                                            quiet=r['kind'] in ('white', 'colored'))
            frames = [dict(status=e['frame']['status'], header_crc=e['frame']['header_crc'],
                           data_crc=e['frame']['data_crc'], sync_start=e['sync_start_raw'],
                           correct=False) for e in events]
            self.record('negatives', r['id'], r, events, frames, cost, [])

    def finish(self):
        assert sha(Path(__file__)) == self.driver_hash, 'Driver changed during replay'
        assert all(sha(Path(p)) == h for p, h in self.sources.items()), 'Source changed during replay'
        summary = {}
        for group in sorted({r['group'] for r in self.rows}):
            rows = [r for r in self.rows if r['group'] == group]
            summary[group] = dict(inputs=len(rows), correct=sum(r['correct'] for r in rows),
                baseline_correct=sum(r['baseline_correct'] for r in rows),
                lost_successes=sum(len(r['lost_success_positions']) for r in rows),
                certificate_inputs=sum(r['cost']['certificate'] is not None for r in rows),
                charged_samples=sum(r['cost']['continuous_work_charged_samples'] for r in rows),
                max_center_delta=max(r['max_matched_center_delta'] for r in rows))
            summary[group]['captures'] = sum(r['captures'] for r in rows)
            summary[group]['header_crc_ok'] = sum(f.get('header_crc') is True for r in rows for f in r['frames'])
            summary[group]['data_crc_ok'] = sum(f.get('data_crc') is True for r in rows for f in r['frames'])
            summary[group]['statuses'] = dict(Counter(f['status'] for r in rows for f in r['frames']))
            numeric = [r['cost']['numeric_diagnostics'] for r in rows if r['cost']['certificate'] is None]
            summary[group]['numeric'] = dict(
                max_score_error=max((v['max_score_error'] for v in numeric), default=None),
                max_normalized_rank_loss=max((v['max_normalized_rank_loss'] for v in numeric), default=None),
                query_gate_disagreements=sum(v['query_gate_disagreements'] for v in numeric),
                budget_violations=sum(v['budget_violations'] for v in numeric))
            summary[group]['candidate_losses'] = sum(len(r['candidate_loss_positions']) for r in rows)
            summary[group]['unexpected_zero_target_candidates'] = sum(
                r['unexpected_zero_target_candidates'] for r in rows)
            summary[group]['paired_negative_candidates'] = sum(
                (r['extra'] or {}).get('negative_candidates', 0) for r in rows)
        write(self.out/'summary.json', summary)
        write(self.out/'failures.json', [r for r in self.rows if any(
            f['status'] != 'valid' or not f.get('whole_packet_ok', f.get('correct', False))
            for f in r['frames']) or r['lost_success_positions'] or
            (r['known_packets'] is not None and r['correct'] < r['known_packets']) or
            (not r['frames'] and r['group'] == 'weak-hardware')])
        acceptance = dict(
            numerical_scope='coarse correlation bank; RRC format gate is separate',
            rrc_numeric_gate_passed=(None if getattr(self.args,'rrc_backend','reference')
                                    == 'reference' else False),
            receiver_promotion=False,
            functional_budget_passed=not any(r['lost_success_positions'] or
                r['max_matched_center_delta'] > .25 or r['unexpected_zero_target_candidates'] or
                (r['extra'] or {}).get('negative_candidates', 0) for r in self.rows),
            numerical_budget_passed=all(r['numeric_budget_passed'] for r in self.rows),
            note='Functional and numeric acceptance are independent; every failed input retained')
        write(self.out/'acceptance.json', acceptance)
        write(self.out/'numeric-budget-failures.json', [r for r in self.rows if
                                                       not r['numeric_budget_passed']])
        write(self.out/'regressions.json', [r for r in self.rows if r['lost_success_positions'] or
            r['max_matched_center_delta'] > .25 or r['unexpected_zero_target_candidates'] or
            (r['extra'] or {}).get('negative_candidates', 0)])
        write(self.out/'manifest.json', {str(p.relative_to(self.out)): sha(p) for p in
                                        self.out.rglob('*') if p.is_file()})
        print(json.dumps(summary, indent=2), flush=True)
        return (not all((acceptance['functional_budget_passed'], acceptance['numerical_budget_passed']))
                or acceptance['rrc_numeric_gate_passed'] is False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-root', 'dataset', 'quiet-dataset', 'selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--tone-radius', type=float, choices=[.9375, .96875])
    parser.add_argument('--backend', choices=['structure', 'fixed', 'scaled'], default='fixed')
    parser.add_argument('--sync-method',choices=['reference','local-parabolic'],default='reference')
    parser.add_argument('--normalized-coarse',action='store_true')
    parser.add_argument('--rrc-backend',choices=['reference','word-full12-c18'],default='reference')
    parser.add_argument('--carrier-weight-kappa',type=int,choices=[2,4,8])
    parser.add_argument('--carrier-weight-fixed',action='store_true')
    parser.add_argument('--group', choices=['original', 'uncoded', 'held', 'coded', 'weak', 'negatives'],
                        action='append')
    args = parser.parse_args()
    args.group = args.group or ['original', 'uncoded', 'held', 'coded', 'weak', 'negatives']
    replay = Replay(args)
    for group in args.group:
        getattr(replay, group)()
    raise SystemExit(replay.finish())
