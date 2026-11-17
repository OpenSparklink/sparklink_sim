"""Develop/freeze/validate bounded spline words; incomplete system gate returns1."""

import argparse
import ast
import hashlib
import importlib.metadata
import inspect
import itertools
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from audit_rx_soft_capture import diagnostic_bits
from scipy.interpolate import CubicSpline
from tests.test_psk_field_layout import independent_wire
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy import sync_candidates
from nearlink_sdr.phy.control_info import ControlInfoA2, ControlInfoB1
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.spline_fixed import (
    FixedUniformCubicEighth,
    SplineFixedFormat,
    fixed_preparation_calendar,
    fixed_spline_cost,
)
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import ToneNotchConfig
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_head, encode_payload, head_crc_parameters
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions

POLICY_PATH = Path(__file__).resolve().parents[1]/'tests/data/spline_fixed_format.json'
POLICY = json.loads(POLICY_PATH.read_text())


def independent_packet(profile, payload):
    """Existing standard encoders; independent bit/angle layout and pulse reference."""
    cfg = profile.config
    if cfg.frame_type == 2:
        head = ControlInfoA2(0, 0, 0, 0, 0, 0, 0, len(payload))
        raw = head.pack(head_crc_parameters(cfg)[2], lqi=0)
    else:
        raw = ControlInfoB1(0, 0, 0, cfg.mcs_index, len(payload), 0, 0).pack(cfg.head_llid)
    data = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder=profile.payload_bitorder)
    sync = independent_wire(profile.sync_bits(), 'QPSK' if cfg.frame_type == 2 else 'BPSK', 40, 0, True)
    head = independent_wire(encode_head(raw, cfg), 'BPSK' if cfg.frame_type == 4 else 'QPSK',
                            40+len(sync), 16 if cfg.frame_type == 2 else 4, cfg.pilot_interval == 0)
    data = independent_wire(encode_payload(data, cfg), cfg.mod_str, 40+len(sync)+len(head),
                            cfg.pilot_interval, True)
    return np.r_[np.exp(1j*np.tile([np.pi/4, 0], 20)), sync, head, data]


def streams(held=False):
    """New payload, configuration and interference combinations, not seed-only split."""
    rng = np.random.default_rng(640117 if held else 630117)
    configs = ([(2, 12, 8), (3, 8, 0), (4, 12, 8), (2, 6, 0),
                (3, 0, 8), (4, 8, 0), (2, 8, 8), (3, 12, 0)] if held else
               [(2, 8, 0), (3, 8, 8), (4, 0, 0), (2, 12, 0),
                (3, 12, 8), (4, 12, 0), (2, 6, 8), (4, 8, 8)])
    lengths = [29, 137, 401, 797] if held else [17, 61, 203, 619]
    kinds = ['tone_burst', 'double_tone', 'wrong_noise', 'burst_step'] if held else ['noise', 'tone', 'wrong_sync', 'burst']
    for j, (ft, mcs, pilot) in enumerate(configs):
        cfg = TxConfig(frame_type=ft, mcs_index=mcs, pilot_interval=pilot,
            pid=3 if held else 1, whitening_seed=0x52, crc_seed=0x123456,
            ctrl_bits_len=28 if ft == 2 else 27, head_llid=0x234567 if ft != 2 else None,
            symbol_rate_mhz=4)
        profile = PSKReceiveProfile.ordinary(cfg)
        length = lengths[j % 4]
        values = (np.arange(length)*73+j*19) % 256 if j % 3 == 0 else (
            np.resize([0, 255, 0, 3, 128], length) if j % 3 == 1 else rng.integers(0, 256, length))
        payload = np.asarray(values, np.uint8).tobytes()
        cfo, ppm = ((-19000, 75) if held else (27000, -50)) if j % 2 else ((41000, -25) if held else (-33000, 100))
        clean, _, truth, step = iq_record(independent_packet(profile, payload),
            ppm=ppm, cfo=cfo, offset=.641 if held else .193)
        prefix = 211+j*13 if held else 89+j*17
        n = len(clean)+prefix+256
        sigma = 1.5 if j < 4 else 2.5
        quiet = sigma/np.sqrt(2)*(rng.normal(size=2048)+1j*rng.normal(size=2048))
        x = sigma/np.sqrt(2)*(rng.normal(size=n)+1j*rng.normal(size=n))
        kind = kinds[j % 4]
        k = np.arange(n)
        if 'tone' in kind:
            x += (4 if held else 3)*np.exp(2j*np.pi*(.113713 if held else -.081731)*k+.73j)
            if kind == 'double_tone':
                x += 1.3*np.exp(-2j*np.pi*.03791*k)
        if 'wrong' in kind:
            other = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, pid=7,
                whitening_seed=0x52, crc_seed=0x123456, symbol_rate_mhz=4))
            wrong, _, _, _ = iq_record(independent_packet(other, b'wrong-profile-new-content'), cfo=9000, offset=.883)
            x += .07*np.resize(wrong, n)
        if 'burst' in kind:
            for start in range(173, n, 1423 if held else 1877):
                count = min(5 if held else 11, n-start)
                x[start:start+count] += 25*np.exp(1j*rng.uniform(-np.pi, np.pi, count))
        background = x.copy()
        amp = (.12, .35, .7)[j % 3]
        x[prefix:prefix+len(clean)] += amp*clean
        if kind == 'burst_step':
            x[n//2:] *= 1.7
        meta = dict(id=f'{"validation" if held else "development"}-{j}', kind=kind,
                    positive=True, frame_type=ft, mcs=mcs, pilot=pilot, length=length,
                    payload_hex=payload.hex(), cfo_hz=cfo, sro_ppm=ppm,
                    sync_start=prefix+truth, step=step, amplitude=amp,
                    complex_noise_variance=sigma*sigma)
        yield meta, profile, x, quiet, payload
        if j < 4:
            yield dict(meta, id=meta['id']+'-negative', positive=False,
                       payload_hex=None, sync_start=None), profile, background, quiet, b''


def numeric_windows(held=False):
    rng = np.random.default_rng(640119 if held else 630119)
    lengths = (5, 23, 127, 255, 639) if held else (4, 17, 64, 256, 640)
    scales = (-219, -17, 3, 71, 219) if held else (-220, -31, 0, 29, 220)
    for n, power in itertools.product(lengths, scales):
        k, t = np.arange(n), np.linspace(0, 1, n)
        noise = rng.normal(size=n)+1j*rng.normal(size=n)
        waves = [t**3+.3*t**2-.7*t+1j*(t**3-t),
                 np.exp(2j*np.pi*rng.integers(0, 8, n)/8+1j*k*np.pi/8),
                 .13*noise+np.exp(1j*(.193*k+.0007*k*k)),
                 noise*np.where(k < n//3, 1., 2**-32)]
        if held:
            waves = [waves[0], waves[1]*np.where(k < n//2, .125, 2),
                     waves[2]+.47*np.exp(-.371j*k), np.cumsum(noise)/n]
        for j, x in enumerate(waves):
            x = np.asarray(x*2.**power, complex)
            times = np.unique(np.r_[np.arange(min(64, 8*(n-1)+1))/8,
                                     np.linspace(0, n-2, 32, dtype=int)+.5, n-1])
            yield f'{n}-{power}-{j}', x, times


def node_audit(folder, fmt, held=False):
    folder.mkdir()
    rows, arrays = [], {}
    for ident, x, times in numeric_windows(held):
        node = FixedUniformCubicEighth(x, fmt)
        actual = node(times)
        reference = CubicSpline(np.arange(len(x)), x)(times)
        scale = float(np.max(abs(x)))
        error = float(np.max(abs(actual-reference)))/scale if scale else 0.
        passed = error <= POLICY['node_limit'] and node.stats['saturations'] == 0
        rows.append(dict(id=ident, error=error, passed=passed, exponent=node.exponent, stats=node.stats,
                         input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
        arrays[ident+'-input'], arrays[ident+'-times'] = x, times
        arrays[ident+'-reference'], arrays[ident+'-fixed'] = reference, actual
    np.savez_compressed(folder/'nodes.npz', **arrays)
    write(folder/'nodes.json', rows)
    return dict(windows=len(rows), failures=sum(not r['passed'] for r in rows),
                max_error=max(r['error'] for r in rows),
                saturated_windows=sum(r['stats']['saturations'] != 0 for r in rows))


def receive(folder, meta, profile, x, quiet, wanted, *, fmt=None, chunking='whole', record_windows=False):
    """Actual selected result enters receiver; optional reference is postselection only."""
    rx = MacStreamReceiver(profile, quiet, quiet_batching=False, defer_quiet_fir=False,
        correlation_backend='scaled', normalized_coarse=True, tone_notch=ToneNotchConfig(radius=31/32),
        sync_search_method='bounded-grid', sync_candidate_count=1, sync_coherent_threshold=.45)
    original = sync_candidates.search_candidate_grid
    windows, comparisons, events, packets = [], [], [], []
    def selected(*args, **kwargs):
        if record_windows and len(windows) < 8:
            windows.append((np.array(args[0], copy=True), np.array(args[2], copy=True),
                {k: kwargs[k] for k in ('coarse_starts', 'candidate_count', 'min_start', 'lookahead') if k in kwargs}))
        if fmt is None:
            return original(*args, **kwargs)
        result = original(*args, **kwargs, interpolator='word-eighth', spline_format=fmt)
        # Never pass this reference result back to the receiver or retry CRC.
        d = {}
        original(*args, **{**kwargs, 'diagnostics': d})
        a, b = d, kwargs['diagnostics']
        score_error = abs(a['best_score']-b['best_score']) if a['best_score'] is not None and b['best_score'] is not None else None
        start_error = abs(a['best_start']-b['best_start']) if 'best_start' in a and 'best_start' in b else None
        comparisons.append(dict(reference=a, fixed=b.copy(), score_error=score_error, start_error=start_error,
            passed=(a['disposition'] == b['disposition'] and (score_error is None or score_error <= POLICY['score_limit'])
                    and (start_error is None or start_error <= POLICY['start_limit']))))
        return result
    rng, offset = np.random.default_rng(630121), 0
    with patch.object(sync_candidates, 'search_candidate_grid', selected):
        while offset < len(x):
            count = len(x) if chunking == 'whole' else 257 if chunking == 'fixed' else int(rng.integers(1, 1001))
            result = rx.feed(x[offset:offset+count])
            packets += result.packets
            events += result.diagnostics['events']
            offset += count
        result = rx.finish()
        packets += result.packets
        events += result.diagnostics['events']
    positive = meta['positive']
    arrays, frames = {}, []
    for j, e in enumerate(events):
        f = e['frame']
        row = {k: f.get(k) for k in ('status', 'header_crc', 'data_crc', 'length_ok', 'mcs', 'header')}
        row.update(sync_start=e['sync_start_raw'], cfo_hz=e['cfo_hz'])
        associated = bool(positive and meta.get('sync_start') is not None
                      and abs(e['sync_start_raw']-meta['sync_start']) < 2*len(profile.sync_symbols()))
        row['truth_associated_for_diagnostics_only'] = associated
        row['timing_truth_max_error'] = row['cfo_truth_error_hz'] = None
        if associated:
            truth_times = meta['sync_start']+np.arange(len(e['symbol_centers_raw']))*meta['step']
            errors = e['symbol_centers_raw']-truth_times
            row['timing_truth_max_error'] = float(np.max(abs(errors)))
            row['cfo_truth_error_hz'] = e['cfo_hz']-meta['cfo_hz']
            arrays[f'{j}-timing-truth-error'] = errors
        row['bit_errors'] = row['ber_bits'] = None
        if associated and f.get('mcs') == profile.config.mcs_index and f.get('header', {}).get('data_length') == len(wanted):
            bits = diagnostic_bits(e, profile)
            if bits is not None:
                truth = np.unpackbits(np.frombuffer(wanted, np.uint8), bitorder=profile.payload_bitorder)
                row['bit_errors'], row['ber_bits'] = int(np.count_nonzero(bits[:len(truth)] != truth)), len(truth)
        frames.append(row)
        arrays[f'{j}-centers'], arrays[f'{j}-symbols'] = e['symbol_centers_raw'], e['symbols']
    folder.mkdir()
    np.savez_compressed(folder/'events.npz', **arrays)
    write(folder/'frames.json', frames)
    write(folder/'searches.json', comparisons)
    row = dict(id=meta['id'], samples=len(x), positive=positive, chunking=chunking,
        captured=len(events), head_crc=sum(f['header_crc'] is True for f in frames),
        data_crc=sum(f['data_crc'] is True for f in frames),
        correct=sum(positive and p.mac_payload == wanted for p in packets),
        false_deliveries=sum(not positive or p.mac_payload != wanted for p in packets),
        payload_hex=[p.mac_payload.hex() for p in packets],
        bit_errors=sum(f['bit_errors'] for f in frames if f['bit_errors'] is not None),
        ber_bits=sum(f['ber_bits'] for f in frames if f['ber_bits'] is not None),
        stats=rx.phy.capture_stats, search_comparison_failures=sum(not r['passed'] for r in comparisons))
    write(folder/'summary.json', row)
    return row, arrays, windows


def compare_stream(base, test, base_arrays, test_arrays):
    # Truth-only diagnostics never enter an adoption/pass decision.
    base_arrays = {k:v for k,v in base_arrays.items() if k.endswith(('-centers', '-symbols'))}
    test_arrays = {k:v for k,v in test_arrays.items() if k.endswith(('-centers', '-symbols'))}
    same_shapes = base_arrays.keys() == test_arrays.keys() and all(
        base_arrays[k].shape == test_arrays[k].shape for k in base_arrays)
    center_error = max((float(np.max(abs(base_arrays[k]-test_arrays[k])))
                       for k in base_arrays if k.endswith('-centers') and len(base_arrays[k])), default=0.) if same_shapes else None
    return dict(reference=base['id'], fixed=test['id'], max_center_error=center_error,
        capture_loss=max(0, base['captured']-test['captured']),
        correct_loss=max(0, base['correct']-test['correct']),
        passed=(test['captured'] >= base['captured'] and test['head_crc'] >= base['head_crc']
                and test['data_crc'] >= base['data_crc'] and test['correct'] >= base['correct']
                and test['false_deliveries'] <= base['false_deliveries']
                and (test['positive'] or test['captured'] <= base['captured'])
                and test['search_comparison_failures'] == 0
                and same_shapes and center_error <= POLICY['center_limit']))


def resource_audit(output, fmt):
    """Finite address/transport replay; score/packet arithmetic remains unclosed."""
    costs = [fixed_spline_cost(fmt, n) for n in (256, 376, 632)]
    for n in (256, 376, 632):
        write(output/f'preparation-{n}.json', list(fixed_preparation_calendar(n)))
    graph = costs[0]
    cycles = graph['serial_preparation_cycles_scenario']
    words = np.random.default_rng(630122).integers(0, 2**46, (4096, 2), dtype=np.uint64)
    np.save(output/'transport-words.npy', words)
    scenarios = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        for interval in (32, 256):
            events = []
            checked = 0
            def sink(event, data):
                nonlocal checked
                if data is not None:
                    stop, path = event['after_samples'], event['path']
                    assert np.array_equal(data, words[stop-256:stop, path])
                    checked += 1
                events.append(event)
            flow = CaptureTransport(CaptureTransportAssumptions(clock_hz=clock,
                fine_service_cycles=cycles))
            result = flow.simulate(words, [(j, (j//interval)%2) for j in range(256, 4097, interval)], sink=sink)
            name = f'queue-{clock}-{interval}.json'
            write(output/name, events)
            scenarios.append(dict(clock_hz=clock, request_interval_samples=interval,
                preparation_only_cycles=cycles, score_and_confirm_and_payload_work_charged=False,
                result=result, checked_windows=checked, events_file=name,
                interpretation='optimistic queue lower load with scoring free;drops already disprove this serial every32-request schedule;not a lost-packet rate'))
    partial = graph['source_and_work_logical_bits']+graph['live_data_control_monitor_reservation_bits']
    partial += graph['scan_pipeline_bits_scenario']+graph['alignment_pipeline_bits_scenario']
    partial += graph['maximum_state_bits']
    partial += graph['coefficient_bits_after_compression']
    blocks = graph['source_tdp36_bram36']+graph['work_tdp36_bram36']
    old_transport = CaptureTransport(CaptureTransportAssumptions(owned=3)).estimate()
    four_transport = CaptureTransport(CaptureTransportAssumptions(owned=4)).estimate()
    extra_transport = four_transport['transport_bram36']-old_transport['transport_bram36']
    extra_controls = (four_transport['register_reservation_total_bits']
                      -old_transport['register_reservation_total_bits'])
    extra_raw_bits = four_transport['logical_sample_bits']-old_transport['logical_sample_bits']
    return dict(nodes=costs, scenarios=scenarios, additional_serial_engine_information_bits=partial,
        existing_capture_information_lower_bits=150124, combined_information_lower_bits=150124+partial,
        original16kib_budget_bits=131072, original_budget_failed=150124+partial > 131072,
        additional_serial_engine_bram36=blocks, prior_conditional_system_bram36=89,
        one_serial_engine_system_subtotal=89+blocks, four_engine_system_subtotal=89+4*blocks+extra_transport,
        fourth_owned_raw_slot_bram36=extra_transport, fourth_descriptor_control_bits=extra_controls,
        four_engine_information_lower_bits=150124+4*partial+extra_controls+extra_raw_bits,
        four_engine_ports_or_deadlines_verified=False,
        unapproved_reserve25_bram36=35, remaining_after_four_engine_and_unapproved_reserve=140-35-89-4*blocks-extra_transport,
        coefficient_rom_bits=graph['coefficient_bits_after_compression'],
        rom_mapping='tiny compile-time prefix/tail constants,possible LUT implementation;LUT/mux/decode timing unverified,not BRAM-free synthesis evidence',
        required_preparation_engines_every32={str(clock):{
            str(n):int(np.ceil((6*n+8)/(clock/500_000))) for n in (256,376,632)}
            for clock in (100_000_000,160_000_000,200_000_000)},
        no_concurrency_credit='RX preparation,Polar,TX,MAC/AES/DMA share only with proved deadlines;TDD alone gives no credit',
        protected_unallocated=['coarse/tone ports and fixed stability','fine differential/coherent score formats and normalizers',
            'packet timing/carrier/LLR/Polar','TX preparation/output','realtime MAC/AES/DMA/queues'],
        full16msps=False,synthesis_evidence=False)


def summarize(rows):
    """Count unique streams only; alternate chunking is not new observation time."""
    unique = [r for r in rows if r['chunking'] == 'whole']
    samples = sum(r['samples'] for r in unique)
    positive = [r for r in unique if r['positive']]
    negative = [r for r in unique if not r['positive']]
    keys = ('energy_triggers','integer_screen_rejections','full_searches','sync_candidates',
            'header_crc_ok','data_crc_ok')
    counters = {k:sum(r['stats'][k] for r in unique) for k in keys}
    negative_samples = sum(r['samples'] for r in negative)
    return dict(positive_streams=len(positive), negative_streams=len(negative),
        captures=sum(r['captured'] for r in positive), heads=sum(r['head_crc'] for r in positive),
        data_crc=sum(r['data_crc'] for r in positive), full=sum(r['correct'] for r in positive),
        bit_errors=sum(r['bit_errors'] for r in positive), ber_bits=sum(r['ber_bits'] for r in positive),
        negative_samples=negative_samples, negative_seconds=negative_samples/16e6,
        negative_candidates=sum(r['captured'] for r in negative),
        false_deliveries=sum(r['false_deliveries'] for r in unique), total_samples=samples,
        counters=counters, per_million_samples={k:v*1e6/samples for k,v in counters.items()})


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    frozen_path, frozen = Path('tests/data/spline_fixed_format.json'), POLICY
    tree = ast.parse(Path('src/nearlink_sdr/phy/spline_fixed.py').read_text())
    names = {'_rne_ratio','_rne_shift','_tables','_prefix_tail','SplineFixedFormat','FixedUniformCubicEighth'}
    tree.body = [n for n in tree.body if getattr(n, 'name', None) in names]
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == frozen['datapath_ast_sha256']
    source = {str(p): sha(p) for p in [Path(__file__), Path('pyproject.toml'), Path('uv.lock'), frozen_path,
        Path('tests/test_strict_receive.py'), Path('tests/test_psk_field_layout.py'),
        Path('scripts/rx_timing_audit.py'), *Path('src').rglob('*.py')]}
    write(output/'protocol.json', dict(source=source, node_limit=frozen['node_limit'], score_limit=frozen['score_limit'],
        actual_scipy_cubic_source_sha256=sha(Path(inspect.getfile(CubicSpline))),
        start_limit=frozen['start_limit'], center_limit=frozen['center_limit'], source_bits=[18, 21, 24], coefficient_fraction_bits=[14,16,17],
        adoption='node/score+MAC strict zero loss;minimum source width then coefficient fraction bits;no retune on validation',
        all_older_sets='fixed regressions only', max_development_saved_windows_per_stream=8,
        fixed_16msps_accepted=False))
    cases, windows, inputs = [], [], {}
    for meta, profile, x, quiet, wanted in streams():
        path = output/(meta['id']+'-input.npz')
        np.savez_compressed(path, iq=x, quiet=quiet, payload=np.frombuffer(wanted, np.uint8))
        meta['input_sha256'] = sha(path)
        write(output/(meta['id']+'-metadata.json'), meta)
        base, arrays, new_windows = receive(output/(meta['id']+'-reference'), meta, profile, x, quiet, wanted, record_windows=True)
        cases.append((meta, profile, x, quiet, wanted, base, arrays))
        windows.extend((meta['id'], *row) for row in new_windows)
        inputs[meta['id']] = meta
    window_arrays, window_metadata = {}, []
    for j, (ident, x, reference, options) in enumerate(windows):
        window_arrays[f'{j}-input'], window_arrays[f'{j}-reference'] = x, reference
        window_metadata.append(dict(id=j, source_id=ident, options=options,
            input_sha256=hashlib.sha256(x.tobytes()).hexdigest()))
    np.savez_compressed(output/'development-search-windows.npz', **window_arrays)
    write(output/'development-search-windows.json', window_metadata)
    rows = []
    for ib, fb in itertools.product((18, 21, 24), (14, 16, 17)):
        fmt = SplineFixedFormat(ib, fb)
        name = f'format-{ib}-{fb}'
        numeric = node_audit(output/name, fmt)
        searches = []
        for j, (ident, x, reference, options) in enumerate(windows):
            a, b = {}, {}
            sync_candidates.search_candidate_grid(x, 4, reference, 8, diagnostics=a, **options)
            sync_candidates.search_candidate_grid(x, 4, reference, 8, diagnostics=b, **options,
                interpolator='word-eighth', spline_format=fmt)
            score = abs(a['best_score']-b['best_score']) if a['best_score'] is not None and b['best_score'] is not None else 0.
            start = abs(a.get('best_start',0)-b.get('best_start',0))
            searches.append(dict(id=j, source_id=ident, score_error=score, start_error=start,
                passed=a['disposition'] == b['disposition'] and score <= frozen['score_limit'] and start <= frozen['start_limit'],
                reference=a, fixed=b))
        write(output/name/'searches.json', searches)
        rows.append(dict(input_bits=ib, coefficient_fraction_bits=fb, numeric=numeric,
                         search_failures=sum(not r['passed'] for r in searches),
                         cost=fixed_spline_cost(fmt),
                         eligible=numeric['failures'] == 0 and all(r['passed'] for r in searches)))
    write(output/'development-formats.json', rows)
    eligible = [r for r in rows if r['eligible']]
    selected = eligible[0] if eligible else None
    if selected:
        assert all(selected[k] == frozen[k] for k in ('input_bits','coefficient_fraction_bits')), 'Frozen format changed;do not retune the viewed validation set'
    dev_comparisons, selected_development, validation = [], [], None
    if selected:
        fmt = SplineFixedFormat(selected['input_bits'], selected['coefficient_fraction_bits'])
        for meta, profile, x, quiet, wanted, base, arrays in cases:
            row, test_arrays, _ = receive(output/(meta['id']+'-selected'), meta, profile, x, quiet, wanted, fmt=fmt)
            dev_comparisons.append(compare_stream(base, row, arrays, test_arrays))
            selected_development.append(row)
    old_comparisons = []
    if selected:
        fixture = Path('tests/data/sync_carrier_tone_success.npz')
        frozen = json.loads(fixture.with_suffix('.json').read_text())
        assert sha(fixture) == frozen['derived_npz_sha256']
        with np.load(fixture) as ar:
            old_x, old_quiet, old_payload = ar['iq'], ar['quiet'], ar['expected'].tobytes()
        old_profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4,
            whitening_seed=0x52, crc_seed=0x123456))
        old_meta = dict(id='old-success', positive=True, input_sha256=sha(fixture),
                        original_input_sha256=frozen['source_input']['sha256'])
        write(output/'old-success-metadata.json', old_meta)
        base, arrays, _ = receive(output/'old-success-reference', old_meta, old_profile,
                                 old_x, old_quiet, old_payload)
        assert base['correct'] == 1
        for chunking in ('whole', 'fixed', 'random'):
            row, test_arrays, _ = receive(output/('old-success-'+chunking), old_meta, old_profile,
                old_x, old_quiet, old_payload, fmt=fmt, chunking=chunking)
            old_comparisons.append(compare_stream(base, row, arrays, test_arrays))
    development_passed = selected is not None and all(r['passed'] for r in dev_comparisons+old_comparisons)
    write(output/'selection.json', dict(source=source, selected=selected,
        dev_comparisons=dev_comparisons, old_regression_veto_only=old_comparisons, frozen=True,
        proceed_to_new_validation=development_passed))
    # Freeze is physically written before even generating independent validation.
    if development_passed:
        numeric = node_audit(output/'validation-nodes', fmt, held=True)
        comparisons, results = [], []
        for meta, profile, x, quiet, wanted in streams(held=True):
            path = output/(meta['id']+'-input.npz')
            np.savez_compressed(path, iq=x, quiet=quiet, payload=np.frombuffer(wanted, np.uint8))
            meta['input_sha256'] = sha(path)
            write(output/(meta['id']+'-metadata.json'), meta)
            base, arrays, _ = receive(output/(meta['id']+'-reference'), meta, profile, x, quiet, wanted)
            for chunking in ('whole', 'fixed', 'random'):
                row, test_arrays, _ = receive(output/(meta['id']+'-'+chunking), meta, profile, x, quiet, wanted, fmt=fmt, chunking=chunking)
                comparisons.append(compare_stream(base, row, arrays, test_arrays))
                results.append(row)
        validation = dict(numeric=numeric, comparisons=comparisons, results=results, summary=summarize(results),
                          passed=numeric['failures'] == 0 and all(r['passed'] for r in comparisons))
        write(output/'validation.json', validation)
    resources = resource_audit(output, fmt) if selected else None
    failed_streams = [dict(id=r['id'], stage='development',
        reason='miss' if not r['captured'] else 'head_crc' if not r['head_crc'] else 'data_crc_or_content',
        bit_errors=r['bit_errors'], ber_bits=r['ber_bits']) for r in selected_development
        if r['positive'] and not r['correct']]
    if validation:
        failed_streams += [dict(id=r['id'], stage='validation',
            reason='miss' if not r['captured'] else 'head_crc' if not r['head_crc'] else 'data_crc_or_content',
            bit_errors=r['bit_errors'], ber_bits=r['ber_bits']) for r in validation['results']
            if r['positive'] and r['chunking'] == 'whole' and not r['correct']]
    report = dict(formats=rows, resources=resources, selected_format=None if selected is None else {
        k:selected[k] for k in ('input_bits','coefficient_fraction_bits')},
        node_candidate_accepted=bool(validation and validation['passed']),
        validation=validation, development_inputs=inputs, captured_development_windows=len(windows),
        development_whole_results=[r[-2] for r in cases],
        development_selected_results=selected_development,
        development_summary=summarize(selected_development) if selected else None,
        all_remaining_new_stream_failures=failed_streams,
        new_reception_gain_claim=None, original_rrc_2percent_failure_preserved=True,
        original_16kib_failure_preserved=True, main_rx_default_changed=False,
        complete_capture_cycle_verified=False, whole_system_accepted=False,
        source=source, dependencies={k:importlib.metadata.version(k) for k in ('numpy','scipy')})
    write(output/'model.json', report)
    (output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    write(output/'manifest.json', {str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(dict(selected=report['selected_format'], windows=len(windows),
        candidate_accepted=report['node_candidate_accepted'],
        formats=[dict(ib=r['input_bits'],fb=r['coefficient_fraction_bits'],node=r['numeric'],search_failures=r['search_failures']) for r in rows]),indent=2))
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
