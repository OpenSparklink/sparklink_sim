"""Frozen global-spline structure audit; exit1 retains the incomplete resource gate."""

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import math
import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from scipy.interpolate import CubicSpline

from nearlink_sdr.phy import sync_candidates
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import ToneNotchConfig
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.phy.uniform_spline import (
    UniformCubicEighth,
    preparation_calendar,
    spline_work,
)
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions


def windows(held=False):
    """Window-level validation, NOT a new unknown-start RF/SNR acceptance set."""
    rng = np.random.default_rng(620219 if held else 620117)
    lengths = [5, 13, 65, 127, 255, 513, 639] if held else [4, 8, 17, 64, 128, 256, 512, 640]
    exponents = [-219, -17, 3, 71, 219] if held else [-220, -40, 0, 40, 220]
    for n in lengths:
        t, k = np.linspace(0, 1, n), np.arange(n)
        polynomial = (t**3-2*t**2+.7*t+1)+1j*(.2*t**3+.3*t**2-t)
        chirp = np.exp(2j*np.pi*(.039*k+.00091*k*k))
        noise = rng.normal(size=n)+1j*rng.normal(size=n)
        symbols = np.exp(2j*np.pi*rng.integers(0, 8, n)/8+1j*k*np.pi/8)
        if held:
            # Different waveform/configuration combinations, not just another seed.
            pulse = np.zeros(n, complex)
            pulse[n//3:min(n, n//3+3)] = [1j, -3, 2j][:min(3, n-n//3)]
            waves = dict(polynomial=polynomial, paired_tones=chirp+.73*np.exp(-.37j*k),
                         mixed_modulation=symbols*np.where(k < n//3, .03125, 4),
                         random_walk=np.cumsum(noise)/n, short_burst=pulse+.03*noise)
        else:
            pulse = np.zeros(n, complex)
            pulse[n//2] = 3-2j
            waves = dict(polynomial=polynomial, chirp=chirp, rotating8psk=symbols,
                         strong_weak=noise*np.where(k < n//2, 1., 2**-40),
                         impulse=pulse, tone_noise=chirp+.11*noise,
                         constant=np.full(n, 1+.3j), zero=np.zeros(n, complex))
        for kind, wave in waves.items():
            for exponent in exponents:
                yield f'{kind}-{n}-{exponent}', wave*2.**exponent, kind


def window_audit(output, held):
    rows, data, failures = [], {}, []
    for ident, y, kind in windows(held):
        n = len(y)
        times = np.arange(8*(n-1)+1)/8
        node = UniformCubicEighth(y)
        actual = np.concatenate([node(times[i:i+128]) for i in range(0, len(times), 128)])
        reference = CubicSpline(np.arange(n), y)(times)
        scale = float(np.max(abs(y)))
        error = float(np.max(abs(actual-reference)))/scale if scale else 0.
        analytic_error = None
        if kind == 'polynomial':
            t = times/(n-1)
            truth = (t**3-2*t**2+.7*t+1)+1j*(.2*t**3+.3*t**2-t)
            # y[0] is exactly the construction's scale; not inferred from output.
            analytic_error = float(np.max(abs(actual-truth*y[0])))/scale
        row = dict(id=ident, samples=n, input_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                   relative_peak_error=error, independent_polynomial_error=analytic_error,
                   passed=error <= 2e-12 and (analytic_error is None or analytic_error <= 2e-12))
        rows.append(row)
        if not row['passed']:
            failures.append(row)
        # Keep all inputs and both results, not only passing summaries.
        data[ident+'-input'], data[ident+'-reference'], data[ident+'-structure'] = y, reference, actual
    cohort = 'validation' if held else 'development'
    np.savez_compressed(output/f'{cohort}-windows.npz', **data)
    write(output/f'{cohort}-windows.json', dict(rows=rows, failures=failures))
    return dict(windows=len(rows), failures=len(failures),
                max_relative_peak_error=max(r['relative_peak_error'] for r in rows),
                independent_polynomials=sum(r['independent_polynomial_error'] is not None for r in rows),
                iq_to_mac_gain_measured=False)


def search_audit(output):
    rng = np.random.default_rng(620117)
    ref = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4)).sync_symbols()
    rows, traces = [], {}
    for ident in range(24):
        y = rng.normal(size=256)+1j*rng.normal(size=256)
        if ident % 3 == 0:
            y += 10*np.exp(2j*np.pi*.037125*np.arange(256))
        elif ident % 3 == 1:
            y[79:86] += 20j
        for peaks in (1, 2):
            results, ds = [], []
            for backend in ('reference', 'uniform-eighth'):
                d = {}
                r = sync_candidates.search_candidate_grid(y, 4, ref, 8,
                    candidate_count=peaks, lookahead=16, diagnostics=d, interpolator=backend)
                ds.append(d)
                results.append(r)
                if r is not None:
                    traces[f'{ident}-{peaks}-{backend}'] = r['phase_symbols']
            a, b = results
            passed = ((a is None) == (b is None) and ds[0]['disposition'] == ds[1]['disposition']
                      and ds[0]['best_start'] == ds[1]['best_start']
                      and abs(ds[0]['best_score']-ds[1]['best_score']) <= 2e-12)
            rows.append(dict(id=ident, peaks=peaks, passed=passed, reference=ds[0], structure=ds[1]))
    np.savez_compressed(output/'search-trajectories.npz', **traces)
    write(output/'searches.json', rows)
    return dict(paired_searches=len(rows), failures=sum(not r['passed'] for r in rows),
                scope='unchanged integer candidate selection/score/order/wait;random windows,not packet recall')


def mac_audit(output):
    folder = Path(__file__).resolve().parents[1]/'tests/data'
    path = folder/'sync_carrier_tone_success.npz'
    meta = json.loads((folder/'sync_carrier_tone_success.json').read_text())
    assert sha(path) == meta['derived_npz_sha256']
    with np.load(path) as ar:
        iq, quiet, wanted = ar['iq'], ar['quiet'], ar['expected'].tobytes()
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4,
                                                whitening_seed=0x52, crc_seed=0x123456))
    rows, arrays = [], {}
    original = sync_candidates.search_candidate_grid
    for kind, x in [('old-success', iq), ('zero', np.zeros(16384, complex)),
                    ('noise', np.resize(quiet, 16384))]:
        for chunking in ('whole', 'fixed', 'random'):
            for backend in ('reference', 'uniform-eighth'):
                rx = MacStreamReceiver(profile, quiet, quiet_batching=False, defer_quiet_fir=False,
                    correlation_backend='scaled', normalized_coarse=True,
                    tone_notch=ToneNotchConfig(radius=31/32), sync_search_method='bounded-grid',
                    sync_candidate_count=1, sync_coherent_threshold=.45)
                def selected(*args, **kwargs):
                    return original(*args, **kwargs, interpolator=backend)
                rng, offset, packets, events = np.random.default_rng(620120), 0, [], []
                with patch.object(sync_candidates, 'search_candidate_grid', selected):
                    while offset < len(x):
                        count = len(x) if chunking == 'whole' else 257 if chunking == 'fixed' else int(rng.integers(1, 1001))
                        r = rx.feed(x[offset:offset+count])
                        packets += r.packets
                        events += r.diagnostics['events']
                        offset += count
                    r = rx.finish()
                    packets += r.packets
                    events += r.diagnostics['events']
                payloads = [p.mac_payload.hex() for p in packets]
                frames = [e['frame'] for e in events]
                ident = f'{kind}-{chunking}-{backend}'
                for i, e in enumerate(events):
                    for k, v in e.items():
                        if isinstance(v, np.ndarray):
                            arrays[f'{ident}-{i}-{k}'] = v
                summaries = [{k: e[k] for k in ('sync_start_raw', 'sync_score', 'cfo_hz',
                                                'emitted_after_samples', 'used_end_padding')}
                             for e in events]
                for summary, frame in zip(summaries, frames, strict=True):
                    summary['frame'] = {k: v.hex() if isinstance(v, bytes) else
                                        v.tolist() if isinstance(v, np.ndarray) else
                                        v.item() if isinstance(v, np.generic) else v
                                        for k, v in frame.items()}
                rows.append(dict(id=ident, samples=len(x), packets=payloads, events=summaries,
                    capture_stats=rx.phy.capture_stats,
                    captures=len(events), head_crc=sum(f.get('header_crc') is True for f in frames),
                    data_crc=sum(f.get('data_crc') is True for f in frames),
                    passed=payloads == ([wanted.hex()] if kind == 'old-success' else [])))
    comparisons = []
    for kind in ('old-success', 'zero', 'noise'):
        reference = next(r for r in rows if r['id'] == f'{kind}-whole-reference')
        for row in [r for r in rows if r['id'].startswith(kind+'-')]:
            base, current = reference['id'], row['id']
            same_decisions = all(row[k] == reference[k] for k in ('packets', 'captures', 'head_crc', 'data_crc'))
            max_center_error = max((float(np.max(abs(value-arrays[current+key[len(base):]])))
                for key, value in arrays.items() if key.startswith(base+'-')
                and key.endswith('-symbol_centers_raw') and len(value)), default=0.) if same_decisions else None
            comparisons.append(dict(reference=base, current=current, same_decisions=same_decisions,
                                    max_center_error=max_center_error,
                                    passed=same_decisions and max_center_error <= .25))
    np.savez_compressed(output/'mac-event-arrays.npz', **arrays)
    write(output/'mac.json', dict(rows=rows, comparisons=comparisons))
    return dict(cases=len(rows), failures=sum(not r['passed'] for r in rows)
                +sum(not r['passed'] for r in comparisons),
                positive_input_sha256=sha(path), original_iq_sha256=meta['iq_bytes_sha256'],
                negative_samples_per_backend=32768, negative_duration_seconds_per_backend=32768/16e6,
                unique_positive_packets=1, new_reception_gain_claim=None,
                scope='fixed old synthetic IQ regression;not independent RF acceptance or real capture')


def schedules(output):
    calendar = list(preparation_calendar())
    write(output/'preparation-calendar.json', calendar)
    words = np.random.default_rng(620121).integers(0, 2**46, (4096, 2), dtype=np.uint64)
    np.save(output/'transport-words.npy', words)
    rows = []
    graph = spline_work()
    for clock in (100_000_000, 160_000_000, 200_000_000):
        for every in (32, 256):
            # Same phase scoring; its arithmetic, normalizers and ports remain
            # optimistic reservations. Preparation cannot overlap another job
            # on this ONE engine; no sharing credit with RRC/TX/decoder.
            units = {100_000_000: 4, 160_000_000: 6, 200_000_000: 7}[clock]
            scoring = max(288, math.ceil(1413/units), math.ceil(3200/11))
            confirmation = max(36, math.ceil(324/units))
            service = len(calendar)+scoring+confirmation
            model = CaptureTransport(CaptureTransportAssumptions(clock_hz=clock,
                fine_service_cycles=service))
            events = []
            checked = 0
            def sink(event, received):
                nonlocal checked
                if received is not None:
                    stop, path = event['after_samples'], event['path']
                    assert np.array_equal(received, words[stop-256:stop, path])
                    checked += 1
                events.append(event)
            report = model.simulate(words, [(j, (j//every) % 2) for j in range(256, 4097, every)], sink=sink)
            name = f'queue-{clock}-{every}.json'
            write(output/name, events)
            rows.append(dict(clock_hz=clock, request_interval_samples=every,
                preparation_cycles=len(calendar), optimistic_scoring_cycles=scoring,
                optimistic_confirmation_cycles=confirmation, total_reserved_cycles=service,
                fine_real_product_units_scenario=units, fine_wide_add_units_scenario=11,
                service_word_format_verified=False, report=report, checked_windows=checked,
                events_file=name, transport=model.estimate()))
    # No retroactive budget change; this is a separate, additional workspace.
    return dict(graph=graph, scenarios=rows, all_preparation_cycles=len(calendar),
        source_read_ports=2, work_read_ports=1, work_write_ports=1,
        q_workspace_scenario_bits=256*64, q_workspace_tdp36_bram36_per_context=2,
        q_workspace_single_serial_engine_bram36=2,
        q_workspace_for_three_parallel_contexts_bram36=6,
        preparation_pipeline_control_ff_scenario=1024,
        pipeline_reservation='eight64bit live words,256bit control,256bit counters;not synthesized',
        held_last_second_and_feedback_bits_unverified=True,
        coefficient_rom_bits_scenario=1276*18, coefficient_format_verified=False,
        bram_conditional_prior_system_subtotal=89, unapproved_chip_reserve=35,
        incremental_scratch_scenario_bram36=2, subtotal_plus_serial_scratch=91,
        subtotal_plus_three_context_scratch=95,
        missing=['coefficient port/width mapping', 'fixed recurrence stability and phase accuracy',
                 'score normalizer/angle/compare pipeline', 'packet loops/Polar/TX/realtime MAC',
                 'interpolation source/M two-read arbitration and parallel preparation'],
        synthesis_evidence=False, full_capture_schedule_closed=False,
        original_16kib_failure_preserved=True)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(p): sha(p) for p in [Path(__file__), Path('pyproject.toml'), Path('uv.lock'),
                                      *Path('src').rglob('*.py')]}
    # Written before generating ANY development/validation inputs. Parameter-free
    # candidate and limits cannot be selected against the validation results.
    reference_source_hash = sha(Path(inspect.getfile(CubicSpline)))
    write(output/'protocol.json', dict(sources=sources,
        actual_scipy_cubic_source_sha256=reference_source_hash, float_error_limit=2e-12,
        amplitude_exponents=[-220, 220], grid='existing nine eighths per1/2peaks',
        fixed_format_accepted=False, default_promotion=False,
        adoption='float structure only;no retuning on validation',
        budget_policy='prospective reasoned changes allowed by user;old2percent/16KiB failures retained;no new ceilings set here'))
    results = dict(development=window_audit(output, False), validation=window_audit(output, True),
                   search=search_audit(output), mac=mac_audit(output), cost=schedules(output))
    numeric = all(results[k]['failures'] == 0 for k in ('development', 'validation', 'search', 'mac'))
    results.update(float_structure_pass=numeric, whole_system_accepted=False,
                   fixed_format_accepted=False, default_rx_changed=False,
                   actual_scipy_cubic_source_sha256=reference_source_hash,
                   source_hashes=sources, dependencies={k: importlib.metadata.version(k) for k in ('numpy', 'scipy')},
                   python=platform.python_version())
    write(output/'model.json', results)
    (output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    write(output/'manifest.json', {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps({k: results[k] for k in ('development', 'validation', 'search', 'mac', 'float_structure_pass')}, indent=2))
    return 1  # Incomplete fixed/resource gate; CI separately requires the float gate.


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
