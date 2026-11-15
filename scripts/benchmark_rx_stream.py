"""Paired development-IQ benchmark for the profiled quiet batching optimization."""
import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
from profile_rx_stream import load, number

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL
from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    original = json.loads((args.original/'frozen-plan.json').read_text())
    selected = [r for r in original['real_inputs'] if r['sample_id'] in args.sample_id]
    if len(selected) != len(args.sample_id) or any(not r['sample_id'].endswith('rx-00') for r in selected):
        raise ValueError('Only explicit frozen development IDs are allowed')
    root = Path(__file__).resolve().parents[1]
    plan = dict(rust_accel=_HAS_RUST_ACCEL, repeats=args.repeats,
                sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (root/'src').rglob('*.py')}, inputs=selected,
                parameters=dict(subdivisions=8, energy_window_symbols=8, bandwidth=.003,
                                 max_rate_ppm=500, quiet_batch_quanta=128),
                comparison_tolerance=dict(symbol_abs=1e-12,center_samples_abs=1e-10,rate_ppm_abs=1e-8))
    (out/'frozen-plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False)+'\n')
    results = []
    profile = PSKReceiveProfile.ws63_ft2_a3(sync_word=number(original['sync_word']),
                                          data_crc_seed=number(original['crc_seed']), pilot_interval=0)
    for r in selected:
        x = load(args.dataset, r)
        quiet_records = [s for s in original['quiet_inputs'] if s['sample_id'] in r['quiet_control_ids']]
        q = np.concatenate([load(args.dataset, s) for s in quiet_records])
        expected = np.packbits(prbs9(r['requested']['length']*8), bitorder='little').tobytes()
        times = {False: [], True: []}
        measurements, reference = [], None
        for trial in range(args.repeats+1):
            for enabled in ([False, True] if trial % 2 == 0 else [True, False]):
                t = time.perf_counter()
                rx = PSKStreamReceiver(profile, q, quiet_batching=enabled)
                initialization = time.perf_counter()-t
                t = time.perf_counter()
                events = rx.feed(x)+rx.finish()
                seconds = time.perf_counter()-t
                if reference is None:
                    reference = events
                assert len(events) == len(reference)
                for a, b in zip(reference, events, strict=True):
                    assert a['frame'] == b['frame']
                    assert a['emitted_after_samples'] == b['emitted_after_samples']
                    for field, tolerance in [('symbols', 1e-12), ('symbol_centers_raw', 1e-10),
                                             ('timing_rate_ppm', 1e-8)]:
                        np.testing.assert_allclose(a[field], b[field], atol=tolerance, rtol=0)
                if trial:
                    times[enabled].append(seconds)
                    latency = [e['emitted_after_samples']-(e['symbol_centers_raw'][-1]+1) for e in events]
                    measurements.append(dict(trial=trial, enabled=enabled, init_seconds=initialization,
                         seconds=seconds, correct_packets=sum(e['frame']['payload'] == expected for e in events),
                         quiet_fast_path_samples=rx.quiet_fast_path_samples,
                         state_array_bytes=sum(a.nbytes for a in vars(rx).values() if isinstance(a, np.ndarray)),
                         availability_delay_samples_percentiles=dict(zip(['min','p50','p95','max'],
                             np.percentile(latency, [0,50,95,100]).tolist(), strict=True))))
        summary = dict(id=r['sample_id'], input_sha256=r['sha256'], samples=len(x),
                       baseline_median_seconds=float(np.median(times[False])),
                       optimized_median_seconds=float(np.median(times[True])),
                       median_speedup=float(np.median(times[False])/np.median(times[True])),
                       optimized_Msamples_per_second=float(len(x)/np.median(times[True])/1e6),
                       target_Msamples_per_second=16, trajectories_match=True, measurements=measurements)
        results.append(summary)
        np.savez_compressed(out/(r['sample_id'].replace('/','-')+'-baseline-traces.npz'),
                            **{f'{i}-{k}': e[k] for i,e in enumerate(reference)
                               for k in ('symbols','symbol_centers_raw','timing_rate_ppm','timing_detector_error')})
        print({k:v for k,v in summary.items() if k!='measurements'}, flush=True)
    assert all(hashlib.sha256((root/f).read_bytes()).hexdigest() == h for f,h in plan['sources'].items())
    (out/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    (out/'memory.json').write_text(json.dumps(dict(cumulative_process_peak_rss_KiB=
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, includes_input_calibration_and_both_methods=True), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original','dataset','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--sample-id', action='append', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    run(parser.parse_args())
