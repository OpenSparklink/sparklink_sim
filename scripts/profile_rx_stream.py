"""Profile one hash-checked development IQ capture; baseline timings include no profiler."""
import argparse
import cProfile
import hashlib
import io
import json
import platform
import pstats
import resource
import time
from pathlib import Path

import numpy as np

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL
from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.mac_interface import iq_to_mac_capture
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile


def number(value):
    return int(value, 0) if isinstance(value, str) else int(value)


def load(root, record):
    raw = (root/record['path']).read_bytes()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256']
    v = np.frombuffer(raw, dtype='<i2').reshape(-1, 2)
    return v[:, 0].astype(float)+1j*v[:, 1]


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.original/'frozen-plan.json').read_text())
    row = next(r for r in plan['real_inputs'] if r['sample_id'] == args.sample_id)
    if not row['sample_id'].endswith('rx-00'):
        raise ValueError('Profile only a declared development input; do not tune on held-out captures')
    x = load(args.dataset, row)
    quiets = [r for r in plan['quiet_inputs'] if r['sample_id'] in row['quiet_control_ids']]
    q = np.concatenate([load(args.dataset, r) for r in quiets])
    profile = PSKReceiveProfile.ws63_ft2_a3(sync_word=number(plan['sync_word']),
                                          data_crc_seed=number(plan['crc_seed']), pilot_interval=0)
    t = time.perf_counter()
    old = iq_to_mac_capture(x, q, profile, energy_window_symbols=8)
    batch_seconds = time.perf_counter()-t
    t = time.perf_counter()
    rx = PSKStreamReceiver(profile, q)
    init_seconds = time.perf_counter()-t
    t = time.perf_counter()
    events = rx.feed(x)+rx.finish()
    seconds = time.perf_counter()-t
    diagnostic = {}
    frames = []
    expected = np.packbits(prbs9(row['requested']['length']*8), bitorder='little').tobytes()
    for i, e in enumerate(events):
        for key in ('symbols', 'symbol_centers_raw', 'timing_detector_error', 'timing_rate_ppm'):
            diagnostic[f'{i}-{key}'] = e[key]
        f = {k: v for k, v in e['frame'].items() if k != 'payload'}
        f.update(sync_start_raw=e['sync_start_raw'], content_ok=e['frame']['payload'] == expected,
                 emitted_after_samples=e['emitted_after_samples'])
        frames.append(f)
    np.savez_compressed(out/'diagnostics.npz', **diagnostic)
    profiler = cProfile.Profile()
    profiled = PSKStreamReceiver(profile, q)
    profiler.enable()
    profiled.feed(x)
    profiled.finish()
    profiler.disable()
    profiler.dump_stats(str(out/'stream.pstats'))
    report = io.StringIO()
    pstats.Stats(profiler, stream=report).sort_stats('cumulative').print_stats(30)
    (out/'profile.txt').write_text(report.getvalue())
    root = Path(__file__).resolve().parents[1]
    result = dict(python=platform.python_version(), rust_accel=_HAS_RUST_ACCEL,
                  sources={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (root/'src').rglob('*.py')},
                  input=row, quiet_inputs=quiets, parameters=dict(timing_subdivisions=8,
                  energy_window_symbols=8, bandwidth=.003, max_rate_ppm=500),
                  init_seconds=init_seconds, stream_seconds=seconds, batch_seconds=batch_seconds,
                  stream_Msamples_per_second=len(x)/seconds/1e6, target_Msamples_per_second=16,
                  actual_realtime=seconds <= len(x)/16e6,
                  frames=frames, batch_correct_packets=sum(p.mac_payload == expected for p in old.packets),
                  peak_history_samples=rx.peak_history_samples, max_frame_symbols=rx.max_frame_symbols,
                  state_array_bytes=sum(a.nbytes for a in vars(rx).values() if isinstance(a, np.ndarray)),
                  process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out/'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'input', 'quiet_inputs', 'frames')},
                     indent=2))
    print('frame statuses:', [r['status'] for r in frames])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--sample-id', required=True)
    run(parser.parse_args())
