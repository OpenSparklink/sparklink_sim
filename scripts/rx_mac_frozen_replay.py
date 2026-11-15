"""Replay frozen IQ through the public strict MAC entry; do not read a mutable dataset index."""
import argparse
import hashlib
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rx_timing_audit as audit

from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.mac_interface import iq_to_mac_capture
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile


def checked_raw(root, record):
    path = root / record['path']
    raw = path.read_bytes()
    assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], path
    v = np.frombuffer(raw, dtype='<i2').reshape(-1, 2)
    return v[:, 0].astype(float) + 1j*v[:, 1]


def verify_manifest(root):
    for p, expected in json.loads((root/'manifest.json').read_text()).items():
        observed = audit.identity(root/p)
        assert observed == expected or observed['sha256'] == expected, p


def run(args):
    root = Path(__file__).resolve().parents[1]
    out, replay, original = args.output, args.replay, args.original
    out.mkdir(parents=True, exist_ok=False)
    verify_manifest(replay)
    verify_manifest(original)
    rows = [json.loads(line) for line in (replay/'results.jsonl').read_text().splitlines()]
    original_plan = json.loads((original/'frozen-plan.json').read_text())
    old_quiet = {r['sample_id']: r for r in original_plan['quiet_inputs']}
    new_quiet = {r['input']['sample_id']: r['input'] for r in rows if r['source'] == 'quiet'}
    plan = dict(replay_manifest=audit.identity(replay/'manifest.json'),
                original_manifest=audit.identity(original/'manifest.json'),
                commit=audit.git('rev-parse', 'HEAD'),
                sources={str(p.relative_to(root)): audit.identity(p) for p in (root/'src').rglob('*.py')},
                driver=audit.identity(Path(__file__)), parameters=dict(
                    sync_word=audit.WORD, data_crc_seed=0x555555, pilot_interval=0,
                    timing_subdivisions=8, timing_tracking=True, energy_window_symbols=8,
                    carrier_method='coherent'),
                expected_content_used_only_in_scoring=True, inputs=[])
    for p in rows:
        item = {k: v for k, v in p.items() if k not in ('arms', 'regressions', 'carrier_regressions')}
        plan['inputs'].append(item)
    audit.write(out/'frozen-plan.json', plan)
    profile = PSKReceiveProfile.ws63_ft2_a3(sync_word=audit.WORD, data_crc_seed=0x555555,
                                          pilot_interval=0)
    results, failures, regressions = [], [], []
    for i, p in enumerate(rows):
        if p['source'] == 'simulation':
            path = (replay if p['id'].startswith('new-noise-') else original)/p['input_iq']
            expected = p.get('input_identity') or audit.identity(path)
            assert audit.identity(path) == expected
            with np.load(path) as data:
                x, q = data['iq'], data['quiet']
        elif p['source'] in ('real', 'real-derived'):
            r = p['input']
            q = np.concatenate([checked_raw(args.dataset, old_quiet[s]) for s in r['quiet_control_ids']])
            if p['source'] == 'real':
                x = checked_raw(args.dataset, r)
            else:
                path = original/(p['id']+'.npz')
                assert audit.identity(path) == p['derivative']
                with np.load(path) as data:
                    x = data['iq']
        else:
            r = p['input']
            other = next(new_quiet[s] for s in r['quiet_control_ids'] if s != r['sample_id'])
            x, q = checked_raw(args.quiet_dataset, r), checked_raw(args.quiet_dataset, other)
        t = time.perf_counter()
        received = iq_to_mac_capture(x, q, profile, energy_window_symbols=8)
        seconds = time.perf_counter()-t
        expected = np.packbits(prbs9(p['length']*8), bitorder='little').tobytes()
        frames = []
        for f in received.diagnostics['frames']:
            payload = f['payload']
            valid_content = payload == expected
            frame = {k: v for k, v in f.items() if k != 'payload'}
            frame.update(content_ok=valid_content, payload_sha256=None if payload is None else
                         hashlib.sha256(payload).hexdigest(), whole_packet_ok=bool(
                             f['status'] == 'valid' and valid_content and f['mcs'] ==
                             (12 if p['kind'] == '8PSK' else 8)))
            frames.append(frame)
        old = [r['sync_start_raw'] for r in p['arms']['coherent']['rows'] if r.get('whole_packet_ok')]
        new = [r['sync_start_raw'] for r in frames if r['whole_packet_ok']]
        same = len(old) == len(new) and np.allclose(sorted(old), sorted(new), atol=1e-9, rtol=0)
        row = dict(id=p['id'], source=p['source'], input=p.get('input_identity', p.get('input', p.get('input_iq'))),
                   seconds=seconds, iq_samples=len(x), frames=frames, successful_sync_positions=new,
                   previous_success_positions=old, same_acceptance=bool(same),
                   diagnostic_reference=str(replay/p['arms']['coherent']['evidence']),
                   source_timing_truth=p.get('truth'), original_sro=p.get('original_sro', 'unknown'),
                   injected_ppm=p.get('injected_ppm'),
                   mac_delivered_packets=len(received.packets),
                   energy_segment_count=len(received.diagnostics['acquisition']['bursts']))
        results.append(row)
        if not same:
            regressions.append(row)
        if any(r['status'] != 'valid' or not r['whole_packet_ok'] for r in frames) or (
                p['source'] != 'quiet' and not new):
            failures.append(row)
        if (i+1) % 50 == 0:
            print(i+1, 'frozen inputs replayed', flush=True)
    summary = []
    for group in ('simulation', 'real', 'real-derived', 'quiet'):
        selected = [r for r in results if r['source'] == group]
        summary.append(dict(source=group, inputs=len(selected),
                            correct_packets=sum(len(r['successful_sync_positions']) for r in selected),
                            mac_delivered_packets=sum(r['mac_delivered_packets'] for r in selected),
                            statuses=dict(Counter(f['status'] for r in selected for f in r['frames'])),
                            seconds=sum(r['seconds'] for r in selected)))
    assert all(audit.identity(root/p) == v for p, v in plan['sources'].items())
    audit.write(out/'results.json', results)
    audit.write(out/'summary.json', dict(groups=summary, acceptance_mismatches=len(regressions),
                source_integrity=True, process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    audit.write(out/'failures.json', failures)
    audit.write(out/'regressions.json', regressions)
    audit.write(out/'manifest.json', {str(p.relative_to(out)): audit.identity(p)
                                    for p in out.rglob('*') if p.is_file()})
    return bool(regressions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('replay', 'original', 'dataset', 'quiet-dataset', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
