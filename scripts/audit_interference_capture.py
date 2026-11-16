"""Independent new interference inputs and frozen opt-in tone-suppression comparisons."""

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audit_capture_hardware import KINDS, profile, score, sha, vector, write

from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def generate(out, split):
    out.mkdir(parents=True, exist_ok=False)
    seed = 370117 if split == 'development' else 380117
    cases = []
    for kind in KINDS:
        for positive in (False, True) if kind != 'zero' else (False,):
            for rep in range(8 if split == 'development' else 12):
                rng = np.random.default_rng(seed+rep)
                options = (dict(tone_frequency=float(rng.uniform(-.2, .2)), tone_phase=1.03)
                           if kind == 'tone' else {})
                x, q, meta = vector(kind, rep, seed, positive, **options)
                ident = f'{kind}-{int(positive)}-{rep}'
                path = out/(ident+'.npz')
                np.savez_compressed(path, iq=x, quiet=q)
                cases.append(dict(id=ident, path=str(path.resolve()), sha256=sha(path),
                                  samples=len(x), **meta))
    write(out/'frozen-inputs.json', dict(seed=seed, split=split, cases=cases,
        generator_sha256=sha(Path(__file__)), transmitter='independent analytic RRC/angle mapper'))
    (out/'generator.executed.py').write_bytes(Path(__file__).read_bytes())


def evaluate(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'evidence').mkdir()
    plan = json.loads((args.inputs/'frozen-inputs.json').read_text())
    config = None if args.radius == 'off' else ToneNotchConfig(
        radius=float(args.radius), release_blocks=args.release_blocks)
    options = dict(quiet_batching=False, tone_notch=config, correlation_backend=args.backend,
                   correlation_input_bits=14, correlation_input_full_scale=1024.,
                   correlation_mantissa_bits=args.mantissa_bits,
                   sync_search_method=args.sync_method,normalized_coarse=args.normalized_coarse)
    sources = {str(p): sha(p) for p in Path('src').rglob('*.py')}
    write(out/'frozen-plan.json', dict(input_plan=str(args.inputs/'frozen-inputs.json'),
        input_plan_sha256=sha(args.inputs/'frozen-inputs.json'),
        tone=None if config is None else asdict(config), backend=args.backend,
        mantissa_bits=args.mantissa_bits if args.backend == 'scaled' else None,
        sync_search_method=args.sync_method,normalized_coarse=args.normalized_coarse,
        chunks=args.chunks, chunk_seed=370120, source_hashes=sources,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        driver_sha256=sha(Path(__file__))))
    (out/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta in plan['cases']:
        path = Path(meta['path'])
        assert sha(path) == meta['sha256']
        with np.load(path) as data:
            x, q = data['iq'], data['quiet']
        rx = PSKStreamReceiver(profile(meta['mcs']), q, **options)
        events, start, rng = [], 0, np.random.default_rng(370120)
        while start < len(x):
            size = (len(x) if args.chunks == 'whole' else 257 if args.chunks == 'fixed'
                    else int(rng.integers(1, 1001)))
            events += rx.feed(x[start:start+size])
            start += size
        events += rx.finish()
        traces = {f'{i}-{name}': e[name] for i, e in enumerate(events) for name in
                  ('symbols', 'symbol_centers_raw', 'timing_rate_ppm', 'timing_detector_error')}
        np.savez_compressed(out/'evidence'/(meta['id']+'.npz'), **traces)
        result = score(events, meta)
        rows.append(dict(input=meta, **result, stats=rx.capture_stats,
            timing_backend=rx.timing_backend, integer_backend=rx.integer_screen_backend,
            tone_stats=None if rx.tone_notch is None else rx.tone_notch.stats,
            tone_operation_estimate=None if rx.tone_notch is None else rx.tone_notch.operation_estimate(),
            per_million={k: v*1e6/len(x) for k, v in rx.capture_stats.items()},
            bank_stats=None if rx.correlation_bank is None else rx.correlation_bank.stats))
        write(out/'results.json', rows)
    summary = {key: sum(r[key] for r in rows) for key in
               ('known_packets', 'captured', 'correct', 'missed', 'false_candidates', 'false_deliveries')}
    summary.update(inputs=len(rows), samples=sum(r['input']['samples'] for r in rows),
        full_searches=sum(r['stats']['full_searches'] for r in rows),
        active_tone_samples=sum((r['tone_stats'] or {}).get('active_samples', 0) for r in rows))
    write(out/'summary.json', summary)
    write(out/'failures.json', [r for r in rows if r['correct'] < r['known_packets'] or r['false_candidates']])
    assert sources == {str(p): sha(p) for p in Path('src').rglob('*.py')}
    write(out/'manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', choices=['development', 'held-out'])
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--radius', choices=['off', '0.9375', '0.96875'], default='off')
    parser.add_argument('--release-blocks', type=int, default=128)
    parser.add_argument('--backend', choices=['reference', 'structure', 'fixed', 'scaled'],
                        default='reference')
    parser.add_argument('--mantissa-bits', type=int, default=10)
    parser.add_argument('--sync-method',choices=['reference','local-parabolic'],default='reference')
    parser.add_argument('--normalized-coarse',action='store_true')
    parser.add_argument('--chunks', choices=['whole', 'fixed', 'random'], default='whole')
    args = parser.parse_args()
    if args.generate:
        generate(args.output, args.generate)
    elif args.inputs:
        evaluate(args)
    else:
        parser.error('Need --generate or --inputs')
