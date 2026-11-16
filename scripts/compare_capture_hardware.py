"""Compare acquisition-model outcomes with frozen budgets, retaining every changed input."""
import argparse
import json
from pathlib import Path

import numpy as np


def run(before, after, output):
    original = {r['input']['id']: r for r in json.loads((before/'results.json').read_text())}
    candidate = {r['input']['id']: r for r in json.loads((after/'results.json').read_text())}
    assert original.keys() == candidate.keys()
    regressions, changes = [], []
    maxima = dict(centers=0., symbols=0., detector=0., rate=0.)
    for ident, old in original.items():
        new = candidate[ident]
        assert old['input']['sha256'] == new['input']['sha256']
        lost = [f['sync_start'] for f in old['frames'] if f['correct'] and not any(
            g['correct'] and abs(g['sync_start']-f['sync_start']) <= .125 for g in new['frames'])]
        delta = dict(id=ident, input_sha256=old['input']['sha256'], lost_successes=lost,
                     captures=[old['captured'], new['captured']],
                     correct=[old['correct'], new['correct']],
                     false_candidates=[old['false_candidates'], new['false_candidates']],
                     false_deliveries=[old['false_deliveries'], new['false_deliveries']])
        if lost or new['false_candidates'] > old['false_candidates'] or new['false_deliveries']:
            regressions.append(delta)
        if old['frames'] != new['frames']:
            changes.append(delta)
        with (np.load(before/'evidence'/(ident+'.npz')) as a,
              np.load(after/'evidence'/(ident+'.npz')) as b):
            for i, f in enumerate(old['frames']):
                pairs = [(j, g) for j, g in enumerate(new['frames'])
                         if abs(g['sync_start']-f['sync_start']) <= 4]
                if len(pairs) != 1:
                    continue
                j, _ = pairs[0]
                for field, label in [('symbols', 'symbols'), ('symbol_centers_raw', 'centers'),
                                     ('timing_rate_ppm', 'rate'), ('timing_detector_error', 'detector')]:
                    x, y = a[f'{i}-{field}'], b[f'{j}-{field}']
                    n = min(len(x), len(y))
                    maxima[label] = max(maxima[label], float(np.max(abs(x[:n]-y[:n]), initial=0)))
    report = dict(inputs=len(original), regressions=regressions, changed_inputs=changes,
                  matched_trajectory_max_deltas=maxima,
                  trajectory_budget_samples=.25,
                  budget_passed=not regressions and maxima['centers'] <= .25)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    return not report['budget_passed']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.before, args.after, args.output))
