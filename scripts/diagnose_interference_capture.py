"""Read-only acquisition funnel and bypass experiment; oracle data never selects an answer."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, score, sha, write

from nearlink_sdr.phy.psk_sync import PSKCaptureConfig, search_psk_sync
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver


class FunnelObserver(PSKStreamReceiver):
    """Offline-only trace after actual processing; production state has no truth input."""

    def __init__(self, *args, trace_path=None, trace_ranges=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.trace_path, self.trace_ranges = trace_path, tuple(trace_ranges)

    def _near(self):
        # Used only to limit diagnostic logging, never any receiver decision.
        return self.trace_path is not None and any(
            a <= self._processed <= b for a, b in self.trace_ranges)

    def _log(self, row):
        with self.trace_path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')

    def _search(self):
        snapshot = None
        if self._near() and len(self._filtered) >= self.search_samples:
            z = self._filtered[-self.search_samples:].copy()
            origin = self._origin+len(self._filtered)-len(z)
            snapshot = z, origin, max(1., self._skip_until-origin)
        super()._search()
        if snapshot is None:
            return
        z, origin, lower = snapshot
        result = dict(type='search', after=self._processed, origin=origin,
            lower_local=lower, upper_raw=origin+len(z)-1-(len(self.reference)-1)*self.sps,
            skip_until=self._skip_until, energy_peak=float(np.max(self._energy, initial=0)),
            energy_threshold=self.threshold, energy_open=bool(np.any(self._energy > self.threshold)),
            coarse_threshold=self.capture_config.integer_screen, fine_threshold=.85,
            active_start=None if self._active is None else self._active['start'])
        for name, subdivisions in [('integer', 1), ('fine', self.subdivisions)]:
            diag = {}
            best = search_psk_sync(z, self.sps, self.reference, subdivisions,
                min_start=lower, lookahead=0, diagnostics=diag)
            start = None if best is None else origin+best['phase']+best['offset']*self.sps
            future = best is not None and (start-origin+(len(self.reference)-1)*self.sps+
                                           4*self.sps > len(z)-1)
            result[name] = dict(best_score=diag['best_score'], best_start=start,
                needs_future=bool(future), positions=diag['scored_positions'],
                disposition=diag['disposition'])
        result['reason'] = ('energy_closed' if not result['energy_open'] else
            'integer_rejected' if self.capture_config.integer_screen is not None and
                (result['integer']['best_score'] is None or
                 result['integer']['best_score'] < self.capture_config.integer_screen) else
            'fine_below_threshold' if result['fine']['best_score'] is None or
                result['fine']['best_score'] < .85 else
            'maximum_needs_future' if result['fine']['needs_future'] else 'ready_qualified')
        self._log(result)

    def _process(self, block):
        before = None if self._active is None else self._active['start']
        skip = self._skip_until
        events = super()._process(block)
        if self._near():
            self._log(dict(type='state', after=self._processed, active_before=before,
                active_after=None if self._active is None else self._active['start'],
                skip_before=skip, skip_after=self._skip_until,
                events=[dict(start=e['sync_start_raw'], status=e['frame']['status']) for e in events]))
        return events


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out/'evidence').mkdir()
    old = json.loads((args.baseline/'results.json').read_text())
    sources = {str(p): sha(p) for p in Path('src').rglob('*.py')}
    write(out/'frozen-plan.json', dict(baseline=str(args.baseline),
        baseline_sha256=sha(args.baseline/'results.json'), source_hashes=sources,
        driver_sha256=sha(Path(__file__)),
        diagnostic_only='Bypass skips integer screening only. Truth limits logs/associates outputs.'))
    (out/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for old_row in old:
        meta = old_row['input']
        path = Path(meta['path'])
        assert sha(path) == meta['sha256']
        with np.load(path) as data:
            x, q = data['iq'], data['quiet']
        arms, traces = {}, {}
        inspect = bool(old_row['missed'] or (meta['kind'] == 'burst' and
                       old_row['correct'] < old_row['known_packets']))
        ranges = [(p['sync_start']-32, p['sync_start']+320) for p in meta['packets']]
        for arm, gate in [('baseline', .6), ('bypass', None)]:
            rx = FunnelObserver(profile(meta['mcs']), q, quiet_batching=False,
                capture_config=PSKCaptureConfig(2., gate),
                trace_path=out/'evidence'/(meta['id']+'-'+arm+'.jsonl') if inspect else None,
                trace_ranges=ranges)
            events = rx.feed(x)+rx.finish()
            scored = score(events, meta)
            arms[arm] = dict(**scored, stats=rx.capture_stats,
                per_million={k: v*1e6/len(x) for k, v in rx.capture_stats.items()},
                candidates_per_million=len(events)*1e6/len(x),
                false_deliveries_per_million=scored['false_deliveries']*1e6/len(x))
            for i, e in enumerate(events):
                for name in ('symbols', 'symbol_centers_raw', 'timing_rate_ppm', 'timing_detector_error'):
                    traces[f'{arm}-{i}-{name}'] = e[name]
        assert arms['baseline']['frames'] == old_row['frames'], meta['id']
        np.savez_compressed(out/'evidence'/(meta['id']+'.npz'), **traces)
        rows.append(dict(input=meta, arms=arms,
                         diagnostic_recovered=arms['bypass']['captured']-arms['baseline']['captured']))
        write(out/'results.json', rows)
    summary = {arm: dict(inputs=len(rows), captured=sum(r['arms'][arm]['captured'] for r in rows),
        correct=sum(r['arms'][arm]['correct'] for r in rows),
        false_candidates=sum(r['arms'][arm]['false_candidates'] for r in rows),
        false_deliveries=sum(r['arms'][arm]['false_deliveries'] for r in rows),
        full_searches=sum(r['arms'][arm]['stats']['full_searches'] for r in rows),
        phase_hypotheses=sum(r['arms'][arm]['stats']['computed_phase_hypotheses'] for r in rows),
        scored_positions=sum(r['arms'][arm]['stats']['computed_scored_positions'] for r in rows))
        for arm in ('baseline', 'bypass')}
    write(out/'summary.json', summary)
    write(out/'failures.json', [r for r in rows if r['arms']['baseline']['missed'] or
                               r['arms']['bypass']['correct'] < len(r['input']['packets'])])
    assert sources == {str(p): sha(p) for p in Path('src').rglob('*.py')}
    write(out/'manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
