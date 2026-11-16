"""Development-only fixed-width scan with independent direct-score/rank and packet budgets."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import evaluate, profile, sha, write
from compare_capture_hardware import run as compare
from scipy.signal import lfilter

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, FixedDifferentialBank
from nearlink_sdr.phy.psk import rrc_filter


def survey(inputs, output, bits, full_scale=1024.):
    output.mkdir(parents=True, exist_ok=False)
    plan = json.loads((inputs/'frozen-inputs.json').read_text())
    rows = []
    for meta in plan['cases']:
        path = Path(meta['path'])
        assert sha(path) == meta['sha256']
        with np.load(path) as data:
            filtered = lfilter(rrc_filter(.4, 4), [1.], data['iq'])
        ref = profile(meta['mcs']).sync_symbols()
        dr = ref[1:]*ref[:-1].conj()
        count = len(filtered)-4*(len(ref)-1)
        wanted, clipped = np.zeros(count), np.zeros(count, bool)
        limit = 1 << (bits-1)
        qr, qi = np.rint(filtered.real/full_scale*limit), np.rint(filtered.imag/full_scale*limit)
        clipped_samples = (qr < -limit) | (qr >= limit) | (qi < -limit) | (qi >= limit)
        for phase in range(4):
            y = filtered[phase::4]
            d = y[1:]*y[:-1].conj()
            c = np.correlate(d, dr, 'valid')
            e = np.convolve(abs(d)**2, np.ones(len(dr)), 'valid')*sum(abs(dr)**2)
            score = np.zeros(len(e))
            np.divide(abs(c), np.sqrt(e), out=score, where=e > 0)
            wanted[phase::4] = score
            clipped[phase::4] = np.convolve(clipped_samples[phase::4],
                                           np.ones(len(ref)), 'valid') > 0
        bank = FixedDifferentialBank(ref, 4, 256, input_full_scale=full_scale,
                                     format=FixedBankFormat(bits))
        actual, fixed_hits = [], []
        for pos in range(0, len(filtered), 32):
            points = bank.advance(filtered[pos:pos+32], inspect=True)
            actual.extend(np.sqrt(p.score_squared) for p in points)
            fixed_hits.extend(p.energy > 0 and (p.numerator << bank.format.threshold_fractional_bits)
                              >= bank.widths['threshold_integer']*p.energy for p in points)
        actual = np.array(actual)
        assert len(actual) == len(wanted)
        good = (wanted >= .55) & ~clipped
        loss = []
        for end in range(256, len(filtered)+1, 32):
            a, b = end-256+1, end-4*(len(ref)-1)
            if np.any(clipped[a:b]) or np.max(wanted[a:b]) < .55:
                continue
            chosen = int(actual[a:b].argmax())
            loss.append(float(np.max(wanted[a:b])-wanted[a+chosen]))
        row = dict(id=meta['id'], input_sha256=meta['sha256'], bits=bits,
            eligible_points=int(np.count_nonzero(good)), saturated_points=int(clipped.sum()),
            max_score_error=float(np.max(abs(actual[good]-wanted[good]), initial=0)),
            max_normalized_rank_loss=max(loss, default=0.),
            threshold_disagreements=int(np.sum(np.array(fixed_hits) != (wanted >= .6))),
            saturations=bank.stats['input_saturations'], overflows=bank.stats['intermediate_overflows'],
            widths=bank.widths, logical_storage_bits=bank.logical_storage_bits)
        rows.append(row)
        np.savez_compressed(output/(meta['id']+'.npz'), reference_score=wanted,
                            fixed_score=actual, fixed_gate=np.array(fixed_hits), saturated_window=clipped)
        write(output/'results.json', rows)
    summary = dict(bits=bits, points=sum(r['eligible_points'] for r in rows),
        max_score_error=max(r['max_score_error'] for r in rows),
        max_normalized_rank_loss=max(r['max_normalized_rank_loss'] for r in rows),
        threshold_disagreements=sum(r['threshold_disagreements'] for r in rows),
        saturations=sum(r['saturations'] for r in rows), overflows=sum(r['overflows'] for r in rows))
    summary['budget_passed'] = (summary['max_score_error'] <= .005 and
                               summary['max_normalized_rank_loss'] <= .005 and not summary['overflows'])
    write(output/'summary.json', summary)
    write(output/'manifest.json', {str(p.relative_to(output)): sha(p) for p in output.iterdir()
                                   if p.is_file()})
    return summary


def run(args):
    inputs = json.loads((args.inputs/'frozen-inputs.json').read_text())
    if inputs['split'] != 'development':
        raise ValueError('Width selection cannot use held-out inputs')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    results = []
    for bits in (10, 12, 14, 16, 18):
        out = args.output/f'width{bits}'
        evaluate(args.inputs, out, dict(correlation_backend='fixed', correlation_input_bits=bits,
                                        correlation_input_full_scale=1024.))
        failed = compare(args.baseline, out, args.output/f'width{bits}-packet-comparison.json')
        stats = survey(args.inputs, args.output/f'width{bits}-score-survey', bits)
        stats['packet_budget_passed'] = not failed
        stats['accepted'] = stats['budget_passed'] and not failed
        results.append(stats)
        write(args.output/'widths.json', results)
        print(json.dumps(stats), flush=True)
    selected = next((r['bits'] for r in results if r['accepted']), None)
    write(args.output/'frozen-selection.json', dict(selected_input_bits=selected,
        hardware_input_bits=None if selected is None else selected+8,
        synthetic_input_full_scale=1024., hardware_input_full_scale=262144.,
        scale_conversion='262144/1024=256: add8 input bits to preserve absolute quantizer LSB',
        threshold_fractional_bits=24, rounding='nearest-even input; saturate input; full-growth intermediates',
        input_plan=str((args.inputs/'frozen-inputs.json').resolve()),
        input_plan_sha256=sha(args.inputs/'frozen-inputs.json'),
        budgets=dict(score_error=.005, normalized_rank_loss=.005, old_success_loss=0),
        results=results, held_out_seen=False))
    return selected is None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('inputs', 'baseline', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
