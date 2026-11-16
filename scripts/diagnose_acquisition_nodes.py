"""Attribute frozen numeric failures at their exact input/rounding/difference/energy nodes."""

import argparse
import json
from pathlib import Path

import numpy as np
import rx_stream_frozen_replay as frozen_stream
from audit_capture_hardware import sha, write
from profile_rx_stream import number
from scipy.signal import lfilter

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, FixedDifferentialBank
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile


def analyze(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original = args.evidence_root/'rx-timing-tracking/final'
    archived = args.evidence_root/'rx-chain-hardening/parser-replay'
    frozen_stream.frozen.verify_manifest(original)
    frozen_stream.frozen.verify_manifest(archived)
    plan = json.loads((original/'frozen-plan.json').read_text())
    rows = [json.loads(s) for s in (archived/'results.jsonl').read_text().splitlines()]
    old_quiet = {r['sample_id']: r for r in plan['quiet_inputs']}
    new_quiet = {r['input']['sample_id']: r['input'] for r in rows if r['source'] == 'quiet'}
    p = PSKReceiveProfile.ws63_ft2_a3(sync_word=number(plan['sync_word']),
                                    data_crc_seed=number(plan['crc_seed']), pilot_interval=0)
    failed = json.loads((args.frozen/'frozen-signoff/regression-original/numeric-budget-failures.json').read_text())
    out = []
    for case in failed:
        meta = next(r for r in rows if r['id'] == case['id'])
        x, _ = frozen_stream.load_input(meta, archived, original, args.dataset,
                                        args.quiet_dataset, old_quiet, new_quiet)
        h, ref = rrc_filter(.4, 4), p.sync_symbols()
        z = lfilter(h, [1.], x)
        bank = FixedDifferentialBank(ref, 4, 256, input_full_scale=262144., format=FixedBankFormat(22))
        dr = ref[1:]*ref[:-1].conj()
        observations = [json.loads(s) for s in Path(case['cost']['numeric_change_log']).read_text().splitlines()]
        for start in sorted({r['start'] for r in observations if 'start' in r}):
            first = start+(len(h)-1)//2
            y = z[first:first+4*len(ref):4]
            real, imag = bank.quantize(y)
            q = real+1j*imag
            d, dq = y[1:]*y[:-1].conj(), q[1:]*q[:-1].conj()
            c, cq = np.vdot(dr, d), np.vdot(dr, dq)
            e, eq = float(sum(abs(d)**2)), float(sum(abs(dq)**2))
            actual = 0 if eq == 0 else abs(cq)/np.sqrt(eq*sum(abs(dr)**2))
            independent = 0 if e == 0 else abs(c)/np.sqrt(e*sum(abs(dr)**2))
            record = dict(id=meta['id'], start=start, source=meta.get('input_identity', meta.get('source_identity')),
                max_filtered_component=float(max(abs(y.real).max(), abs(y.imag).max())),
                nonzero_quantized_components=int(np.count_nonzero(real)+np.count_nonzero(imag)),
                input_components=2*len(y), input_LSB=.125,
                independent_reference_score=float(independent), fixed_score=float(actual),
                difference_energy=e, quantized_difference_energy=eq,
                numerator=float(abs(c)**2), quantized_numerator=float(abs(cq)**2),
                original_observations=[r for r in observations if r.get('start') == start],
                reason='input underflow to zero' if eq == 0 else 'input rounding propagated through normalized energy/correlation')
            out.append(record)
            np.savez_compressed(args.output/(meta['id']+f'-{start}.npz'),
                filtered=y, quantized_real=real, quantized_imag=imag, differences=d,
                quantized_differences=dq, reference=dr)
    write(args.output/'results.json', out)
    (args.output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    write(args.output/'manifest.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps([{k: v for k, v in r.items() if k != 'original_observations'} for r in out], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-root', 'frozen', 'dataset', 'quiet-dataset', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    analyze(parser.parse_args())
