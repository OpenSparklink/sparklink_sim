"""Truth timing/carrier controls for burst damage only; excluded from acceptance rates."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, sha, transmitted, write
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.strict_rx import decode_psk_frame


def oracle_decode(x, p, centers, cfo):
    """Known-center diagnostic, sync-only gain/phase; no expected-data/CRC selection."""
    h = rrc_filter(.4, p.config.sps)
    z = lfilter(h, [1.], x)
    y = CubicSpline(np.arange(len(z)), z)(centers+(len(h)-1)//2)
    y *= np.exp(-2j*np.pi*cfo*centers/16e6)
    ref = p.sync_symbols()
    gain = np.vdot(ref, y[:len(ref)])/np.vdot(ref, ref)
    if abs(gain) < 1e-30:
        return dict(status='zero_sync_gain', header_crc=None, data_crc=None, payload=None)
    return decode_psk_frame(y/gain, p)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = json.loads((args.funnel/'results.json').read_text())
    rows = []
    for r in results:
        meta = r['input']
        if meta['kind'] != 'burst' or not meta['packets']:
            continue
        path = Path(meta['path'])
        assert sha(path) == meta['sha256']
        with np.load(path) as data:
            x = data['iq']
        p = profile(meta['mcs'])
        with np.load(args.funnel/'evidence'/(meta['id']+'.npz')) as traces:
            for target in meta['packets']:
                wire, _ = transmitted(p, length=target['length'], payload_seed=1)
                truth = target['sync_start']+np.arange(len(wire)-40)*target['step']
                matched = [(i, f) for i, f in enumerate(r['arms']['baseline']['frames'])
                           if abs(f['sync_start']-target['sync_start']) <= 4]
                controls = dict(truth_timing_truth_carrier=(truth, target['cfo_hz']))
                if matched:
                    i, f = matched[0]
                    controls['actual_timing_truth_carrier'] = (
                        traces[f'baseline-{i}-symbol_centers_raw'], target['cfo_hz'])
                    controls['truth_timing_actual_carrier'] = (
                        truth, f['carrier_error']['sync_nominal_hz'])
                arms = {}
                for label, (centers, cfo) in controls.items():
                    f = oracle_decode(x, p, centers, cfo)
                    arms[label] = {k: v for k, v in f.items() if k != 'payload'}
                    arms[label]['correct'] = bool(f['status'] == 'valid' and
                                                  f['payload'].hex() == target['expected_hex'])
                rows.append(dict(input=meta, baseline=r['arms']['baseline']['frames'],
                    diagnostic_only=True, truth=target, arms=arms))
    write(args.output/'results.json', rows)
    write(args.output/'summary.json', dict(controls_excluded_from_acceptance=True,
        inputs=len(rows), truth_timing_truth_carrier_correct=sum(
            r['arms']['truth_timing_truth_carrier']['correct'] for r in rows)))
    (args.output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    write(args.output/'manifest.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print((args.output/'summary.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--funnel', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
