"""Frozen interference/weak-packet evidence for acquisition structure and fixed-point models."""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_rx_soft_capture import diagnostic_bits
from tests.test_strict_receive import iq_record, transmitted


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def profile(mcs=8, pid=0):
    return PSKReceiveProfile.ordinary(TxConfig(mcs_index=mcs, pid=pid, symbol_rate_mhz=4,
                                             whitening_seed=0x52, crc_seed=0x123456))


def vector(kind, repeat, seed, positive):
    """Analytic independent TX pulses; no receiver interpolation or expected-data feedback."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, repeat, int(positive),
                                                        KINDS.index(kind)]))
    mcs = (8, 12, 6, 8)[repeat % 4]
    length = (37, 255, 511, 73)[repeat % 4]
    amplitude = (.25, .125, .5, 1.)[repeat % 4]
    cfo, ppm = (0, 30000, -25000, 45000)[repeat % 4], (0, 100, -100, 25)[repeat % 4]
    wire, expected = transmitted(profile(mcs), length=length,
                                 payload_seed=int(rng.integers(0, 2**31)))
    packet, _, truth, step = iq_record(wire, ppm=ppm, cfo=cfo,
                                      offset=float(rng.uniform(0, 4)))
    prefix = int(rng.integers(0, 513))
    copies = 4 if kind == 'dense' else 1
    # Dense starts omit redundant leading/trailing zeros, leaving a short legal guard.
    first, last = 400, len(packet)-400
    if copies > 1:
        packet = packet[first:last]
        truth -= first
    n = prefix+copies*len(packet)+1024
    sigma = (2., 3., 5., 2.)[repeat % 4]

    def noise(count):
        return sigma/np.sqrt(2)*(rng.normal(size=count)+1j*rng.normal(size=count))

    x = noise(n) if kind != 'zero' else np.zeros(n, complex)
    q = noise(2048)
    if kind == 'tone':
        tone_amplitude = (2., 10., 20., 50.)[repeat % 4]
        x += tone_amplitude*np.exp(2j*np.pi*.037125*np.arange(n)+.281j)
    elif kind == 'wrong-sync':
        wrong, _ = transmitted(profile(8, 7), length=55, payload_seed=seed+repeat)
        signal, _, _, _ = iq_record(wrong, offset=.793, cfo=-17000)
        x += .25*np.resize(signal, n)
    elif kind == 'burst':
        for start in range(97+repeat, n, 997):
            end = min(start+31+repeat, n)
            x[start:end] += 100*np.exp(1j*rng.uniform(-np.pi, np.pi, end-start))
    packets = []
    if positive:
        for i in range(copies):
            start = prefix+i*len(packet)
            background_power = float(np.mean(abs(x[start:start+len(packet)])**2))
            signal_power = float(np.mean(abs(amplitude*packet)**2))
            x[start:start+len(packet)] += amplitude*packet
            packets.append(dict(sync_start=start+truth, step=step, length=length, mcs=mcs,
                                expected_hex=expected.hex(), cfo_hz=cfo, ppm=ppm,
                                signal_window_power=signal_power,
                                background_window_power=background_power,
                                measured_sinr_db=10*np.log10(signal_power/background_power)))
    return x, q, dict(kind=kind, positive=positive, repeat=repeat, mcs=mcs, length=length,
                     amplitude=amplitude, noise_complex_variance=sigma**2, packets=packets)


KINDS = ('zero', 'awgn', 'tone', 'wrong-sync', 'burst', 'dense')


def generate(output, split):
    output.mkdir(parents=True, exist_ok=False)
    seed = 340116 if split == 'development' else 350116
    cases = []
    for kind in KINDS:
        for positive in (False, True) if kind != 'zero' else (False,):
            for repeat in range(4 if split == 'development' else 8):
                x, q, meta = vector(kind, repeat, seed, positive)
                ident = f'{kind}-{int(positive)}-{repeat}'
                p = output/(ident+'.npz')
                np.savez_compressed(p, iq=x, quiet=q)
                cases.append(dict(id=ident, path=str(p.resolve()), sha256=sha(p), samples=len(x),
                                  quiet_samples=len(q), **meta))
    write(output/'frozen-inputs.json', dict(seed=seed, split=split, cases=cases,
          source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
          driver_sha256=sha(Path(__file__)),
          reference='independent analytical RRC and angle mapper, 16-symbol TX pulse tails'))
    (output/'generator.executed.py').write_bytes(Path(__file__).read_bytes())
    return cases


def score(events, meta):
    frames, matched = [], set()
    for e in events:
        f = e['frame']
        # Truth is used only after the receiver has selected sync and completed processing.
        nearest = min(meta['packets'], key=lambda p: abs(p['sync_start']-e['sync_start_raw']),
                      default=None)
        target = nearest if nearest and abs(nearest['sync_start']-e['sync_start_raw']) <= 4 else None
        row = {k: v for k, v in f.items() if k != 'payload'}
        row.update(sync_start=e['sync_start_raw'], matched=target is not None,
                   payload_hex=None if f.get('payload') is None else f['payload'].hex())
        row['correct'] = bool(target and f['status'] == 'valid' and
                              f['payload'].hex() == target['expected_hex'] and
                              f['header']['data_length'] == target['length'] and
                              f['mcs'] == target['mcs'])
        if target:
            matched.add(target['sync_start'])
            error = e['symbol_centers_raw']-(target['sync_start']+
                                           np.arange(len(e['symbols']))*target['step'])
            row['timing_error'] = dict(rms=float(np.sqrt(np.mean(error**2))),
                head_mid_tail=[float(error[i]) for i in (0, len(error)//2, len(error)-1)])
            if f.get('header_crc') and f.get('mcs') == target['mcs']:
                actual = diagnostic_bits(e, profile(meta['mcs']))
                wanted = np.unpackbits(np.frombuffer(bytes.fromhex(target['expected_hex']), np.uint8))
                if actual is not None and len(actual) == len(wanted):
                    row['bit_errors'] = int(np.sum(actual != wanted))
                    row['ber'] = row['bit_errors']/len(wanted)
        frames.append(row)
    return dict(frames=frames, known_packets=len(meta['packets']), captured=len(matched),
                missed=len(meta['packets'])-len(matched), correct=sum(r['correct'] for r in frames),
                false_candidates=sum(not r['matched'] for r in frames),
                false_deliveries=sum(not r['matched'] and r['status'] == 'valid' for r in frames))


def evaluate(inputs, output, options=None):
    output.mkdir(parents=True, exist_ok=False)
    (output/'evidence').mkdir()
    plan = json.loads((inputs/'frozen-inputs.json').read_text())
    options = options or {}
    write(output/'frozen-plan.json', dict(input_plan=str((inputs/'frozen-inputs.json').resolve()),
          input_plan_sha256=sha(inputs/'frozen-inputs.json'), options=options,
          source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
          source_sha256={str(p): sha(p) for p in Path('src').rglob('*.py')},
          driver_sha256=sha(Path(__file__)), association_tolerance_samples=4))
    (output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta in plan['cases']:
        path = Path(meta['path'])
        assert sha(path) == meta['sha256']
        with np.load(path) as data:
            # No quiet jumping: continuous detector work must remain visible in hardware studies.
            rx = PSKStreamReceiver(profile(meta['mcs']), data['quiet'], quiet_batching=False,
                                   **options)
            events = rx.feed(data['iq'])+rx.finish()
        traces = {f'{i}-{k}': e[k] for i, e in enumerate(events) for k in
                  ('symbols', 'symbol_centers_raw', 'timing_rate_ppm', 'timing_detector_error')}
        np.savez_compressed(output/'evidence'/(meta['id']+'.npz'), **traces)
        rows.append(dict(input=meta, **score(events, meta), stats=rx.capture_stats,
                         peak_history_samples=rx.peak_history_samples,
                         correlation_bank=None if rx.correlation_bank is None else dict(
                             stats=rx.correlation_bank.stats,
                             state_array_bytes=rx.correlation_bank.state_array_bytes,
                             operations=rx.correlation_bank.worst_operations_per_sample())))
        write(output/'results.json', rows)
    summary = dict(inputs=len(rows), samples=sum(r['input']['samples'] for r in rows),
                   known_packets=sum(r['known_packets'] for r in rows),
                   captured=sum(r['captured'] for r in rows), correct=sum(r['correct'] for r in rows),
                   missed=sum(r['missed'] for r in rows),
                   false_candidates=sum(r['false_candidates'] for r in rows),
                   false_deliveries=sum(r['false_deliveries'] for r in rows),
                   statuses=dict(Counter(f['status'] for r in rows for f in r['frames'])))
    write(output/'summary.json', summary)
    write(output/'failures.json', [r for r in rows if r['correct'] != r['known_packets'] or
                                   r['false_candidates']])
    write(output/'manifest.json', {str(p.relative_to(output)): sha(p) for p in output.rglob('*')
                                   if p.is_file()})
    print(json.dumps(summary, indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', choices=['development', 'held-out'])
    parser.add_argument('--inputs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--correlation-backend', choices=['reference', 'structure'],
                        default='reference')
    args = parser.parse_args()
    if args.generate:
        generate(args.output, args.generate)
    elif args.inputs:
        evaluate(args.inputs, args.output, dict(correlation_backend=args.correlation_backend))
    else:
        parser.error('Need --generate or --inputs')
