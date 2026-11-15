#!/usr/bin/env python3
"""Offline timing boundary experiment. Nonzero exit means observed failures, not a broken run."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from nearlink_sdr.common.crc import CRC24A_POLY, crc_attach
from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.control_info import ControlInfoA3, polar_encode_control
from nearlink_sdr.phy.frame import decode_ft2_a3_capture
from nearlink_sdr.phy.pilot import pilot_symbol
from nearlink_sdr.phy.psk import (
    PSKDemodulator,
    PSKModulator,
    acquire_psk_capture,
    refine_psk_carrier,
)

WORD = 0x6A28C4BD9D8EB6C5
FS = 16_000_000
RS = 4_000_000
GRIDS = (1, 2, 4, 8, 16)
SYNC = PSKModulator('QPSK', sps=1).map_symbols(
    np.array([(WORD >> i) & 1 for i in range(64)]))


def identity(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], text=True,
                                   cwd=Path(__file__).resolve().parents[1]).strip()


def packet(kind, length):
    """FT2/A3 no whitening, no DATA pilots/coding; control retains mandatory coding/pilot."""
    if not 1 <= length <= 2047:
        raise ValueError('A3 length must fit its 11-bit field')
    bits = prbs9(length * 8)
    wirebits = crc_attach(bits, CRC24A_POLY, 24, 0x555555)
    mcs = 12 if kind == '8PSK' else 8
    head = ControlInfoA3(0, 0, 0, 0, 0, 0, 0, length).pack(WORD & 0xFFF, lqi=mcs)
    control = PSKModulator('QPSK', sps=1).map_symbols(polar_encode_control(head), 32)
    control = np.r_[control[:16], pilot_symbol('QPSK', 48),
                    PSKModulator('QPSK', sps=1).map_symbols(
                        polar_encode_control(head)[32:], 49)]
    data = PSKModulator(kind, sps=1).map_symbols(wirebits, 65)
    # 10 us FT2 preamble at 4 Msymbol/s; absolute parity agrees with even 40 symbols.
    preamble = np.exp(1j * np.tile([np.pi / 4, 0], 20))
    return np.r_[preamble, SYNC, control, data], bits


def pulse(t):
    """Independent continuous RRC closed form, unit symbol energy (not RX tap generator)."""
    beta = .4
    with np.errstate(divide='ignore', invalid='ignore'):
        h = (np.sin(np.pi * (1-beta)*t) + 4*beta*t*np.cos(np.pi*(1+beta)*t))
        h /= np.pi*t*(1-(4*beta*t)**2)
    h = np.where(abs(t) < 1e-10, 1-beta+4*beta/np.pi, h)
    special = beta/np.sqrt(2)*((1+2/np.pi)*np.sin(np.pi/(4*beta))
                             + (1-2/np.pi)*np.cos(np.pi/(4*beta)))
    return np.where(abs(abs(t)-1/(4*beta)) < 1e-10, special, h)


def reference(symbols, offset, ppm):
    """Positive ppm = more observed ADC samples per transmitted symbol; no RX interpolation."""
    step = 4 * (1 + ppm * 1e-6)
    origin = 512 + offset
    n = np.arange(int(np.ceil(origin + (len(symbols)+16)*step)) + 512)
    t = (n-origin)/step
    nearest = np.floor(t).astype(int)
    y = np.zeros(len(n), complex)
    # Independently truncated 16-symbol pulse tails; receiver has span 10.
    for lag in range(-16, 17):
        k = nearest + lag
        valid = (k >= 0) & (k < len(symbols)) & (abs(t-k) <= 16)
        y[valid] += symbols[k[valid]] * pulse(t[valid]-k[valid]) / 2
    return 100*y, origin+40*step, step


def simulate(case):
    symbols, bits = packet(case['kind'], case['length'])
    x, sync_start, step = reference(symbols, case['offset'], case['ppm'])
    rng = np.random.default_rng(case['seed'])
    sigma = 50 * 10 ** (-case.get('snr_db', 80)/20)
    def noise(n):
        return sigma / np.sqrt(2) * (rng.normal(size=n)+1j*rng.normal(size=n))
    q = noise(8192)
    envelope = np.linspace(1, case.get('fade_end', 1), len(x))
    x *= envelope * case.get('amplitude', 1)
    x *= np.exp(1j*(.37+2*np.pi*case.get('cfo_hz', 0)*np.arange(len(x))/FS))
    x += noise(len(x))
    return x, q, dict(sync_start=sync_start, step=step), bits


def score_symbols(y, frame, kind, expected_length):
    """Post-selection diagnostic, including bad-CRC bytes; never feeds acquisition/decoder."""
    nbits = expected_length*8
    bps = 3 if kind == '8PSK' else 2
    ndata = -(-(nbits+24)//bps)
    start = 65
    if kind not in ('8PSK', 'QPSK') or len(y) < start+ndata:
        return dict(ber=None, content_ok=False, recovered_length=None)
    corrected, _ = refine_psk_carrier(y[start:start+ndata], kind, start)
    hard = PSKDemodulator(kind, sps=1).demap_symbols(corrected, start)[:nbits]
    reference_bits = prbs9(nbits)
    errors = hard != reference_bits
    windows = [dict(first_bit=int(ix[0]), end_bit=int(ix[-1]+1), errors=int(errors[ix].sum()),
                    ber=float(errors[ix].mean())) for ix in np.array_split(np.arange(nbits), 3)]
    payload = frame.get('payload')
    return dict(ber=float(errors.mean()), bit_errors=int(errors.sum()), ber_windows=windows,
                diagnostic_payload_hex=np.packbits(hard, bitorder='little').tobytes().hex(),
                # Acceptance content only for actual decoded payload, not diagnostic fixed length.
                content_ok=payload == np.packbits(reference_bits, bitorder='little').tobytes(),
                recovered_length=None if payload is None else len(payload))


def analyze(x, q, kind, length, grid, truth=None, *, timing_tracking=False):
    t0 = time.perf_counter()
    acquisition = acquire_psk_capture(x, q, FS, RS, SYNC, timing_subdivisions=grid,
                                      timing_tracking=timing_tracking)
    ta = time.perf_counter()
    rows, arrays = [], {}
    for i, burst in enumerate(acquisition['bursts']):
        y = burst.pop('symbols', None)
        centers = burst.pop('symbol_centers_raw', None)
        for field in ('timing_rate_ppm', 'timing_detector_error'):
            values = burst.pop(field, None)
            if values is not None:
                arrays[f'{field}_{i}'] = values
        if centers is not None:
            arrays[f'centers_{i}'] = centers
        row = dict(burst_index=i, timing_loop_loss_of_lock=None, cycle_slips=None, **burst)
        if y is not None:
            frame = decode_ft2_a3_capture(y, sync_word=WORD, data_crc_seed=0x555555,
                                         pilot_interval=0)
            diag = score_symbols(y, frame, kind, length)
            payload = frame.pop('payload')
            row.update(frame=frame, **diag)
            row['payload_hex'] = None if payload is None else payload.hex()
            row['length_ok'] = frame.get('header', {}).get('data_length') == length
            row['mcs_ok'] = frame.get('mcs') == (12 if kind == '8PSK' else 8)
            row['whole_packet_ok'] = bool(frame['data_crc'] and row['length_ok']
                                          and row['mcs_ok'] and row['content_ok'])
            if truth:
                count = 65 + -(-(length*8+24)//(3 if kind == '8PSK' else 2))
                target = count
                count = min(count, len(y))
                j = np.arange(count)
                actual = row['sync_start_raw']+4*j if centers is None else centers[:count]
                error = actual-(truth['sync_start']+truth['step']*j)
                row['timing_error_observed_symbols'] = count
                row['timing_error_covers_packet'] = count == target
                bad = np.flatnonzero(abs(error) > .05)
                settled = int(bad[-1]+1) if len(bad) else 0
                row['settling_symbol_0p05_samples'] = (settled if count-settled >= 128
                                                       and count == target else None)
                row['timing_error_samples_head_mid_tail'] = error[[0, count//2, -1]].tolist()
                row['timing_error_rms_samples'] = float(np.sqrt(np.mean(error**2)))
                row['timing_error_max_abs_samples'] = float(abs(error).max())
                row['half_symbol_boundary_crossings'] = int(np.count_nonzero(np.diff(
                    np.floor((error+2)/4))))
                arrays[f'error_{i}'] = error
            arrays[f'symbols_{i}'] = y
        else:
            row['whole_packet_ok'] = False
        rows.append(row)
    elapsed = time.perf_counter()-t0
    return dict(grid=grid, acquisition_seconds=ta-t0, total_seconds=elapsed,
                iq_Msamples_per_second=len(x)/elapsed/1e6,
                payload_Mbit_per_second=length*8/elapsed/1e6,
                process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                energy_segments=len(rows), rows=rows), arrays


def plan(stage):
    cases = []
    for split, seed, offsets in [('development', 11015, [0,.125,.25,.375,.5,.625,.75,.875,.97]),
                                 ('held-out', 22015, [.031,.193,.419,.683,.823])]:
        for kind in ('8PSK', 'QPSK'):
            for offset in offsets:
                cases.append(dict(split=split, seed=seed, family='static', kind=kind,
                                  length=255, offset=offset, ppm=0, grids=GRIDS))
            if stage == 'minimal':
                continue
            for ppm in (0,-10,10,-25,25,-50,50,-100,100):
                cases.append(dict(split=split, seed=seed, family='sro', kind=kind,
                                  length=255, offset=.193, ppm=ppm, grids=GRIDS))
            for length in (511,1023,2047):
                for ppm in (0,-50,50,-100,100):
                    cases.append(dict(split=split, seed=seed, family='length', kind=kind,
                                      length=length, offset=.419, ppm=ppm, grids=(1,8,16)))
            factors = ([dict(cfo_hz=c) for c in (-100000,-50000,50000,100000)]
                       + [dict(snr_db=s) for s in (30,20,15,10)]
                       + [dict(amplitude=a) for a in (.4,2)] + [dict(fade_end=.2)])
            for factor in [*factors, dict(ppm=100,cfo_hz=50000,snr_db=20,fade_end=.5)]:
                c = dict(split=split, seed=seed, family='impairment', kind=kind,
                         length=255, offset=.419, ppm=0, grids=(1,8,16))
                c.update(factor)
                cases.append(c)
    return cases


def load_iq(path):
    v = np.fromfile(path, dtype='<i2').reshape(-1,2)
    return v[:,0].astype(float)+1j*v[:,1].astype(float)


def aggregate(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['source'], r.get('family',''), r['split'], r['kind'], r['grid'])].append(r)
    result = []
    for key, rs in groups.items():
        rows = [b for r in rs for b in r['rows'] if b['status'] != 'short_energy_segment']
        # Synthetic denominator is known transmitted packets, even when there is no energy segment.
        denom = len(rs) if key[0] == 'simulation' else len(rows)
        result.append(dict(source=key[0], family=key[1], split=key[2], kind=key[3], grid=key[4],
                           captures=len(rs), packet_denominator=denom,
                           whole_packet_ok=sum(b['whole_packet_ok'] for b in rows),
                           data_crc_ok=sum(b.get('frame',{}).get('data_crc') is True for b in rows),
                           length_ok=sum(b.get('length_ok',False) for b in rows),
                           content_ok=sum(b.get('content_ok',False) for b in rows),
                           seconds=sum(r['total_seconds'] for r in rs),
                           complete_packet_denominator=sum(not b['edge_truncated'] for b in rows),
                           complete_packet_ok=sum(b['whole_packet_ok'] and not b['edge_truncated']
                                                  for b in rows),
                           truncated=sum(b.get('frame',{}).get('status','').startswith('truncated')
                                         for b in rows),
                           max_abs_timing_error=max((b['timing_error_max_abs_samples']
                                                     for b in rows if
                                                     'timing_error_max_abs_samples' in b),
                                                    default=None)))
    return result


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    import nearlink_sdr.phy.psk as psk
    if not Path(psk.__file__).resolve().is_relative_to(root/'src'):
        raise RuntimeError('PYTHONPATH must point to this worktree/src')
    cases = plan(args.stage)
    dataset = args.dataset.resolve()
    records = [json.loads(s) for s in (dataset/'samples.jsonl').read_text().splitlines()]
    selected = [r for r in records if r['classification']=='formal_on'
                and r['requested']['phy']==4 and r['requested']['rate'] in (2,3)
                and r['requested']['pilot']==0 and r['requested']['polar']==0
                and r['requested']['length']==255 and r['requested']['payload']==0]
    controls = {i for r in selected for i in r['quiet_control_ids']}
    inputs = selected + [r for r in records if r['sample_id'] in controls]
    before = {r['path']: identity(dataset/r['path']) for r in inputs}
    if any(before[r['path']] != {k:r[k] for k in ('bytes','sha256')} for r in inputs):
        raise ValueError('Input hash/size mismatch')
    frozen = dict(stage=args.stage, cases=cases, real_inputs=inputs, input_hashes=before,
                  model_commit=git('rev-parse','HEAD'), model_diff=git('diff','HEAD'),
                  driver=identity(Path(__file__)), python=sys.version, platform=platform.platform(),
                  prior_evidence={str(p):identity(p) for p in [
                      dataset.parents[3]/'scripts/hardware/ws63_psk_diagnose.py',
                      dataset.parents[3]/'verification/analysis/ws63/20261115-rx2/diagnosis-90052e0/report/summary.json',
                      dataset/'README.md', dataset/'manifest.json'] if p.exists()},
                  dependencies={k:importlib.metadata.version(k) for k in
                                ('numpy','scipy','matplotlib','pytest','ruff')},
                  source_hashes={str(p.relative_to(root)):identity(p) for p in
                                 sorted((root/'src').rglob('*.py'))},
                  command=sys.argv, sync_word=hex(WORD), crc_seed=hex(0x555555),
                  timing_tracking=False, sro_estimation=False, convergence_time=None,
                  split_policy='rx-00 development; rx-01/02 held-out; seeds11015/22015; no tuning',
                  synthetic_truth='analytic span16 RRC at t=(n-origin)/[4*(1+ppm*1e-6)]',
                  crc_content_selection=False)
    write(out/'frozen-plan.json', frozen)
    (out/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    all_results = []
    stream = (out/'results.jsonl').open('w')

    def save(meta, x, q, truth=None, grids=GRIDS):
        for grid in grids:
            result, arrays = analyze(x, q, meta['kind'], meta.get('length',255), grid, truth)
            result.update(meta, grid=grid)
            artifact = f"evidence/{meta['id']}-g{grid}.npz"
            p = out/artifact
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p, **arrays)
            result['evidence'] = artifact
            result['failures'] = [i for i,b in enumerate(result['rows'])
                                  if not b['whole_packet_ok']]
            all_results.append(result)
            stream.write(json.dumps(result)+'\n')
            stream.flush()

    # Full protocol/parameters frozen above, before either development or held-out processing.
    for split in ('development','held-out'):
        for index, c in enumerate(cases):
            if c['split'] != split:
                continue
            x,q,truth,_ = simulate(c)
            case_id = f'sim-{index:04}'
            np.savez_compressed(out/(case_id+'.npz'), iq=x, quiet=q)
            save(dict(c, source='simulation', id=case_id, truth=truth, input_iq=case_id+'.npz'), x,q,truth,c['grids'])
        for r in selected:
            if ('development' if r['sample_id'].endswith('rx-00') else 'held-out') != split:
                continue
            x = load_iq(dataset/r['path'])
            quiets = [load_iq(dataset/s['path']) for s in records
                      if s['sample_id'] in r['quiet_control_ids']]
            q = np.concatenate(quiets)
            kind = '8PSK' if r['requested']['rate']==3 else 'QPSK'
            meta = dict(source='real',family='original',split=split,kind=kind,length=255,
                        id=r['sample_id'].replace('/','-'),input=r, timing_truth=None)
            save(meta,x,q)
            if args.stage == 'full' and r['sample_id'].endswith(('rx-00','rx-02')):
                for ppm in (-100,100):
                    # Independent quintic B-spline resampling, no cubic RX interpolator.
                    coords = np.arange(len(x))/(1+ppm*1e-6)
                    derived = (map_coordinates(x.real,[coords],order=5,mode='constant')
                               +1j*map_coordinates(x.imag,[coords],order=5,mode='constant'))
                    derived_id = meta['id']+f'-injected{ppm}'
                    p = out/(derived_id+'.npz')
                    np.savez_compressed(p,iq=derived)
                    save(dict(meta,source='real-derived',family='injected-sro',id=derived_id,
                              injected_ppm=ppm,original_sro='unknown',derivative=identity(p)),
                         derived,q,grids=(1,8,16))
        for r in inputs:
            if r['sample_id'] not in controls:
                continue
            if split != 'held-out':
                continue
            x = load_iq(dataset/r['path'])
            other = next(s for s in inputs if s['sample_id'] in controls
                         and s['requested']['id']==r['requested']['id']
                         and s['sample_id']!=r['sample_id'])
            save(dict(source='quiet',family='negative-control',split=split,kind='none',length=255,
                      id=r['sample_id'].replace('/','-'),input=r),x,load_iq(dataset/other['path']))
        print(split, 'complete;',len(all_results),'arms',flush=True)
    stream.close()
    summary = aggregate(all_results)
    write(out/'summary.json', summary)
    failures = [dict(id=r['id'],grid=r['grid'],source=r['source'],evidence=r['evidence'],
                     input=r.get('input'),rows=[b for b in r['rows'] if not b['whole_packet_ok']])
                for r in all_results if r['failures'] or
                (r['source']=='simulation' and not any(b['whole_packet_ok'] for b in r['rows']))]
    write(out/'failures.json', failures)
    if any(identity(dataset/p)!=v for p,v in before.items()):
        raise RuntimeError('Original IQ changed during run')
    if any(identity(root/p) != v for p,v in frozen['source_hashes'].items()):
        raise RuntimeError('Receiver sources changed during run')
    write(out/'integrity-after.json',dict(inputs_unchanged=True, sources_unchanged=True,
                                         process_peak_rss_KiB=
                                         resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    plot(out,all_results)
    report(out, all_results)
    write(out/'manifest.json',{str(p.relative_to(out)):identity(p) for p in
                              sorted(out.rglob('*')) if p.is_file()})
    failed = sum(not any(b['whole_packet_ok'] for b in r['rows']) for r in all_results
                 if r['source']=='simulation')
    print('synthetic failing arms:',failed, '; failure evidence:',len(failures),flush=True)
    return int(bool(failures))



def report(out, records):
    """Human-readable results and exact same-segment regression associations."""
    import csv
    summary = aggregate(records)
    with (out/'summary.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    paired = []
    for base in records:
        if base['source'] != 'real' or base['grid'] != 1:
            continue
        for b in base['rows']:
            if b['status'] == 'short_energy_segment':
                continue
            arms = {}
            for arm in records:
                if arm['id'] != base['id'] or arm['source'] != 'real':
                    continue
                match = [z for z in arm['rows'] if z['start_raw']==b['start_raw']
                         and z['end_raw_exclusive']==b['end_raw_exclusive']]
                arms[str(arm['grid'])] = [dict(status=z.get('frame',{}).get('status'),
                                              whole_packet_ok=z['whole_packet_ok'],
                                              ber=z.get('ber'), evidence=arm['evidence'])
                                         for z in match]
            paired.append(dict(sample_id=base['input']['sample_id'],input=base['input'],
                               start_raw=b['start_raw'],end_raw_exclusive=b['end_raw_exclusive'],
                               baseline_ok=b['whole_packet_ok'],arms=arms))
    write(out/'paired-original-segments.json', paired)
    text = [
        '# 离线接收定时专项结果', '',
        '实际基线 90052e0。接收算法未修改：已知同步差分相关选一次相位，之后固定4样本抽取。',
        '没有 SRO 估计或持续定时跟踪。步长不等于实际精度。完整计划、依赖和源码身份见 frozen-plan.json。',
        '', '## 原始真实 IQ 的最小对照', '',
        'mode-065 为8PSK，mode-057为QPSK，各三份65.536 ms完整采集。',
        '255字节PRBS9、4Msymbol/s、16MS/s、无数据导频/编码/加扰；控制头编码/导频保留。',
        '下表分母为非边界完整能量段；QPSK另有一个尾部截断候选，各网格均保留失败。',
        '', '|调制|搜索步长/采样|整包正确|数据CRC正确|整文件平均ms|IQ吞吐 Msamples/s|',
        '|---|---:|---:|---:|---:|---:|']
    for kind in ('8PSK','QPSK'):
        for grid in GRIDS:
            rs=[r for r in records if r['source']=='real' and r['kind']==kind and r['grid']==grid]
            rows=[b for r in rs for b in r['rows'] if b['status']!='short_energy_segment'
                  and not b['edge_truncated']]
            sec=sum(r['total_seconds'] for r in rs)
            ok=sum(b['whole_packet_ok'] for b in rows)
            crc=sum(b.get('frame',{}).get('data_crc') is True for b in rows)
            text.append(f'|{kind}|1/{grid}|{ok}/{len(rows)}|{crc}/{len(rows)}|'
                        f'{sec/len(rs)*1000:.2f}|{len(rs)*1048576/sec/1e6:.2f}|')
    regressions=[p for p in paired if p['baseline_ok'] and any(
        not all(a['whole_packet_ok'] for a in v) or not v for v in p['arms'].values())]
    quiet=[r for r in records if r['source']=='quiet']
    text += ['',f'同段关联见 paired-original-segments.json；原成功段退化数 {len(regressions)}。',
             f'静默对照 {len(quiet)} 个文件×网格臂，出现同步候选 '
             f'{sum("frame" in b for r in quiet for b in r["rows"])}，有效帧 '
             f'{sum(b["whole_packet_ok"] for r in quiet for b in r["rows"])}。',
             '', '## 仿真静态偏移：保留集', '',
             '两调制各5个非网格偏移，开发集另各9个偏移覆盖一个采样周期。',
             '|步长/采样|整包正确/10|最大绝对真值误差/采样|', '|---|---:|---:|']
    for g in GRIDS:
        rs=[r for r in records if r['source']=='simulation' and r['family']=='static'
            and r['split']=='held-out' and r['grid']==g]
        good=sum(any(b['whole_packet_ok'] for b in r['rows']) for r in rs)
        err=max(b['timing_error_max_abs_samples'] for r in rs for b in r['rows']
                if 'timing_error_max_abs_samples' in b)
        text.append(f'|1/{g}|{good}/{len(rs)}|{err:.6f}|')
    text += ['', '## SRO 与包长：1/8搜索的保留集', '',
             '各点是单包可重复实验，不能解释为统计PER或标准合格限值。',
             '|调制|字节|注入ppm|整包|BER|包头/中/尾真值误差（采样）|',
             '|---|---:|---:|---|---:|---|']
    for r in records:
        if (r['source']!='simulation' or r['family'] not in ('sro','length')
                or r['grid']!=8 or r['split']!='held-out'):
            continue
        b=next((b for b in r['rows'] if 'frame' in b),{})
        error=b.get('timing_error_samples_head_mid_tail')
        error='未知' if error is None else '/'.join(f'{e:.4f}' for e in error)
        ber=b.get('ber')
        text.append(f'|{r["kind"]}|{r["length"]}|{r["ppm"]}|'
                    f'{"正确" if b.get("whole_packet_ok") else "失败"}|'
                    f'{ber if ber is not None else "不可计算"}|{error}|')
    if any(r['family']=='impairment' for r in records if r['source']=='simulation'):
        text += ['', '## CFO、噪声和幅度：1/8搜索的保留集', '',
                 '|调制|单因素或组合参数|整包|长能量段状态|', '|---|---|---|---|']
        for r in records:
            if (r['source']!='simulation' or r['family']!='impairment'
                    or r['grid']!=8 or r['split']!='held-out'):
                continue
            factors={k:r[k] for k in ('ppm','cfo_hz','snr_db','amplitude','fade_end') if k in r}
            states=[b.get('frame',{}).get('status',b['status']) for b in r['rows']
                    if b['end_raw_exclusive']-b['start_raw']>=256]
            good=any(b['whole_packet_ok'] for b in r['rows'])
            text.append(f'|{r["kind"]}|{factors}|{"正确" if good else "失败"}|{states}|')
        text += ['', 'SNR按稳态平均信号功率2500数码平方定义，噪声为复AWGN，静默使用同功率噪声。',
                 '15/10 dB失败含能量段切碎/截断；组合中8PSK也出现截断。不能将这些全部归因为定时。',
                 '幅度渐变施加在整个采集坐标上；实际包内端点由冻结长度与公式共同决定。']
    text += ['', '## 证据与限制', '',
             '完整逐臂结果为 results.jsonl，表格为 summary.csv，全部失败为 failures.json。',
             '输入/依赖/参数见 frozen-plan.json；派生符号、真值轨迹与生成IQ均另存。',
             'manifest.json 为全部结果哈希，integrity-after.json 复核原始输入未变。',
             '没有跟踪环可比较；收敛时间、环路失锁、周期滑移为null。另报半符号边界跨越，含义不同。',
             '已知同步32符号窗口为8us，但居中滤波和整包载波细化使用未来信息，不能称作实时锁定。',
             '真实IQ精确定时未知，定时误差为null；注入±100ppm只代表附加偏差，原未知SRO单独保留。',
             '长包只保证A3字段与模型可表示，不宣称厂测设备或某个调度配置支持。',
             '时间统计不含生成/压缩/写盘；RSS是进程累计高水位，含存档开销，不能当每臂内存。',
             '主接收机、设备、SDK固件、RTL与Release均未改动；不作实时FPGA或完整SLE验收。', '',
             '![定时、长度和成本](timing-boundaries.png)', '']
    (out/'REPORT.md').write_text('\n'.join(text))


def plot(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for grid in GRIDS:
        rs = [r for r in records if r['source']=='simulation' and r['family']=='static'
              and r['kind']=='8PSK' and r['split']=='held-out' and r['grid']==grid]
        axes[0].plot([r['offset'] for r in rs],
                     [next((b['timing_error_samples_head_mid_tail'][0] for b in r['rows']
                            if 'timing_error_samples_head_mid_tail' in b),np.nan) for r in rs],
                     'o-',label=f'1/{grid} sample')
    axes[0].set(xlabel='Injected static delay (samples)',ylabel='Selected timing error (samples)')
    axes[0].legend(fontsize=7)
    for kind in ('8PSK','QPSK'):
        rs = [r for r in records if r['source']=='simulation' and r['family']=='length'
              and r['kind']==kind and r['split']=='held-out' and r['grid']==8 and r['ppm']==100]
        axes[1].plot([r['length'] for r in rs],
                     [next((b.get('ber',np.nan) for b in r['rows'] if 'ber' in b),np.nan)
                      for r in rs],'o-',label=kind)
    axes[1].set(xlabel='Payload bytes (+100 ppm, grid 1/8)',ylabel='Payload BER')
    axes[1].legend()
    for kind in ('8PSK','QPSK'):
        rs=[r for r in records if r['source']=='real' and r['kind']==kind]
        times=[sum(r['total_seconds'] for r in rs if r['grid']==g)/3 for g in GRIDS]
        axes[2].plot(GRIDS,times,'o-',label=kind)
    axes[2].set(xlabel='Subdivisions per ADC sample',ylabel='Seconds per 65.536 ms capture')
    axes[2].legend()
    fig.savefig(out/'timing-boundaries.png',dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--stage',choices=('minimal','full'),default='full')
    sys.exit(run(parser.parse_args()))
