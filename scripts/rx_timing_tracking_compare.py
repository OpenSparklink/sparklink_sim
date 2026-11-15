#!/usr/bin/env python3
"""Frozen fixed-phase versus packet timing-loop comparison; exit1 retains observed failures."""
import argparse
import json
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rx_timing_audit as audit
from scipy.ndimage import map_coordinates

from nearlink_sdr.phy.timing import PSKTimingConfig


def comparison(meta, x, q, out, truth=None):
    pair = dict(meta, arms={})
    for tracking in (False, True):
        label = 'tracked' if tracking else 'fixed'
        result, arrays = audit.analyze(x, q, meta['kind'], meta['length'], 8, truth,
                                       timing_tracking=tracking)
        artifact = f"evidence/{meta['id']}-{label}.npz"
        np.savez_compressed(out/artifact, **arrays)
        result['evidence'] = artifact
        pair['arms'][label] = result
    # Boundaries are energy detector output, unaffected by phase tracking or CRC.
    old = pair['arms']['fixed']['rows']
    new = pair['arms']['tracked']['rows']
    assert [(b['start_raw'], b['end_raw_exclusive']) for b in old] == [
        (b['start_raw'], b['end_raw_exclusive']) for b in new]
    pair['regressions'] = [i for i,(a,b) in enumerate(zip(old,new,strict=True))
                           if a['whole_packet_ok'] and not b['whole_packet_ok']]
    pair['recoveries'] = [i for i,(a,b) in enumerate(zip(old,new,strict=True))
                          if not a['whole_packet_ok'] and b['whole_packet_ok']]
    return pair


def synthetic_plan():
    cases = audit.plan('full')
    for c in cases:
        c.pop('grids')
    # Fresh seed/offset and carrier+SRO combinations are frozen before all replay.
    for kind in ('8PSK', 'QPSK'):
        for length in (255,1023,2047):
            for ppm in (-100,0,100):
                cases.append(dict(split='fresh-held-out', seed=33016, family='fresh',kind=kind,
                                  length=length,offset=.317,ppm=ppm,cfo_hz=50000,snr_db=30))
    return cases


def summarize(results):
    groups = defaultdict(list)
    for r in results:
        groups[(r['source'],r['split'],r['kind'],r['family'])].append(r)
    summary=[]
    for (source,split,kind,family), pairs in groups.items():
        for arm in ('fixed','tracked'):
            rows = [b for p in pairs for b in p['arms'][arm]['rows']
                    if b['status']!='short_energy_segment']
            whole = sum(b['whole_packet_ok'] for b in rows)
            full = [b for b in rows if not b['edge_truncated']]
            summary.append(dict(source=source,split=split,kind=kind,family=family,arm=arm,
                inputs=len(pairs),known_tx_packets=len(pairs) if source=='simulation' else None,
                any_packet_correct=sum(any(b['whole_packet_ok'] for b in p['arms'][arm]['rows'])
                                       for p in pairs),
                complete_energy_segments=len(full),complete_packets_correct=sum(
                    b['whole_packet_ok'] for b in full),all_packets_correct=whole,
                statuses=dict(Counter(b.get('frame',{}).get('status',b['status']) for b in rows)),
                seconds=sum(p['arms'][arm]['total_seconds'] for p in pairs),
                recoveries=sum(len(p['recoveries']) for p in pairs),
                regressions=sum(len(p['regressions']) for p in pairs)))
    return summary


def run(args):
    root=Path(__file__).resolve().parents[1]
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'evidence').mkdir()
    import nearlink_sdr.phy.psk as psk
    if not Path(psk.__file__).resolve().is_relative_to(root/'src'):
        raise RuntimeError('PYTHONPATH must select this worktree')
    dataset=args.dataset.resolve()
    records=[json.loads(s) for s in (dataset/'samples.jsonl').read_text().splitlines()]
    selected=[r for r in records if r['classification']=='formal_on'
              and all(r['requested'][k]==v for k,v in
                      dict(phy=4,length=255,payload=0,polar=0,pilot=0).items())
              and r['requested']['rate'] in (2,3)]
    quiet_ids={i for r in selected for i in r['quiet_control_ids']}
    quiets=[r for r in records if r['sample_id'] in quiet_ids]
    hashes={r['path']:audit.identity(dataset/r['path']) for r in [*selected,*quiets]}
    assert all(hashes[r['path']]=={k:r[k] for k in ('bytes','sha256')}
               for r in [*selected,*quiets])
    cases=synthetic_plan()
    config=PSKTimingConfig()
    frozen=dict(model_commit=audit.git('rev-parse','HEAD'),source_diff=audit.git('diff','HEAD'),
                sources={str(p.relative_to(root)):audit.identity(p) for p in
                         sorted((root/'src').rglob('*.py'))},
                driver=audit.identity(Path(__file__)),sender_and_scorer=audit.identity(
                    root/'scripts/rx_timing_audit.py'),
                cases=cases,real_inputs=selected,quiet_inputs=quiets,input_hashes=hashes,
                selected_grid=8,tracking_config=vars(config),sync_word=hex(audit.WORD),
                crc_seed='0x555555',sample_hz=audit.FS,symbol_hz=audit.RS,
                parameters_frozen_before_replay=True,
                development_selection='seed11015 only; bandwidth .0015/.003/.006 by true timing RMS',
                holdout='seed22015 original held-out; seed33016 fresh with CFO50kHz/SNR30dB',
                acceptance='actual header/data CRC, exact length/MCS and full unchanged PRBS9',
                lock_caveat='no independent receiver lock classifier; true-error settling is evaluation only',
                platform=audit.platform.platform(),python=audit.sys.version,
                dependencies={k:audit.importlib.metadata.version(k)
                              for k in ('numpy','scipy','matplotlib','ruff','pytest')})
    audit.write(out/'frozen-plan.json',frozen)
    (out/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    (out/'sender.executed.py').write_bytes((root/'scripts/rx_timing_audit.py').read_bytes())
    results=[]
    stream=(out/'results.jsonl').open('w')
    def save(meta,x,q,truth=None):
        pair=comparison(meta,x,q,out,truth)
        results.append(pair)
        stream.write(json.dumps(pair)+'\n')
        stream.flush()
    t0=time.perf_counter()
    for split in ('development','held-out','fresh-held-out'):
        for i,c in enumerate(cases):
            if c['split']!=split:
                continue
            x,q,truth,_=audit.simulate(c)
            case_id=f'sim-{i:04}'
            np.savez_compressed(out/(case_id+'.npz'),iq=x,quiet=q)
            save(dict(c,source='simulation',id=case_id,truth=truth,input_iq=case_id+'.npz'),x,q,truth)
        for r in selected:
            if ('development' if r['sample_id'].endswith('rx-00') else 'held-out')!=split:
                continue
            x=audit.load_iq(dataset/r['path'])
            q=np.concatenate([audit.load_iq(dataset/s['path']) for s in quiets
                              if s['sample_id'] in r['quiet_control_ids']])
            meta=dict(source='real',family='original',split=split,
                      kind='8PSK' if r['requested']['rate']==3 else 'QPSK',length=255,
                      id=r['sample_id'].replace('/','-'),input=r,timing_truth=None)
            save(meta,x,q)
            if r['sample_id'].endswith(('rx-00','rx-02')):
                for ppm in (-100,100):
                    t=np.arange(len(x))/(1+ppm*1e-6)
                    v=(map_coordinates(x.real,[t],order=5,mode='constant')
                       +1j*map_coordinates(x.imag,[t],order=5,mode='constant'))
                    derived_id=meta['id']+f'-injected{ppm}'
                    p=out/(derived_id+'.npz')
                    np.savez_compressed(p,iq=v)
                    save(dict(meta,source='real-derived',family='injected-sro',id=derived_id,
                              injected_ppm=ppm,original_sro='unknown',derivative=audit.identity(p)),
                         v,q)
        print(split, len(results),'pairs complete',flush=True)
    for r in quiets:
        other=next(s for s in quiets if s['requested']['id']==r['requested']['id']
                   and s['sample_id']!=r['sample_id'])
        save(dict(source='quiet',family='negative-control',split='held-out',kind='none',length=255,
                  id=r['sample_id'].replace('/','-'),input=r),
             audit.load_iq(dataset/r['path']),audit.load_iq(dataset/other['path']))
    stream.close()
    summary=summarize(results)
    audit.write(out/'summary.json',summary)
    failures=[]
    for p in results:
        for label,a in p['arms'].items():
            failed=[b for b in a['rows'] if not b['whole_packet_ok']]
            no_packet=p['source']=='simulation' and not any(b['whole_packet_ok'] for b in a['rows'])
            if failed or no_packet:
                failures.append(dict(id=p['id'],source=p['source'],arm=label,evidence=a['evidence'],
                                     input=p.get('input',p.get('input_iq')),rows=failed))
    audit.write(out/'failures.json',failures)
    audit.write(out/'regressions.json',[p for p in results if p['regressions']])
    audit.write(out/'recoveries.json',[p for p in results if p['recoveries']])
    assert all(audit.identity(dataset/p)==v for p,v in hashes.items())
    assert all(audit.identity(root/p)==v for p,v in frozen['sources'].items())
    audit.write(out/'integrity-after.json',dict(inputs_unchanged=True,sources_unchanged=True,
                total_wall_seconds=time.perf_counter()-t0,
                process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    report(out,results,summary)
    audit.write(out/'manifest.json',{str(p.relative_to(out)):audit.identity(p)
                                    for p in sorted(out.rglob('*')) if p.is_file()})
    print('regressed packets:',sum(len(p['regressions']) for p in results),flush=True)
    return int(any(p['regressions'] for p in results) or any(f['arm']=='tracked' for f in failures))


def report(out,pairs,summary):
    import csv
    with (out/'summary.csv').open('w') as stream:
        w=csv.DictWriter(stream,fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    text=['# PSK 包内持续定时跟踪对照','',
          '两臂只改变持续定时跟踪开关，均使用1/8采样的初始同步搜索。',
          '同步、能量分段、载波细化、CRC和期待载荷规则相同；实际采样坐标逐符号记录。','',
          '|来源/集合/调制/因素|固定整包正确|跟踪整包正确|输入数|恢复段|退化段|',
          '|---|---:|---:|---:|---:|---:|']
    for old,new in zip(summary[::2],summary[1::2],strict=True):
        label='/'.join(str(old[k]) for k in ('source','split','kind','family'))
        text.append(f'|{label}|{old["all_packets_correct"]}|{new["all_packets_correct"]}|'
                    f'{old["inputs"]}|{old["recoveries"]}|{old["regressions"]}|')
    text += ['','## 长包真值误差与BER（原保留集）','',
             '|调制|字节|ppm|固定/跟踪CRC|固定/跟踪BER|固定/跟踪尾误差/采样|',
             '|---|---:|---:|---|---|---|']
    for p in pairs:
        if p['source']!='simulation' or p['family']!='length' or p['split']!='held-out':
            continue
        arms=[next((b for b in p['arms'][a]['rows'] if 'frame'in b),{})
              for a in ('fixed','tracked')]
        crc='/'.join(str(b.get('frame',{}).get('data_crc')) for b in arms)
        ber='/'.join(f'{b["ber"]:.6g}' if b.get('ber') is not None else 'null' for b in arms)
        err='/'.join(f'{b["timing_error_samples_head_mid_tail"][-1]:.6f}'
                     if 'timing_error_samples_head_mid_tail'in b else 'null' for b in arms)
        text.append(f'|{p["kind"]}|{p["length"]}|{p["ppm"]}|{crc}|{ber}|{err}|')
    text += ['','所有失败、恢复和退化分别见failures.json、recoveries.json、regressions.json。',
             '真实IQ原始SRO未知，没有真值定时误差。附加重采样偏差单独标注。',
             '收敛诊断定义为真值误差进入±0.05采样且余包至少128符号一直保持；',
             '仅为离线评价，不能作为真实IQ环路锁定或标准限值。限幅事件不是周期滑移。',
             '低SNR的能量分段切碎与截断仍保留，不能全部归因定时。',
             'RSS为整个进程高水位，含生成和存档；逐臂耗时不含生成和写盘。',
             '![真值轨迹与处理成本](tracking.png)','']
    (out/'REPORT.md').write_text('\n'.join(text))
    plot(out,pairs)


def plot(out,pairs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    p=next(p for p in pairs if p['source']=='simulation' and p['family']=='length'
           and p['split']=='held-out' and p['kind']=='8PSK' and p['length']==2047 and p['ppm']==100)
    for arm in ('fixed','tracked'):
        arr=np.load(out/p['arms'][arm]['evidence'])
        key=next(k for k in arr.files if k.startswith('error_'))
        axes[0].plot(arr[key],label=arm)
    axes[0].set(xlabel='Symbol from sync',ylabel='True timing error (ADC samples)',
                title='8PSK / 2047 bytes / +100 ppm')
    axes[0].legend()
    for kind in ('8PSK','QPSK'):
        for arm in ('fixed','tracked'):
            times=[p['arms'][arm]['total_seconds']*1000 for p in pairs
                   if p['source']=='real' and p['kind']==kind]
            axes[1].bar(kind+' '+arm,np.mean(times))
    axes[1].set(ylabel='Mean ms per 65.536 ms capture')
    fig.savefig(out/'tracking.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    raise SystemExit(run(args))
