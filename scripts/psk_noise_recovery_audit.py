#!/usr/bin/env python3
"""Replay frozen failures with isolated energy and carrier changes, retaining all failures."""
import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rx_timing_audit as audit

ARMS = {
    'legacy': dict(energy_window_symbols=1,carrier_method='phase_regression'),
    'energy8': dict(energy_window_symbols=8,carrier_method='phase_regression'),
    'coherent': dict(energy_window_symbols=8,carrier_method='coherent'),
}


def passed(arm):
    return sum(b['whole_packet_ok'] for b in arm['rows'])


def regressions(before,after):
    """Associate accepted packets by sync coordinate after selection, never by content."""
    reg=[]
    for i,b in enumerate(before['rows']):
        if not b['whole_packet_ok']:
            continue
        matches=[z for z in after['rows']
                 if abs(z.get('sync_start_raw',-1e9)-b['sync_start_raw'])<=2]
        if len(matches)!=1 or not matches[0]['whole_packet_ok']:
            reg.append(i)
    return reg


def fresh_cases():
    return [dict(kind=kind,length=255,offset=.277,ppm=100,cfo_hz=30000,snr_db=snr,
                 seed=seed,split='new-held-out',family='noise-boundary')
            for seed in (44016,55016) for kind in ('8PSK','QPSK') for snr in (8,10,12,15)]


def compare(meta,x,q,out,truth=None):
    pair=dict(meta,arms={})
    for name,options in ARMS.items():
        result,arrays=audit.analyze(x,q,meta['kind'],meta['length'],8,truth,
                                    timing_tracking=True,**options)
        evidence=f"evidence/{meta['id']}-{name}.npz"
        np.savez_compressed(out/evidence,**arrays)
        result['evidence']=evidence
        pair['arms'][name]=result
    pair['regressions']=regressions(pair['arms']['legacy'],pair['arms']['coherent'])
    pair['carrier_regressions']=regressions(pair['arms']['energy8'],pair['arms']['coherent'])
    return pair


def run(args):
    root=Path(__file__).resolve().parents[1]
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'evidence').mkdir()
    baseline=args.baseline.resolve()
    manifest=json.loads((baseline/'manifest.json').read_text())
    for p,ident in manifest.items():
        if audit.identity(baseline/p)!=ident:
            raise ValueError('Baseline artifact changed: '+p)
    archived=[json.loads(s) for s in (baseline/'results.jsonl').read_text().splitlines()]
    original_failures=[p['id'] for p in archived if p['source']=='simulation'
                       and not passed(p['arms']['tracked'])]
    dataset=args.dataset.resolve()
    original_plan=json.loads((baseline/'frozen-plan.json').read_text())
    records=[*original_plan['real_inputs'],*original_plan['quiet_inputs']]
    index={r['sample_id']:r for r in records}
    quiet_dataset=(args.quiet_dataset or dataset).resolve()
    quiet_records=[json.loads(s)for s in (quiet_dataset/'samples.jsonl').read_text().splitlines()]
    quiet_index={r['sample_id']:r for r in quiet_records}
    quiets=[r for r in quiet_records if r['classification']=='formal_quiet']
    if len(quiets)!=144:
        raise ValueError('Expected144 formal quiet recordings in explicit quiet dataset')
    raw_inputs={str((dataset/r['path']).resolve()):r for r in records}
    raw_inputs.update({str((quiet_dataset/r['path']).resolve()):r for r in quiets})
    hashes={p:audit.identity(Path(p)) for p in raw_inputs}
    assert all(hashes[p]=={k:r[k]for k in ('bytes','sha256')}for p,r in raw_inputs.items())
    import nearlink_sdr.phy.psk as psk
    if not Path(psk.__file__).resolve().is_relative_to(root/'src'):
        raise ValueError('PYTHONPATH must select this checkout')
    plan=dict(commit=audit.git('rev-parse','HEAD'),diff=audit.git('diff','HEAD'),
              sources={str(p.relative_to(root)):audit.identity(p)for p in (root/'src').rglob('*.py')},
              driver=audit.identity(Path(__file__)),scorer=audit.identity(root/'scripts/rx_timing_audit.py'),
              baseline=str(baseline),baseline_manifest=audit.identity(baseline/'manifest.json'),
              target_failures=original_failures,arms=ARMS,fresh_cases=fresh_cases(),
              quiet_scope=str(quiet_dataset),
              quiet_samples_metadata=audit.identity(quiet_dataset/'samples.jsonl'),
              positive_scope="original frozen baseline inputs only, checked against archived hashes",
              inputs=hashes,quiet_ids=[r['sample_id']for r in quiets],
              parameters_frozen_before_replay=True,
              parameter_selection='energy window8 from seed11015 power fragmentation; coherent fit3 bounded Newton iterations; no held-out tuning',
              acceptance='actual header/data CRC, exact length/MCS, unchanged full PRBS9; no CRC retry',
              dependencies={k:audit.importlib.metadata.version(k)for k in ('numpy','scipy','matplotlib','pytest','ruff')},
              python=audit.sys.version,platform=audit.platform.platform())
    audit.write(out/'frozen-plan.json',plan)
    (out/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    (out/'scorer.executed.py').write_bytes((root/'scripts/rx_timing_audit.py').read_bytes())
    results=[]
    stream=(out/'results.jsonl').open('w')
    def save(meta,x,q,truth=None,archived_baseline=None):
        p=compare(meta,x,q,out,truth)
        if archived_baseline is not None:
            replay=p['arms']['legacy']
            p['archived_baseline_reproduced']=(passed(replay)==passed(archived_baseline)
                and not regressions(archived_baseline,replay)
                and not regressions(replay,archived_baseline))
        results.append(p)
        stream.write(json.dumps(p)+'\n')
        stream.flush()
        return p
    t0=time.perf_counter()
    baseline_matches=True
    for p in archived:
        if p['source']=='quiet':
            continue
        meta={k:v for k,v in p.items()if k not in ('arms','regressions','recoveries')}
        if p['source']=='simulation':
            inp=baseline/p['input_iq']
            v=np.load(inp)
            x,q=v['iq'],v['quiet']
            meta['input_identity']=audit.identity(inp)
        else:
            r=p['input']
            q=np.concatenate([audit.load_iq(dataset/index[i]['path'])for i in r['quiet_control_ids']])
            if p['source']=='real':
                x=audit.load_iq(dataset/r['path'])
            else:
                inp=baseline/(p['id']+'.npz')
                x=np.load(inp)['iq']
                assert audit.identity(inp)==p['derivative']
        pair=save(meta,x,q,p.get('truth'),archived_baseline=p['arms']['tracked'])
        baseline_matches &= pair['archived_baseline_reproduced']
    print('old positive inputs:',len(results),'baseline reproduced:',baseline_matches,flush=True)
    for i,c in enumerate(fresh_cases()):
        x,q,truth,_=audit.simulate(c)
        name=f'new-noise-{i:02}'
        np.savez_compressed(out/(name+'.npz'),iq=x,quiet=q)
        save(dict(c,source='simulation',id=name,truth=truth,input_iq=name+'.npz',
                  input_identity=audit.identity(out/(name+'.npz'))),x,q,truth)
    for i,r in enumerate(quiets):
        other=quiet_index[next(i for i in r['quiet_control_ids']
                               if i!=r['sample_id'] and i in quiet_index)]
        save(dict(source='quiet',split='all-formal-quiet',family='negative-control',kind='none',
                  length=255,id=r['sample_id'].replace('/','-'),input=r),
             audit.load_iq(quiet_dataset/r['path']),audit.load_iq(quiet_dataset/other['path']))
        if (i+1)%36==0:
            print('quiet inputs:',i+1,flush=True)
    stream.close()
    failures=[]
    for p in results:
        for name,arm in p['arms'].items():
            rows=[b for b in arm['rows']if not b['whole_packet_ok']]
            if rows or (p['source']=='simulation'and not passed(arm)):
                failures.append(dict(id=p['id'],arm=name,source=p['source'],
                                     input=p.get('input',p.get('input_iq')),
                                     evidence=arm['evidence'],rows=rows))
    audit.write(out/'failures.json',failures)
    audit.write(out/'target-nine.json',[p for p in results if p['id']in original_failures])
    audit.write(out/'regressions.json',[p for p in results if p['regressions'] or p['carrier_regressions']])
    assert all(audit.identity(Path(p))==v for p,v in hashes.items())
    assert all(audit.identity(root/p)==v for p,v in plan['sources'].items())
    assert audit.identity(quiet_dataset/'samples.jsonl')==plan['quiet_samples_metadata']
    summary=[]
    for group,subset in [
        ('original-nine',[p for p in results if p['id']in original_failures]),
        ('old-simulation',[p for p in results if p['source']=='simulation'and not p['id'].startswith('new-')]),
        ('new-noise',[p for p in results if p['id'].startswith('new-')]),
        ('real',[p for p in results if p['source']=='real']),
        ('real-derived',[p for p in results if p['source']=='real-derived']),
        ('quiet',[p for p in results if p['source']=='quiet'])]:
        for name in ARMS:
            rows=[b for p in subset for b in p['arms'][name]['rows']]
            summary.append(dict(group=group,arm=name,inputs=len(subset),
                whole_packets_correct=sum(passed(p['arms'][name])for p in subset),
                any_packet_correct=sum(bool(passed(p['arms'][name]))for p in subset),
                sync_candidates=sum('frame'in b for b in rows),
                statuses=dict(Counter(b.get('frame',{}).get('status',b['status'])for b in rows)),
                seconds=sum(p['arms'][name]['total_seconds']for p in subset)))
    audit.write(out/'summary.json',summary)
    audit.write(out/'integrity-after.json',dict(inputs_unchanged=True,sources_unchanged=True,
        archived_baseline_reproduced=baseline_matches,total_wall_seconds=time.perf_counter()-t0,
        process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    report(out,results,summary,original_failures,baseline)
    audit.write(out/'manifest.json',{str(p.relative_to(out)):audit.identity(p)for p in out.rglob('*')if p.is_file()})
    print('target-nine',[(r['arm'],r['any_packet_correct'])for r in summary if r['group']=='original-nine'],flush=True)
    print('regressions',sum(bool(p['regressions'] or p['carrier_regressions'])for p in results),flush=True)
    return int(not baseline_matches or any(p['regressions'] or p['carrier_regressions']for p in results)
               or any(f['arm']=='coherent'for f in failures))


def report(out,pairs,summary,target,baseline):
    lines=['# PSK噪声分段与载波相干拟合对照','',
           '三臂在处理输入前固定。legacy复现393fcef，energy8只改能量平均，coherent再改载波细化。',
           '所有臂均启用持续定时跟踪，初始1/8搜索、同步门限0.85和CRC/长度/PRBS9规则不变。','',
           '|输入集合|旧算法|仅能量窗8|再相干拟合|输入数|','|---|---:|---:|---:|---:|']
    for i in range(0,len(summary),3):
        rows=summary[i:i+3]
        lines.append('|'+rows[0]['group']+'|'+ '|'.join(str(r['whole_packets_correct'])for r in rows)
                     +'|'+str(rows[0]['inputs'])+'|')
    lines+=['','## 原九个失败的逐项关联','',
            '|ID|调制|SNR|legacy|energy8|coherent|最终BER|','|---|---|---:|---|---|---|---:|']
    for p in pairs:
        if p['id']not in target:
            continue
        status=[]
        for arm in p['arms'].values():
            counts=Counter(b.get('frame',{}).get('status',b['status'])for b in arm['rows']
                           if b['status']!='short_energy_segment')
            status.append(str(dict(counts))if counts else '无足够长能量段')
        bers=[b.get('ber')for b in p['arms']['coherent']['rows']if 'frame'in b]
        lines.append(f'|{p["id"]}|{p["kind"]}|{p.get("snr_db")}|'+ '|'.join(status)+f'|{bers}|')
    lines+=['','完整输入、三臂符号、实际采样坐标和失败见results.jsonl、evidence、failures.json。',
            '原九项见target-nine.json；原成功包按同步坐标±2采样关联，退化见regressions.json。',
            '相干拟合仅使用已去交替旋转的M次方样本，不搜索CRC合格的相位旋转。',
            '新保留集含两种调制、8/10/12/15dB、两随机种子和CFO30kHz/SRO100ppm。',
            '不因旧九项恢复而宣称噪声极限消失；新噪声失败继续保留。',
            '真实IQ无精确定时或SRO真值，不把判决残差当作真实定时误差。',
            '显式静默数据源144份正式静默均以原关联的另一份静默定门限，不使用被测文件自行定门限。',
            '8符号能量平均会降低分段时间分辨率，默认仍为1，不能推广到任意紧邻包间隔。',
            '整文件尾部截断仍保留；没有按期待长度补齐或修改输入。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    plot(out,pairs,baseline)


def plot(out,pairs,baseline):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=next(p for p in pairs if p['id']=='sim-0040')
    xq=np.load(baseline/p['input_iq'])
    x,q=xq['iq'],xq['quiet']
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for w in (4,32):
        kernel=np.ones(w)/w
        ep=np.convolve(abs(x)**2,kernel,'same')
        threshold=4*np.quantile(np.convolve(abs(q)**2,kernel,'same'),.999)
        axes[0].plot(ep,label=f'{w//4} symbol avg',alpha=.7)
        axes[0].axhline(threshold,ls='--',label=f'{w//4} symbol threshold')
    axes[0].set(xlabel='Original ADC sample',ylabel='Power (codes squared)',title='8PSK / 10dB')
    axes[0].legend(fontsize=8)
    for name in ('energy8','coherent'):
        arm=p['arms'][name]
        b=next(b for b in arm['rows']if 'frame'in b)
        arrays=np.load(out/arm['evidence'])
        y=arrays[f'symbols_{b["burst_index"]}']
        first,last=b['frame']['data_start_symbol'],b['frame']['frame_end_symbol']
        y,_=audit.refine_psk_carrier(y[first:last],'8PSK',first,
                                     method=ARMS[name]['carrier_method'])
        symbols,_=audit.packet('8PSK',255)
        error=np.angle(y*symbols[40+first:40+last].conj())
        axes[1].plot(np.convolve(error,np.ones(32)/32,'valid'),label=name)
    axes[1].set(xlabel='Data symbol',ylabel='Mean phase error vs TX reference (rad)',
                title='Evaluation only: no reference used by carrier estimator')
    axes[1].legend()
    fig.savefig(out/'noise-recovery.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--quiet-dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    raise SystemExit(run(parser.parse_args()))
