"""Controlled energy/carrier ablation and post-selection packet regression links."""
import importlib.util
from pathlib import Path


def load_audit(monkeypatch):
    scripts=Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location('noise_audit',scripts/'psk_noise_recovery_audit.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_three_arms_keep_development_failure_and_recover_without_retry(tmp_path,monkeypatch):
    module=load_audit(monkeypatch)
    x,q,truth,_=module.audit.simulate(dict(kind='8PSK',length=255,offset=.419,ppm=0,
                                         snr_db=10,seed=11015))
    (tmp_path/'evidence').mkdir()
    p=module.compare(dict(id='development',kind='8PSK',length=255),x,q,tmp_path,truth)
    assert [module.passed(p['arms'][name])for name in module.ARMS]==[0,0,1]
    assert not p['regressions'] and not p['carrier_regressions']
    old=next(b for b in p['arms']['energy8']['rows']if 'frame'in b)
    new=next(b for b in p['arms']['coherent']['rows']if 'frame'in b)
    assert old['sync_start_raw']==new['sync_start_raw']
    assert old['timing_loop']==new['timing_loop']
    assert (tmp_path/p['arms']['energy8']['evidence']).exists()
    assert (tmp_path/p['arms']['coherent']['evidence']).exists()


def test_regression_identity_uses_time_and_keeps_failed_matching_packet(monkeypatch):
    module=load_audit(monkeypatch)
    old=dict(rows=[dict(sync_start_raw=100.,whole_packet_ok=True),
                   dict(sync_start_raw=500.,whole_packet_ok=False)])
    same=dict(rows=[dict(sync_start_raw=100.125,whole_packet_ok=True)])
    wrong_time=dict(rows=[dict(sync_start_raw=200.,whole_packet_ok=True)])
    failed=dict(rows=[dict(sync_start_raw=100.,whole_packet_ok=False)])
    assert module.regressions(old,same)==[]
    assert module.regressions(old,wrong_time)==[0]
    assert module.regressions(old,failed)==[0]
