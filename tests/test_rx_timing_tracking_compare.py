"""Regression evidence must retain failed fixed timing, not replace its verdict."""
import importlib.util
from pathlib import Path


def test_pair_retains_baseline_and_saved_actual_centers(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('timing_compare',scripts/'rx_timing_tracking_compare.py')
    compare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare)
    case = dict(kind='8PSK',length=2047,offset=.419,ppm=-100,seed=11015)
    x,q,truth,_ = compare.audit.simulate(case)
    (tmp_path/'evidence').mkdir()
    row = compare.comparison(dict(case,source='simulation',family='length',split='development',
                                  id='test'),x,q,tmp_path,truth)
    assert not row['regressions'] and len(row['recoveries']) == 1
    assert any(b.get('frame',{}).get('status')=='bad_data_crc' for b in row['arms']['fixed']['rows'])
    assert any(b['whole_packet_ok'] for b in row['arms']['tracked']['rows'])
    for label in ('fixed','tracked'):
        assert (tmp_path/row['arms'][label]['evidence']).exists()
    summary = compare.summarize([row])
    assert [s['all_packets_correct'] for s in summary] == [0,1]
