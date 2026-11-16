"""Offline funnel observations cannot alter synchronization or CRC delivery."""

import runpy
from pathlib import Path

import numpy as np

from nearlink_sdr.phy.psk_sync import PSKCaptureConfig
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from tests.test_stream_receive import equal_events
from tests.test_strict_receive import iq_record, transmitted


def test_observer_and_integer_bypass_keep_peak_waiting_and_bad_crc(tmp_path, monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    module = runpy.run_path(scripts/'diagnose_interference_capture.py')
    p = module['profile']()
    wire, _ = transmitted(p, bad_data=True)
    x, q, _, _ = iq_record(wire, offset=12.277, cfo=30000, ppm=100)
    for gate in (.6, None):
        cfg = PSKCaptureConfig(2., gate)
        ordinary = PSKStreamReceiver(p, q, quiet_batching=False, capture_config=cfg)
        observed = module['FunnelObserver'](p, q, quiet_batching=False, capture_config=cfg,
            trace_path=tmp_path/f'{gate}.jsonl', trace_ranges=((0, len(x)+100),))
        assert ordinary.feed(x[:864]) == observed.feed(x[:864]) == []
        assert ordinary._active is observed._active is None
        a = ordinary.feed(x[864:])+ordinary.finish()
        b = observed.feed(x[864:])+observed.finish()
        equal_events(a, b)
        assert a[0]['frame']['status'] == 'bad_data_crc'
        assert ordinary.capture_stats == observed.capture_stats
        assert (tmp_path/f'{gate}.jsonl').exists()
        assert np.array_equal(a[0]['symbol_centers_raw'], b[0]['symbol_centers_raw'])


def test_truth_oracle_cannot_turn_damaged_payload_into_success(monkeypatch):
    scripts = Path(__file__).parents[1]/'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    module = runpy.run_path(scripts/'diagnose_burst_oracles.py')
    p = module['profile']()
    for damage, status in [({}, 'valid'), (dict(bad_data=True), 'bad_data_crc')]:
        wire, _ = transmitted(p, **damage)
        x, _, start, step = iq_record(wire, cfo=30000, ppm=100)
        centers = start+np.arange(len(wire)-40)*step
        assert module['oracle_decode'](x, p, centers, 30000)['status'] == status
