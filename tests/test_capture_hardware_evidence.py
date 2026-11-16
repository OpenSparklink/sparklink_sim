"""Independent interference vectors and post-selection metrics preserve failed CRCs."""
import runpy
from pathlib import Path

import numpy as np


def test_new_development_vectors_are_deterministic_and_have_independent_packet_truth():
    module = runpy.run_path(Path(__file__).parents[1]/'scripts/audit_capture_hardware.py')
    vector = module['vector']
    for kind in ('zero', 'tone', 'wrong-sync', 'burst', 'dense'):
        x, q, meta = vector(kind, 0, 340116, kind != 'zero')
        y, r, other = vector(kind, 0, 340116, kind != 'zero')
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(q, r)
        assert meta == other
        assert len(meta['packets']) == (0 if kind == 'zero' else 4 if kind == 'dense' else 1)
        assert np.isfinite(x).all()
        if meta['packets']:
            assert all(0 < t['sync_start'] < len(x) for t in meta['packets'])
    a = vector('awgn', 0, 340116, True)[2]['packets'][0]['expected_hex']
    b = vector('awgn', 0, 350116, True)[2]['packets'][0]['expected_hex']
    assert a != b


def test_bad_crc_keeps_diagnostic_ber_without_becoming_a_success():
    module = runpy.run_path(Path(__file__).parents[1]/'scripts/audit_capture_hardware.py')
    p = module['profile']()
    wire, expected = module['transmitted'](p, bad_data=True, payload_seed=340117)
    x, q, start, step = module['iq_record'](wire)
    rx = module['PSKStreamReceiver'](p, q, quiet_batching=False)
    events = rx.feed(x)+rx.finish()
    row = module['score'](events, dict(mcs=8, packets=[dict(sync_start=start, step=step,
        expected_hex=expected.hex(), length=37, mcs=8)]))
    assert row['captured'] == 1 and row['correct'] == 0
    f, = row['frames']
    assert f['status'] == 'bad_data_crc' and f['data_crc'] is False
    assert f['bit_errors'] == 16 and f['ber'] == 16/(37*8)
