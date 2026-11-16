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
        expected_hex=expected.hex(), length=37, mcs=8, ppm=0, cfo_hz=0)]))
    assert row['captured'] == 1 and row['correct'] == 0
    f, = row['frames']
    assert f['status'] == 'bad_data_crc' and f['data_crc'] is False
    assert f['bit_errors'] == 16 and f['ber'] == 16/(37*8)
    assert 'after_payload_error_hz' in f['carrier_error']


def test_frozen_audit_chunks_preserve_bad_crc_and_sampling_trajectory(tmp_path):
    import json

    module = runpy.run_path(Path(__file__).parents[1]/'scripts/audit_capture_hardware.py')
    p = module['profile']()
    wire, expected = module['transmitted'](p, bad_data=True)
    x, q, start, step = module['iq_record'](wire)
    inputs = tmp_path/'inputs'
    inputs.mkdir()
    path = inputs/'packet.npz'
    np.savez_compressed(path, iq=x, quiet=q)
    meta = dict(id='bad-crc', path=str(path), sha256=module['sha'](path), samples=len(x),
        mcs=8, packets=[dict(sync_start=start, step=step, expected_hex=expected.hex(),
                           length=37, mcs=8, ppm=0, cfo_hz=0)])
    (inputs/'frozen-inputs.json').write_text(json.dumps(dict(cases=[meta])))
    opts = dict(correlation_backend='fixed', correlation_input_bits=14)
    a = module['evaluate'](inputs, tmp_path/'whole', opts)
    b = module['evaluate'](inputs, tmp_path/'random', opts, chunk_pattern='random')
    assert a[0]['frames'] == b[0]['frames']
    assert a[0]['frames'][0]['status'] == 'bad_data_crc'
    assert a[0]['correct'] == b[0]['correct'] == 0
    with (np.load(tmp_path/'whole/evidence/bad-crc.npz') as first,
          np.load(tmp_path/'random/evidence/bad-crc.npz') as second):
        assert first.files == second.files
        for name in first.files:
            np.testing.assert_array_equal(first[name], second[name])
