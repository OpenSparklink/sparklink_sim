"""The offline quiet shortcut proves closed gating; it cannot certify strong interference."""

import runpy
from pathlib import Path

import numpy as np


def test_quiet_certificate_is_conservative_and_never_uses_packet_truth():
    module = runpy.run_path(Path(__file__).parents[1]/'scripts/replay_acquisition_regression.py')
    certify = module['closed_gate_certificate']
    rng = np.random.default_rng(340124)
    q = rng.normal(size=4096)+1j*rng.normal(size=4096)
    x = rng.normal(size=8192)+1j*rng.normal(size=8192)
    proof = certify(x, q)
    assert proof is not None
    assert proof['max_causal_energy']+proof['summation_error_bound'] < proof['threshold']
    assert certify(np.zeros(1000, complex), np.zeros(4096, complex)) is not None
    assert certify(100*np.exp(.19j*np.arange(8192)), q) is None
    impulse = np.zeros(8192, complex)
    impulse[250] = 1000
    assert certify(impulse, q) is None
    assert certify(np.array([np.nan]), q) is None
    assert certify(np.full(8192, 1e200), q) is None


def test_diagnostic_observer_is_bounded_and_does_not_feed_packet_truth(tmp_path):
    module = runpy.run_path(Path(__file__).parents[1]/'scripts/replay_acquisition_regression.py')
    fixture = runpy.run_path(Path(__file__).parents[1]/'scripts/audit_capture_hardware.py')
    p = fixture['profile']()
    wire, _ = fixture['transmitted'](p, bad_data=True)
    x, q, _, _ = fixture['iq_record'](wire)
    kwargs = dict(quiet_batching=False, correlation_backend='fixed',
                  correlation_input_bits=14, correlation_input_full_scale=1024.)
    ordinary = module['PSKStreamReceiver'](p, q, **kwargs)
    observed = module['DiagnosticReceiver'](p, q, diagnostic_file=tmp_path/'numeric.jsonl', **kwargs)
    a, b = ordinary.feed(x)+ordinary.finish(), observed.feed(x)+observed.finish()
    assert len(a) == len(b) == 1
    assert a[0]['frame']['status'] == b[0]['frame']['status'] == 'bad_data_crc'
    np.testing.assert_array_equal(a[0]['symbol_centers_raw'], b[0]['symbol_centers_raw'])
    assert len(observed.correlation_bank.points) <= 256
    assert observed.numeric['eligible_score_observations'] > 0
    assert observed.numeric['budget_violations'] == 0
    # Tiny input is intentionally outside the fixed absolute resolution: keep failures.
    tiny = module['DiagnosticReceiver'](p, q*.001, diagnostic_file=tmp_path/'tiny.jsonl', **kwargs)
    tiny_plain = module['PSKStreamReceiver'](p, q*.001, **kwargs)
    low = tiny.feed(x*.001)+tiny.finish()
    plain = tiny_plain.feed(x*.001)+tiny_plain.finish()
    assert [e['frame']['status'] for e in low] == [e['frame']['status'] for e in plain]
    assert all(e['frame']['status'] == 'bad_data_crc' for e in low)
    assert tiny.numeric['budget_violations'] > 0
    assert (tmp_path/'tiny.jsonl').exists()
