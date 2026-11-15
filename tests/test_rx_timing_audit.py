"""Validate the audit oracle and legal wire format, without requiring stressed RX success."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'rx_timing_audit', Path(__file__).parents[1] / 'scripts/rx_timing_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('kind', ['QPSK', '8PSK'])
@pytest.mark.parametrize('length', [255, 511, 1023, 2047])
def test_legal_a3_lengths_and_exact_content(kind, length):
    symbols, bits = audit.packet(kind, length)
    row = audit.decode_ft2_a3_capture(symbols[40:], sync_word=audit.WORD,
                                    data_crc_seed=0x555555, pilot_interval=0)
    assert row['data_crc'] and row['header']['data_length'] == length
    assert row['payload'] == np.packbits(bits, bitorder='little').tobytes()


def test_analytic_pulse_symmetry_singularities_and_true_clock_sign():
    t = np.array([0, .625, 1, 2, 3])
    assert np.isfinite(audit.pulse(t)).all()
    np.testing.assert_allclose(audit.pulse(t), audit.pulse(-t))
    symbols = np.ones(120)
    _, start, step = audit.reference(symbols, .193, 100)
    assert step == pytest.approx(4.0004)
    assert start == pytest.approx(512.193 + 40 * 4.0004)
    # Independent linear identity: fixed RX steps accumulate -4*ppm samples/symbol.
    j = np.array([0, 1000, 5000])
    error = start + 4*j - (start + step*j)
    np.testing.assert_allclose(error, [0, -.4, -2], atol=1e-10)


def test_crc_content_are_independent_and_bad_payload_never_accepted():
    y, _ = audit.packet('8PSK', 255)
    frame = audit.decode_ft2_a3_capture(y[40:], sync_word=audit.WORD,
                                      data_crc_seed=0x555555, pilot_interval=0)
    frame['payload'] = b'\0' * 255
    scored = audit.score_symbols(y[40:], frame, '8PSK', 255)
    assert scored['ber'] == 0 and not scored['content_ok']
    with pytest.raises(ValueError):
        audit.packet('8PSK', 2048)


def test_tracking_audit_measures_actual_centers_and_preserves_failed_baseline():
    case = dict(kind='8PSK', length=2047, offset=.419, ppm=100, seed=11015)
    x, q, truth, _ = audit.simulate(case)
    old, _ = audit.analyze(x, q, '8PSK', 2047, 8, truth)
    new, arrays = audit.analyze(x, q, '8PSK', 2047, 8, truth, timing_tracking=True)
    a = next(b for b in old['rows'] if 'frame' in b)
    b = next(b for b in new['rows'] if 'frame' in b)
    assert a['frame']['status'] == 'bad_data_crc' and b['whole_packet_ok']
    centers = arrays[f'centers_{b["burst_index"]}']
    error = arrays[f'error_{b["burst_index"]}']
    expected = centers[:len(error)] - (truth['sync_start'] + np.arange(len(error))*truth['step'])
    np.testing.assert_array_equal(error, expected)
    assert abs(error[-1]) < .02 and abs(a['timing_error_samples_head_mid_tail'][-1]) > 2
    assert b['timing_error_covers_packet']


@pytest.mark.parametrize('kind,snr', [('QPSK',10),('QPSK',15),('8PSK',15)])
def test_longer_energy_average_recovers_development_noise_fragmentation(kind, snr):
    x,q,truth,_=audit.simulate(dict(kind=kind,length=255,offset=.419,ppm=0,
                                   snr_db=snr,seed=11015))
    original=x.copy()
    old,_=audit.analyze(x,q,kind,255,8,truth,timing_tracking=True,energy_window_symbols=1)
    new,_=audit.analyze(x,q,kind,255,8,truth,timing_tracking=True,energy_window_symbols=8)
    assert not any(b['whole_packet_ok'] for b in old['rows'])
    assert sum(b['whole_packet_ok'] for b in new['rows'])==1
    np.testing.assert_array_equal(x,original)


def test_coherent_carrier_recovers_noisy_full_frame_without_crc_retry():
    x,q,truth,_=audit.simulate(dict(kind='8PSK',length=255,offset=.419,ppm=0,
                                   snr_db=10,seed=11015))
    old,_=audit.analyze(x,q,'8PSK',255,8,truth,timing_tracking=True,
                        energy_window_symbols=8,carrier_method='phase_regression')
    new,_=audit.analyze(x,q,'8PSK',255,8,truth,timing_tracking=True,
                        energy_window_symbols=8,carrier_method='coherent')
    a=next(b for b in old['rows']if 'frame'in b)
    b=next(b for b in new['rows']if 'frame'in b)
    assert a['frame']['status']=='bad_data_crc' and a['ber']>.4
    assert b['whole_packet_ok'] and b['ber']==0
    assert a['sync_start_raw']==b['sync_start_raw'] and a['sync_score']==b['sync_score']
    assert a['timing_loop']==b['timing_loop']
