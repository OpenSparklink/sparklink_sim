"""Independent continuous-time TX truth for packet-local clock tracking."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from nearlink_sdr.phy.psk import acquire_psk_capture, rrc_filter
from nearlink_sdr.phy.timing import PSKTimingConfig, track_psk_timing

spec = importlib.util.spec_from_file_location(
    'rx_audit_reference', Path(__file__).parents[1] / 'scripts/rx_timing_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('kind', ['QPSK', '8PSK'])
@pytest.mark.parametrize('ppm', [-100, -50, 0, 50, 100])
def test_long_packet_crc_content_and_true_drift(kind, ppm):
    case = dict(kind=kind, length=2047, offset=.419, ppm=ppm, seed=11015)
    x, q, truth, bits = audit.simulate(case)
    original = x.copy()
    fixed = acquire_psk_capture(x, q, audit.FS, audit.RS, audit.SYNC, timing_subdivisions=8)
    tracked = acquire_psk_capture(x, q, audit.FS, audit.RS, audit.SYNC,
                                  timing_subdivisions=8, timing_tracking=True)
    b0 = next(b for b in fixed['bursts'] if 'symbols' in b)
    b = next(b for b in tracked['bursts'] if 'symbols' in b)
    # The acquisition winner is identical: neither payload nor CRC selects the timing.
    assert (b['sync_start_raw'], b['sync_score']) == (b0['sync_start_raw'], b0['sync_score'])
    np.testing.assert_allclose(b['symbols'][:32], b0['symbols'][:32], atol=1e-10)
    frame = audit.decode_ft2_a3_capture(b['symbols'], sync_word=audit.WORD,
                                      data_crc_seed=0x555555, pilot_interval=0)
    assert frame['header_crc'] and frame['data_crc'] and frame['header']['data_length'] == 2047
    assert frame['payload'] == np.packbits(bits, bitorder='little').tobytes()
    n = frame['frame_end_symbol']
    error = b['symbol_centers_raw'][:n] - (truth['sync_start'] + np.arange(n)*truth['step'])
    assert np.max(abs(error)) < .1
    assert np.sqrt(np.mean(error[-512:]**2)) < .025
    assert np.median(b['timing_rate_ppm'][n-512:n]) == pytest.approx(ppm, abs=8)
    assert b['symbol_sample_step'] is None
    np.testing.assert_array_equal(x, original)
    if ppm:
        old = audit.decode_ft2_a3_capture(b0['symbols'], sync_word=audit.WORD,
                                        data_crc_seed=0x555555, pilot_interval=0)
        assert old['status'] == 'bad_data_crc'


@pytest.mark.parametrize('kind', ['QPSK', '8PSK'])
def test_detector_s_curve_sign_and_rotation_invariance(kind):
    x, _, truth, _ = audit.simulate(
        dict(kind=kind, length=2047, offset=.419, ppm=0, seed=11015))
    z = fftconvolve(x, rrc_filter(.4, 4), 'same')
    f = CubicSpline(np.arange(len(z)), z)
    t = truth['sync_start'] + 4*np.arange(100, 4000)
    for delta, sign in [(-.25, 1), (0, 0), (.25, -1)]:
        current, previous, mid = f(t+delta), f(t+delta-4), f(t+delta-2)
        e = np.mean(((previous-current)*mid.conj()).real) / np.mean(abs(current)**2)
        if sign:
            assert sign*e > .06
        else:
            assert abs(e) < .001
    a = track_psk_timing(z, 4, truth['sync_start'], truth['sync_start']+2000*4)
    b = track_psk_timing(z*2.3*np.exp(1.3j), 4, truth['sync_start'],
                         truth['sync_start']+2000*4)
    np.testing.assert_allclose(a['centers'], b['centers'], atol=1e-10)


def test_hold_bounds_zero_input_and_rate_saturation():
    z = np.zeros(100, complex)
    a = track_psk_timing(z, 4, .193, 95, hold_symbols=32)
    np.testing.assert_allclose(a['centers'], np.arange(.193, 95, 4))
    assert np.isfinite(a['symbols']).all()
    assert a['diagnostics']['rate_limit_events'] == 0
    x, _, truth, _ = audit.simulate(
        dict(kind='8PSK', length=1023, offset=.419, ppm=100, seed=11015))
    z = fftconvolve(x, rrc_filter(.4, 4), 'same')
    r = track_psk_timing(z, 4, truth['sync_start'], len(z)-1,
                         config=PSKTimingConfig(max_rate_ppm=1))
    assert r['diagnostics']['rate_limit_events'] > 0
    assert abs(r['rate_ppm']).max() < 1.00001
    assert np.all(np.diff(r['centers']) > 0)
    assert r['centers'][-1] < len(z)-1


@pytest.mark.parametrize('kwargs', [dict(bandwidth=0), dict(bandwidth=.1),
                                    dict(bandwidth=np.nan), dict(max_rate_ppm=0),
                                    dict(max_rate_ppm=np.inf), dict(max_rate_ppm=10001)])
def test_invalid_loop_parameters(kwargs):
    with pytest.raises(ValueError):
        PSKTimingConfig(**kwargs)


@pytest.mark.parametrize('kwargs', [dict(sps=1), dict(sps=True), dict(start=-1),
                                    dict(stop=100), dict(start=np.nan), dict(hold_symbols=0),
                                    dict(hold_symbols=True), dict(config='bad')])
def test_invalid_tracking_inputs(kwargs):
    values = dict(sps=4, start=2., stop=98.)
    values.update(kwargs)
    with pytest.raises(ValueError):
        track_psk_timing(np.ones(100), **values)


def test_default_is_unchanged_and_quiet_is_not_tracked():
    x, q, _, _ = audit.simulate(
        dict(kind='8PSK', length=255, offset=.419, ppm=0, seed=11015))
    for options in ({}, dict(timing_tracking=False)):
        result = acquire_psk_capture(x, q, audit.FS, audit.RS, audit.SYNC, **options)
        b = next(b for b in result['bursts'] if 'symbols' in b)
        assert b['symbol_sample_step'] == 4 and 'timing_loop' not in b
    assert acquire_psk_capture(q, q, audit.FS, audit.RS, audit.SYNC,
                               timing_tracking=True)['bursts'] == []
    for options in (dict(timing_tracking=1), dict(timing_config='bad')):
        with pytest.raises(ValueError):
            acquire_psk_capture(x, q, audit.FS, audit.RS, audit.SYNC, **options)
