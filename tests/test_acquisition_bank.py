"""Independent direct dots, normalized ordering and causal finite history contracts."""

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_bank import (
    CorrelationPoint,
    DifferentialBank,
    normalized_greater,
    reference_integer_operations,
    sign_coefficients,
)
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


@pytest.mark.parametrize('sps', [4, 8])
@pytest.mark.parametrize('pattern', [[1], [7], [32]])
def test_continuous_bank_matches_independent_direct_normalization(sps, pattern):
    ref = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols()
    rng = np.random.default_rng(340118)
    x = rng.normal(size=411)+1j*rng.normal(size=411)
    bank = DifferentialBank(ref, sps, 64*sps, origin=-40)
    pos, points, capacity = 0, [], bank.state_array_bytes
    while pos < len(x):
        n = pattern[pos % len(pattern)]
        points.extend(bank.advance(x[pos:pos+n], inspect=True))
        pos += n
        assert bank.state_array_bytes == capacity
    dr = ref[1:]*ref[:-1].conj()
    for p in points:
        start = p.start+40
        y = x[start:start+len(ref)*sps:sps]
        dy = y[1:]*y[:-1].conj()
        expected = abs(np.vdot(dr, dy))**2/(sum(abs(dy)**2)*sum(abs(dr)**2))
        assert abs(p.score_squared-expected) < 1e-10
        assert bank.stats['new_correlations'] == len(x)-(len(ref)-1)*sps
    lo, hi = len(x)-64*sps+1, len(x)-1-(len(ref)-1)*sps
    values = [p.score_squared for p in points if lo-40 <= p.start <= hi-40]
    assert bank.gate(lo-40, hi-40) == (max(values) >= .6**2, len(values))
    assert bank.worst_operations_per_sample()['sqrt'] == 0


def test_squared_normalization_orders_energy_not_raw_amplitude_and_handles_zero():
    strong_wrong = CorrelationPoint(0, 1000, 10000, 1)
    weak_right = CorrelationPoint(1, 1, 1, 1)
    zero = CorrelationPoint(2, 0, 0, 1)
    assert normalized_greater(weak_right, strong_wrong)
    assert not normalized_greater(strong_wrong, weak_right)
    assert not normalized_greater(zero, zero)
    assert zero.score_squared == 0
    assert normalized_greater(weak_right, zero)
    assert not normalized_greater(zero, weak_right)
    # Common coefficient energy is also part of the normalized ordering.
    assert normalized_greater(CorrelationPoint(0, 10, 1, 1), CorrelationPoint(0, 20, 1, 4))


def test_threshold_boundary_and_dynamic_range_do_not_replace_normalized_score():
    ref = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols()
    rng = np.random.default_rng(340119)
    signal = ref+.3*(rng.normal(size=32)+1j*rng.normal(size=32))
    dr, dy = ref[1:]*ref[:-1].conj(), signal[1:]*signal[:-1].conj()
    score = abs(np.vdot(dr, dy))/np.sqrt(sum(abs(dr)**2)*sum(abs(dy)**2))
    for amplitude in (1e-6, 1., 1000.):
        for delta in (-1e-8, 1e-8):
            bank = DifferentialBank(ref, 4, 256, score+delta)
            x = np.zeros(128, complex)
            x[::4] = amplitude*signal
            for pos in range(0, len(x), 32):
                bank.advance(x[pos:pos+32])
            assert bank.gate(0, 0) == (delta < 0, 1)
    costs = reference_integer_operations()
    assert costs['new_differences'] == 252 and costs['normalized_positions'] == 131
    assert costs['divide'] == costs['sqrt'] == 131 and costs['hypot'] == 414
    assert costs['multiply'] == 17790
    assert reference_integer_operations(min_start=1000)['divide'] == 0
    with pytest.raises(ValueError):
        reference_integer_operations(sps=0)


def test_silence_bounds_and_rejected_reference_are_explicit():
    ref = np.exp(1j*np.pi/4*np.tile([0, 1], 16))
    bank = DifferentialBank(ref, 4, 256)
    for _ in range(20):
        bank.advance(np.zeros(32, complex))
    assert bank.gate(400, 500) == (False, 101)
    assert bank.gate(10000, 20000) == (False, 0)
    with pytest.raises(ValueError, match='bounded'):
        bank.gate(0, 5)
    with pytest.raises(ValueError, match='finite query'):
        bank.gate(np.nan, 1)
    for x in [np.zeros(33), [np.nan], np.zeros((2, 2)), []]:
        with pytest.raises(ValueError):
            bank.advance(x)
    for ref in [np.zeros(32), [1], [1, np.nan], np.exp(1j*np.arange(32)**2*.13)]:
        with pytest.raises(ValueError):
            sign_coefficients(ref)
    for kwargs in [dict(sps=True), dict(window_samples=2), dict(threshold=0),
                   dict(origin=1.5), dict(input_full_scale=np.inf)]:
        options = dict(sps=4, window_samples=256) | kwargs
        with pytest.raises(ValueError):
            DifferentialBank(np.ones(32), **options)
    bank.samples = 2**63-1
    with pytest.raises(OverflowError):
        bank.advance([1])


def test_stream_structure_keeps_existing_peak_waiting_and_bad_header_semantics():
    from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
    from tests.test_stream_receive import equal_events
    from tests.test_strict_receive import iq_record, transmitted

    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    for damage in ({}, dict(bad_head=True), dict(bad_data=True)):
        wire, _ = transmitted(profile, **damage)
        x, q, _, _ = iq_record(wire, cfo=30000, ppm=100, offset=12.277)
        reference = PSKStreamReceiver(profile, q, quiet_batching=False)
        events = reference.feed(x[:864])
        assert reference._active is None
        events += reference.feed(x[864:])+reference.finish()
        rx = PSKStreamReceiver(profile, q, quiet_batching=False, correlation_backend='structure')
        assert rx.integer_screen_backend == 'structure'
        actual = rx.feed(x[:864])
        assert rx._active is None  # The observed fine maximum still needs future samples.
        for pos in range(864, len(x), 513):
            actual.extend(rx.feed(x[pos:pos+513]))
        actual.extend(rx.finish())
        equal_events(events, actual)
        assert rx.correlation_bank.stats['samples'] == len(x)+rx.future_samples
    with pytest.raises(ValueError, match='correlation_backend'):
        PSKStreamReceiver(profile, q, correlation_backend='invalid')
    with pytest.raises(ValueError, match='quiet_batching'):
        PSKStreamReceiver(profile, q, correlation_backend='structure')
