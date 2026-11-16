"""Bounded causal tone detection, independent filter identity and integration failures."""

import numpy as np
import pytest
from scipy.signal import lfilter

from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.tone_notch import CausalToneNotch, ToneNotchConfig
from tests.test_stream_receive import equal_events
from tests.test_strict_receive import iq_record, transmitted


def test_tone_filter_matches_independent_transfer_function_and_arbitrary_chunks():
    rng = np.random.default_rng(370119)
    x = 100*np.exp(2j*np.pi*.07123*np.arange(4096))+.1*(rng.normal(size=4096)+
                                                                      1j*rng.normal(size=4096))
    whole, chunked = CausalToneNotch(), CausalToneNotch()
    first = whole.advance(x)
    result, start = [], 0
    while start < len(x):
        size = int(rng.integers(1, 258))
        result.extend(chunked.advance(x[start:start+size]))
        start += size
    np.testing.assert_array_equal(first, result)
    assert whole.stats == chunked.stats
    assert whole.stats['activations'] == 1
    a = whole.coefficient
    expected, _ = lfilter([1, -a], [1, -31/32*a], x[256:], zi=[-a*x[255]])
    np.testing.assert_allclose(first[256:], expected, atol=1e-10, rtol=0)
    assert np.mean(abs(first[512:])**2) < np.mean(abs(x[512:])**2)*.001
    assert whole.state_array_bytes == 96


def test_zero_energy_release_bounds_and_bad_inputs_are_explicit():
    model = CausalToneNotch(ToneNotchConfig(release_blocks=2))
    assert np.array_equal(model.advance(np.zeros(512)), np.zeros(512))
    assert model.stats['activations'] == 0
    model.advance(np.exp(.21j*np.arange(512)))
    assert model.active
    for _ in range(10):
        model.advance(np.zeros(128))
    assert not model.active and model.bad <= 2 and model.qualified <= 2
    assert model.stats['releases'] == 1
    for value in [[], [np.nan], np.zeros((2, 2)), np.zeros(4097)]:
        with pytest.raises(ValueError):
            model.advance(value)
    for kwargs in [dict(radius=.5), dict(coherence_squared=0), dict(block_samples=12),
                   dict(release_blocks=129)]:
        with pytest.raises(ValueError):
            ToneNotchConfig(**kwargs)
    with pytest.raises(ValueError):
        CausalToneNotch(sps=8)
    with pytest.raises(ValueError):
        CausalToneNotch(config='wrong')
    with pytest.raises(OverflowError):
        model.advance([1e100])
    model.samples = 2**63-1
    with pytest.raises(OverflowError):
        model.advance([1])


def test_opt_in_preserves_clean_frame_crc_and_peak_waiting_across_chunks():
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tx_pipeline import TxConfig
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    for damage, status in [({}, 'valid'), (dict(bad_data=True), 'bad_data_crc')]:
        wire, _ = transmitted(p, **damage)
        x, q, _, _ = iq_record(wire, offset=12.277, ppm=100, cfo=30000)
        baseline = PSKStreamReceiver(p, q, quiet_batching=False)
        expected = baseline.feed(x)+baseline.finish()
        notch = PSKStreamReceiver(p, q, quiet_batching=False, tone_notch=ToneNotchConfig())
        events = notch.feed(x[:864])
        assert notch._active is None
        for start in range(864, len(x), 257):
            events += notch.feed(x[start:start+257])
        events += notch.finish()
        equal_events(expected, events)
        assert events[0]['frame']['status'] == status
        assert notch.tone_notch.stats['activations'] == 0
    with pytest.raises(ValueError, match='quiet_batching'):
        PSKStreamReceiver(p, q, tone_notch=ToneNotchConfig())


def test_cancelling_lag1_fit_restarts_a_bounded_two_block_acquisition():
    model = CausalToneNotch()
    # Inject exact detector accumulator nodes: lag4 coherence alone does not
    # prohibit opposite lag1 phases in consecutive detector blocks.
    model.qualified, model.acquire_sum = 1, -128+0j
    model.c1 = model.c4 = 128+0j
    model.e0 = model.e4 = 128.
    model.block_count = 127
    model.advance([0])
    assert model.qualified == 0 and model.acquire_sum == 0 and not model.active
    model.advance(np.exp(.21j*np.arange(512)))
    assert model.active and model.stats['activations'] == 1


def test_filtered_fallback_recovers_weak_tone_packet_without_crc_path_selection():
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tx_pipeline import TxConfig
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4,
                                          whitening_seed=0x52, crc_seed=0x123456))
    for bad_head in (False, True):
        rng = np.random.default_rng(370122)
        wire, expected = transmitted(p, length=511, payload_seed=370122, bad_head=bad_head)
        x, _, _, _ = iq_record(wire, offset=.619, ppm=-100, cfo=-25000)
        x = (.5*x+5/np.sqrt(2)*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))+
             20*np.exp(2j*np.pi*.037125*np.arange(len(x))+1.03j))
        q = 5/np.sqrt(2)*(rng.normal(size=2048)+1j*rng.normal(size=2048))
        original = PSKStreamReceiver(p, q, quiet_batching=False)
        assert original.feed(x)+original.finish() == []
        kwargs = dict(quiet_batching=False, tone_notch=ToneNotchConfig())
        whole, chunked = PSKStreamReceiver(p, q, **kwargs), PSKStreamReceiver(p, q, **kwargs)
        a = whole.feed(x)+whole.finish()
        b = []
        for start in range(0, len(x), 257):
            b += chunked.feed(x[start:start+257])
        b += chunked.finish()
        equal_events(a, b)
        assert a[0]['acquisition_path'] == 'tone'
        assert whole.capture_stats['tone_candidates'] == 1
        assert whole.peak_history_samples <= 2*whole.history_limit+len(whole.h)+len(whole._box)
        if bad_head:
            assert a[0]['frame']['status'] == 'bad_header_crc'
            assert a[0]['frame']['payload'] is None
        else:
            assert a[0]['frame']['payload'] == expected
