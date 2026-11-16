"""Predeclared numerical tolerances and arbitrary-block PSK state invariance."""
from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.rx_pipeline import receive_capture
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted

# Frozen before implementation in rx-chain-hardening/streaming-baseline.json.
SYMBOL_ATOL, CENTER_ATOL, RATE_ATOL = 1e-12, 1e-10, 1e-8
OFFLINE_CENTER_ATOL = .25


def consume(profile, quiet, x, sizes):
    rx = PSKStreamReceiver(profile, quiet)
    events = []
    pos = index = 0
    while pos < len(x):
        n = int(sizes[index % len(sizes)])
        events.extend(rx.feed(x[pos:pos+n]))
        pos += n
        index += 1
    before_eof = len(events)
    events.extend(rx.finish())
    return events, rx, before_eof


def equal_events(a, b):
    assert len(a) == len(b)
    for expected, actual in zip(a, b, strict=True):
        assert expected['frame'] == actual['frame']
        assert expected['emitted_after_samples'] == actual['emitted_after_samples']
        for key, atol in [('symbols', SYMBOL_ATOL), ('symbol_centers_raw', CENTER_ATOL),
                          ('timing_rate_ppm', RATE_ATOL), ('timing_detector_error', SYMBOL_ATOL)]:
            np.testing.assert_allclose(actual[key], expected[key], atol=atol, rtol=0)


@pytest.mark.parametrize('mcs', [8, 12])
def test_all_chunk_boundaries_preserve_packet_state_and_trajectory(mcs):
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=mcs, symbol_rate_mhz=4))
    wire, expected = transmitted(p)
    x, quiet, _, _ = iq_record(wire, ppm=100, cfo=30000)
    baseline, _, before = consume(p, quiet, x, [len(x)])
    assert before == 1 and baseline[0]['frame']['payload'] == expected
    for sizes in [[1], [7], [31], [32], [37], [511],
                  np.random.default_rng(66016).integers(1, 257, 100)]:
        actual, rx, before = consume(p, quiet, x, sizes)
        assert before == 1
        equal_events(baseline, actual)
        assert rx.buffered_symbols == 0
        assert rx.peak_history_samples <= rx.history_limit+len(rx.h)-1+len(rx._box)-1+rx.quantum
        assert rx.finish() == []
        with pytest.raises(ValueError, match='after finish'):
            rx.feed(np.empty(0))


def test_contiguous_packets_are_emitted_before_eof_and_reset_clock():
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, expected = transmitted(p)
    x, q, truth, step = iq_record(np.r_[wire, wire])
    events, rx, before = consume(p, q, x, [113])
    assert before == 2 and [e['frame']['payload'] for e in events] == [expected, expected]
    np.testing.assert_allclose([e['sync_start_raw'] for e in events],
                               [truth, truth+len(wire)*step], atol=.125, rtol=0)
    assert all(e['timing_loop']['hold_symbols'] == 32 for e in events)
    assert all(e['timing_rate_ppm'][0] == 0 for e in events)
    # Pipeline emits after its fixed FIR/interpolation and quantum delay, not after file EOF.
    for e in events:
        assert e['emitted_after_samples'] < len(x)
        delay = e['emitted_after_samples'] - (e['symbol_centers_raw'][-1]+1)
        assert delay <= rx.future_samples+rx.quantum


def test_bad_header_then_good_contiguous_packet_and_end_truncation():
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    bad, _ = transmitted(p, bad_head=True)
    good, expected = transmitted(p)
    x, q, _, _ = iq_record(np.r_[bad, good])
    events, _, _ = consume(p, q, x, [17, 29, 7])
    assert [e['frame']['status'] for e in events] == ['bad_header_crc', 'valid']
    assert events[0]['frame']['payload'] is None and events[1]['frame']['payload'] == expected
    x, q, truth, step = iq_record(good)
    for symbols, status in [(45, 'truncated_header'), (len(good)-40-15, 'truncated_data')]:
        cut = int(truth+symbols*step)
        whole, _, _ = consume(p, q, x[:cut], [cut])
        divided, _, _ = consume(p, q, x[:cut], [1, 23, 101])
        assert len(whole) == 1 and whole[0]['frame']['status'] == status
        assert whole[0]['frame']['payload'] is None
        equal_events(whole, divided)


@pytest.mark.parametrize('ft,mcs', [(2, 6), (3, 8), (4, 0)])
def test_supported_coded_frames_and_offline_reference(ft, mcs):
    cfg = TxConfig(frame_type=ft, mcs_index=mcs, ctrl_bits_len=28 if ft == 2 else 27,
                   head_llid=7 if ft >= 3 else None, symbol_rate_mhz=4)
    p = PSKReceiveProfile.ordinary(cfg)
    wire, expected = transmitted(p)
    x, q, truth, step = iq_record(wire, ppm=100)
    streamed, _, _ = consume(p, q, x, [13, 257, 3])
    batch = receive_capture(x, q, p, energy_window_symbols=8)
    assert [e['frame']['payload'] for e in streamed] == [expected]
    assert [r['payload'] for r in batch['frames']] == [expected]
    centers = batch['acquisition']['bursts'][batch['frames'][0]['burst_index']]['symbol_centers_raw']
    actual = streamed[0]['symbol_centers_raw']
    np.testing.assert_allclose(actual, centers[:len(actual)], atol=OFFLINE_CENTER_ATOL, rtol=0)
    # Separately compare to independent transmit truth, not hard-decision residuals.
    assert np.max(abs(actual-(truth+np.arange(len(actual))*step))) < .25


def test_silence_bound_is_independent_of_total_input_length():
    p = PSKReceiveProfile.ordinary(TxConfig())
    rx = PSKStreamReceiver(p, np.zeros(1024))
    block = np.zeros(1024, complex)
    for _ in range(256):
        assert rx.feed(block) == []
        assert rx.buffered_samples <= rx.history_limit+len(rx.h)-1+len(rx._box)-1+rx.quantum
    assert rx.finish() == [] and rx.input_samples == 256*1024
    assert rx.sync_searches == 0


def test_wrong_context_invalid_input_and_config_are_not_success():
    p = PSKReceiveProfile.ordinary(TxConfig())
    with pytest.raises(ValueError):
        PSKStreamReceiver(p, np.zeros(64))
    with pytest.raises(ValueError):
        PSKStreamReceiver(p, np.zeros(1024), timing_subdivisions=0)
    with pytest.raises(ValueError):
        PSKStreamReceiver(p, np.zeros(1024), carrier_method='crc_retry')
    rx = PSKStreamReceiver(p, np.zeros(1024))
    with pytest.raises(ValueError):
        rx.feed(np.zeros((32, 2)))
    with pytest.raises(ValueError):
        rx.feed(np.full(32, np.nan))
    wire, _ = transmitted(p)
    x, q, _, _ = iq_record(wire)
    wrong = replace(p, config=replace(p.config, head_crc_seed=1))
    events, _, _ = consume(wrong, q, x, [43])
    assert all(e['frame']['payload'] is None for e in events)


def test_mac_stream_adapter_delivers_only_verified_packets_before_finish():
    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    bad, _ = transmitted(p, bad_head=True)
    good, expected = transmitted(p)
    x, q, _, _ = iq_record(np.r_[bad, good])
    receiver = MacStreamReceiver(p, q)
    received = receiver.feed(x)
    assert [r.mac_payload for r in received.packets] == [expected]
    assert [r['status'] for r in received.diagnostics['frames']] == ['bad_header_crc', 'valid']
    assert receiver.finish().packets == []


@pytest.mark.parametrize('ppm', [-100, 100])
def test_maximum_a3_payload_tracks_known_clock_and_finishes_before_eof(ppm):
    from tests.test_rx_timing_audit import audit
    p = PSKReceiveProfile.ws63_ft2_a3(sync_word=audit.WORD, data_crc_seed=0x555555,
                                      pilot_interval=0)
    p = replace(p, config=replace(p.config, mcs_index=12))
    wire, expected = transmitted(p, length=2047)
    x, q, truth, step = iq_record(wire, ppm=ppm, cfo=30000)
    events, rx, before = consume(p, q, x, [73, 5, 1024])
    assert before == 1 and events[0]['frame']['payload'] == expected
    assert events[0]['frame']['header']['data_length'] == 2047
    centers = events[0]['symbol_centers_raw']
    error = centers-(truth+np.arange(len(centers))*step)
    assert np.max(abs(error)) < .25
    assert len(centers) <= rx.max_frame_symbols
    assert rx.buffered_symbols == 0


def test_mac_checks_crc_flags_even_if_a_broken_producer_labels_status_valid(monkeypatch):
    import nearlink_sdr.phy.mac_interface as mac
    row = dict(status='valid', payload=b'unverified', header={}, header_crc=False,
               data_crc=True, length_ok=True)
    monkeypatch.setattr(mac, 'receive_capture', lambda *a, **k: dict(frames=[row]))
    assert mac.iq_to_mac_capture(None, None, None).packets == []


def test_iterator_yields_before_consuming_large_input_and_does_not_queue_output():
    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, payload = transmitted(p)
    x, q, _, _ = iq_record(np.r_[wire, wire])
    receiver = MacStreamReceiver(p, q)
    output = receiver.iter_feed(x)
    first = next(output)
    assert [p.mac_payload for p in first.packets] == [payload]
    assert receiver.phy._processed < len(x)
    assert sum(len(r.packets) for r in output) == 1
    assert receiver.finish().packets == []


def test_profiled_quiet_batching_preserves_canonical_filter_and_packet_trajectories():
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=12, symbol_rate_mhz=4))
    wire, expected = transmitted(p)
    x, q, _, _ = iq_record(wire, ppm=100, cfo=30000)
    quiet = np.tile(q, 20)
    joined = np.r_[quiet, x, quiet, x, quiet]
    baseline = PSKStreamReceiver(p, q, quiet_batching=False)
    reference = baseline.feed(joined)+baseline.finish()
    optimized = PSKStreamReceiver(p, q, quiet_batching=True)
    actual = optimized.feed(joined)+optimized.finish()
    assert optimized.quiet_fast_path_samples > 0
    assert [r['frame']['payload'] for r in actual] == [expected, expected]
    equal_events(reference, actual)
    np.testing.assert_array_equal(optimized._filtered, baseline._filtered)
    np.testing.assert_array_equal(optimized._fir_state, baseline._fir_state)
    for sizes in [[1], [37], [4095], [4096], [8192],
                  np.random.default_rng(67016).integers(1, 8193, 100)]:
        other, _, _ = consume(p, q, joined, sizes)
        equal_events(reference, other)


def test_quiet_tail_fuses_dispatch_without_changing_canonical_fir_state(monkeypatch):
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    rng = np.random.default_rng(174016)
    quiet = rng.normal(size=8192) + 1j*rng.normal(size=8192)
    block = .01*(rng.normal(size=4096) + 1j*rng.normal(size=4096))
    reference = PSKStreamReceiver(p, quiet, quiet_batching=False)
    assert reference.feed(block) == []
    actual = PSKStreamReceiver(p, quiet)
    monkeypatch.setattr(actual, '_process', lambda _: pytest.fail('per-quantum quiet dispatch'))
    assert actual.feed(block) == []
    for field in ('_filtered', '_energy', '_fir_state', '_energy_state'):
        np.testing.assert_array_equal(getattr(actual, field), getattr(reference, field))
    assert actual._origin == reference._origin
    assert actual._processed == reference._processed
    assert actual.buffered_samples == reference.buffered_samples


@pytest.mark.parametrize('dtype', [np.complex128, np.complex64, np.float64, np.float32])
def test_deferred_quiet_fir_replaces_cache_and_keeps_original_dtype_arithmetic(monkeypatch, dtype):
    import nearlink_sdr.phy.stream_rx as module
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    rng = np.random.default_rng(204016)
    quiet = .001*(rng.normal(size=8192) + 1j*rng.normal(size=8192))
    raw = .00001*(rng.normal(size=3*4096) + 1j*rng.normal(size=3*4096))
    raw = (raw.real if np.issubdtype(dtype, np.floating) else raw).astype(dtype)
    eager = PSKStreamReceiver(p, quiet, defer_quiet_fir=False)
    deferred = PSKStreamReceiver(p, quiet, defer_quiet_fir=True)
    assert eager.feed(raw) == []
    original = module._fir_block

    def checked(taps, samples, state):
        assert taps is not deferred.h, 'proven-quiet complex FIR must wait until needed'
        return original(taps, samples, state)

    with monkeypatch.context() as context:
        context.setattr(module, '_fir_block', checked)
        assert deferred.feed(raw) == []
    assert deferred.quiet_deferred_fir_samples == len(raw)
    assert deferred._quiet_tail.dtype == dtype
    assert deferred.buffered_samples <= eager.buffered_samples
    assert sum(v.nbytes for v in vars(deferred).values() if isinstance(v, np.ndarray)) <= sum(
        v.nbytes for v in vars(eager).values() if isinstance(v, np.ndarray))
    for field in ('_filtered', '_energy', '_fir_state', '_energy_state'):
        np.testing.assert_array_equal(getattr(deferred, field), getattr(eager, field))
    wire, expected = transmitted(p)
    x, _, _, _ = iq_record(wire, ppm=100, cfo=30000)
    actual = deferred.feed(x) + deferred.finish()
    reference = eager.feed(x) + eager.finish()
    equal_events(reference, actual)
    assert [e['frame']['payload'] for e in actual] == [expected]


@pytest.mark.parametrize('window', [1, 4, 8, 16, 64])
def test_deferred_filter_respects_original_cache_budget_and_falls_back_for_long_energy(window):
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    rng = np.random.default_rng(205016)
    quiet = rng.normal(size=8192) + 1j*rng.normal(size=8192)
    raw = .001*(rng.normal(size=4096+17) + 1j*rng.normal(size=4096+17))
    eager = PSKStreamReceiver(p, quiet, defer_quiet_fir=False, energy_window_symbols=window)
    deferred = PSKStreamReceiver(p, quiet, energy_window_symbols=window)
    assert eager.feed(raw) == deferred.feed(raw) == []
    assert deferred.buffered_samples <= eager.buffered_samples
    if window != 8:
        assert deferred.quiet_deferred_fir_samples == 0
    for field in ('_filtered', '_energy', '_fir_state', '_energy_state', '_pending'):
        np.testing.assert_array_equal(getattr(deferred, field), getattr(eager, field))
    assert eager.finish() == deferred.finish() == []
    with pytest.raises(ValueError, match='defer_quiet_fir'):
        PSKStreamReceiver(p, quiet, defer_quiet_fir=1)


def test_adaptive_quiet_prefix_preserves_packet_timing_and_probe_budget():
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=12, symbol_rate_mhz=4))
    wire, expected = transmitted(p, length=255)
    x, q, _, _ = iq_record(wire, ppm=100, cfo=30000)
    joined = np.r_[np.tile(q, 6), x, np.tile(q, 5), x, np.tile(q, 4)]
    plain = PSKStreamReceiver(p, q, quiet_batching=False)
    reference = plain.feed(joined)+plain.finish()
    assert [e['frame']['payload'] for e in reference] == [expected, expected]
    aligned = PSKStreamReceiver(p, q, adaptive_quiet=False)
    equal_events(reference, aligned.feed(joined)+aligned.finish())
    for sizes in ([len(joined)], [4096], [1024], [513],
                  np.random.default_rng(264016).integers(1, 8193, 100)):
        rx = PSKStreamReceiver(p, q)
        events, pos, i = [], 0, 0
        while pos < len(joined):
            n = int(sizes[i % len(sizes)])
            events.extend(rx.feed(joined[pos:pos+n]))
            assert rx.quiet_probe_samples <= rx.input_samples+rx.quiet_batch_samples
            assert 0 <= rx._quiet_probe_credit <= rx._quiet_credit_limit
            pos, i = pos+n, i+1
        events.extend(rx.finish())
        equal_events(reference, events)
        if sizes[0] == len(joined):
            assert rx.quiet_prefix_jumps > 0
        assert rx.quiet_probe_state_samples <= rx.quantum*rx.quiet_probe_calls
        assert rx.peak_history_samples <= plain.peak_history_samples
    with pytest.raises(ValueError, match='adaptive_quiet'):
        PSKStreamReceiver(p, q, adaptive_quiet=1)


def test_continuous_tone_cannot_repeat_future_energy_probes():
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    rng = np.random.default_rng(265016)
    q = rng.normal(size=8192)+1j*rng.normal(size=8192)
    tone = 100*np.exp(2j*np.pi*.1375*np.arange(32768))
    rx = PSKStreamReceiver(p, q)
    assert rx.feed(tone)+rx.finish() == []
    assert rx.quiet_probe_calls == 1
    assert rx.quiet_probe_samples == rx.quiet_batch_samples
    assert rx.capture_stats['sync_candidates'] == 0


def test_unready_true_sync_peak_is_deferred_without_substituting_an_earlier_phase():
    from scipy.signal import fftconvolve

    from nearlink_sdr.phy.psk import rrc_filter
    from nearlink_sdr.phy.psk_sync import search_psk_sync
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=12, symbol_rate_mhz=4))
    wire, _ = transmitted(p)
    x, _, _, _ = iq_record(wire, offset=12.277, ppm=100, cfo=30000)
    filtered = fftconvolve(x, rrc_filter(.4,4), mode='same')
    window = filtered[824-256:824]
    maximum = search_psk_sync(window,4,p.sync_symbols(),8)
    assert maximum['score'] > .99
    # The full known-sync maximum exists, but its 16-sample readiness guard does not.
    start = maximum['phase']+4*maximum['offset']
    assert start+31*4+16 > len(window)-1
    assert search_psk_sync(window,4,p.sync_symbols(),8,lookahead=16) is None


@pytest.mark.parametrize('offset', [.277,4.277,8.277,12.277,16.277,20.277,24.277,28.277])
def test_sync_readiness_is_not_biased_by_internal_quantum_alignment(offset):
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=12, symbol_rate_mhz=4))
    wire, expected = transmitted(p)
    x,q,truth,_ = iq_record(wire,offset=offset,ppm=100,cfo=30000)
    events,_,_ = consume(p,q,x,[37])
    assert [e['frame']['payload'] for e in events] == [expected]
    assert abs(events[0]['sync_start_raw']-truth) < .125
