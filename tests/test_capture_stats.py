"""Capture funnel counts are observational; bad CRC is never a search selector."""

import numpy as np
import pytest

from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


@pytest.mark.parametrize(
    "damage,head_ok,head_bad,data_ok,data_bad",
    [({}, 1, 0, 1, 0), ({"bad_head": True}, 0, 1, 0, 0), ({"bad_data": True}, 1, 0, 0, 1)],
)
def test_funnel_counts_each_head_and_data_once(damage, head_ok, head_bad, data_ok, data_bad):
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, _ = transmitted(profile, **damage)
    x, q, _, _ = iq_record(wire)
    rx = PSKStreamReceiver(profile, q)
    events = rx.feed(x) + rx.finish()
    assert len(events) == 1
    s = rx.capture_stats
    assert s["sync_candidates"] == 1
    assert s["energy_triggers"] == rx.sync_searches > 0
    assert s["phase_hypotheses"] == 32 * s["full_searches"]
    assert s["header_crc_ok"] == head_ok and s["header_crc_failed"] == head_bad
    assert s["data_crc_ok"] == data_ok and s["data_crc_failed"] == data_bad


def test_silence_and_preparation_maximum_have_distinct_counters():
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8))
    rx = PSKStreamReceiver(profile, np.zeros(1024, complex))
    assert rx.feed(np.zeros(2048, complex)) + rx.finish() == []
    assert rx.capture_stats["energy_triggers"] == rx.sync_searches == 0
    ref = profile.sync_symbols()
    y = np.zeros(130, complex)
    y[2:130:4] = ref
    diagnostic = {}
    assert search_psk_sync(y, 4, ref, 1, lookahead=16, diagnostics=diagnostic) is None
    assert diagnostic["best_score"] == pytest.approx(1)
    assert diagnostic["disposition"] == "maximum_needs_future"


def test_quiet_calibrated_gate_scales_and_policy_is_explicit():
    from nearlink_sdr.phy.psk_sync import PSKCaptureConfig

    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4))
    rng = np.random.default_rng(151016)
    sigma = 50 * 10 ** (-5 / 20)
    q = sigma / np.sqrt(2) * (rng.normal(size=8192) + 1j * rng.normal(size=8192))
    wire, expected = transmitted(profile)
    x, _, _, _ = iq_record(wire, cfo=30000, ppm=100, offset=1.731)
    x += sigma / np.sqrt(2) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    old = PSKStreamReceiver(profile, q, capture_config=PSKCaptureConfig.legacy())
    new = PSKStreamReceiver(profile, q)
    assert new.threshold == old.threshold / 2
    assert old.feed(x) + old.finish() == []
    assert old.sync_searches == 0
    events = new.feed(x) + new.finish()
    assert [e["frame"]["payload"] for e in events] == [expected]
    assert new.capture_stats["full_searches"] > 0
    assert new.history_limit == old.history_limit and new.max_frame_symbols == old.max_frame_symbols
    scaled = PSKStreamReceiver(profile, q * 0.01)
    assert scaled.threshold == pytest.approx(new.threshold * 1e-4)
    for kwargs in [
        dict(energy_multiplier=0),
        dict(energy_multiplier=np.nan),
        dict(integer_screen=2),
    ]:
        with pytest.raises(ValueError):
            PSKCaptureConfig(**kwargs)


def test_integer_screen_avoids_fractional_preparation_for_a_tone(monkeypatch):
    import nearlink_sdr.phy.psk_sync as sync

    profile = PSKReceiveProfile.ordinary(TxConfig())
    monkeypatch.setattr(sync, "CubicSpline", lambda *a, **k: pytest.fail("unnecessary spline"))
    diagnostic = {}
    assert (
        sync.search_psk_sync(
            np.ones(256, complex),
            4,
            profile.sync_symbols(),
            8,
            integer_screen=0.6,
            diagnostics=diagnostic,
        )
        is None
    )
    assert diagnostic["disposition"] == "integer_screen_rejected"
    assert diagnostic["integer_phase_hypotheses"] == 4
    assert diagnostic["phase_hypotheses"] == 0


def test_weak_coded_capture_reaches_mac_and_bad_head_is_never_delivered():
    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    from nearlink_sdr.phy.psk_sync import PSKCaptureConfig

    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4))
    rng = np.random.default_rng(151016)
    sigma = 50*10**(-5/20)
    quiet = sigma/np.sqrt(2)*(rng.normal(size=8192)+1j*rng.normal(size=8192))
    wire, expected = transmitted(profile)
    x, _, _, _ = iq_record(wire, ppm=100, cfo=30000, offset=1.731)
    noise = sigma/np.sqrt(2)*(rng.normal(size=len(x))+1j*rng.normal(size=len(x)))
    old = MacStreamReceiver(profile, quiet, capture_config=PSKCaptureConfig.legacy())
    assert old.feed(x+noise).packets+old.finish().packets == []
    for bad_head in (False, True):
        symbols, _ = transmitted(profile, bad_head=bad_head)
        signal, _, _, _ = iq_record(symbols, ppm=100, cfo=30000, offset=1.731)
        receiver = MacStreamReceiver(profile, quiet)
        packets, frames = [], []
        for start in range(0, len(signal), 513):
            for result in receiver.iter_feed((signal+noise)[start:start+513]):
                packets.extend(result.packets)
                frames.extend(result.diagnostics['frames'])
        result = receiver.finish()
        packets.extend(result.packets)
        frames.extend(result.diagnostics['frames'])
        if bad_head:
            assert packets == [] and [f['status'] for f in frames] == ['bad_header_crc']
        else:
            assert [p.mac_payload for p in packets] == [expected]
            assert packets[0].crc_ok and packets[0].head_crc_ok
