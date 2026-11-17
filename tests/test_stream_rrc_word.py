"""Explicit experimental RRC frontend -> actual strict MAC; no truth in decisions."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from tests.test_receive_scope import make_profile
from tests.test_strict_receive import iq_record, transmitted


def receiver(profile, quiet):
    return MacStreamReceiver(
        profile, quiet, quiet_batching=False, defer_quiet_fir=False, rrc_backend="word-full12-c18"
    )


def consume(rx, x, chunks):
    events, packets, start = [], [], 0
    rng = np.random.default_rng(510120)
    while start < len(x):
        n = len(x) if chunks == "whole" else int(rng.integers(1, 258))
        result = rx.feed(x[start : start + n])
        events += result.diagnostics["events"]
        packets += result.packets
        start += n
        assert rx.phy.buffered_samples <= (
            rx.phy.history_limit + 80 + len(rx.phy._box) - 1 + rx.phy.quantum
        )
    before = len(packets)
    result = rx.finish()
    return packets + result.packets, events + result.diagnostics["events"], before


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("mcs", range(13))
@pytest.mark.parametrize("pilot", [0, 8])
def test_word_frontend_all_configured_psk_mcs_crc_payload_and_chunk_state(ft, mcs, pilot):
    p = make_profile(ft, mcs, pilot)
    wire, payload = transmitted(p)
    x, q, truth, step = iq_record(
        wire,
        offset=((ft * 13 + mcs) * 0.137) % 4,
        cfo=30000 if mcs % 2 else -25000,
        ppm=100 if ft % 2 else -100,
    )
    arms = []
    for chunks in ("whole", "random"):
        rx = receiver(p, q)
        packets, events, before = consume(rx, x, chunks)
        assert before == 1 and [p.mac_payload for p in packets] == [payload]
        assert len(events) == 1
        event = events[0]
        frame = event["frame"]
        assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
        assert frame["mcs"] == mcs and frame["header"]["data_length"] == 37
        assert (
            np.max(
                abs(event["symbol_centers_raw"] - (truth + np.arange(len(event["symbols"])) * step))
            )
            <= 0.25
        )
        word = rx.phy._rrc_word_model
        assert len(rx.phy._fir_state) == 0 and rx.phy.fir_array_bytes == 1696
        assert word._r.dtype == word._i.dtype == np.dtype("int16")
        assert word.closed and word.stats["samples"] == len(x) + rx.phy.future_samples
        assert word.stats["zero_flush_samples"] == rx.phy.future_samples == 42
        assert rx.finish().packets == []
        with pytest.raises(ValueError, match="after finish"):
            rx.feed(np.empty(0))
        arms.append(events)
    assert arms[0][0]["frame"] == arms[1][0]["frame"]
    for key in ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(arms[0][0][key], arms[1][0][key])


def test_word_frontend_bad_crcs_and_eof_never_deliver_payload():
    p = make_profile(2, 12, 0)
    for damage, status in [
        (dict(bad_head=True), "bad_header_crc"),
        (dict(bad_data=True), "bad_data_crc"),
    ]:
        wire, _ = transmitted(p, **damage)
        x, q, _, _ = iq_record(wire)
        packets, events, _ = consume(receiver(p, q), x, "random")
        assert not packets and events[0]["frame"]["status"] == status
        assert events[0]["frame"]["payload"] is None
    wire, _ = transmitted(p)
    x, q, truth, step = iq_record(wire)
    for symbols, status in [(45, "truncated_header"), (len(wire) - 55, "truncated_data")]:
        packets, events, _ = consume(receiver(p, q), x[: int(truth + symbols * step)], "random")
        assert not packets and len(events) == 1 and events[0]["frame"]["status"] == status


def test_same_stream_keeps_rrc_history_across_dense_packets_and_silence():
    p = make_profile(2, 8, 8)
    wire, payload = transmitted(p)
    x, q, _, _ = iq_record(np.r_[wire, wire])
    rx = receiver(p, q)
    packets, events, before = consume(rx, x, "random")
    assert before == 2 and [p.mac_payload for p in packets] == [payload, payload]
    assert all(e["emitted_after_samples"] < len(x) for e in events)
    quiet = receiver(p, q)
    packets, events, _ = consume(quiet, np.zeros(9001), "random")
    assert not packets and not events and quiet.phy.sync_searches == 0
    assert quiet.phy.quiet_fast_path_samples == quiet.phy.quiet_deferred_fir_samples == 0


@pytest.mark.parametrize(
    "changes", [dict(quiet_batching=True), dict(defer_quiet_fir=True), dict(rrc_backend="unknown")]
)
def test_explicit_word_path_rejects_unverified_skip_and_invalid_mode(changes):
    opts = dict(quiet_batching=False, defer_quiet_fir=False, rrc_backend="word-full12-c18")
    opts.update(changes)
    with pytest.raises(ValueError):
        PSKStreamReceiver(make_profile(2, 8), np.zeros(1024), **opts)


@pytest.mark.parametrize("config", [dict(sps=8), dict(symbol_rate_mhz=2)])
def test_word_kernel_rejects_geometry_outside_frozen_sps4_4msym_scope(config):
    p = make_profile(2, 8)
    p = replace(p, config=replace(p.config, **config))
    with pytest.raises(ValueError, match="SPS4/4Msym"):
        receiver(p, np.zeros(1024))


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("mcs", [8, 12])
@pytest.mark.parametrize("length", [1, 2047])
def test_word_filter_remains_continuous_through_legal_uncoded_length_extremes(ft, mcs, length):
    p = make_profile(ft, mcs)
    wire, payload = transmitted(p, length=length)
    x, q, _, _ = iq_record(wire, offset=0.619, cfo=30000, ppm=100)
    packets, events, before = consume(receiver(p, q), x, "random")
    assert before == 1 and [p.mac_payload for p in packets] == [payload]
    assert events[0]["frame"]["header"]["data_length"] == length


def test_factory_word_frontend_keeps_explicit_sync_crc_and_header_mcs_context():
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from tests.test_rx_timing_audit import audit

    p = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=audit.WORD, data_crc_seed=0x555555, pilot_interval=0
    )
    transmitted_profile = replace(p, config=replace(p.config, mcs_index=12))
    wire, payload = transmitted(transmitted_profile, length=255)
    x, q, _, _ = iq_record(wire, cfo=30000, ppm=100)
    packets, events, before = consume(receiver(p, q), x, "random")
    assert before == 1 and [r.mac_payload for r in packets] == [payload]
    assert p.config.mcs_index == 8 and events[0]["frame"]["mcs"] == 12
