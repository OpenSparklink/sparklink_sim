"""Explicit strict-stream scope, not standards or fixed-point throughput certification.

Independent pulse/angle fixtures still share the existing bit encoder. Separate
standard vectors remain necessary; this matrix cannot prove both ends error-free.
"""

import hashlib
import json
import os
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


def make_profile(ft, mcs, pilot=8):
    return PSKReceiveProfile.ordinary(TxConfig(
        frame_type=ft, mcs_index=mcs, ctrl_bits_len=28 if ft == 2 else 27,
        head_llid=0x234567 if ft >= 3 else None, pilot_interval=pilot,
        symbol_rate_mhz=4, crc_seed=0x123456, whitening_seed=0x52,
    ), length_contract="standard" if ft == 2 else "legacy-b1-payload-length")


def receive(profile, x, quiet, chunks):
    rx = MacStreamReceiver(profile, quiet)
    packets, events, start = [], [], 0
    rng = np.random.default_rng(440117)
    while start < len(x):
        size = len(x) if chunks == "whole" else int(rng.integers(1, 514))
        result = rx.feed(x[start:start + size])
        packets += result.packets
        events += result.diagnostics["events"]
        start += size
        assert rx.phy.buffered_samples <= (
            rx.phy.history_limit + len(rx.phy.h) - 1 + len(rx.phy._box) - 1 + rx.phy.quantum
        )
    before = len(packets)
    result = rx.finish()
    packets += result.packets
    events += result.diagnostics["events"]
    # Post-selection diagnostics only. CI stays self-contained unless the caller
    # explicitly requests independent evidence files outside the source tree.
    if folder := os.environ.get("RX_SCOPE_EVIDENCE"):
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        config = {f.name: getattr(profile.config, f.name) for f in fields(profile.config)
                  if not f.name.startswith("_")}
        descriptor = dict(config=config, max_payload_bytes=profile.max_payload_bytes)
        key = hashlib.sha256(x.tobytes() + quiet.tobytes() +
                             json.dumps(descriptor, sort_keys=True).encode()).hexdigest()
        path = out / (key + "-input.npz")
        if not path.exists():
            np.savez_compressed(path, iq=x, quiet=quiet)
        arrays = {f"{i}-{name}": e[name] for i, e in enumerate(events) for name in
                  ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]}
        np.savez_compressed(out / (key + "-" + chunks + "-trace.npz"), **arrays)
        report = dict(**descriptor, chunks=chunks, input_sha256=hashlib.sha256(
            path.read_bytes()).hexdigest(), iq_dtype=str(x.dtype), samples=len(x),
            before_eof=before, frames=[e["frame"] for e in events],
            delivered_payloads=[p.mac_payload.hex() for p in packets],
            statistics=rx.phy.capture_stats,
            floating_frame_array_bytes=sum(getattr(rx.phy, k).nbytes for k in
                                           ["_values", "_centers", "_rates", "_errors"]))
        (out / (key + "-" + chunks + ".json")).write_text(json.dumps(
            report, indent=2, default=lambda value: value.hex() if isinstance(value, bytes)
            else value.item()) + "\n")
    return packets, events, before


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("mcs", range(13))
@pytest.mark.parametrize("pilot", [0, 8])
def test_all_configured_psk_mcs_reach_mac_with_unknown_start_and_clock(ft, mcs, pilot):
    p = make_profile(ft, mcs, pilot)
    wire, payload = transmitted(p)
    x, q, truth, step = iq_record(
        wire, offset=((ft * 13 + mcs) * .137) % 4,
        cfo=30000 if mcs % 2 else -25000, ppm=100 if ft % 2 else -100,
    )
    arms = []
    for chunk in ["whole", "random"]:
        packets, events, before = receive(p, x, q, chunk)
        assert [packet.mac_payload for packet in packets] == [payload]
        assert before == 1 and len(events) == 1
        event = events[0]
        frame = event["frame"]
        assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
        assert frame["header"]["data_length"] == 37 and frame["mcs"] == mcs
        assert np.max(abs(event["symbol_centers_raw"] - (
            truth + np.arange(len(event["symbols"])) * step))) <= .25
        arms.append(events)
    assert arms[0][0]["frame"] == arms[1][0]["frame"]
    np.testing.assert_array_equal(arms[0][0]["symbol_centers_raw"],
                                  arms[1][0]["symbol_centers_raw"])


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("mcs", [8, 12])
@pytest.mark.parametrize("length", [1, 2047])
def test_psk_field_length_extremes_deliver_before_eof(ft, mcs, length):
    p = make_profile(ft, mcs)
    wire, payload = transmitted(p, length=length)
    x, q, _, _ = iq_record(wire, offset=.619, cfo=30000, ppm=100)
    packets, events, before = receive(p, x, q, "random")
    assert before == 1 and [packet.mac_payload for packet in packets] == [payload]
    assert events[0]["frame"]["header"]["data_length"] == length


@pytest.mark.parametrize("ft", [3, 4])
@pytest.mark.parametrize("actual_mcs", [6, 12])
def test_b1_payload_mcs_comes_from_validated_header_not_caller_length(ft, actual_mcs):
    transmitted_profile = make_profile(ft, actual_mcs)
    wire, payload = transmitted(transmitted_profile)
    x, q, _, _ = iq_record(wire)
    p = make_profile(ft, 8)
    packets, events, before = receive(p, x, q, "random")
    assert before == 1 and [packet.mac_payload for packet in packets] == [payload]
    assert events[0]["frame"]["header_crc"] and events[0]["frame"]["mcs"] == actual_mcs


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("damage,status", [({"bad_head": True}, "bad_header_crc"),
                                           ({"bad_data": True}, "bad_data_crc")])
def test_every_psk_frame_type_rejects_actual_bad_crc_at_mac(ft, damage, status):
    p = make_profile(ft, 8)
    wire, _ = transmitted(p, **damage)
    x, q, _, _ = iq_record(wire)
    packets, events, before = receive(p, x, q, "random")
    assert packets == [] and before == 0
    assert [e["frame"]["status"] for e in events] == [status]


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_each_frame_rejects_validated_length_exceeding_profile_bound(ft):
    p = make_profile(ft, 8)
    wire, _ = transmitted(p)
    x, q, _, _ = iq_record(wire)
    packets, events, _ = receive(replace(p, max_payload_bytes=32), x, q, "random")
    assert packets == [] and events[0]["frame"]["header_crc"]
    assert events[0]["frame"]["status"] == "invalid_length"


def test_gfsk_cannot_be_mislabeled_as_supported_psk_stream():
    with pytest.raises(ValueError, match="GFSK"):
        PSKReceiveProfile.ordinary(TxConfig(frame_type=1))
