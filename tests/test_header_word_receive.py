"""Explicit word head in strict IQ→MAC with unchanged payload processing."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import decode_psk_frame
from nearlink_sdr.sim.header_word_flow import HeaderWordService
from tests.test_receive_scope import make_profile, receive
from tests.test_strict_receive import iq_record, transmitted


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (2, 12, 8), (3, 6, 16), (4, 12, 4)])
def test_unknown_start_word_head_to_MAC_uses_one_CRC_gate_per_packet(monkeypatch, ft, mcs, pilot):
    profile = make_profile(ft, mcs, pilot)
    wire, payload = transmitted(profile, length=83, payload_seed=711057 + ft)
    x, quiet, _, _ = iq_record(wire, ppm=-83, cfo=23000, offset=0.649)
    expected, old, _ = receive(profile, x, quiet, "whole")
    calls = []
    original = HeaderWordService.issue

    def count(self, bits):
        calls.append(self.cycle)
        return original(self, bits)

    monkeypatch.setattr(HeaderWordService, "issue", count)
    for layout in ["word-dense", "word-depth-parity"]:
        p = replace(profile, head_backend=layout)
        for blocks in ["whole", "random"]:
            before = len(calls)
            packets, events, eof = receive(p, x, quiet, blocks)
            assert len(calls) == before + 1  # Valid receipt reused at payload end.
            assert [r.mac_payload for r in packets] == [payload]
            assert packets[0].mac_payload == expected[0].mac_payload and eof == 1
            row = events[0]["frame"].copy()
            stats = row.pop("head_word_model")
            assert row == old[0]["frame"]
            assert stats["demap_service_cycles"] is None
            assert not stats["whole_RX_cycle_acceptance"]
            np.testing.assert_array_equal(
                events[0]["symbol_centers_raw"], old[0]["symbol_centers_raw"]
            )


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_real_errors_and_EOF_do_not_become_success(damage, status):
    profile = make_profile(2, 8)
    wire, _ = transmitted(profile, **damage)
    word = replace(profile, head_backend="word-dense")
    a = decode_psk_frame(wire[40:], profile)
    b = decode_psk_frame(wire[40:], word)
    b.pop("head_word_model")
    assert a == b and b["status"] == status and b["payload"] is None
    assert decode_psk_frame(wire[40:60], word)["header_crc"] is None


def test_multiple_packets_bad_header_and_silence_preserve_packet_ownership():
    profile = make_profile(2, 8)
    first, p1 = transmitted(profile, length=17, payload_seed=714111)
    bad, _ = transmitted(profile, length=83, bad_head=True, payload_seed=714112)
    last, p2 = transmitted(profile, length=41, payload_seed=714113)
    pieces = [iq_record(wire)[0] for wire in (first, bad, last)]
    quiet = iq_record(first)[1]
    x = np.concatenate(
        [
            np.zeros(1024),
            pieces[0],
            np.zeros(1024),
            pieces[1],
            np.zeros(1024),
            pieces[2],
            np.zeros(1024),
        ]
    )
    baseline, old, _ = receive(profile, x, quiet, "whole")
    packets, events, _ = receive(
        replace(profile, head_backend="word-depth-parity"), x, quiet, "random"
    )
    assert [p.mac_payload for p in packets] == [p1, p2]
    assert [p.mac_payload for p in baseline] == [p1, p2]
    assert [e["frame"]["status"] for e in events] == ["valid", "bad_header_crc", "valid"]
    for left, right in zip(events, old, strict=True):
        frame = left["frame"].copy()
        frame.pop("head_word_model")
        assert frame == right["frame"]


def test_silence_observation_is_bounded_and_does_not_invoke_header_decoder(monkeypatch):
    profile = replace(make_profile(3, 8), head_backend="word-dense")
    monkeypatch.setattr(HeaderWordService, "issue", lambda *a: pytest.fail("Silent header"))
    x = np.zeros(8192, dtype=complex)
    packets, events, _ = receive(profile, x, np.ones(1024) * 0.001, "random")
    assert packets == events == []
