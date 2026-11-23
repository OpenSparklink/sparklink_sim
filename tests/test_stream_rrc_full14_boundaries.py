"""Standard legal lengths,finite EOF and genuine CRC failures on the new frontend."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from tests.captured_frame_reference import packet
from tests.full14_stream_reference import fixture, receive
from tests.maximum_reference import channel

MODE = "word-full14-c18"


@pytest.mark.parametrize("ft,mcs", [(2, 8), (2, 12), (3, 8), (3, 12), (4, 8), (4, 12)])
@pytest.mark.parametrize("maximum", [False, True])
def test_minimum_and_maximum_standard_wire_lengths_are_not_legacy_payload_lengths(ft, mcs, maximum):
    crc_bytes = 3 if mcs == 8 else 4
    size = (2047-(crc_bytes if ft >= 3 else 0)) if maximum else 1
    p, x, q, payload, *_ = fixture(ft, mcs, 8, size=size, index=3157 if mcs == 8 else 3158)
    arms = [receive(p, x, q, MODE, mode) for mode in ("whole", "random")]
    for a in arms:
        assert [r.mac_payload for r in a["packets"]] == [payload] and a["before_EOF"] == 1
        frame = a["events"][0]["frame"]
        assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
        assert frame["header"]["data_length"] == (2047 if maximum else
                                                  1+(crc_bytes if ft >= 3 else 0))
    np.testing.assert_array_equal(arms[0]["events"][0]["symbol_centers_raw"],
                                  arms[1]["events"][0]["symbol_centers_raw"])


@pytest.mark.parametrize("damage,status", [(dict(bad_head=True), "bad_header_crc"),
                                           (dict(bad_data=True), "bad_data_crc")])
def test_bad_crc_stays_a_failure_and_does_not_reach_mac(damage, status):
    p, x, q, *_ = fixture(2, 12, 0, damage=damage)
    a = receive(p, x, q, MODE, "random")
    assert not a["packets"] and len(a["events"]) == 1
    assert a["events"][0]["frame"]["status"] == status
    assert a["events"][0]["frame"]["payload"] is None


def test_eof_truncation_seals_once_and_never_invents_a_payload():
    p, x, q, _, truth, step = fixture(2, 12, 0, size=127)
    for symbols, status in [(45, "truncated_header"), (120, "truncated_data")]:
        a = receive(p, x[:int(truth+symbols*step)], q, MODE, "random")
        assert not a["packets"] and len(a["events"]) == 1
        assert a["events"][0]["frame"]["status"] == status
        rx = a["receiver"].phy
        assert rx._rrc_word_model.stats["zero_flush_samples"] == rx.future_samples == 42
        assert not a["receiver"].finish().packets


def test_dense_packets_keep_one_history_and_quiet_stream_does_not_search():
    p, _, q, first, *_ = fixture(3, 8, 4, size=59)
    second = bytes((k*31+23) & 255 for k in range(17))
    symbols = np.r_[packet(p, first), np.zeros(8), packet(p, second)]
    x, _, _, _, _ = channel(symbols, offset=.481, ppm=75, cfo=-37000, prefix=173,
                            amplitude=.17, family="clean", seed=1033791)
    a = receive(p, x, q, MODE, "random")
    assert a["before_EOF"] == 2 and [r.mac_payload for r in a["packets"]] == [first, second]
    assert all(e["emitted_after_samples"] < len(x) for e in a["events"])
    assert a["receiver"].phy._rrc_word_model.stats["samples"] == len(x)+42
    silent = receive(p, np.zeros(9001), q, MODE, "random")
    assert not silent["packets"] and not silent["events"]
    assert silent["receiver"].phy.sync_searches == 0


def test_factory_profile_keeps_explicit_sync_crc_and_actual_header_mcs():
    from tests.test_rx_timing_audit import audit
    from tests.test_strict_receive import iq_record, transmitted

    p = PSKReceiveProfile.ws63_ft2_a3(sync_word=audit.WORD, data_crc_seed=0x555555,
                                     pilot_interval=0)
    wire, payload = transmitted(replace(p, config=replace(p.config, mcs_index=12)), length=255)
    x, q, _, _ = iq_record(wire, cfo=30000, ppm=100)
    a = receive(p, x, q, MODE, "random")
    assert a["before_EOF"] == 1 and [r.mac_payload for r in a["packets"]] == [payload]
    assert p.config.mcs_index == 8 and a["events"][0]["frame"]["mcs"] == 12
    assert p.sync_word == audit.WORD and p.config.crc_seed == 0x555555
