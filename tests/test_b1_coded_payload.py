"""Standard FT3/4 logical CRC groups, tail Polar words and real MAC delivery."""

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.polar import PolarEncoder
from nearlink_sdr.common.scrambler import scramble_sequence
from nearlink_sdr.phy.rx_pipeline import decode_payload, payload_coded_length
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_payload
from nearlink_sdr.sim.header_geometry_flow import HeaderGeometryService
from tests.test_receive_scope import receive
from tests.test_strict_receive import iq_record, transmitted


def polynomial_crc(data, polynomial, width, seed):
    n = len(data)
    info = 0
    for value in data:
        info = (info << 1) | int(value)
    value = (info << width) ^ ((seed & ((1 << width) - 1)) << n)
    divisor = (1 << width) | polynomial
    while value.bit_length() > width:
        value ^= divisor << (value.bit_length() - width - 1)
    return [(value >> k) & 1 for k in range(width - 1, -1, -1)]


def independent_encoded_payload(data, cfg, sizes, *, bad=None, bad_frame_crc=False):
    """Use literal reference (N,K) placements, independent of the plan under test."""
    crc_width = cfg.crc_len
    outer_poly = 0x00065B if crc_width == 24 else 0x04C11DB7
    frame_crc = polynomial_crc(data, outer_poly, crc_width, cfg.crc_seed)
    if bad_frame_crc:
        frame_crc[0] ^= 1
    frame = np.r_[data, frame_crc]
    split = sizes[0][1] - 24
    words = []
    for i, (group, dimensions) in enumerate(zip((frame[:split], frame[split:]),
                                                (sizes[:1], sizes[1:]), strict=True)):
        seed = cfg.crc_seed if cfg.b1_code_block_mode == "standard" else 0
        trailer = polynomial_crc(group, 0xB2B117, 24, seed)
        if bad == i:
            trailer[-1] ^= 1
        prep = np.r_[np.zeros(sum(k for _, k in dimensions) - len(group) - 24, np.int8),
                     group, trailer]
        pos = 0
        for n, k in dimensions:
            words.append(PolarEncoder(n, k).encode(prep[pos:pos+k]))
            pos += k
    coded = np.concatenate(words)
    return coded ^ scramble_sequence(256 + len(coded), cfg.whitening_seed)[256:]


@pytest.mark.parametrize("b,words", [
    (32, ((128, 24), (64, 12))),
    (184, ((1024, 192),)),
    (192, ((1024, 192),)),
    (200, ((1024, 256),)),
    (256, ((1024, 256),)),
])
def test_literal_appendix_a_rate_quarter_tail_geometry(b, words):
    # V2.2.0 Appendix A covers a logical tail's K_r, including its CRC if present.
    single, = b1_code_block_plan(b, "1/4", standard=True)
    assert (single.data_bits, single.crc_bits, single.codewords) == (b, 0, words)
    _, full, tail = b1_code_block_plan(464 + b - 24, "1/4", standard=True)
    assert full.codewords == ((1024, 256),) and full.crc_bits == 24
    assert (tail.data_bits, tail.crc_bits, tail.codewords) == (b - 24, 24, words)


@pytest.mark.parametrize(("mcs", "kcb", "unit"), [
    (0, 256, 12), (1, 384, 20), (4, 512, 28),
    (5, 640, 36), (6, 768, 44), (7, 896, 52),
])
def test_finite_standard_geometry_at_ordinary_adjusted_and_full_blocks(mcs, kcb, unit):
    g = HeaderGeometryService(3, 24, 8, b1_code_block_mode="standard")
    for bit_count, count in [(kcb, 1024), (kcb + 8, 1024 + 64 * ((32 + 23 + unit)//unit)),
                             (16 * unit, 1024), (16 * unit + 8, 1024),
                             (kcb - 24 + 16 * unit, 2048), (2 * (kcb - 24), 2048)]:
        assert g.issue(bit_count // 8, mcs)
        start = g.cycle
        while not g.ready and g.cycle - start < 128:
            g.step()
        result = g.take()
        assert result is not None and result["coded_bits"] == count
        assert result["payload_length_bytes"] == bit_count//8 - 3
    assert g.cost()["registers_bits"]["register_file"] == 24 * 24
    assert g.cost()["DSP"] == g.cost()["RAMB18"] == 0


@pytest.mark.parametrize(("ft", "mcs", "length", "crc_width", "tail"), [
    (3, 4, 65, 24, ((1024, 512), (128, 56), (64, 28))),
    (4, 9, 83, 32, ((1024, 640), (128, 72), (64, 36))),
])
@pytest.mark.parametrize("mode", ["legacy", "standard"])
def test_independent_crc_and_coded_layout_at_mac(ft, mcs, length, crc_width, tail, mode):
    cfg = TxConfig(frame_type=ft, mcs_index=mcs, symbol_rate_mhz=4, ctrl_bits_len=27,
                   head_llid=0x234567, whitening_seed=0x63, crc_len=crc_width,
                   crc_seed=0x123456 if crc_width == 24 else 0x36123456,
                   pilot_interval=8, b1_code_block_mode=mode)
    dims = tail if mode == "standard" else (tail[0], tail[0])
    profile = replace(PSKReceiveProfile.ordinary(cfg), head_backend="word-dense")
    for bad, bad_frame_crc in ((None, False), (0, False), (1, False), (None, True)):
        def payload_encoder(data, actual):
            assert actual == cfg
            written = independent_encoded_payload(data, cfg, dims, bad=bad,
                                                  bad_frame_crc=bad_frame_crc)
            assert len(written) == payload_coded_length(length*8, cfg)
            checks = decode_payload(written, cfg, length*8, details=True)
            assert checks.data_crc is (not bad_frame_crc)
            assert checks.code_block_crcs == (bad != 0, bad != 1)
            assert checks.passed is (bad is None and not bad_frame_crc)
            np.testing.assert_array_equal(checks.data_bits, data)
            if bad is None and not bad_frame_crc:
                np.testing.assert_array_equal(encode_payload(data, cfg), written)
            return written

        with patch("tests.test_strict_receive.encode_payload", payload_encoder):
            wire, payload = transmitted(profile, length=length, payload_seed=778917 + ft)
        iq, quiet, _, _ = iq_record(wire, offset=.513, cfo=-29000, ppm=85)
        packets, events, before = receive(profile, iq, quiet, "random")
        valid = bad is None and not bad_frame_crc
        assert before == int(valid)
        frame, = [e["frame"] for e in events]
        assert frame["header_crc"] and frame["length_ok"]
        assert frame["data_crc"] is (not bad_frame_crc)
        assert frame["code_block_crcs"] == (bad != 0, bad != 1)
        assert frame["coded_bits"] == sum(n for n, _ in dims)
        assert frame["payload_length_bytes"] == length and frame["mcs"] == mcs
        status = ("bad_code_block_crc" if bad is not None else
                  "bad_data_crc" if bad_frame_crc else "valid")
        assert frame["status"] == status
        assert [packet.mac_payload for packet in packets] == ([payload] if valid else [])
        if mode == "standard" and valid:
            full_packets, full_events, full_before = receive(profile, iq, quiet, "whole")
            assert full_before == before and full_events[0]["frame"] == frame
            assert [packet.mac_payload for packet in full_packets] == [payload]
            np.testing.assert_array_equal(full_events[0]["symbol_centers_raw"],
                                          events[0]["symbol_centers_raw"])


@pytest.mark.parametrize("ft", [3, 4])
@pytest.mark.parametrize("mcs", [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])
def test_standard_coded_ft3_ft4_support_matrix(ft, mcs):
    cfg = TxConfig(frame_type=ft, mcs_index=mcs, symbol_rate_mhz=4, ctrl_bits_len=27,
                   head_llid=0x134625, whitening_seed=0x53, crc_len=24 if mcs % 2 else 32,
                   crc_seed=0x435347, pilot_interval=16 if mcs % 2 else 4,
                   b1_code_block_mode="standard")
    p = replace(PSKReceiveProfile.ordinary(cfg), head_backend="word-dense")
    wire, payload = transmitted(p, length=113 + 2*mcs, payload_seed=32089 + ft*mcs)
    iq, quiet, _, _ = iq_record(wire, offset=(ft*mcs + .1) % 4, ppm=85 if ft == 3 else -70,
                               cfo=31000 if mcs % 2 else -20500)
    packets, events, before = receive(p, iq, quiet, "random")
    assert before == 1 and [packet.mac_payload for packet in packets] == [payload]
    frame, = [e["frame"] for e in events]
    assert frame["mcs"] == mcs and frame["payload_length_bytes"] == len(payload)
    assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
    assert len(frame["code_block_crcs"]) > 1 and all(frame["code_block_crcs"])
