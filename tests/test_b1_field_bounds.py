"""§6.4.2.2.2: reject overflow before it aliases a CRC-valid empty B1."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.control_info import ControlInfoB1
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.captured_frame_reference import packet

BASELINE = json.loads((Path(__file__).parent / "data/b1-field-bounds/baseline.json")
                      .read_text())


@pytest.mark.parametrize("row", [r for r in BASELINE["rows"] if 0 <= r["length"] <= 2047])
@pytest.mark.parametrize("integer", [int, np.int64])
def test_original_valid_head_bits_and_real_CRC_are_unchanged(row, integer):
    head = ControlInfoB1(0, 0, 0, BASELINE["MCS"], integer(row["length"]), 0, 0)
    bits = head.pack(BASELINE["LLID"])
    assert bits.tolist() == row["bits"]
    assert ControlInfoB1.unpack(bits, BASELINE["LLID"]).data_length == row["length"]
    assert sum(int(bits[14 + k]) << k for k in range(11)) == row["length"]


@pytest.mark.parametrize("length", [-1, 2048, 4095, 2**100, True, np.bool_(False), 1.5, None])
def test_bad_field_never_reaches_CRC_or_becomes_a_different_valid_head(length, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid length must be rejected before CRC generation")

    monkeypatch.setattr("nearlink_sdr.phy.control_info._b_group_pack", forbidden)
    with pytest.raises(ValueError, match="11-bit"):
        ControlInfoB1(0, 0, 0, 8, length, 0, 0).pack(0x173925)


def test_original_overflow_alias_is_preserved_as_a_bad_input_not_new_success():
    bad = next(r for r in BASELINE["rows"] if r["length"] == 2048)
    assert bad["decoded_length"] == 0 and bad["real_CRC_accepted"]
    received = ControlInfoB1.unpack(np.asarray(bad["bits"]), BASELINE["LLID"])
    assert received.data_length == 0
    with pytest.raises(ValueError):
        ControlInfoB1(0, 0, 0, BASELINE["MCS"], bad["length"], 0, 0).pack(BASELINE["LLID"])


def profile(ft, width):
    return PSKReceiveProfile.ordinary(TxConfig(
        frame_type=ft, mcs_index=8 if width == 24 else 12, ctrl_bits_len=27,
        head_llid=0x173925, whitening_seed=0, crc_len=width, crc_seed=0x123456,
        pilot_interval=8, symbol_rate_mhz=4, sps=4))


@pytest.mark.parametrize("ft", [3, 4])
@pytest.mark.parametrize("width", [24, 32])
def test_generator_rejects_first_illegal_payload_before_encoding(ft, width, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Out-of-range B1 cannot reach header encoding or IQ generation")

    monkeypatch.setattr("tests.maximum_reference.encode_head", forbidden)
    with pytest.raises(ValueError, match="11-bit"):
        packet(profile(ft, width), bytes(2048 - width // 8))


@pytest.mark.parametrize("ft", [3, 4])
@pytest.mark.parametrize("width", [24, 32])
def test_legal_maximum_only_in_ideal_independent_constellation_not_full_IQ(ft, width):
    p = profile(ft, width)
    payload = bytes((k * 73 + (k >> 5) + ft * 17 + width) & 255
                    for k in range(2047 - width // 8))
    symbols = packet(p, payload)[{3: 48, 4: 64}[ft]:]
    for backend in ("reference", "word-depth-parity"):
        row = decode_psk_frame(symbols, replace(p, head_backend=backend))
        assert row["header"]["data_length"] == 2047 and row["length_ok"]
        assert row["header_crc"] is True and row["data_crc"] is True
        assert row["status"] == "valid" and row["payload"] == payload
