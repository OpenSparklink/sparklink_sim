"""Bit-level acquisition, CRC isolation and recovery without payload-assisted search."""
import numpy as np
import pytest

from nearlink_sdr.common.crc import CRC24A_POLY, crc_attach
from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.common.scrambler import scramble
from nearlink_sdr.phy.control_info import ControlInfoA3
from nearlink_sdr.phy.frame import decode_ft1_a3_candidates

SYNC = 0x9D8EB6C5
ARGS = dict(sync_word=SYNC, data_crc_seed=0x555555, whitening_seed=None)


def frame(length=37, whitening=None):
    sync = np.array([(SYNC >> i) & 1 for i in range(32)])
    head = ControlInfoA3(0, 0, 0, 0, 0, 0, 0, length).pack(SYNC & 0xFFF)
    data = crc_attach(prbs9(length * 8), CRC24A_POLY, 24, 0x555555)
    tail = np.concatenate([head, data]).astype(np.uint8)
    if whitening is not None:
        tail = scramble(tail, whitening)
    return np.concatenate([sync, tail])


@pytest.mark.parametrize("length", [0, 37, 255])
@pytest.mark.parametrize("seed", [None, 73])
def test_full_frames_and_no_input_mutation(length, seed):
    b = np.concatenate([np.zeros(19, dtype=int), frame(length, seed), np.zeros(8)])
    original = b.copy()
    rows = decode_ft1_a3_candidates(b, **(ARGS | dict(whitening_seed=seed)))
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "valid" and row["header_crc"] and row["data_crc"]
    assert row["header"]["data_length"] == length
    assert row["payload"] == np.packbits(prbs9(length * 8), bitorder="little").tobytes()
    assert row["sync_start"] == 19 and row["payload_start"] == 91
    assert row["frame_end"] == len(b) - 8
    np.testing.assert_array_equal(b, original)


def test_literal_header_vector_and_crc_rejection():
    # SDK-sync/A3 real-IQ observation, fixed independently of this test's packer.
    head = np.unpackbits(np.frombuffer(bytes.fromhex("0000fec1f6"), dtype="u1"),
                         bitorder="little")
    b = np.concatenate([frame(255)[:32], head])
    row = decode_ft1_a3_candidates(b, **ARGS)[0]
    assert row["header_crc"] and row["header"]["data_length"] == 255
    assert row["status"] == "truncated_data" and row["payload"] is None
    b[34] ^= 1
    row = decode_ft1_a3_candidates(b, **ARGS)[0]
    assert row["status"] == "bad_header_crc" and row["payload"] is None
    assert "header" not in row


def test_bad_candidate_then_good_in_same_input():
    bad_head = frame()
    bad_head[35] ^= 1
    bad_data = frame()
    bad_data[80] ^= 1
    b = np.concatenate([bad_head, bad_data, frame()])
    rows = decode_ft1_a3_candidates(b, **ARGS)
    assert [r["status"] for r in rows] == ["bad_header_crc", "bad_data_crc", "valid"]
    assert all(r["payload"] is None for r in rows[:2])


def test_truncations_and_wrong_sync_seed_or_whitening():
    b = frame()
    for length, status in [(40, "truncated_header"), (len(b) - 1, "truncated_data")]:
        row = decode_ft1_a3_candidates(b[:length], **ARGS)[0]
        assert row["status"] == status and row["payload"] is None
    assert decode_ft1_a3_candidates(b, **(ARGS | dict(sync_word=0x5A2BDA62))) == []
    for options in [dict(data_crc_seed=0), dict(whitening_seed=1)]:
        row = decode_ft1_a3_candidates(b, **(ARGS | options))[0]
        assert row["status"] != "valid" and row["payload"] is None
    assert decode_ft1_a3_candidates(np.array([], dtype=int), **ARGS) == []


@pytest.mark.parametrize("bits,options", [
    (np.array([[0]]), {}), (np.array([2]), {}),
    (np.array([0]), {"sync_word": -1}),
    (np.array([0]), {"data_crc_seed": 1 << 24}),
    (np.array([0]), {"whitening_seed": 128}),
])
def test_invalid_inputs(bits, options):
    with pytest.raises(ValueError):
        decode_ft1_a3_candidates(bits, **(ARGS | options))
