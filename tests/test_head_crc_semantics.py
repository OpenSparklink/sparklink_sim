"""Real §6.4 header CRC gate, including formerly accepted invalid synthetic heads."""
from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.rx_pipeline import decode_head, rx_chain
from nearlink_sdr.phy.tx_pipeline import (
    TxConfig,
    attach_head_crc,
    encode_head,
    head_crc_parameters,
    tx_chain,
)


@pytest.mark.parametrize("ft,k", [(1, 20), (2, 28), (3, 27), (4, 27)])
def test_actual_control_crc_and_wrong_context(ft, k):
    cfg = TxConfig(frame_type=ft, mcs_index=8, pid=0 if ft >= 3 else 0x879012,
                   ctrl_bits_len=k, head_llid=0x123456 if ft >= 3 else None)
    head = attach_head_crc(np.random.default_rng(63016).integers(0, 2, k), cfg)
    bits, ok = decode_head(encode_head(head, cfg), cfg)
    assert ok
    np.testing.assert_array_equal(bits, head)
    bad = head.copy()
    bad[0] ^= 1  # Corrupt information before channel coding: ECC cannot legitimize its CRC.
    assert not decode_head(encode_head(bad, cfg), cfg)[1]
    other = (replace(cfg, head_llid=cfg.head_llid ^ 1) if ft >= 3 else
             replace(cfg, head_crc_seed=head_crc_parameters(cfg)[2] ^ 1))
    assert not decode_head(encode_head(head, cfg), other)[1]
    # Payload CRC seed has no role in control CRC.
    assert decode_head(encode_head(head, cfg), replace(cfg, crc_seed=0xFEDCBA))[1]
    encoded = encode_head(head, cfg)
    for n in [0, 1, len(encoded) - 1, len(encoded) + 1]:
        truncated = np.pad(encoded, (0, 1))[:n]
        decoded, ok = decode_head(truncated, cfg)
        assert not ok and decoded.size == 0


def test_standard_a_seed_is_sync_low_bits_not_pid():
    cfg = TxConfig(pid=0x879012)
    assert head_crc_parameters(cfg)[2] == 0xAAD  # TV201 actual SyncWord2.
    assert head_crc_parameters(cfg)[2] != (cfg.pid & 0xFFF)


@pytest.mark.parametrize("ft", [3, 4])
def test_missing_b_llid_is_an_error(ft):
    cfg = TxConfig(frame_type=ft)
    with pytest.raises(ValueError, match="head_llid"):
        decode_head(np.zeros(256, dtype=int), cfg)
    with pytest.raises(ValueError, match="head_llid"):
        rx_chain(np.zeros(2000), cfg, 8)


def test_bad_head_stops_payload_processing_and_delivery(monkeypatch):
    import nearlink_sdr.phy.rx_pipeline as rx

    cfg = TxConfig(mcs_index=8, pilot_interval=8)
    head = attach_head_crc(np.zeros(cfg.ctrl_bits_len, dtype=int), cfg)
    head[0] ^= 1
    iq = tx_chain(head, np.ones(64, dtype=int), cfg)

    def forbidden(*args, **kwargs):
        pytest.fail("Payload decoder must not run after a failed header")

    monkeypatch.setattr(rx, "decode_payload", forbidden)
    result = rx_chain(iq, cfg, 8)
    assert not result.head_crc_ok and not result.crc_ok and result.data_bits.size == 0


@pytest.mark.parametrize("length", [-1, 2048, 1.5, True])
def test_illegal_known_length_is_rejected_before_allocation(length):
    with pytest.raises(ValueError, match="payload length"):
        rx_chain(np.zeros(2000), TxConfig(), length)


def test_invalid_header_lengths_and_bits():
    for k in [-1, 0, 65]:
        with pytest.raises(ValueError, match="information length"):
            decode_head(np.zeros(64), TxConfig(ctrl_bits_len=k))
    for bits in [np.zeros((8, 8)), np.full(64, 2), np.full(64, np.nan)]:
        with pytest.raises(ValueError, match="binary"):
            decode_head(bits, TxConfig())
    with pytest.raises(ValueError, match="12-bit"):
        decode_head(np.zeros(64), TxConfig(head_crc_seed=4096))


def test_arbitrary_parity_is_a_failure_instead_of_a_positive_fixture():
    cfg = TxConfig(frame_type=3, ctrl_bits_len=27, head_llid=0)
    previous_fixture = np.random.default_rng(301).integers(0, 2, 51).astype(np.int8)
    decoded, ok = decode_head(encode_head(previous_fixture, cfg), cfg)
    np.testing.assert_array_equal(decoded, previous_fixture)
    assert not ok
