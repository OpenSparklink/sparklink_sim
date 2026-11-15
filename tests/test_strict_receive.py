"""Independent transmitted pulses/angles → strict PHY → MAC, without oracle-assisted sync."""
from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.control_info import ControlInfoA2, ControlInfoA3, ControlInfoB1
from nearlink_sdr.phy.mac_interface import iq_to_mac_capture
from nearlink_sdr.phy.rx_pipeline import receive_capture
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_head, encode_payload, head_crc_parameters
from tests.test_psk_field_layout import independent_wire
from tests.test_rx_timing_audit import audit


def transmitted(profile, *, bad_head=False, bad_data=False, empty=False, mcs=None):
    cfg = profile.config
    payload = np.random.default_rng(65016).integers(0, 256, 37, dtype=np.uint8).tobytes()
    data = np.unpackbits(np.frombuffer(payload, dtype=np.uint8), bitorder=profile.payload_bitorder)
    if profile.control_format == "B1":
        head = ControlInfoB1(0, 0, 0, cfg.mcs_index if mcs is None else mcs, 37, 0, 0)
        raw = head.pack(cfg.head_llid)
    else:
        parser = ControlInfoA2 if profile.control_format == "A2" else ControlInfoA3
        head = parser(0, int(empty), 0, 0, 0, 0, 0, 37)
        raw = head.pack(head_crc_parameters(cfg)[2],
                        lqi=cfg.mcs_index if profile.has_lqi else None)
    if bad_head:
        raw[0] ^= 1
    encoded = encode_head(raw, cfg)
    wire_bits = encode_payload(data, cfg)
    if bad_data:
        wire_bits = wire_bits.copy()
        wire_bits[:16] ^= 1  # Uncoded tests: corrupt data, preserve actual CRC.
    npre = 40
    sync_kind = "QPSK" if cfg.frame_type == 2 else "BPSK"
    ctrl_kind = "BPSK" if cfg.frame_type == 4 else "QPSK"
    sync = independent_wire(profile.sync_bits(), sync_kind, npre, 0, True)
    control = independent_wire(encoded, ctrl_kind, npre + len(sync),
                               16 if cfg.frame_type == 2 else 4, cfg.pilot_interval == 0)
    data_wire = independent_wire(wire_bits, cfg.mod_str, npre + len(sync) + len(control),
                                 cfg.pilot_interval, True)
    pre = np.exp(1j * np.tile([np.pi / 4, 0], npre // 2))
    return np.r_[pre, sync, control, data_wire], payload


def iq_record(symbols, *, ppm=0, cfo=0, offset=.277):
    x, sync_start, step = audit.reference(symbols, offset, ppm)
    x *= np.exp(2j * np.pi * cfo * np.arange(len(x)) / 16e6)
    rng = np.random.default_rng(65017)
    q = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * .001
    return x, q, sync_start, step


@pytest.mark.parametrize("ft,mcs", [(2, 6), (2, 8), (2, 12), (3, 0), (3, 8), (4, 0), (4, 12)])
@pytest.mark.parametrize("pilot", [0, 8])
def test_unknown_start_and_header_length_reach_mac(ft, mcs, pilot):
    cfg = TxConfig(frame_type=ft, mcs_index=mcs, pid=0, whitening_seed=0x52,
                   crc_seed=0x123456, ctrl_bits_len=28 if ft == 2 else 27,
                   head_llid=0x234567 if ft >= 3 else None,
                   pilot_interval=pilot, symbol_rate_mhz=4)
    profile = PSKReceiveProfile.ordinary(cfg)
    symbols, payload = transmitted(profile)
    direct = decode_psk_frame(symbols[40:], profile)
    assert direct["status"] == "valid" and direct["payload"] == payload
    x, q, _, _ = iq_record(symbols)
    result = iq_to_mac_capture(x, q, profile, energy_window_symbols=8)
    assert [p.mac_payload for p in result.packets] == [payload]
    row, = result.diagnostics["frames"]
    assert row["header_crc"] and row["data_crc"] and row["length_ok"]
    assert row["header"]["data_length"] == 37 and row["mcs"] == mcs
    assert result.packets[0].header == row["header"]


def test_factory_profile_is_explicit_and_uses_same_parser():
    from nearlink_sdr.phy.frame import decode_ft2_a3_capture
    cfg = TxConfig(mcs_index=12, ctrl_bits_len=36, crc_seed=0x555555,
                   pilot_interval=0, symbol_rate_mhz=4)
    profile = PSKReceiveProfile.ws63_ft2_a3(sync_word=audit.WORD,
                                          data_crc_seed=cfg.crc_seed, pilot_interval=0)
    # Factory profile MCS is selected only by validated header, never by expected bytes.
    wire, payload = transmitted(replace(profile, config=replace(profile.config, mcs_index=12)))
    direct = decode_ft2_a3_capture(wire[40:], sync_word=audit.WORD,
                                   data_crc_seed=cfg.crc_seed, pilot_interval=0)
    assert direct["status"] == "valid" and direct["payload"] == payload
    x, q, truth, step = iq_record(wire, ppm=100, cfo=30000)
    result = iq_to_mac_capture(x, q, profile, energy_window_symbols=8)
    assert [p.mac_payload for p in result.packets] == [payload]
    burst = result.diagnostics["acquisition"]["bursts"][result.diagnostics["frames"][0]["burst_index"]]
    centers = burst["symbol_centers_raw"]
    assert len(centers) > 100
    # This is independent known transmitted time, not a decision residual.
    assert np.max(abs(centers[:100] - (truth + np.arange(100)*step))) < .2


@pytest.mark.parametrize("damage,expected", [({"bad_head": True}, "bad_header_crc"),
                                            ({"bad_data": True}, "bad_data_crc"),
                                            ({"empty": True}, "invalid_length")])
def test_bad_frames_never_reach_mac(damage, expected):
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, _ = transmitted(profile, **damage)
    x, q, _, _ = iq_record(wire)
    result = iq_to_mac_capture(x, q, profile, energy_window_symbols=8)
    assert result.packets == []
    assert [r["status"] for r in result.diagnostics["frames"]] == [expected]


def test_bad_header_does_not_read_untrusted_length_or_decode_payload(monkeypatch):
    import nearlink_sdr.phy.rx_pipeline as pipeline
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8))
    wire, _ = transmitted(profile, bad_head=True)
    monkeypatch.setattr(pipeline, "payload_coded_length", lambda *a: pytest.fail("Read untrusted length"))
    row = decode_psk_frame(wire[40:], profile)
    assert row["status"] == "bad_header_crc" and "header" not in row and row["payload"] is None


def test_length_limit_truncation_unsupported_mcs_and_silence():
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8))
    wire, _ = transmitted(profile)
    assert decode_psk_frame(wire[40:80], profile)["status"] == "truncated_header"
    assert decode_psk_frame(wire[40:-10], profile)["status"] == "truncated_data"
    row = decode_psk_frame(wire[40:], replace(profile, max_payload_bytes=32))
    assert row["status"] == "invalid_length" and row["header_crc"] and not row["length_ok"]
    b = PSKReceiveProfile.ordinary(TxConfig(frame_type=3, ctrl_bits_len=27, head_llid=7))
    bad, _ = transmitted(b, mcs=15)
    assert decode_psk_frame(bad[40:], b)["status"] == "unsupported_mcs"
    quiet = np.zeros(4096, complex)
    assert receive_capture(quiet, quiet, profile)["frames"] == []


def test_profiles_do_not_guess_missing_context_or_factory_layout():
    with pytest.raises(ValueError, match="head_llid"):
        PSKReceiveProfile.ordinary(TxConfig(frame_type=3, ctrl_bits_len=27))
    with pytest.raises(ValueError, match="ctrl_bits_len"):
        PSKReceiveProfile.ordinary(TxConfig(ctrl_bits_len=36))
    with pytest.raises(ValueError, match="Sync word"):
        PSKReceiveProfile.ws63_ft2_a3(sync_word=2**64, data_crc_seed=0, pilot_interval=0)


def test_two_separated_packets_bad_first_header_and_input_immutability():
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    good, payload = transmitted(profile)
    bad, _ = transmitted(profile, bad_head=True)
    first, q, _, _ = iq_record(bad)
    second, _, _, _ = iq_record(good)
    joined = np.r_[first, np.zeros(512, complex), second]
    original = joined.copy()
    result = iq_to_mac_capture(joined, q, profile, energy_window_symbols=8)
    assert [p.mac_payload for p in result.packets] == [payload]
    assert [r["status"] for r in result.diagnostics["frames"]] == ["bad_header_crc", "valid"]
    np.testing.assert_array_equal(joined, original)
    all_good = iq_to_mac_capture(np.r_[second, np.zeros(512), second], q, profile,
                                 energy_window_symbols=8)
    assert [p.mac_payload for p in all_good.packets] == [payload, payload]


def test_raw_end_truncation_and_wrong_seed_are_rejections():
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, _ = transmitted(profile)
    x, q, start, step = iq_record(wire)
    cut = int(start + (len(wire) - 40 - 15) * step)
    result = iq_to_mac_capture(x[:cut], q, profile, energy_window_symbols=8)
    assert not result.packets
    assert result.diagnostics["frames"][0]["status"] == "truncated_data"
    wrong = replace(profile, config=replace(profile.config,
                    head_crc_seed=head_crc_parameters(profile.config)[2] ^ 1))
    result = iq_to_mac_capture(x, q, wrong, energy_window_symbols=8)
    assert not result.packets
    assert result.diagnostics["frames"][0]["status"] == "bad_header_crc"
