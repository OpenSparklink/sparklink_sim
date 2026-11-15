"""Actual A3 CRC gating with frame-wide alternating PSK phase and pilots."""
import numpy as np
import pytest

from nearlink_sdr.phy.control_info import ControlInfoA3, polar_encode_control
from nearlink_sdr.phy.frame import decode_ft2_a3_capture
from nearlink_sdr.phy.pilot import pilot_symbol
from nearlink_sdr.phy.psk import PSKModulator
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_payload

SYNC = 0x6A28C4BD9D8EB6C5
ARGS = dict(sync_word=SYNC, data_crc_seed=0x555555)


def wire(bits, kind, start, interval, omit_last):
    mod = PSKModulator(kind, sps=1)
    bps = mod.bits_per_symbol
    bits = np.r_[bits, np.zeros((-len(bits)) % bps, dtype=int)]
    out = []
    for i, block in enumerate(bits.reshape(-1, bps)):
        out.append(mod.map_symbols(block, start + len(out))[0])
        if interval and (i + 1) % interval == 0 and (not omit_last or i < len(bits) // bps - 1):
            out.append(pilot_symbol(kind, start + len(out)))
    return np.array(out)


def packet(mcs, interval, *, bad_head=False, bad_data=False):
    rng = np.random.default_rng(29)
    data = rng.integers(0, 2, 37 * 8)
    head = ControlInfoA3(0, 0, 0, 0, 0, 0, 0, 37).pack(SYNC & 0xFFF, lqi=mcs)
    if bad_head:
        head[10] ^= 1
    coded_head = polar_encode_control(head)
    cfg = TxConfig(frame_type=2, mcs_index=mcs, whitening_seed=0, crc_seed=0x555555)
    encoded = encode_payload(data, cfg)
    if bad_data:
        # A whole unknown RF field, not one correctable coded bit.
        encoded[:] = rng.integers(0, 2, len(encoded))
    sync = PSKModulator("QPSK", sps=1).map_symbols(np.array([(SYNC >> i) & 1 for i in range(64)]))
    control = wire(coded_head, "QPSK", 32, 16, interval == 0)
    payload = wire(encoded, cfg.mod_str, 32 + len(control), interval, True)
    return np.r_[sync, control, payload], data


@pytest.mark.parametrize("mcs", [6, 8, 10, 12])
@pytest.mark.parametrize("interval", [0, 1, 4, 16])
def test_real_crc_and_length_without_expected_payload(mcs, interval):
    symbols, bits = packet(mcs, interval)
    original = symbols.copy()
    row = decode_ft2_a3_capture(symbols, pilot_interval=interval, **ARGS)
    assert row["status"] == "valid" and row["header_crc"] and row["data_crc"]
    assert row["header"]["data_length"] == 37 and row["mcs"] == mcs
    assert row["payload"] == np.packbits(bits, bitorder="little").tobytes()
    assert row["frame_end_symbol"] == len(symbols)
    if interval:
        assert row["pilot_count"] > 0 and row["pilot_phase_rms_rad"] < 1e-10
    np.testing.assert_array_equal(symbols, original)


def test_bad_crc_and_truncation_do_not_expose_payload():
    for kw, expected in [(dict(bad_head=True), "bad_header_crc"),
                         (dict(bad_data=True), "bad_data_crc")]:
        symbols, _ = packet(8, 0, **kw)
        row = decode_ft2_a3_capture(symbols, pilot_interval=0, **ARGS)
        assert row["status"] == expected and row["payload"] is None
    symbols, _ = packet(8, 0)
    for n, expected in [(40, "truncated_header"), (100, "truncated_data")]:
        row = decode_ft2_a3_capture(symbols[:n], pilot_interval=0, **ARGS)
        assert row["status"] == expected and row["payload"] is None
    with pytest.raises(ValueError):
        decode_ft2_a3_capture(symbols, pilot_interval=3, **ARGS)
    with pytest.raises(ValueError):
        decode_ft2_a3_capture(np.array([float('nan')]), pilot_interval=0, **ARGS)


def test_lqi_unsupported_mcs_remains_unknown():
    head = ControlInfoA3(0, 0, 0, 0, 0, 0, 0, 37).pack(SYNC & 0xFFF, lqi=0)
    control = wire(polar_encode_control(head), "QPSK", 32, 16, True)
    row = decode_ft2_a3_capture(np.r_[np.ones(32), control], pilot_interval=0, **ARGS)
    assert row["status"] == "unsupported_mcs" and row["payload"] is None
