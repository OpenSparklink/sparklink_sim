"""Standard B1 input bytes and zero-data ownership through real finite head CRC."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import decode_psk_frame
from nearlink_sdr.sim.header_word_flow import HeaderWordService
from tests.test_b1_wire_length import BASELINE, ROOT, b1_header_wire, profile, receive
from tests.test_rx_timing_audit import audit


@pytest.mark.parametrize("row", BASELINE["rows"],
                         ids=lambda r: f"ft{r['frame_type']}-crc{r['CRC_width']}")
def test_original_standard_length_bytes_through_both_word_banks(row):
    ft, width = row["frame_type"], row["CRC_width"]
    p = profile(ft, width)
    z = np.load(ROOT / f"ft{ft}-crc{width}-input.npz")
    reference = decode_psk_frame(z["symbols"], p)
    for bank in ("word-dense", "word-depth-parity"):
        word = replace(p, head_backend=bank)
        result = decode_psk_frame(z["symbols"], word)
        metadata = result.pop("head_word_model")
        assert result == reference
        assert result["payload"] == bytes.fromhex(row["expected_payload"])
        assert result["frame_end_symbol"] == row["actual_frame_end"]
        assert result["payload_length_bytes"] == row["payload_bytes"]
        assert metadata["statistics"]["CRC_input_bits"] == p.config.ctrl_bits_len
        assert metadata["geometry_statistics"]["accepted"] == 1
        packets, events, _ = receive(word, z["iq"], z["quiet"], (257, 1, 37))
        assert packets == [z["payload"].tobytes()] and len(events) == 1


@pytest.mark.parametrize("ft", [3, 4])
@pytest.mark.parametrize("pilot", [0, 8])
def test_zero_control_owns_its_entire_end_once_including_experimental_no_pilot(ft, pilot,
                                                                          monkeypatch):
    p = profile(ft)
    p = replace(p, config=replace(p.config, pilot_interval=pilot))
    symbols, _ = b1_header_wire(p, 0)
    if not pilot:
        symbols = symbols[:-1]  # Existing experimental no-pilot control-tail convention.
    npre = 48 if ft == 3 else 64
    pre = np.exp(1j * np.tile([np.pi / 4, 0], npre // 2))
    x, _, _ = audit.reference(np.r_[pre, symbols], .37, 0)
    quiet = np.zeros(1024, complex)
    calls = []
    original = HeaderWordService.issue
    def issue(self, bits):
        calls.append(self.cycle)
        return original(self, bits)
    monkeypatch.setattr(HeaderWordService, "issue", issue)
    for backend in ("reference", "word-dense", "word-depth-parity"):
        before = len(calls)
        packets, events, rx = receive(replace(p, head_backend=backend), x, quiet, (37, 1, 257))
        assert not packets and len(events) == 1
        row = events[0]["frame"]
        assert row["status"] == "valid_empty_control" and row["header_crc"] is True
        assert row["data_crc"] is None and row["payload"] is None
        assert row["data_crc_present"] is False and row["coded_bits"] == 0
        assert row["frame_end_symbol"] == len(symbols)
        assert len(calls) - before == int(backend != "reference")
        assert rx.phy._skip_until > events[0]["symbol_centers_raw"][-1]
