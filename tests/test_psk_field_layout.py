"""Independent angle reference catches field/pilot parity resets (§6.2.1.2, §6.7)."""
import numpy as np
import pytest

from nearlink_sdr.phy.frame import FrameConfig, FrameFields, frame_to_symbols, symbols_to_data_bits


def independent_wire(bits, kind, start, interval, omit_last):
    bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3}[kind]
    angles = {"BPSK": [90, -90], "QPSK": [45, 135, -45, -135],
              "8PSK": [22.5, 67.5, 157.5, 112.5, -22.5, -67.5, -157.5, -112.5]}[kind]
    rotation = {"BPSK": -90, "QPSK": -45, "8PSK": -22.5}[kind]
    padded = np.pad(bits, (0, (-len(bits)) % bps))
    out = []
    for j, group in enumerate(padded.reshape(-1, bps)):
        index = sum(int(b) << (bps - i - 1) for i, b in enumerate(group))
        angle = angles[index] + rotation * ((start + len(out)) % 2)
        out.append(np.exp(1j * np.deg2rad(angle)))
        if interval and (j + 1) % interval == 0 and (not omit_last or j + 1 < len(padded)//bps):
            # Standard pilots: BPSK 90°, QPSK 45°, 8PSK 22.5°.
            angle = angles[0] + rotation * ((start + len(out)) % 2)
            out.append(np.exp(1j * np.deg2rad(angle)))
    return np.asarray(out)


@pytest.mark.parametrize("kind", ["QPSK", "8PSK"])
@pytest.mark.parametrize("interval", [0, 4, 8, 16])
def test_frame_matches_independent_global_symbol_reference(kind, interval):
    from nearlink_sdr.phy.preamble import generate_preamble
    from nearlink_sdr.phy.sync_sequence import sync_signal_2

    rng = np.random.default_rng(64016)
    cfg = FrameConfig(frame_type=2, mod_type=kind, pilot_interval=interval)
    fields = FrameFields(generate_preamble(2, 1), sync_signal_2(0),
                         rng.integers(0, 2, 64), rng.integers(0, 2, 80), 2)
    before = len(fields.preamble_bits)
    sync = independent_wire(fields.sync_bits, "QPSK", before, 0, True)
    control = independent_wire(fields.ctrl_bits, "QPSK", before + len(sync), 16, interval == 0)
    payload = independent_wire(fields.data_bits, kind, before + len(sync) + len(control),
                               interval, True)
    actual = frame_to_symbols(fields, cfg)
    np.testing.assert_allclose(actual[before:], np.r_[sync, control, payload], atol=1e-14, rtol=0)
    head, data = symbols_to_data_bits(actual, cfg, n_ctrl_coded_bits=64, n_data_bits=80)
    np.testing.assert_array_equal(head, fields.ctrl_bits)
    np.testing.assert_array_equal(data, fields.data_bits)
