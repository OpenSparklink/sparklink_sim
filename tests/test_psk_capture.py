"""Known-sync acquisition independent of payload/CRC, preserving frame parity."""
import numpy as np
import pytest
from scipy.signal import fftconvolve

from nearlink_sdr.phy.psk import (
    PSKDemodulator,
    PSKModulator,
    acquire_psk_capture,
    refine_psk_carrier,
    rrc_filter,
)


@pytest.mark.parametrize("kind", ["QPSK", "8PSK"])
@pytest.mark.parametrize("sps", [4, 8, 16])
def test_full_capture_blind_carrier_and_frame_parity(kind, sps):
    rng = np.random.default_rng(144)
    sync_bits = np.array([(0x6A28C4BD9D8EB6C5 >> i) & 1 for i in range(64)])
    qpsk = PSKModulator("QPSK", sps=1)
    sync = qpsk.map_symbols(sync_bits, start_symbol_index=16)
    head = qpsk.map_symbols(rng.integers(0, 2, 64), start_symbol_index=48)
    bps = 2 if kind == "QPSK" else 3
    bits = rng.integers(0, 2, 1200 * bps)
    data = PSKModulator(kind, sps=1).map_symbols(bits, start_symbol_index=80)
    preamble = np.exp(1j * np.tile([np.pi / 4, 0], 8))
    all_symbols = np.r_[preamble, sync, head, data]
    up = np.zeros(len(all_symbols) * sps, complex)
    up[::sps] = all_symbols
    waveform = 60 * fftconvolve(up, rrc_filter(.4, sps), "same")
    offset = 1703
    x = np.zeros(offset + len(waveform) + 1000, complex)
    x[offset:offset + len(waveform)] = waveform
    fs, rs, cfo = sps * 1000000, 1000000, -73000
    x *= np.exp(1j * (0.73 + 2 * np.pi * cfo * np.arange(len(x)) / fs))
    x += .1 * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    quiet = .1 * (rng.normal(size=20000) + 1j * rng.normal(size=20000))
    original = x.copy()
    result = acquire_psk_capture(x, quiet, fs, rs, sync)
    bursts = [b for b in result["bursts"] if b["status"] == "synchronized_unparsed"]
    assert len(bursts) == 1
    b = bursts[0]
    assert abs(b["sync_start_raw"] - (offset + 16 * sps)) <= 1
    assert b["sync_score"] > .9
    assert b["cfo_hz"] == pytest.approx(cfo, abs=3000)
    y, fine = refine_psk_carrier(b["symbols"][64:64 + len(data)], kind, 80)
    decoded = PSKDemodulator(kind, sps=1).demap_symbols(y, 80)
    assert np.mean(decoded[20:-20] != bits[20:-20]) < .01
    assert abs(fine["residual_cycles_per_symbol"]) < .005
    np.testing.assert_array_equal(x, original)


def test_no_sync_short_and_edge_are_reported():
    rng = np.random.default_rng(1)
    sync = PSKModulator("QPSK", sps=1).map_symbols(rng.integers(0, 2, 64))
    q = .01 * (rng.normal(size=20000) + 1j * rng.normal(size=20000))
    assert acquire_psk_capture(q, q, 16000000, 1000000, sync)["bursts"] == []
    x = np.zeros(20000, complex)
    x[100:150] = 10
    x[10000:] = 10
    rows = acquire_psk_capture(x, q, 16000000, 1000000, sync)["bursts"]
    assert rows[0]["status"] == "short_energy_segment"
    assert rows[-1]["edge_truncated"] and rows[-1]["status"] == "no_sync"


@pytest.mark.parametrize("kind", ["BPSK", "QPSK", "8PSK", "BPSK_NOROT"])
def test_segment_demap_retains_parity_across_pilot_or_field(kind):
    bits = np.tile([0, 1, 1, 0, 0, 0], 8)
    symbols = PSKModulator(kind, sps=1).map_symbols(bits, start_symbol_index=13)
    np.testing.assert_array_equal(
        PSKDemodulator(kind, sps=1).demap_symbols(symbols, 13), bits)


@pytest.mark.parametrize("fs,rs", [(0, 1), (1, 0), (float("nan"), 1), (15500000, 1000000)])
def test_invalid_rates_fail(fs, rs):
    with pytest.raises(ValueError):
        acquire_psk_capture(np.ones(100), np.ones(100), fs, rs, np.ones(32))


def test_invalid_shapes_or_carrier_arguments():
    with pytest.raises(ValueError):
        acquire_psk_capture(np.ones((10, 10)), np.ones(100), 16000000, 1000000, np.ones(32))
    with pytest.raises(ValueError):
        refine_psk_carrier(np.ones(10), "QPSK", 0)
    with pytest.raises(ValueError):
        refine_psk_carrier(np.ones(100), "BPSK_NOROT", 0)


@pytest.mark.parametrize("subdivisions", [0, 33, 1.5, True])
def test_invalid_fractional_grid_rejected(subdivisions):
    with pytest.raises(ValueError):
        acquire_psk_capture(np.ones(200), np.ones(200), 4000000, 1000000,
                            np.ones(32), timing_subdivisions=subdivisions)


def test_fractional_timing_uses_same_adc_samples_and_known_sync_only():
    rng = np.random.default_rng(81)
    sync = PSKModulator("QPSK", sps=1).map_symbols(rng.integers(0, 2, 64))
    symbols = np.r_[np.ones(20), sync, np.ones(120)]
    # Independent dense transmit grid, then a single native4SPS sampling operation.
    up = np.zeros(len(symbols) * 64, complex)
    up[::64] = symbols
    dense = fftconvolve(up, rrc_filter(.4, 64), "same")
    native = dense[6::16] * 1000
    x = np.r_[np.zeros(512), native, np.zeros(512)]
    q = .01 * (rng.normal(size=4096) + 1j * rng.normal(size=4096))
    original = x.copy()
    integer = acquire_psk_capture(x, q, 4000000, 1000000, sync)
    fractional = acquire_psk_capture(x, q, 4000000, 1000000, sync,
                                     timing_subdivisions=8)
    a = next(b for b in integer["bursts"] if "symbols" in b)
    b = next(b for b in fractional["bursts"] if "symbols" in b)
    assert b["sync_score"] > a["sync_score"]
    assert b["sync_evm"] < a["sync_evm"]
    assert b["sync_start_raw"] == pytest.approx(512 + 20 * 4 - 6 / 16, abs=.125)
    assert fractional["sample_hz"] == integer["sample_hz"] == 4000000
    np.testing.assert_array_equal(x, original)
    assert acquire_psk_capture(q, q, 4000000, 1000000, sync,
                               timing_subdivisions=8)["bursts"] == []


@pytest.mark.parametrize('window', [0,65,True,1.5])
def test_invalid_energy_window_rejected(window):
    with pytest.raises(ValueError):
        acquire_psk_capture(np.ones(1000),np.ones(1000),16000000,4000000,
                            np.ones(32),energy_window_symbols=window)


def test_energy_window_must_fit_both_recordings():
    with pytest.raises(ValueError):
        acquire_psk_capture(np.ones(1000),np.ones(65),16000000,4000000,
                            np.ones(32),energy_window_symbols=32)
