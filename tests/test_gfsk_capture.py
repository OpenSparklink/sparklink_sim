"""Real-capture acquisition must not need known payload/sync alignment."""
import numpy as np
import pytest

from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.gfsk import GFSKModulator, analyze_gfsk_capture


@pytest.mark.parametrize("sps,cfo", [(8, 73000), (16, -73000)])
def test_blind_acquisition_reuses_demodulator_and_maps_samples(sps, cfo):
    rng = np.random.default_rng(20261115)
    bits = rng.integers(0, 2, 1000)
    waveform = 50 * GFSKModulator(sps=sps).modulate(bits)
    start = 1703
    signal = np.zeros(start + len(waveform) + 1000, complex)
    signal[start:start + len(waveform)] = waveform
    fs = 1000000 * sps
    signal *= np.exp(2j * np.pi * cfo * np.arange(len(signal)) / fs)
    signal += rng.normal(size=len(signal)) + 1j * rng.normal(size=len(signal))
    quiet = rng.normal(size=10000) + 1j * rng.normal(size=10000)
    original = signal.copy()
    result = analyze_gfsk_capture(signal, quiet, fs, 1000000)
    burst, = result["bursts"]
    assert burst["status"] == "demodulated_unparsed"
    assert not burst["edge_truncated"]
    assert burst["cfo_hz"] == pytest.approx(cfo, abs=10000)
    assert result["sps"] == sps
    assert result["crc"].startswith("unknown")
    assert burst["bit0_raw_reference"] == burst["bit0_filtered_index"] - 32
    # Compare at the known injected sample coordinates; no search feeds acquisition.
    raw_centers = burst["bit0_raw_reference"] + (np.arange(len(burst["bits"])) + 0.5) * sps
    indexes = np.floor((raw_centers - start) / sps).astype(int)
    valid = (indexes >= 5) & (indexes < len(bits) - 5)
    assert np.mean(burst["bits"][valid] != bits[indexes[valid]]) < 0.01
    np.testing.assert_array_equal(signal, original)


def test_quiet_and_short_and_truncated_segments_are_not_fake_packets():
    rng = np.random.default_rng(1)
    quiet = rng.normal(size=20000) + 1j * rng.normal(size=20000)
    assert analyze_gfsk_capture(quiet, quiet, 16000000, 1000000)["bursts"] == []
    signal = np.zeros(10000, complex)
    signal[100:200] = 30
    signal[5000:] = 30 * GFSKModulator(sps=16).modulate(prbs9(313))[:5000]
    result = analyze_gfsk_capture(signal, quiet, 16000000, 1000000)
    assert result["bursts"][0]["status"] == "short_energy_segment"
    assert result["bursts"][-1]["edge_truncated"]
    assert result["crc"].startswith("unknown")


def test_single_tone_cannot_determine_both_cfo_and_fsk_deviation():
    x = np.ones(10000, complex)
    result = analyze_gfsk_capture(x, np.zeros(10000, complex), 16000000, 1000000)
    assert result["bursts"][0]["status"] == "insufficient_frequency_states"


@pytest.mark.parametrize("fs,rs", [(0, 1), (1, 0), (float("nan"), 1), (15500000, 1000000)])
def test_unsupported_rates_fail_explicitly(fs, rs):
    with pytest.raises(ValueError):
        analyze_gfsk_capture(np.ones(100), np.ones(100), fs, rs)


def test_prbs9_convention_period_and_polynomial():
    bits = prbs9(1022)
    np.testing.assert_array_equal(bits[:511], bits[511:])
    assert len({tuple(bits[i:i + 9]) for i in range(511)}) == 511
    np.testing.assert_array_equal(bits[:-9] ^ bits[4:-5] ^ bits[9:], 0)
    assert np.packbits(bits[:32], bitorder="little").tobytes().hex() == "ffc1fbe8"
    assert len(prbs9(0)) == 0
    for seed in (0, 512, -1):
        with pytest.raises(ValueError):
            prbs9(1, seed)
