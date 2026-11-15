"""Independent Gaussian/angle references for §6.2.1.2/6.7/6.10 soft payload RX."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.psk import PSKDemodulator
from nearlink_sdr.phy.psk_fields import soft_demap_psk_field
from nearlink_sdr.phy.rx_pipeline import decode_payload
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_payload
from tests.test_psk_field_layout import independent_wire
from tests.test_strict_receive import iq_record, transmitted


@pytest.mark.parametrize("kind,bps", [("BPSK", 1), ("QPSK", 2), ("8PSK", 3)])
@pytest.mark.parametrize("start", [0, 1, 65])
@pytest.mark.parametrize("method", ["maxlog", "logmap"])
def test_soft_llr_matches_independent_gaussian_likelihood(kind, bps, start, method):
    rng = np.random.default_rng(71016)
    y = rng.normal(size=17) + 1j * rng.normal(size=17)
    variance = 0.37
    expected = []
    for i, z in enumerate(y):
        metrics = []
        for label in range(2**bps):
            bits = np.array([(label >> b) & 1 for b in range(bps - 1, -1, -1)])
            ref = independent_wire(bits, kind, start + i, 0, True)[0]
            metrics.append(-(abs(z - ref) ** 2) / variance)
        for bit in range(bps - 1, -1, -1):
            zero = [v for label, v in enumerate(metrics) if not (label >> bit) & 1]
            one = [v for label, v in enumerate(metrics) if (label >> bit) & 1]
            reduce = max if method == "maxlog" else np.logaddexp.reduce
            expected.append(np.clip(reduce(zero) - reduce(one), -64, 64))
    original = y.copy()
    actual = PSKDemodulator(kind).soft_demap_symbols(y, variance, start, method=method)
    np.testing.assert_allclose(actual, expected, atol=3e-14, rtol=2e-14)
    np.testing.assert_array_equal(y, original)


@pytest.mark.parametrize("kind,bps", [("BPSK", 1), ("QPSK", 2), ("8PSK", 3)])
@pytest.mark.parametrize("interval", [0, 4, 8, 16])
def test_pilots_and_padding_do_not_reset_soft_rotation(kind, bps, interval):
    bits = np.random.default_rng(71017).integers(0, 2, 137)
    wire = independent_wire(bits, kind, 65, interval, True)
    llr = soft_demap_psk_field(wire, kind, 65, 0.1, interval)
    np.testing.assert_array_equal((llr[: len(bits)] < 0).astype(int), bits)
    assert len(llr) == -(-len(bits) // bps) * bps


def test_soft_variance_validation_and_numerical_bounds():
    demod = PSKDemodulator("QPSK")
    for variance in (0, -1, np.nan, np.inf, [1]):
        with pytest.raises(ValueError):
            demod.soft_demap_symbols(np.ones(8), variance)
    for method in ("maxlog", "logmap"):
        llr = demod.soft_demap_symbols(np.array([1e308 + 1e308j, 0]), 1e-300, method=method)
        assert np.isfinite(llr).all() and abs(llr).max() <= 64
        np.testing.assert_array_equal(demod.soft_demap_symbols(np.array([], complex), 0.1), [])
    with pytest.raises(ValueError):
        demod.soft_demap_symbols(np.array([np.nan]), 1)
    with pytest.raises(ValueError):
        PSKReceiveProfile.ordinary(TxConfig()).__class__(TxConfig(), noise_variance=0)


@pytest.mark.parametrize("ft,mcs", [(2, 6), (2, 10), (3, 6), (4, 0)])
@pytest.mark.parametrize("whitening", [0, 0x52])
def test_soft_descrambling_and_polar_block_alignment(ft, mcs, whitening):
    cfg = TxConfig(
        frame_type=ft,
        mcs_index=mcs,
        whitening_seed=whitening,
        ctrl_bits_len=28 if ft == 2 else 27,
        head_llid=7 if ft != 2 else None,
    )
    bits = np.random.default_rng(71016).integers(0, 2, 255 * 8)
    encoded = encode_payload(bits, cfg)
    llr = (1 - 2 * encoded.astype(float)) * np.linspace(0.05, 20, len(encoded))
    actual, ok = decode_payload(encoded, cfg, len(bits), llr=llr)
    assert ok
    np.testing.assert_array_equal(actual, bits)
    with pytest.raises(ValueError):
        decode_payload(encoded, cfg, len(bits), llr=llr[:-1])
    with pytest.raises(ValueError):
        decode_payload(encoded, cfg, len(bits), llr=np.full(len(llr), np.inf))


def test_strict_coded_raw_iq_soft_metadata_and_chunk_equivalence():

    from nearlink_sdr.phy.stream_rx import PSKStreamReceiver

    profile = replace(
        PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4)), llr_method="logmap"
    )
    wire, expected = transmitted(profile)
    x, q, _, _ = iq_record(wire, ppm=100, cfo=30000)
    all_at_once = PSKStreamReceiver(profile, q)
    a = all_at_once.feed(x) + all_at_once.finish()
    chunked = PSKStreamReceiver(profile, q)
    b = []
    for pos in range(0, len(x), 37):
        b.extend(chunked.feed(x[pos : pos + 37]))
    b.extend(chunked.finish())
    assert len(a) == len(b)
    for first, second in zip(a, b, strict=True):
        assert first["frame"] == second["frame"]
        assert first["emitted_after_samples"] == second["emitted_after_samples"]
        for key, tolerance in [
            ("symbols", 1e-12),
            ("symbol_centers_raw", 1e-10),
            ("timing_rate_ppm", 1e-8),
            ("timing_detector_error", 1e-12),
        ]:
            np.testing.assert_allclose(first[key], second[key], atol=tolerance, rtol=0)
    assert len(a) == 1 and a[0]["frame"]["payload"] == expected
    assert a[0]["frame"]["soft_information"]["variance_source"].startswith("known-sync")
    uncoded = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8))
    wire, payload = transmitted(uncoded)
    old = decode_psk_frame(wire[40:], replace(uncoded, soft_payload=False))
    new = decode_psk_frame(wire[40:], uncoded)
    assert old == new and new["payload"] == payload and "soft_information" not in new
