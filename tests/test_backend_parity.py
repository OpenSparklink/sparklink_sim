"""Both DSP implementations must execute when the optional extension is available."""

import importlib
import importlib.util

import numpy as np
import pytest

if importlib.util.find_spec("nearlink_sdr_accel") is None:
    pytest.skip("optional DSP Rust extension is not built", allow_module_level=True)
importlib.import_module("nearlink_sdr_accel")  # Installed-but-broken extensions must fail.

from nearlink_sdr.common import crc, polar  # noqa: E402


@pytest.mark.parametrize(
    "n,k",
    [(64, 32), (64, 48), (128, 72), (256, 156), (512, 320), (1024, 768), (1024, 1), (1024, 1023)],
)
def test_rust_python_encoder_and_sc_match_with_variable_reliability(n, k, monkeypatch):
    assert polar._HAS_RUST_ACCEL, "A Rust-enabled test must not silently exercise fallback"
    with monkeypatch.context() as context:
        context.setattr(polar, "_HAS_RUST_ACCEL", False)
        python_encoder = polar.PolarEncoder(n, k)
        python_decoder = polar.PolarDecoder(n, k)
    rust_encoder = polar.PolarEncoder(n, k)
    rust_decoder = polar.PolarDecoder(n, k)
    assert python_encoder._rust_encoder is None and python_decoder._rust_decoder is None
    assert rust_encoder._rust_encoder is not None and rust_decoder._rust_decoder is not None
    rng = np.random.default_rng(71016 + n + k)
    for _ in range(8):
        bits = rng.integers(0, 2, k, dtype=np.int8)
        expected = python_encoder.encode(bits)
        np.testing.assert_array_equal(rust_encoder.encode(bits), expected)
        llr = (1 - 2 * expected.astype(float)) * rng.uniform(0.01, 5, n) + rng.normal(0, 1, n)
        np.testing.assert_array_equal(rust_decoder.decode(llr), python_decoder.decode(llr))
    np.testing.assert_array_equal(
        rust_decoder.decode(np.zeros(n)), python_decoder.decode(np.zeros(n))
    )


@pytest.mark.parametrize(
    "poly,width,seed",
    [
        (crc.CRC12_POLY, 12, 0xAAD),
        (crc.CRC24A_POLY, 24, 0x555555),
        (crc.CRC24B_POLY, 24, 0x555555),
        (crc.CRC32_POLY, 32, 0x12345678),
    ],
)
def test_rust_python_seeded_crc_match(poly, width, seed, monkeypatch):
    assert crc._HAS_RUST_CRC
    rng = np.random.default_rng(71017 + width)
    for length in (0, 1, 40, 255 * 8):
        bits = rng.integers(0, 2, length, dtype=np.int8)
        with monkeypatch.context() as context:
            context.setattr(crc, "_HAS_RUST_CRC", False)
            expected = crc.crc_calculate(bits, poly, width, seed)
        np.testing.assert_array_equal(crc.crc_calculate(bits, poly, width, seed), expected)
