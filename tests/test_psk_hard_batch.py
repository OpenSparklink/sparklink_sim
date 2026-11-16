"""Batched hard decisions retain scalar mapping, ties, wire parity and input ownership."""

import numpy as np
import pytest

from nearlink_sdr.phy.psk import PSKDemodulator


def scalar_reference(demod, symbols, start):
    bits = []
    for i, value in enumerate(symbols):
        if demod._rotation is not None and (i + start) % 2 == 1:
            value *= np.conj(demod._rotation)
        distances = np.abs(value - demod._constellation_vals)
        label = demod._constellation_keys[np.argmin(distances)]
        bits.extend((label >> b) & 1 for b in range(demod._bits_per_symbol - 1, -1, -1))
    return np.asarray(bits, dtype=int)


@pytest.mark.parametrize("kind", ["BPSK", "QPSK", "8PSK", "BPSK_NOROT"])
@pytest.mark.parametrize("start", [-1, 0, 1])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_batch_matches_scalar_at_random_values_ties_and_batch_edges(kind, start, dtype):
    demod = PSKDemodulator(kind, sps=1)
    rng = np.random.default_rng(214016)
    ref = demod._constellation_vals
    midpoints = [(a + b) / 2 for a in ref for b in ref]
    symbols = np.r_[
        rng.normal(size=2051) + 1j * rng.normal(size=2051), ref, midpoints, 0j, 1e-30j, -1e30j
    ].astype(dtype)
    symbols.setflags(write=False)
    for n in (0, 1, 127, 128, 129, len(symbols)):
        before = symbols[:n].copy()
        expected = scalar_reference(demod, symbols[:n], start)
        actual = demod.demap_symbols(symbols[:n], start)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(symbols[:n], before)
