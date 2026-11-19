"""Independent integer projections; no floating tie/CRC/expected-data selection."""

import itertools

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import pack_normalized_word
from nearlink_sdr.phy.psk import PSKDemodulator
from nearlink_sdr.phy.psk_hard_word import hard_psk_word


def projection_reference(real, imag, kind, index):
    """Enumerate all labeled constellation directions using exact integer dots."""
    if kind == "BPSK":
        directions = [(1, 0), (-1, 0)] if index & 1 else [(0, 1), (0, -1)]
    else:
        directions = (
            [(1, 0), (0, 1), (0, -1), (-1, 0)]
            if index & 1
            else [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        )
    scores = [real * a + imag * b for a, b in directions]
    maximum = max(scores)
    winners = [k for k, value in enumerate(scores) if value == maximum]
    label = winners[0]
    width = 1 if kind == "BPSK" else 2
    bits = tuple((label >> bit) & 1 for bit in range(width - 1, -1, -1))
    return label, bits, len(winners) > 1


@pytest.mark.parametrize("kind", ["BPSK", "QPSK"])
@pytest.mark.parametrize("index", [0, 1, 73, 131070, 131071])
def test_full_signed_domain_and_exponent_independence(kind, index):
    rng = np.random.default_rng(265019)
    edge = [-(1 << 21), -(1 << 21) + 1, -1, 0, 1, (1 << 21) - 1]
    pairs = list(itertools.product(edge, repeat=2))
    pairs += [tuple(map(int, row)) for row in rng.integers(-(1 << 21), 1 << 21, (97, 2))]
    for real, imag in pairs:
        expected = projection_reference(real, imag, kind, index)
        for exponent in [-1024, -20, 0, 4, 1023]:
            row = hard_psk_word(pack_normalized_word(real, imag, exponent), kind, index)
            assert (row["label"], row["bits"], row["mathematical_tie"]) == expected
            assert row["index"] == index and row["zero_word"] == (real == imag == 0)


def test_float_comparison_uses_well_scaled_nonties_and_preserves_tie_differences():
    rng = np.random.default_rng(265020)
    differences = []
    for kind, index in itertools.product(["BPSK", "QPSK"], [0, 1]):
        demap = PSKDemodulator(kind)
        for real, imag in rng.integers(-(1 << 21), 1 << 21, (200, 2)):
            real, imag = int(real), int(imag)
            row = hard_psk_word(pack_normalized_word(real, imag, -21), kind, index)
            if row["mathematical_tie"]:
                continue
            sample = complex(real, imag) * 2**-21
            # Explicit start index includes the prescribed alternating rotation.
            expected = tuple(
                map(int, demap.demap_symbols(np.array([sample]), start_symbol_index=index))
            )
            assert row["bits"] == expected
        # The old NumPy Euclidean routine's ulp-sensitive ties are not bit requirements.
        for real, imag in [(0, 0), (1, 0), (0, 1), (1, 1), (-1, -1)]:
            row = hard_psk_word(pack_normalized_word(real, imag, 0), kind, index)
            actual = tuple(
                map(
                    int,
                    demap.demap_symbols(np.array([complex(real, imag)]), start_symbol_index=index),
                )
            )
            if row["bits"] != actual:
                assert row["mathematical_tie"]
                differences.append((kind, index, real, imag))
    assert differences  # Explicitly retain the mathematical-tie algorithm difference.


@pytest.mark.parametrize(
    "args",
    [
        (-1, "BPSK", 0),
        (1 << 55, "BPSK", 0),
        (True, "BPSK", 0),
        (1 << 44, "QPSK", 0),
        (0, "8PSK", 0),
        (0, None, 0),
        (0, "BPSK", -1),
        (0, "BPSK", 131072),
        (0, "BPSK", True),
    ],
)
def test_invalid_format_is_explicit(args):
    with pytest.raises((ValueError, OverflowError)):
        hard_psk_word(*args)
