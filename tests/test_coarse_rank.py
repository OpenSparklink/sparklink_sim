"""Normalized integer score and bounded ranking counterexamples."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.coarse_rank import RankedCoarseBank, normalize_squared
from tests.test_sync_refine import profile


def test_serial_ratio_normalization_matches_independent_integer_division_and_range_rules():
    rng = np.random.default_rng(410121)
    for bits in [8, 12, 16]:
        for _ in range(100):
            e = int(rng.integers(0, 2**48))
            n = int(rng.integers(0, 2**57))
            expected = min(1 << bits, (n << bits) // (62 * e)) if e else 0
            assert normalize_squared(n, e, 62, bits) == expected
    assert normalize_squared(1, 0, 62) == 0
    assert normalize_squared(0, 17, 62) == 0
    assert normalize_squared(31, 1, 62) == 2048
    assert Fraction(normalize_squared(10, 7, 62), 4096) <= Fraction(10, 7 * 62)
    for args in [(-1, 1, 62), (1, -1, 62), (1, 1, 0), (1.0, 1, 62), (1, 1, 62, 7)]:
        with pytest.raises(ValueError):
            normalize_squared(*args)


def test_normalized_ring_is_bounded_and_ranking_is_normalized_not_raw_magnitude():
    rng = np.random.default_rng(410122)
    ref = profile().sync_symbols()
    x = np.tile(ref, 10).repeat(4) + 0.1 * (rng.normal(size=1280) + 1j * rng.normal(size=1280))
    bank = RankedCoarseBank(ref, 4, 256)
    points = {}
    for start in range(0, len(x), 32):
        for point in bank.advance(x[start : start + 32], inspect=True):
            points[point.start] = point
        lower, upper = max(0, bank.samples - 255), bank.samples - 125
        enabled, _, chosen = bank.query_best(lower, upper, phase_origin=0)
        if enabled:
            assert lower <= chosen <= upper
            selected = points[chosen].score_squared
            ideal = max(p.score_squared for t, p in points.items() if lower <= t <= upper)
            assert ideal - selected <= 1 / 4096 + 1e-12
    assert bank._rank_scores.size == 256 and bank.ranking_storage_bits > 256 * 13
    assert bank.stats["normalization_words"] > 1000
    with pytest.raises(ValueError):
        bank.query_best(0, 100)
    with pytest.raises(ValueError):
        bank.query_best(1025, 1155, phase_origin=0.5)
    for kwargs in [dict(rank_fractional_bits=7), dict(rank_fractional_bits=True)]:
        with pytest.raises(ValueError):
            RankedCoarseBank(ref, 4, 256, **kwargs)
    zero = RankedCoarseBank(ref, 4, 256)
    for _ in range(8):
        zero.advance(np.zeros(32))
    assert zero.query_best(1, 131) == (False, 131, None)
    assert zero.stats["normalization_zero_energy"] == 132
    assert zero.worst_operations_per_sample()["normalized_ranking"]["worst_interval_cycles"] == 12
