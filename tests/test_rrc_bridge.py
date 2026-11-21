"""Independent Fraction rounding and existing adapter/receiver compatibility."""

import math
import random
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, quantize_scaled_words
from nearlink_sdr.phy.rrc_bridge import rrc_to_scaled_word, rrc_to_scaled_words
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC, PairProductRRC
from nearlink_sdr.phy.rrc_word import FixedRRC, FixedRRCFormat
from nearlink_sdr.sim.capture_fine import pack_filter_word


def oracle(r, i, exponent, bits):
    values = [Fraction(v) * Fraction(2) ** exponent for v in (r, i)]
    peak = max(map(abs, values))
    native = math.frexp(float(peak))[1] - (bits - 2)
    target = min(32, max(-48, native))
    rounded = [round(v / Fraction(2) ** target) for v in values]
    limit = 1 << (bits - 1)
    clipped = [min(limit - 1, max(-limit, v)) for v in rounded]
    return (*clipped, target), dict(
        exponent_underflows=int(peak > 0 and native < -48),
        exponent_overflows=int(native > 32),
        input_saturations=sum(not -limit <= v < limit for v in rounded),
        components_erased=sum(v != 0 and q == 0 for v, q in zip(values, clipped, strict=True)),
    )


@pytest.mark.parametrize("bits", range(8, 15))
def test_fraction_oracle_and_old_float_adapter_words_counters_and_dtypes(bits):
    rng = random.Random(372715 + bits)
    rows = [(0, 0, e) for e in (-96, -48, 0, 32, 48)]
    for k in range(19):
        for sign in (-1, 1):
            v = min(2**18 - 1, (1 << k) - 1) * sign
            rows += [(v, -v, e) for e in (-96, -49, -48, -1, 0, 31, 32, 48)]
    rows += [(-(2**18), 2**18 - 1, e) for e in range(-96, 49)]
    rows += [
        (rng.randrange(-(2**18), 2**18), rng.randrange(-(2**18), 2**18), rng.randrange(-96, 49))
        for _ in range(300)
    ]
    fmt, actual_stats, old_stats = ScaledBankFormat(bits), {}, {}
    for pos in range(0, len(rows), 32):
        group = rows[pos : pos + 32]
        r, i, e = [np.array(x, np.int64) for x in zip(*group, strict=True)]
        before = [a.tobytes() for a in (r, i, e)]
        x = np.ldexp(r.astype(float), e) + 1j * np.ldexp(i.astype(float), e)
        old = quantize_scaled_words(x, fmt, stats=old_stats)
        new = rrc_to_scaled_words(r, i, e, fmt, stats=actual_stats)
        for a, b in zip(new, old, strict=True):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
        for row in group:
            assert rrc_to_scaled_word(pack_filter_word(*row), fmt) == oracle(*row, bits)
        assert before == [a.tobytes() for a in (r, i, e)]
    assert actual_stats == old_stats
    assert actual_stats["exponent_underflows"] and actual_stats["exponent_overflows"]
    assert actual_stats["input_saturations"]


@pytest.mark.parametrize(
    "word,fmt",
    [
        (True, ScaledBankFormat()),
        (2**46, ScaledBankFormat()),
        (pack_filter_word(1, 0, -97), ScaledBankFormat()),
        (pack_filter_word(1, 0, 49), ScaledBankFormat()),
        (0, None),
    ],
)
def test_invalid_word_or_format_is_not_silent_success(word, fmt):
    with pytest.raises(ValueError):
        rrc_to_scaled_word(word, fmt)


@pytest.mark.parametrize(
    "arrays",
    [
        ([], [], []),
        ([0] * 33, [0] * 33, [0] * 33),
        ([1], [1, 2], [0]),
        ([1.0], [0], [0]),
        ([True], [False], [0]),
        ([[1]], [[0]], [[0]]),
        ([1, 2**18], [0, 0], [0, 0]),
        ([1, 0], [0, 0], [0, 49]),
    ],
)
def test_bounded_port_validates_entire_batch_before_any_counter_update(arrays):
    stats = dict(exponent_underflows=11, input_saturations=23)
    before = stats.copy()
    with pytest.raises(ValueError):
        rrc_to_scaled_words(*arrays, ScaledBankFormat(), stats=stats)
    assert stats == before


def test_counter_overflow_and_invalid_counter_type_are_atomic():
    for stats in [dict(input_saturations=2**64 - 1), dict(exponent_overflows=True)]:
        before = stats.copy()
        with pytest.raises((OverflowError, ValueError)):
            rrc_to_scaled_words([2**18 - 1], [2**18 - 1], [48], ScaledBankFormat(), stats=stats)
        assert stats == before
    with pytest.raises(ValueError):
        rrc_to_scaled_words([0], [0], [0], ScaledBankFormat(), stats=None)


@pytest.mark.parametrize("kind", [FixedRRC, PairProductRRC, FullMantissaRRC])
def test_bounded_actual_RRC_output_words_keep_history_values_and_existing_feed(kind):
    rng = np.random.default_rng(372716)
    x = (rng.normal(size=137) + 1j * rng.normal(size=137)) * np.r_[
        np.full(41, 1e-8), np.full(41, 1e3), np.ones(55)
    ]
    old, new = [kind(FixedRRCFormat(12, 18)) for _ in range(2)]
    expected = old.feed(x)
    groups = [new.feed_output_words(x[k : k + 17]) for k in range(0, len(x), 17)]
    r, i, e = [np.concatenate([v[k] for v in groups]) for k in range(3)]
    actual = np.ldexp(r.astype(float), e) + 1j * np.ldexp(i.astype(float), e)
    np.testing.assert_array_equal(actual, expected)
    assert old.stats == new.stats
    for key in ("_r", "_i", "_e"):
        np.testing.assert_array_equal(getattr(old, key), getattr(new, key))
    for bad in ([], np.zeros(33), [[0]], [np.nan]):
        before = new.stats.copy()
        with pytest.raises(ValueError):
            new.feed_output_words(bad)
        assert new.stats == before
    np.testing.assert_array_equal(new.finish(), old.finish())
    assert new.stats == old.stats
    with pytest.raises(RuntimeError, match="after finish"):
        new.feed_output_words([1])
