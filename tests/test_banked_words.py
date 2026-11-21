"""Independent word/RNE checks and exact existing coarse-query regression."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_flow import multiply_signed_tiles, signed_tile_terms
from nearlink_sdr.phy.acquisition_maximum import RollingMaximumRankedBank
from nearlink_sdr.phy.banked_words import (
    BankedDifferenceHistory,
    BankedRankedReference,
    DifferenceWord,
    aligned_piece,
    input_word,
    split_alignment,
)
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


@pytest.mark.parametrize("width", [10, 21, 28, 30, 61])
def test_actual_native_signed_operands_reconstruct_the_wide_integer(width):
    rng = np.random.default_rng(519001 + width)
    values = [-(1 << (width - 1)), (1 << (width - 1)) - 1, -1, 0, 1]
    values += [int(v) for v in rng.integers(-(1 << (width - 1)), 1 << (width - 1), 64)]
    for a, b in zip(values, reversed(values), strict=True):
        terms = signed_tile_terms(a, b, width, width)
        assert all(-(1 << 24) <= x < 1 << 24 and -(1 << 17) <= y < 1 << 17 for x, y, _ in terms)
        assert sum(x * y << s for x, y, s in terms) == a * b
        assert multiply_signed_tiles(a, b, width, width) == a * b
    assert len(signed_tile_terms(1, 1, 21, 21)) == 2
    assert len(signed_tile_terms(1, 1, 30, 30)) == 4


@pytest.mark.parametrize("shift", [0, 1, 19, 20, 21, 42, 43, 59, 60, 160, 320])
def test_scalar_floor_and_rounding_match_independent_fraction(shift):
    values = [-(1 << 58) + 1, -(1 << 20), -5, -3, -1, 0, 1, 3, 5, (1 << 58) - 1]
    for value in values:
        floor, increment = split_alignment(value, shift)
        assert increment in (0, 1)
        assert floor + increment == round(Fraction(value, 1 << shift))
    piece = aligned_piece(DifferenceWord(-3, 5, 34, -11), min(64, shift - 11), -1, 1)
    r = round(Fraction(-3, 1 << min(shift, 75)))
    i = round(Fraction(5, 1 << min(shift, 75)))
    p = round(Fraction(34, 1 << (2 * min(shift, 75))))
    assert (piece[0] + piece[3], piece[1] + piece[4], piece[2] + piece[5]) == (
        -r + i,
        -i - r,
        p,
    )


@pytest.mark.parametrize("terms", [31, 61, 125])
@pytest.mark.parametrize("lanes", [8, 16])
def test_every_wrapped_batch_reads_one_word_per_bank_without_clearing_RAM(terms, lanes):
    memory = BankedDifferenceHistory(terms, lanes)
    records = []
    for n in range(4 * memory.pitch + 179):
        word = DifferenceWord(n, -n, n * n * 2, (n % 161) - 96)
        memory.append(n, word)
        records.append(word)
        for first in range(0, terms, lanes):
            count = min(lanes, terms - first)
            result, addresses = memory.read_batch(n, first, count)
            assert len({a[0] for a in addresses if a is not None}) == sum(
                a is not None for a in addresses
            )
            for j, actual in enumerate(result, first):
                source = n - 4 * (terms - 1 - j)
                assert actual == (records[source] if source >= 0 else DifferenceWord(0, 0, 0, 0))
    with pytest.raises(ValueError, match="expired"):
        memory.read_batch(0, terms - 1, 1)
    # A negative startup address is a known zero independent of overwritten RAM.
    assert memory.read_batch(0, 0, 1)[0] == [DifferenceWord(0, 0, 0, 0)]
    assert memory.estimate()["runtime_RAM_clear_words"] == 0


@pytest.mark.parametrize("frame", [2, 3, 4])
@pytest.mark.parametrize("lanes", [8, 16])
def test_banked_functional_scores_and_normalized_phase_ties_equal_existing_bank(frame, lanes):
    cfg = TxConfig(
        frame_type=frame,
        mcs_index=8,
        symbol_rate_mhz=4,
        pilot_interval=0,
        ctrl_bits_len=28 if frame == 2 else 27,
        head_llid=None if frame == 2 else 0x351709,
    )
    ref = PSKReceiveProfile.ordinary(cfg).sync_symbols()
    window = {2: 256, 3: 376, 4: 632}[frame]
    old = RollingMaximumRankedBank(ref, 4, window, origin=-40)
    new = BankedRankedReference(ref, 4, window, origin=-40, lanes=lanes)
    rng = np.random.default_rng(519183 + frame)
    r, i = [rng.integers(-512, 512, 821) for _ in range(2)]
    e = rng.integers(-48, 33, len(r))
    r[57:161] = i[57:161] = 0
    for pos in range(0, len(r), 19):
        args = [v[pos : pos + 19] for v in (r, i, e)]
        assert old.advance_words(*args, inspect=True) == new.advance_words(*args, inspect=True)
        hi = -40 + old.samples - old.k * 4 - 1
        assert old.query_best(hi - 130, hi) == new.query_best(hi - 130, hi)
        assert old.query_peaks(hi - 130, hi, count=2) == new.query_peaks(hi - 130, hi, count=2)
        assert old.stats == new.stats
    assert new.emulator_state_bytes > new.banked_history.estimate()["logical_data_bits"] // 8
    assert not any(hasattr(new, name) for name in ("_difference_real", "_powers"))
    assert new.worst_operations_per_sample()["outgoing_cache_bits"] == 292


def test_invalid_formats_expired_history_and_stale_cache_cannot_continue_silently():
    for args in [(512, 0, 0), (0, -513, 0), (0, 0, -49), (True, 0, 0), (0.1, 0, 0)]:
        with pytest.raises(ValueError):
            input_word(*args)
    for args in [(1 << 20, 0, 0, 0), (0, 0, 1 << 43, 0), (0, 0, 0, 65)]:
        with pytest.raises(ValueError):
            DifferenceWord(*args)
    with pytest.raises(ValueError):
        BankedDifferenceHistory(32, 8)
    with pytest.raises(ValueError):
        split_alignment(0, 321)
    with pytest.raises(ValueError):
        split_alignment(1 << 59, 0)
    with pytest.raises(ValueError):
        aligned_piece(None, 0, 1, 0)
    ref = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4)).sync_symbols()
    new = BankedRankedReference(ref, 4, 256)
    before = new.emulator_state_bytes
    new.advance_words([1] * 4, [0] * 4, [0] * 4)
    assert new.emulator_state_bytes > before
    new.outgoing[0] = None
    with pytest.raises(ValueError, match="cache"):
        new.advance_words([1], [0], [0])
    with pytest.raises(ValueError, match="epoch"):
        new.advance_words([1], [0], [0])


@pytest.mark.parametrize(
    "arguments", [(1, 1, 1), ([[1]], [[1]], [[1]]), ([], [], []), ([1], [1, 2], [0])]
)
def test_malformed_word_port_is_rejected_before_any_state_update(arguments):
    ref = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4)).sync_symbols()
    node = BankedRankedReference(ref, 4, 256)
    before = node.stats.copy()
    with pytest.raises(ValueError, match="equally sized"):
        node.advance_words(*arguments)
    assert node.samples == 0 and node.stats == before
    assert len(node.advance_words([1], [0], [0], inspect=True)) == 0
