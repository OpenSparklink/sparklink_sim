"""Independent exact products,registered handoff and bounded score contexts."""

from collections import deque
from copy import deepcopy

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat
from nearlink_sdr.phy.banked_words import DifferenceWord
from nearlink_sdr.sim.scaled_native import (
    DifferencePipeline,
    FoldResult,
    Q12Pipeline,
    ScorePipeline,
)


def test_registered_prefix_native_products_and_global_clock_enable_preserve_lag4():
    rng = np.random.default_rng(519901)
    words = [
        (int(r), int(i), int(e))
        for r, i, e in zip(
            rng.integers(-512, 512, 93),
            rng.integers(-512, 512, 93),
            rng.integers(-48, 33, 93),
            strict=True,
        )
    ]
    words[:4] = [(-512, -512, -48), (511, -512, 32), (-512, 511, 0), (0, 0, -48)]
    source = deque((n, *word) for n, word in enumerate(words))
    pending, output = deque(), []
    node = DifferencePipeline()
    for cycle in range(2000):
        old = deepcopy((node.cells, node.iq, node.next_tag))
        full = len(pending) == 3 and node.cells[-1] is not None
        node.step(source, pending, 3)
        if full:
            assert (node.cells, node.iq, node.next_tag) == old
        if cycle % 7 in (0, 1) and pending:
            output.append(pending.popleft())
        if not source and not pending and node.idle:
            break
    assert len(output) == len(words)
    for n, word in output:
        r, i, e = words[n]
        pr, pi, pe = words[n - 4] if n >= 4 else (0, 0, 0)
        a, b = r * pr + i * pi, i * pr - r * pi
        assert word == DifferenceWord(a, b, a * a + b * b, e + pe)
    assert node.stats["native_products"] == 8 * len(words)
    assert node.stats["stalled_edges"] > 0
    assert node.estimate()["DSP"] == 8


@pytest.mark.parametrize("terms,coefficient", [(31, 62), (61, 61), (125, 125)])
def test_native_wide_score_and_restoring_rank_equal_independent_integer_ratio(terms, coefficient):
    widths = FixedBankFormat(10).widths(terms, coefficient, 0.6)
    c, e = widths["correlation"], widths["window_energy"]
    sources = [
        FoldResult(0, 0, 0, 0, 0),
        FoldResult(1, 1, -1, 0, 1),
        FoldResult(2, -(1 << (c - 1)), (1 << (c - 1)) - 1, 1, 0),
        FoldResult(3, 31, -27, 113, 0),
        FoldResult(4, 73, 91, (1 << e) - 1, 2),
        FoldResult(5, 14, 0, 196, 0),
    ]
    score, rank = ScorePipeline(widths, coefficient), Q12Pipeline(widths, terms, coefficient)
    incoming, output = deque(), []
    for cycle in range(300):
        if cycle % 8 == 0 and cycle // 8 < len(sources):
            incoming.append(sources[cycle // 8])
        value = rank.step()
        if value is not None:
            output.append(value)
        score.step(incoming, rank)
        if cycle > 8 * len(sources) and score.idle and rank.idle:
            break
    assert len(output) == len(sources)
    for source, actual in zip(sources, output, strict=True):
        n = source.real * source.real + source.imag * source.imag
        denominator = source.energy * coefficient
        expected = 0 if denominator == 0 else min(4096, (n << 12) // denominator)
        assert actual["source"] == source
        assert actual["numerator"] == n
        assert actual["rank"] == expected
        assert actual["hit"] == bool(
            source.energy and (n << 24) >= (source.energy * widths["threshold_integer"])
        )
    assert score.native.stats["products"] == 8 * len(sources)
    assert score.estimate()["DSP"] == 4 and rank.estimate()["DSP"] == 0


def test_registered_normalizer_cannot_be_overwritten_and_preserves_latency():
    widths = FixedBankFormat(10).widths(31, 62, 0.6)
    source = FoldResult(0, 1, 0, 1, 0)
    rank = Q12Pipeline(widths, 31, 62)
    assert rank.offer(source, 1, 62, False)
    assert not rank.offer(source, 1, 62, False)
    for _ in range(12):
        assert rank.step() is None
    result = rank.step()
    assert result["rank"] == (4096 // 62)
    assert rank.idle
