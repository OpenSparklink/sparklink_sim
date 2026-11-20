"""Frozen node axes from the earlier external experiment;no RF holdout claim."""

import numpy as np

from tests.test_sync_fit_word import packed


def node_vectors(split):
    if split == "development":
        rng = np.random.default_rng(2026112001)
        for k in range(700):
            e = [-128, -23, 0, 127][k % 4]
            r, i = (int(v) for v in rng.integers(-(2**21), 2**21, 2))
            current = packed(r, i, e)
            midpoint = (
                None
                if k < 32
                else (0 if k % 9 == 0 else packed(int(rng.integers(-(2**21), 2**21)), 17, e))
            )
            yield current, midpoint
    elif split == "validation":
        exponents = [127, -128, 7, -61, 0, -24, 43, -1]
        for k in range(768):
            r = [0, 1, -1, (1 << 21) - 1, -(1 << 21), 1 << 16, -31, 1 << 4][(k // 8) % 8]
            i = [0, -11, 1 << 20, -(1 << 21), 1][(k // 17) % 5]
            current = packed(r, i, exponents[k % 8])
            midpoint = (
                None
                if k < 32
                else packed(
                    [0, 1, -1, 1 << 20, -(1 << 21)][(k // 23) % 5],
                    (1 << 21) - 1 if k % 7 else 0,
                    exponents[(k // 27) % 8],
                )
            )
            yield current, midpoint
    else:
        raise ValueError("Need development or validation node axes")
