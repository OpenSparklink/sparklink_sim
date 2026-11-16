"""Compare FIR state with SciPy across arbitrary short/long real/complex blocks."""

import numpy as np
import pytest
from scipy.signal import lfilter

from nearlink_sdr.phy.stream_rx import _fir_block


@pytest.mark.parametrize("kind", ["real", "complex", "real-input-complex-state"])
@pytest.mark.parametrize("tap_count", [1, 32, 81])
def test_fir_state_and_outputs(kind, tap_count):
    rng = np.random.default_rng(164016)
    taps = rng.normal(size=tap_count)
    taps /= sum(abs(taps))
    x = rng.normal(size=8192)
    state = rng.normal(size=tap_count - 1)
    if kind == "complex":
        x = x + 1j * rng.normal(size=len(x))
    if kind != "real":
        state = state + 1j * rng.normal(size=len(state))
    actual, reference = state.copy(), state.copy()
    offset = 0
    for n in [1, 2, 31, 32, 33, 80, 81, 4096, 128, 3708]:
        block = x[offset : offset + n]
        if not len(block):
            break
        a, actual = _fir_block(taps, block, actual)
        b, reference = lfilter(taps, [1.0], block, zi=reference)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(actual, reference)
        assert actual.shape == (tap_count - 1,)
        offset += len(block)
    assert offset == len(x)
