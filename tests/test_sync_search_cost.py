"""Direct short correlations must preserve the previous SciPy-selected answer exactly."""

import numpy as np
import pytest
from scipy.signal import correlate as scipy_correlate

from nearlink_sdr.phy import psk_sync


@pytest.mark.parametrize("length", [32, 128])
@pytest.mark.parametrize("subdivisions", [1, 8])
def test_direct_correlation_preserves_search_answer(monkeypatch, length, subdivisions):
    rng = np.random.default_rng(184016)
    z = rng.normal(size=(length + 32) * 4) + 1j * rng.normal(size=(length + 32) * 4)
    reference = np.exp(1j * np.pi / 2 * rng.integers(0, 4, length))
    measured, prior = {}, {}
    actual = psk_sync.search_psk_sync(
        z, 4, reference, subdivisions, min_start=17, diagnostics=measured
    )
    monkeypatch.setattr(psk_sync, "correlate", scipy_correlate)
    expected = psk_sync.search_psk_sync(
        z, 4, reference, subdivisions, min_start=17, diagnostics=prior
    )
    assert measured == prior
    for key in ("score", "delta", "phase", "offset"):
        assert actual[key] == expected[key]
    np.testing.assert_array_equal(actual["phase_symbols"], expected["phase_symbols"])
