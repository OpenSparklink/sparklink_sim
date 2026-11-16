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


def assert_same_answer(actual, expected):
    assert (actual is None) == (expected is None)
    if actual is not None:
        for key in ("score", "delta", "phase", "offset"):
            assert actual[key] == expected[key]
        np.testing.assert_array_equal(actual["phase_symbols"], expected["phase_symbols"])


def test_reused_integer_scores_trim_the_last_raw_sample_before_fine_selection():
    rng = np.random.default_rng(244016)
    ref = np.exp(1j * rng.uniform(-np.pi, np.pi, 32))
    z = 0.001 * (rng.normal(size=256) + 1j * rng.normal(size=256))
    z[255 - 4 * np.arange(31, -1, -1)] = ref
    expected = psk_sync.search_psk_sync(z, 4, ref, 8, min_start=1)
    diagnostic = {}
    actual = psk_sync.search_psk_sync(
        z, 4, ref, 8, min_start=1, integer_screen=0.6, diagnostics=diagnostic
    )
    assert_same_answer(actual, expected)
    assert actual["phase"] != 3.0  # The perfect coarse edge needs an unavailable fine sample.
    assert diagnostic["reused_phase_hypotheses"] == 4
    assert diagnostic["computed_phase_hypotheses"] == 28


@pytest.mark.skipif(psk_sync._RustIntegerScreen is None, reason="optional native sync not built")
@pytest.mark.parametrize("length", [32, 128])
@pytest.mark.parametrize("sps", [4, 8])
@pytest.mark.parametrize("trim", [0, 1, 3])
def test_native_gate_borderline_uses_exact_reference_with_original_work_ceiling(length, sps, trim):
    rng = np.random.default_rng(245016)
    z = rng.normal(size=(length + 32) * sps - trim) + 1j * rng.normal(
        size=(length + 32) * sps - trim
    )
    ref = np.exp(1j * rng.uniform(-np.pi, np.pi, length))
    coarse = psk_sync.search_psk_sync(z, sps, ref, 1, min_start=17.1)
    # Deliberately place the gate on the independently computed NumPy score;
    # frozen receiver parameter .6 is never adjusted by this kernel unit test.
    threshold = coarse["score"]
    expected = psk_sync.search_psk_sync(
        z, sps, ref, 8, min_start=17.1, integer_screen=threshold, integer_backend="python"
    )
    assert_same_answer(expected, psk_sync.search_psk_sync(z, sps, ref, 8, min_start=17.1))
    diagnostic = {}
    actual = psk_sync.search_psk_sync(
        z,
        sps,
        ref,
        8,
        min_start=17.1,
        integer_screen=threshold,
        integer_backend="rust",
        diagnostics=diagnostic,
    )
    assert_same_answer(actual, expected)
    assert diagnostic["integer_reference_fallbacks"] == 1
    assert diagnostic["integer_evaluations"] == 2 * sps
    assert diagnostic["integer_evaluations"] + diagnostic["computed_phase_hypotheses"] <= sps * 9
    value, phases, positions = psk_sync._RustIntegerScreen(z, sps, ref, 17.1)
    reference = {}
    psk_sync.search_psk_sync(z, sps, ref, 1, min_start=17.1, diagnostics=reference)
    assert abs(value - coarse["score"]) < 1e-12
    assert phases == reference["phase_hypotheses"]
    assert positions == reference["scored_positions"]


def test_missing_native_screen_is_explicit(monkeypatch):
    monkeypatch.setattr(psk_sync, "_RustIntegerScreen", None)
    with pytest.raises(RuntimeError, match="not installed"):
        psk_sync.search_psk_sync(
            np.ones(256, complex), 4, np.ones(32, complex), 8, integer_backend="rust"
        )
