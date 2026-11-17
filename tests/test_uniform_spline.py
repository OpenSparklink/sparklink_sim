"""Independent analytic/Fraction references and unchanged search/receiver decisions."""

from fractions import Fraction
from unittest.mock import patch

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from nearlink_sdr.phy import sync_candidates
from nearlink_sdr.phy.sync_candidates import search_candidate_grid
from nearlink_sdr.phy.uniform_spline import (
    UniformCubicEighth,
    eighth_weights,
    preparation_calendar,
    spline_work,
)
from tests.test_sync_refine import profile


@pytest.mark.parametrize("n", [4, 8, 17, 64, 128, 256, 512, 640])
def test_cubic_polynomial_has_independent_known_interpolation_truth(n):
    def truth(x):
        t = x/(n-1)
        return (t**3-2*t**2+.7*t+1)+1j*(.2*t**3+.3*t**2-t)
    node = UniformCubicEighth(truth(np.arange(n)))
    times = np.arange(8*(n-1)+1)/8
    for k in range(0, len(times), 128):
        assert np.max(abs(node(times[k:k+128])-truth(times[k:k+128]))) < 2e-12


def test_integer_weight_table_is_exact_independent_fraction_algebra():
    for m in range(8):
        t = Fraction(m, 8)
        expected = (1-t, t, ((1-t)**3-(1-t))/6, (t**3-t)/6)
        assert tuple(Fraction(v, 1024) for v in eighth_weights(m)) == expected


def test_global_reference_all_supported_lengths_and_strong_weak_transition():
    rng = np.random.default_rng(620117)
    for n in range(4, 641):
        y = rng.normal(size=n)+1j*rng.normal(size=n)
        y[n//2:] *= 2**-40
        actual = UniformCubicEighth(y)
        ref = CubicSpline(np.arange(n), y)
        t = np.arange(0, (n-1)*8+1, 7)/8
        for k in range(0, len(t), 128):
            assert np.max(abs(actual(t[k:k+128])-ref(t[k:k+128]))) < 2e-12*np.max(abs(y))


def test_window_is_owned_and_readonly_endpoint_empty_and_scalar_contracts():
    y = np.arange(8, dtype=complex)
    node = UniformCubicEighth(y)
    y[:] = 0
    assert node(7) == 7 and node(0) == 0 and node(.125) == pytest.approx(.125)
    assert node([]).shape == (0,)
    assert not node.samples.flags.writeable and not node.second.flags.writeable
    assert np.array_equal(UniformCubicEighth(np.zeros(4))([0, .5, 3]), np.zeros(3))


@pytest.mark.parametrize("samples", [np.ones(3), np.ones(641), np.ones((2, 4)),
                                      [1, 2, 3, np.nan], np.full(4, np.inf)])
def test_unmodeled_or_nonfinite_windows_are_explicit(samples):
    with pytest.raises(ValueError):
        UniformCubicEighth(samples)


def test_existing_sync_range_is_not_silently_expanded():
    with pytest.raises(OverflowError):
        UniformCubicEighth(np.full(8, 1e100))


@pytest.mark.parametrize("positions", [[-1], [4], [.1], [np.nan], [np.inf], [True],
                                         [1+0j], ["1"], np.zeros((2, 2)), np.zeros(129)])
def test_queries_do_not_fetch_future_samples_or_change_phase_grid(positions):
    with pytest.raises(ValueError):
        UniformCubicEighth(np.ones(4))(positions)


@pytest.mark.parametrize("peaks", [1, 2])
def test_unchanged_search_order_scores_phase_and_future_wait(peaks):
    p = profile()
    rng = np.random.default_rng(620119)
    for n, starts, lookahead in [(256, [31, 109][:peaks], 0),
                                (256, [128, 124][:peaks], 16)]:
        y = rng.normal(size=n)+1j*rng.normal(size=n)
        d, e = {}, {}
        a = search_candidate_grid(y, 4, p.sync_symbols(), 8, coarse_starts=starts,
                                  candidate_count=peaks, diagnostics=d, lookahead=lookahead)
        b = search_candidate_grid(y, 4, p.sync_symbols(), 8, coarse_starts=starts,
            candidate_count=peaks, diagnostics=e, lookahead=lookahead,
            interpolator="uniform-eighth")
        assert (a is None) == (b is None) and d["disposition"] == e["disposition"]
        assert d["best_start"] == e["best_start"]
        assert abs(d["best_score"]-e["best_score"]) < 2e-12
        if a is not None:
            assert a["phase"] == b["phase"]
            assert np.max(abs(a["phase_symbols"]-b["phase_symbols"])) < 2e-12
        for k in ["scored_positions", "coarse_starts", "requested_peaks"]:
            assert d[k] == e[k]
    with pytest.raises(ValueError):
        search_candidate_grid(y, 4, p.sync_symbols(), 8, interpolator="unbounded")


def test_exact_zero_ties_have_same_order():
    a, b = {}, {}
    for backend, diag in [("reference", a), ("uniform-eighth", b)]:
        search_candidate_grid(np.zeros(256), 4, profile().sync_symbols(), 8,
            coarse_starts=[32], interpolator=backend, diagnostics=diag)
    assert a == b


def test_old_raw_iq_mac_success_is_preserved_under_fixed_random_chunks():
    import hashlib
    import json
    from pathlib import Path

    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tone_notch import ToneNotchConfig
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    folder = Path(__file__).parent/"data"
    path = folder/"sync_carrier_tone_success.npz"
    meta = json.loads((folder/"sync_carrier_tone_success.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["derived_npz_sha256"]
    with np.load(path) as ar:
        x, quiet, wanted = ar["iq"], ar["quiet"], ar["expected"].tobytes()
    cfg = TxConfig(mcs_index=6, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    rx = MacStreamReceiver(PSKReceiveProfile.ordinary(cfg), quiet,
        quiet_batching=False, defer_quiet_fir=False, correlation_backend="scaled",
        normalized_coarse=True, tone_notch=ToneNotchConfig(radius=31/32),
        sync_search_method="bounded-grid", sync_candidate_count=1, sync_coherent_threshold=.45)
    original = sync_candidates.search_candidate_grid

    def chosen(*args, **kwargs):
        return original(*args, **kwargs, interpolator="uniform-eighth")

    packets, offset, rng = [], 0, np.random.default_rng(620120)
    with patch.object(sync_candidates, "search_candidate_grid", chosen):
        while offset < len(x):
            n = int(rng.integers(1, 1001))
            packets += rx.feed(x[offset:offset+n]).packets
            offset += n
        packets += rx.finish().packets
    assert [p.mac_payload for p in packets] == [wanted]


def test_work_keeps_preparation_coefficients_shift_add_growth_and_unknown_formats():
    w = spline_work()
    assert w["preparation_real_constant_products"] == 1018
    assert w["fractional_real_variable_products"] == 0 and w["fractional_real_adds"] == 3200
    assert w["preparation_real_adds"] == 4092 and w["preparation_source_word_reads"] == 1280
    assert w["rom_words"] == 1276 and w["held_last_second_word"]
    assert not w["coefficient_format_verified"] and not w["work_format_verified"]
    for args in [(True, 32, 1), (256, 128, 1), (641, 32, 1), (256, 32, 3)]:
        with pytest.raises(ValueError):
            spline_work(*args)
    with pytest.raises(ValueError):
        eighth_weights(8)


@pytest.mark.parametrize("n", [4, 256, 640])
def test_finite_calendar_read_before_write_ports_and_inplace_lifetime(n):
    rows = list(preparation_calendar(n))
    assert len(rows) == 5*n+4
    initialized, backward = set(), set()
    yr = qr = qw = 0
    for phase, source, read, write in rows:
        assert len(source) <= 2 and all(0 <= j < n for j in source)
        yr += len(source)
        if read is not None:
            assert read in initialized and read not in backward
            qr += 1
        if write is not None:
            assert read != write
            initialized.add(write)
            qw += 1
            if phase in ("backward_write", "last_second_write"):
                backward.add(write)
    assert backward == set(range(n))
    assert (yr, qr, qw) == (5*n, 2*n-1, 2*n)
    assert rows[-1] == ("last_second_write", (), None, n-1)
    with pytest.raises(ValueError):
        list(preparation_calendar(3))
