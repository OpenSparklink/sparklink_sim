"""Independent pulse/geometry and bounded normalized ranking counterexamples."""

import numpy as np
import pytest
from scipy.interpolate import BarycentricInterpolator
from scipy.signal import lfilter

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.sync_refine import local_cubic_samples, refine_psk_sync
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_stream_receive import equal_events
from tests.test_strict_receive import iq_record, transmitted


def profile():
    return PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))


def test_four_point_interpolation_matches_independent_polynomial_including_edges():
    rng = np.random.default_rng(410120)
    z = rng.normal(size=17) + 1j * rng.normal(size=17)
    t = np.array([0, 0.03, 0.5, 1.137, 8.619, 15.98, 16.0])
    expected = []
    for value in t:
        first = int(np.clip(np.floor(value) - 1, 0, len(z) - 4))
        expected.append(
            BarycentricInterpolator(np.arange(first, first + 4), z[first : first + 4])(value)
        )
    np.testing.assert_allclose(local_cubic_samples(z, t), expected, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(local_cubic_samples(z, np.arange(len(z))), z)
    for window, times in [([0, 1, 2], [0]), (z, [-0.1]), (z, [17]), (z, [np.nan])]:
        with pytest.raises(ValueError):
            local_cubic_samples(window, times)


def test_maximum_needing_future_is_not_replaced_by_ready_center():
    p = profile()
    wire, _ = transmitted(p)
    x, _, truth, _ = iq_record(wire, offset=0.4)
    filtered = lfilter(rrc_filter(0.4, 4), [1.0], x)
    start = int(truth) + 40 - 115
    z = filtered[start : start + 256]
    diagnostic = {}
    best = refine_psk_sync(z, 4, p.sync_symbols(), 8, coarse_start=115, lookahead=0)
    assert 115 < best["phase"] < 115.5 and best["score"] > 0.99
    pending = refine_psk_sync(
        z, 4, p.sync_symbols(), 8, coarse_start=115, lookahead=16, diagnostics=diagnostic
    )
    assert pending is None and diagnostic["disposition"] == "maximum_needs_future"
    assert diagnostic["best_score"] == best["score"] and diagnostic["scored_positions"] <= 4
    with pytest.raises(ValueError):
        refine_psk_sync(z, 8, p.sync_symbols(), 8)
    with pytest.raises(ValueError):
        refine_psk_sync(z, 4, p.sync_symbols(), 8, coarse_start=np.nan)
    assert refine_psk_sync(np.zeros(256), 4, p.sync_symbols(), 8, min_start=1000) is None


@pytest.mark.parametrize("offset", [0.031, 0.277, 0.619, 0.968])
def test_local_float_and_q12_streams_use_actual_header_crc_payload_and_chunk_state(offset):
    p = profile()
    wire, payload = transmitted(p)
    x, q, truth, step = iq_record(wire, offset=offset, ppm=100, cfo=30000)
    for normalized in [False, True]:
        options = dict(
            quiet_batching=False,
            correlation_backend="scaled",
            sync_search_method="local-parabolic",
            normalized_coarse=normalized,
        )
        whole, chunked = PSKStreamReceiver(p, q, **options), PSKStreamReceiver(p, q, **options)
        expected = whole.feed(x) + whole.finish()
        actual = []
        for k in range(0, len(x), 257):
            actual += chunked.feed(x[k : k + 257])
        actual += chunked.finish()
        equal_events(expected, actual)
        (event,) = actual
        assert event["frame"]["payload"] == payload and event["frame"]["status"] == "valid"
        assert event["frame"]["header_crc"] and event["frame"]["data_crc"]
        assert abs(event["sync_start_raw"] - truth) <= 0.125
        assert (
            np.sqrt(
                np.mean(
                    (
                        event["symbol_centers_raw"]
                        - (truth + np.arange(len(event["symbols"])) * step)
                    )
                    ** 2
                )
            )
            < 0.1
        )
        assert (
            whole.capture_stats["computed_scored_positions"]
            <= 4 * whole.capture_stats["full_searches"]
        )
        assert bool(whole.capture_stats["coarse_ranking_recomputed_positions"]) == (not normalized)
    for kwargs in [
        dict(sync_search_method="bad"),
        dict(normalized_coarse=True),
        dict(sync_search_method="local-parabolic"),
        dict(
            sync_search_method="local-parabolic",
            correlation_backend="structure",
            normalized_coarse=True,
        ),
    ]:
        with pytest.raises(ValueError):
            PSKStreamReceiver(p, q, quiet_batching=False, **kwargs)


def test_nonfinite_wait_or_out_of_range_amplitude_cannot_create_a_candidate():
    ref = profile().sync_symbols()
    with pytest.raises(ValueError):
        refine_psk_sync(np.ones(256), 4, ref, 8, lookahead=np.nan)
    with pytest.raises(OverflowError):
        refine_psk_sync(np.full(256, 1e100), 4, ref, 8)


@pytest.mark.parametrize("ft", [3, 4])
def test_longer_sync_ranking_window_contract_and_actual_crc(ft):
    """Longer references are functional smoke checks, not FT2 cost extrapolation."""
    cfg = TxConfig(frame_type=ft, mcs_index=8, ctrl_bits_len=27, head_llid=7,
                   symbol_rate_mhz=4)
    p = PSKReceiveProfile.ordinary(cfg)
    wire, payload = transmitted(p)
    x, q, _, _ = iq_record(wire)
    rx = PSKStreamReceiver(p, q, quiet_batching=False, correlation_backend="scaled",
                           sync_search_method="local-parabolic", normalized_coarse=True)
    events = rx.feed(x) + rx.finish()
    assert len(events) == 1
    f = events[0]["frame"]
    assert f["header_crc"] and f["data_crc"] and f["length_ok"]
    assert f["payload"] == payload and f["mcs"] == 8
    assert rx.correlation_bank.rank_history == 256
