"""Search geometry, normalized ranking and future gates without CRC selection."""

import numpy as np
import pytest
from scipy.signal import lfilter

from nearlink_sdr.phy.coarse_rank import RankedCoarseBank
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.sync_candidates import search_candidate_grid
from tests.test_stream_receive import equal_events
from tests.test_strict_receive import iq_record, transmitted
from tests.test_sync_refine import profile


@pytest.mark.parametrize("peaks", [1, 2])
@pytest.mark.parametrize("offset", [.031, .277, .619, .968])
def test_local_grid_keeps_reference_spline_and_actual_crc(peaks, offset):
    p = profile()
    wire, wanted = transmitted(p)
    x, q, truth, _ = iq_record(wire, offset=offset, ppm=100, cfo=30000)
    opts = dict(quiet_batching=False, correlation_backend="scaled",
                sync_search_method="bounded-grid", sync_candidate_count=peaks)
    for ranked in [False, True]:
        whole = PSKStreamReceiver(p, q, normalized_coarse=ranked, **opts)
        blocks = PSKStreamReceiver(p, q, normalized_coarse=ranked, **opts)
        expected = whole.feed(x) + whole.finish()
        actual = []
        for start in range(0, len(x), 257):
            actual += blocks.feed(x[start:start+257])
        actual += blocks.finish()
        equal_events(expected, actual)
        assert len(actual) == 1 and actual[0]["frame"]["payload"] == wanted
        assert actual[0]["frame"]["status"] == "valid"
        assert abs(actual[0]["sync_start_raw"]-truth) <= .125
        assert whole.capture_stats["scored_positions"] <= 9*peaks*whole.capture_stats["full_searches"]


def test_fine_score_and_tie_order_are_the_existing_full_grid_score():
    ref = profile().sync_symbols()
    rng = np.random.default_rng(550121)
    z = rng.normal(size=256) + 1j*rng.normal(size=256)
    # Prescribed candidate set does not use the reference winner to select in
    # the receiver; this is only a local score equality counterexample.
    full = search_psk_sync(z, 4, ref, 8, min_start=1)
    start = full["phase"]+full["offset"]*4
    d = {}
    local = search_candidate_grid(z, 4, ref, 8, min_start=1,
                                  coarse_starts=[round(start)], diagnostics=d)
    assert local["phase"] == start
    assert abs(local["score"]-full["score"]) < 1e-14
    assert abs(local["delta"]-full["delta"]) < 1e-14
    assert d["scored_positions"] <= 9


def test_stronger_pending_peak_is_never_replaced_by_weaker_ready_peak():
    p = profile()
    wire, _ = transmitted(p)
    x, _, truth, _ = iq_record(wire, offset=.4)
    filtered = lfilter(rrc_filter(.4, 4), [1.], x)
    first = int(truth)+40-115
    z = filtered[first:first+256]
    found = search_candidate_grid(z, 4, p.sync_symbols(), 8, coarse_starts=[115])
    assert found["phase"] > 115
    d = {}
    pending = search_candidate_grid(z, 4, p.sync_symbols(), 8, coarse_starts=[115, 40],
                                    candidate_count=2, lookahead=16, diagnostics=d)
    assert pending is None and d["disposition"] == "maximum_needs_future"
    assert d["best_score"] == found["score"]


def test_shared_q12_ring_top_two_match_independent_greedy_normalized_order():
    ref = profile().sync_symbols()
    rng = np.random.default_rng(550122)
    x = np.repeat(np.tile(ref, 10), 4) + .15*(rng.normal(size=1280)+1j*rng.normal(size=1280))
    bank = RankedCoarseBank(ref, 4, 256)
    points = {}
    for start in range(0, len(x), 32):
        points.update({p.start: p for p in bank.advance(x[start:start+32], inspect=True)})
        lower, upper = max(0, bank.samples-255), bank.samples-125
        enabled, count, actual = bank.query_peaks(lower, upper, count=2, phase_origin=0)
        if not enabled:
            assert actual == []
            continue
        candidates = [p for phase in range(4) for p in range(lower, upper+1)
                      if p % 4 == phase]
        def key(p):
            return min(4096, int(points[p].numerator)*4096 // (
                int(points[p].energy)*bank.coefficient_energy)) if points[p].energy else 0
        candidates.sort(key=key, reverse=True)
        expected = []
        for position in candidates:
            if key(position) and all(abs(position-p) >= 4 for p in expected):
                expected.append(position)
            if len(expected) == 2:
                break
        assert actual == expected
        assert len(actual) <= 2 and count <= 131
        assert bank._rank_scores.size == 256
    with pytest.raises(ValueError):
        bank.query_peaks(0, 100, count=2)
    with pytest.raises(ValueError):
        bank.query_peaks(1025, 1155, count=3)
    with pytest.raises(ValueError):
        bank.query_peaks(1025, 1155, phase_origin=.5)


@pytest.mark.parametrize("kwargs", [dict(candidate_count=0), dict(candidate_count=True),
    dict(coarse_starts=[1, 2], candidate_count=2), dict(coarse_starts=[float("nan")]),
    dict(coarse_starts=[1.1]), dict(coarse_starts=[1, 5])])
def test_invalid_or_unbounded_candidates_are_rejected(kwargs):
    with pytest.raises(ValueError):
        search_candidate_grid(np.zeros(256), 4, profile().sync_symbols(), 8, **kwargs)


def test_zero_nonfinite_and_range_contracts():
    ref = profile().sync_symbols()
    assert search_candidate_grid(np.zeros(256), 4, ref, 8) is None
    assert search_candidate_grid(np.ones(256), 4, ref, 8, min_start=1000) is None
    for z, sps, lookahead in [(np.zeros(641), 4, 0), (np.ones(256), 8, 0),
                              (np.ones(256)*np.nan, 4, 0), (np.ones(256), 4, np.nan)]:
        with pytest.raises(ValueError):
            search_candidate_grid(z, sps, ref, 8, lookahead=lookahead)
    with pytest.raises(OverflowError):
        search_candidate_grid(np.full(256, 1e100), 4, ref, 8)
    for options in [dict(sync_candidate_count=True), dict(sync_candidate_count=2),
                    dict(sync_search_method="bounded-grid")]:
        with pytest.raises(ValueError):
            PSKStreamReceiver(profile(), np.zeros(512), quiet_batching=False, **options)


def test_zero_q12_ring_does_not_invent_peaks():
    bank = RankedCoarseBank(profile().sync_symbols(), 4, 256)
    for _ in range(8):
        bank.advance(np.zeros(32))
    assert bank.query_peaks(1, 131, count=2) == (False, 131, [])


def test_direct_mac_scenario_keeps_drops_and_unverified_costs(tmp_path, monkeypatch):
    import importlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("model_sync_candidate_work")
    assert module.run(SimpleNamespace(output=tmp_path / "cost")) == 1
    report = json.loads((tmp_path / "cost/model.json").read_text())
    assert not report["whole_system_accepted"]
    r = next(r for r in report["scenarios"] if r["clock_hz"] == 200_000_000
             and r["graph"]["peaks"] == 1)
    assert not r["direct_graph_within_partial_product_budget"]
    assert r["queue"]["dropped_requests"] > 0
    assert r["queue"]["peak_owned_windows"] <= 3
    assert r["downstream_stall"]["dropped_requests"] > r["queue"]["dropped_requests"]
    assert not r["graph"]["fine_fixed_format_verified"]
    assert not r["full_chain_schedule_verified"]
    assert "not an impossibility bound" in r["graph"]["evidence"]
    with pytest.raises(ValueError):
        module.estimate(3)
