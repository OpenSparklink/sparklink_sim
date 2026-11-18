"""Independent enumeration,causal wait semantics and unchanged saturation guard."""

import json
from pathlib import Path

import pytest

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_transport import (
    CaptureRequest,
    CaptureTransport,
    CaptureTransportAssumptions,
    FineService,
)
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SplineScoreFlow,
    SyncSignTemplate,
    _geometry,
    spline_score_cost,
)


def enumerate_geometry(peaks, n, count, minimum, lookahead):
    ticks = [
        8 * p + d
        for p in peaks
        for d in range(-4, 5)
        if 8 * p + d >= minimum and 8 * p + d + 32 * (count - 1) < 8 * (n - 1)
    ]
    ready = [t for t in ticks if t + 32 * (count - 1) + lookahead <= 8 * (n - 1)]
    return ticks, ready


def execute_geometry(peaks, n, count, minimum, lookahead):
    events = []
    program = _geometry(
        peaks,
        n,
        count,
        minimum,
        lookahead,
        lambda event, **kw: events.append(dict(event=event, **kw)),
    )
    command = next(program)
    operations = 0
    while True:
        operations += 1
        a, b = command["a"], command["b"]
        code = command["code"]
        assert code in ("sub", "add", "mov", "cmp") and command["bits"] <= 17
        value = {"sub": a - b, "add": a + b, "mov": a, "cmp": (a > b) - (a < b)}[code]
        lo, hi = (
            (-(2 ** (command["bits"] - 1)), 2 ** (command["bits"] - 1))
            if command["signed"]
            else (0, 2 ** command["bits"])
        )
        assert lo <= value < hi
        try:
            command = program.send(value)
        except StopIteration as done:
            return done.value, events, operations


def test_finite_intervals_equal_independent_grid_enumeration_and_all_boundaries():
    cases = []
    for n in (4, 8, 17, 32, 64, 256, 376, 640):
        for count in (2, 4, 32, 62, 126, 128):
            for peaks in ((), (0,), (2,), (0, 4), (630, 639)):
                for minimum in (0, 1, 8 * n):
                    for lookahead in (0, 1, 128, 32768):
                        cases.append((peaks, n, count, minimum, lookahead))
    for args in cases:
        ticks, ready = enumerate_geometry(*args)
        (positions, deferred), events, operations = execute_geometry(*args)
        assert positions == len(ticks) and deferred == (bool(ticks) and not ready)
        assert operations <= 32
        for interval in (e for e in events if e["event"] == "geometry_interval"):
            selected = [
                t for t in ticks if 8 * interval["peak"] - 4 <= t <= 8 * interval["peak"] + 4
            ]
            assert selected == list(range(interval["lower"], interval["upper"] + 1))


@pytest.mark.parametrize("power", [False, True])
def test_all_future_returns_no_score_tick_or_numeric_pass_but_runs_preparation(power):
    tpl = SyncSignTemplate.from_symbols(
        PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4)).sync_symbols()
    )
    words = [(1000, 0, 0)] * 256
    kw = dict(lookahead_ticks=128)
    old = SplineScoreFlow(SplineScoreAssumptions(256, reuse_symbol_power=power)).run(
        words, tpl, [128], **kw
    )
    events = []
    new = SplineScoreFlow(
        SplineScoreAssumptions(256, reuse_symbol_power=power, defer_all_future=True)
    ).run(words, tpl, [128], sink=events.append, **kw)
    assert old["best"]["disposition"] == "maximum_needs_future"
    assert new["geometry_wait"] and new["best"] is None and new["numerical_pass"] is None
    assert new["reason"] == "all_valid_positions_need_future"
    assert (
        not new["resource_drop"] and not new["numerical_rejection"] and not new["scoring_evaluated"]
    )
    assert (
        new["geometry_positions"] == 9
        and new["scores_evaluated"] == 0
        and new["geometry_cycles"] == 18
    )
    assert new["preparation_arithmetic"] == old["preparation_arithmetic"]
    assert new["preparation_cycles"] == old["preparation_cycles"]
    assert new["prepared_source_words"] == old["prepared_source_words"]
    assert new["cycles"] < old["cycles"] and new["scoring_reads"] == 0
    assert all(e["event"] not in ("sample", "score", "selected_sample") for e in events)


def test_any_ready_position_keeps_original_unready_maximum_and_does_not_retry():
    words = [(0, 0, 0)] * 32
    for j, v in zip((2, 6, 10, 14), (50000, 50000, -50000, -50000), strict=True):
        words[j] = (v, 0, 0)
    for j in (7, 11, 15, 19):
        words[j] = (50000, 0, 0)
    tpl = SyncSignTemplate(((1, 0),) * 3)
    traces = []
    results = []
    for guard in (False, True):
        trace = []
        r = SplineScoreFlow(
            SplineScoreAssumptions(32, reuse_symbol_power=True, defer_all_future=guard)
        ).run(
            words,
            tpl,
            [2, 7],
            lookahead_ticks=128,
            sink=lambda e: (
                trace.append({k: v for k, v in e.items() if k != "cycle"})
                if e["event"] in ("sample", "score", "selected_sample")
                else None
            ),
        )
        traces.append(trace)
        results.append(r)
    old, new = results
    assert old["best"]["disposition"] == "maximum_needs_future" and old["best"]["start_tick"] > 24
    assert not new["geometry_wait"] and new["scoring_evaluated"] and new["scores_evaluated"] == 18
    assert new["best"] == old["best"] and traces[0] == traces[1]
    assert new["cycles"] - old["cycles"] == new["geometry_cycles"] == 30


def test_exact_ready_endpoint_empty_and_invalid_grid_are_distinct():
    words = [(91, 47, 0)] * 32
    tpl = SyncSignTemplate(((1, 0),) * 3)
    flow = SplineScoreFlow(SplineScoreAssumptions(32, defer_all_future=True))
    # First valid tick12 plus tail96 plus lookahead140 equals last usable tick248.
    r = flow.run(words, tpl, [2], lookahead_ticks=140)
    assert not r["geometry_wait"] and r["scoring_evaluated"]
    assert flow.run(words, tpl, [2], lookahead_ticks=141)["geometry_wait"]
    for peaks, minimum in (([], 0), ([31], 0), ([2], 256)):
        r = flow.run(words, tpl, peaks, minimum_tick=minimum)
        assert not r["geometry_wait"] and r["geometry_positions"] == r["scores_evaluated"] == 0
        assert r["best"] is None and r["numerical_pass"] is None


@pytest.mark.parametrize("aligned", [False, True])
def test_saturation_rejection_precedes_geometry_even_if_all_positions_need_future(aligned):
    frozen = json.loads((Path(__file__).parent / "data/spline_score_saturation.json").read_text())
    raw = [tuple(w) for w in frozen["words"]]
    words = [w[:2] for w in raw] if aligned else raw
    kw = dict(exponent=0) if aligned else {}
    tpl = SyncSignTemplate(((1, 0),))
    old = SplineScoreFlow(SplineScoreAssumptions(8)).run(
        words, tpl, [1], lookahead_ticks=32768, **kw
    )
    new = SplineScoreFlow(SplineScoreAssumptions(8, defer_all_future=True)).run(
        words, tpl, [1], lookahead_ticks=32768, **kw
    )
    assert old == new and new["numerical_rejection"] and new["numerical_pass"] is False
    assert "geometry_wait" not in new


def test_geometry_cycle_abort_does_not_claim_wait_or_a_numeric_result():
    words = [(100, 0, 0)] * 32
    tpl = SyncSignTemplate(((1, 0),) * 3)
    base = SplineScoreFlow(SplineScoreAssumptions(32)).run(words, tpl, [2])
    r = SplineScoreFlow(SplineScoreAssumptions(32, defer_all_future=True)).run(
        words, tpl, [2], lookahead_ticks=32768, max_cycles=base["preparation_cycles"] + 2
    )
    assert r["resource_drop"] and r["numerical_pass"] is None and not r["geometry_wait"]
    assert r["best"] is None and r["geometry_cycles"] is None


def test_typed_wait_and_bounded_transport_use_no_selected_tick_or_retention_success():
    tpl = SyncSignTemplate.from_symbols(
        PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4)).sync_symbols()
    )
    worker = CaptureFineWorker(
        tpl, 256, confirmation_required=True, reuse_symbol_power=True, defer_all_future=True
    )
    events = []
    source = [(pack_filter_word(1000, 0, 0),) * 2] * 300
    result = CaptureTransport().simulate(
        source,
        [CaptureRequest(256, 0, (128,), 0, 128)],
        fine_worker=worker,
        sink=lambda e, w: events.append(e),
    )
    assert (
        result["worker_results"]["geometry_wait"] == 1
        and result["worker_results"]["confirm_pending"] == 0
    )
    assert events[0]["disposition"] == "geometry_wait" and events[0]["start_tick"] is None
    assert (
        events[0]["retention_necessary_pass"] is None
        and events[0]["continuation_gap_samples"] is None
    )
    assert FineService(1, "geometry_wait").start_tick is None
    with pytest.raises(ValueError):
        FineService(1, "geometry_wait", 0)
    with pytest.raises(ValueError):
        SplineScoreAssumptions(defer_all_future=1)
    old = spline_score_cost(SplineScoreAssumptions(reuse_symbol_power=True))
    new = worker.estimate()["fine"]
    assert (
        new["additional_query_state_upper_bits"] - old["additional_query_state_upper_bits"]
        == 118
        <= 128
    )
    assert new["peak_bit_monitor_slots"] == 77 <= new["peak_bit_monitor_reserved_slots"] == 80
    assert new["shared_DSP25x18_count"] == old["shared_DSP25x18_count"] == 1
    assert (
        CaptureTransport(CaptureTransportAssumptions()).estimate(measured_worker=True)[
            "register_reservation_bits"
        ]["worker_statistics"]
        == 11 * 64
    )
