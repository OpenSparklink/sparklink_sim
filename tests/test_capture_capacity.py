"""Necessary graph work and finite admission/history/EOF behavior, never packet acceptance."""

import numpy as np
import pytest

from nearlink_sdr.phy.capture_capacity import (
    CaptureCapacity,
    CaptureCapacityAssumptions,
    fine_work_floor,
)
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def test_present_grid_dimensions_match_actual_search_and_optimistic_reuse_floor():
    rng = np.random.default_rng(410117)
    ref = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols()
    x = rng.normal(size=256) + 1j * rng.normal(size=256)
    diagnostic = {}
    search_psk_sync(x, 4, ref, 8, min_start=1, diagnostics=diagnostic)
    floor = fine_work_floor()
    assert floor["all_phases"] == diagnostic["phase_hypotheses"] == 32
    assert floor["legal_positions"] == diagnostic["scored_positions"] == 1040
    assert floor["charged_phases"] == 28
    assert floor["differential_products"] == 1757
    assert floor["charged_legal_positions"] == 910
    assert floor["variable_real_products"] == 10605
    assert fine_work_floor(free_integer_phases=False)["variable_real_products"] == 12120
    limits = CaptureCapacity().necessary_capacity()
    assert not limits["fine_graph_all_multipliers_necessary_condition"]
    assert limits["required_multiplier_units_at_clock"] == 27
    assert limits["optimistic_product_demand_per_second"] == 5_302_500_000


def test_fcfs_exact_cycles_owned_windows_history_loss_and_admission():
    flow = CaptureCapacity()
    events = []
    result = flow.simulate(iter([1000, 1001, 1002, 1003]), 10000, sink=events.append)
    assert result["accepted_requests"] == 3 and result["dropped_requests"] == 1
    assert result["peak_owned_windows"] == result["owned_window_capacity"] == 3
    assert events[-1]["status"] == "request_queue_full"
    assert [e["released_cycle"] for e in events[:3]] == [15172, 17824, 20476]
    assert [e["retained_gap_samples"] for e in events[:3]] == [0, 41, 253]
    assert result["history_gaps"] == 2
    assert result["closed_loop_capture_verified"] is False


def test_dense_quiet_stalls_and_eof_do_not_create_free_buffers_or_arrivals():
    flow = CaptureCapacity()
    quiet = flow.simulate(iter([]), 1000000)
    assert quiet["requests"] == 0 and quiet["continuous_input_detection_charged_samples"] == 1000000
    dense = flow.simulate(range(512, 10000, 512), 10000)
    assert dense["dropped_requests"] == dense["history_gaps"] == 0
    assert dense["peak_owned_windows"] == 1
    stalled = flow.simulate(
        [1000, 1001, 1002, 1003], 10000, stalls=[(15000, 17000), (17000, 20000)]
    )
    assert stalled["blocked_results"] == 1 and stalled["dropped_requests"] == 1
    assert stalled["max_latency_cycles"] > dense["max_latency_cycles"]
    # EOF stops new arrivals; a long externally blocked result does not pretend
    # that nonexistent samples overwrote the ring after the last input.
    eof = flow.simulate([10000], 10000, stalls=[(0, 1000000)])
    assert eof["blocked_results"] == 1 and eof["history_gaps"] == 0


def test_invalid_model_dimensions_and_trace_order_are_explicit():
    for kwargs in [
        dict(fine_multipliers=19),
        dict(window_samples=128),
        dict(queue_depth=17),
        dict(input_hz=300000000),
        dict(sample_bits=129),
        dict(snapshot_reads_per_cycle=33),
    ]:
        with pytest.raises(ValueError):
            CaptureCapacityAssumptions(**kwargs)
    with pytest.raises(ValueError):
        CaptureCapacity(0)
    flow = CaptureCapacity()
    for requests, count in [([], 0), ([0], 100), ([101], 100), ([2, 1], 100), ([1.0], 100)]:
        with pytest.raises(ValueError):
            flow.simulate(requests, count)
    for stalls in [[(2, 1)], [(0, 3), (2, 4)], [(-1, 2)], iter([])]:
        with pytest.raises(ValueError):
            flow.simulate([], 100, stalls=stalls)
    with pytest.raises(OverflowError):
        flow.simulate([], 2**63)
    with pytest.raises(ValueError):
        fine_work_floor(sps=0)
    assert fine_work_floor(window=4)["all_phases"] == 0


def test_slow_snapshot_cannot_read_already_overwritten_history():
    assumptions = CaptureCapacityAssumptions(
        history_samples=256, queue_depth=16, snapshot_reads_per_cycle=1
    )
    flow = CaptureCapacity(assumptions)
    records = []
    result = flow.simulate(range(1000, 1017), 10000, sink=records.append)
    assert result["peak_owned_windows"] == 17
    assert result["snapshot_history_gaps"] == 17
    assert all(r["snapshot_history_gap"] for r in records)
    assert not flow.necessary_capacity()["storage_lower_bound_within_budget"]


def test_request_observer_preserves_bad_crc_and_chunked_trajectories(monkeypatch):
    import runpy
    from pathlib import Path

    from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
    from nearlink_sdr.phy.tone_notch import ToneNotchConfig
    from tests.test_stream_receive import equal_events
    from tests.test_strict_receive import iq_record, transmitted

    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    observer = runpy.run_path(scripts / "audit_capture_capacity.py")["RequestObserver"]
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, _ = transmitted(p, bad_data=True)
    x, q, _, _ = iq_record(wire, offset=12.277, ppm=100, cfo=30000)
    options = dict(quiet_batching=False, correlation_backend="scaled", tone_notch=ToneNotchConfig())
    plain = PSKStreamReceiver(p, q, **options)
    expected = plain.feed(x) + plain.finish()
    traces = []
    for chunk in [len(x), 257]:
        requests, events = [], []
        rx = observer(p, q, request_sink=requests.append, **options)
        for k in range(0, len(x), chunk):
            events += rx.feed(x[k : k + chunk])
        events += rx.finish()
        equal_events(expected, events)
        assert events[0]["frame"]["status"] == "bad_data_crc"
        assert len(requests) == rx.capture_stats["full_searches"]
        assert any(r["candidate_sync_start"] is not None for r in requests)
        traces.append(requests)
    assert traces[0] == traces[1]
