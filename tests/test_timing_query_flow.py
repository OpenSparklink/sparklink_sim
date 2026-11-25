"""Exact rational coordinates,registered values,backpressure and finite bounds."""

import math
from fractions import Fraction

import pytest

from nearlink_sdr.phy.timing_word import ONE, WordTimingState
from nearlink_sdr.sim.timing_query_flow import TimingQueryService


def independent(position):
    cell = math.floor(position)
    return dict(cell=cell, phase=round((position - cell) * 65536))


@pytest.mark.parametrize("base", [0, 11, 2**60 + 1])
@pytest.mark.parametrize("fraction", [0, 32768, 98304, ONE - 32768, ONE - 1])
@pytest.mark.parametrize("count", [31, 32, 126])
def test_three_registered_edges_match_independent_fraction_with_large_indices(
    base, fraction, count
):
    state = WordTimingState((1, 0), 0, 32)
    state.next_q32 = base * ONE + fraction
    state.previous_q32 = max(0, state.next_q32 - 4 * ONE - 1)
    state.count = count
    before = state.snapshot()
    t = Fraction(state.next_q32, ONE)
    midpoint = Fraction(state.next_q32 + state.previous_q32, 2 * ONE)
    expected = (independent(t), independent(midpoint) if count >= 32 else None)
    service = TimingQueryService()
    assert service.issue(state)
    for _ in range(3):
        assert not service.ready and service.take() is None
        service.step()
    assert service.ready_cycle - service.first_cycle == 3
    result = service.take()
    assert result["queries"] == expected
    assert result["future_end"] == math.ceil(t + 3)
    assert result["retain_before"] == max(0, min(q["cell"] for q in expected if q) - 1)
    assert result["count"] == count and state.snapshot() == before
    assert service.estimate()["logical_register_bits_upper"] == 1689
    assert not service.estimate()["synthesis"]


def test_query_snapshot_held_reply_one_credit_abort_and_release():
    state = WordTimingState((1, 0), 95, 32)
    service = TimingQueryService()
    assert service.issue(state) and not service.issue(state)
    state.next_q32 += 4 * ONE
    with pytest.raises(RuntimeError):
        service.release()
    for _ in range(7):
        service.step()
    result = service.take()
    assert result["queries"][0] == dict(cell=11, phase=57344)
    assert service.stats["output_stall_edges"] == 4
    assert not service.issue(state)
    service.step()
    assert service.issue(state)
    service.abort()
    assert service.take() is None
    assert service.stats["aborted"] == 1
    service.abort()
    with pytest.raises(RuntimeError):
        service.issue(state)
    other = TimingQueryService()
    other.release()
    with pytest.raises(RuntimeError):
        other.issue(state)
    with pytest.raises(ValueError):
        other.issue(object())


def test_future_coordinate_and_counter_overflow_are_explicit():
    state = WordTimingState((1, 0), 0, 32)
    state.next_q32 = (2**64 - 1) * ONE
    service = TimingQueryService()
    assert service.issue(state)
    service.step()
    with pytest.raises(OverflowError):
        service.step()
    service = TimingQueryService()
    service.stats["accepted"] = 2**64 - 1
    with pytest.raises(OverflowError):
        service.issue(state)
    assert service.mode == "idle"
    service.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        service.step()


@pytest.mark.parametrize("count", [31, 32])
@pytest.mark.parametrize("fraction", [0, 32768, ONE - 1])
def test_compact_query_publishes_exact_same_receipt_one_edge_earlier(count, fraction):
    state = WordTimingState((1, 0), 0, 32)
    state.next_q32 = (2**63 + 3) * ONE + fraction
    state.previous_q32 = state.next_q32 - 4 * ONE - 1
    state.count = count
    out = []
    for compact in (False, True):
        service = TimingQueryService(compact_retirement=compact)
        latency = 2 if compact else 3
        assert service.issue(state)
        for _ in range(latency):
            assert service.take() is None
            service.step()
        assert service.ready_cycle - service.first_cycle == latency
        service.step()  # Hold the exact registered reply through backpressure.
        out.append(service.take())
        assert not service.issue(state)
        assert service.estimate()["minimum_II_with_immediate_take"] == latency + 1
    assert out[0] == out[1]
