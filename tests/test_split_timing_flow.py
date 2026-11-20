"""Actual independent detector/PI clocks and words;no complete streaming claim."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import KI_Q48, KP_Q48, TimingWordFormat, WordTimingState
from nearlink_sdr.sim.feedback_timing_word import FeedbackAgeTimingState
from nearlink_sdr.sim.split_timing_flow import (
    SplitTimingQueryService,
    TimingDetectorService,
    TimingPIService,
    split_timing_states,
)
from nearlink_sdr.sim.timing_query_flow import TimingQueryService
from nearlink_sdr.sim.timing_word_flow import TimingWordService
from tests.split_calendar import run_node_calendar
from tests.test_feedback_timing_word import words


def execute(service, *args, stall=0):
    if service.taken_this_cycle:
        service.step()
    assert service.issue(*args)
    start = service.cycle
    while not service.ready:
        service.step()
        assert service.cycle - start < 256
    result = deepcopy(service.result)
    for _ in range(stall):
        service.step()
        assert service.result == result
    assert service.take() == result and service.take() is None
    return result, service.ready_cycle - start


@pytest.mark.parametrize("age", [0, 1, 2, 3])
@pytest.mark.parametrize("fmt", [TimingWordFormat(30, 24), TimingWordFormat(30, 32),
                                 TimingWordFormat(38, 32)])
@pytest.mark.parametrize("hold", [32, 62, 126])
def test_independent_owners_match_original_fixed_age_and_queries(age, fmt, hold):
    initial = ((1 << 50, -40), 4337, hold, fmt)
    detector, pi = split_timing_states(*initial)
    ds = TimingDetectorService(detector, divider="radix4-prefix")
    ps, qs = TimingPIService(pi), SplitTimingQueryService()
    reference = FeedbackAgeTimingState(*initial, error_age=age)
    queue = [0] * age
    data, middle = list(words(7311 + hold, 192)), list(words(7419 + hold, 192))
    max_detector = max_pi = 0
    for n, current in enumerate(data):
        midpoint = middle[n] if n >= hold else None
        original_pi = pi.snapshot()
        result, latency = execute(ds, current, midpoint, stall=int(n % 29 == 0))
        max_detector = max(latency, max_detector)
        assert pi.snapshot() == original_pi  # Detector never advances future NCO.
        used = result["error_word"] if age == 0 else (queue[0] if midpoint is not None else 0)
        original_detector = detector.snapshot()
        output, latency = execute(ps, used, stall=int(n % 31 == 0))
        max_pi = max(latency, max_pi)
        assert detector.snapshot() == original_detector
        query, latency = execute(qs, pi)
        assert latency == 3
        reference.advance_words(current, midpoint)
        state = reference.state
        if age and midpoint is not None:
            queue = [*queue[1:], result["error_word"]]
        assert queue == reference.pending_errors
        assert used == reference.last_applied_error_word
        assert result == dict(
            count=state.count, power=list(state.power_word), previous_word=current,
            error_word=state.error_word, error_clip_events=state.error_clip_events,
            power_floor_events=state.power_floor_events,
            power_alignment_erased=state.power_alignment_erased,
            ted_alignment_erased=state.ted_alignment_erased,
        )
        for k in ("count", "previous_q32", "next_q32", "omega_q32", "rate_limit_events"):
            assert output[k] == getattr(state, k)
        assert output["applied_error_word"] == used
        assert query["queries"] == state.queries and query["future_end"] == state.required_end
        assert query["count"] == state.count
        if used:
            shift = 48 + fmt.error_bits - 32
            expected = [round(Fraction(c * used, 1 << shift)) for c in (KI_Q48, KP_Q48)]
            assert ps.state.omega_q32 - original_pi["omega_q32"] == expected[0] or (
                pi.rate_limit_events > original_pi["rate_limit_events"]
            )
    assert max_detector <= (31 if fmt.error_bits == 24 else 35)
    assert max_pi <= 17
    cost = ds.estimate(), ps.estimate()
    assert sum(c["DSP25x18"] for c in cost) == 8
    assert all(not c["shared_bank_credit"] for c in cost)
    assert all(c["packet_FIFO_context_and_arbitration_bits"] is None for c in cost)
    assert all(not c["full_receiver16MSps"] and not c["synthesized"] for c in cost)


def test_old_single_owner_services_reject_both_new_owners():
    detector, pi = split_timing_states((1 << 50, -40), 4337, 32)
    for state in (detector, pi):
        with pytest.raises(ValueError):
            TimingWordService(state)
        with pytest.raises(ValueError):
            TimingQueryService().issue(state)
    for service, state in [(TimingDetectorService, pi), (TimingPIService, detector)]:
        with pytest.raises(ValueError):
            service(state)
    with pytest.raises(ValueError):
        SplitTimingQueryService().issue(detector)
    canonical = WordTimingState((1 << 50, -40), 4337, 32)
    with pytest.raises(ValueError):
        SplitTimingQueryService().issue(canonical)
    assert not hasattr(detector, "next_q32") and not hasattr(pi, "power_word")


@pytest.mark.parametrize("error", [True, 1., np.int64(0), (2 << 32) + 1, -(2 << 32) - 1, 1])
def test_invalid_error_and_known_sync_input_do_not_take_credit(error):
    _, pi = split_timing_states((1 << 50, -40), 4337, 32, TimingWordFormat(30, 32))
    service = TimingPIService(pi)
    before = deepcopy(service.__dict__)
    with pytest.raises(ValueError):
        service.issue(error)
    assert service.stats == before["stats"] and service.input_ready
    assert service.job is None and pi.count == 0


@pytest.mark.parametrize("owner,fault", [
    ("detector", "count"), ("detector", "error_clip_events"),
    ("detector", "completed"), ("PI", "count"), ("PI", "NCO"), ("PI", "completed"),
])
def test_final_preflight_does_not_partially_commit_owner(owner, fault):
    d, p = split_timing_states((1 << 50, -40), 4337, 32)
    s = TimingDetectorService(d) if owner == "detector" else TimingPIService(p)
    assert s.issue(0, None) if owner == "detector" else s.issue(0)
    while (s.mode != "detector_commit" if owner == "detector"
           else not (s.mode == "tail" and s.age == 3)):
        s.step()
    if fault == "completed":
        s.stats["completed"] = (1 << 64) - 1
    elif fault == "NCO":
        # The registered addition is already prepared;changing its source is an error.
        s.state.next_q32 += 1
    else:
        setattr(s.state, fault, (1 << 64) - 1)
        if fault == "error_clip_events":
            s.job["clipped"] = True
    before, stats, clock = s.state.snapshot(), s.stats.copy(), s.cycle
    with pytest.raises(OverflowError):
        s.step()
    assert s.state.snapshot() == before and s.stats == stats and s.cycle == clock
    s.abort()
    assert s.mode == "aborted" and s.result is None


def test_abort_release_backpressure_and_count_domains_are_explicit():
    d, p = split_timing_states((1 << 50, -40), 4337, 32)
    ds, ps = TimingDetectorService(d), TimingPIService(p)
    assert ps.issue(0)
    assert not ps.issue(0)
    with pytest.raises(RuntimeError):
        ps.release()
    while not ps.ready:
        ps.step()
    assert ps.take()["count"] == 1 and d.count == 0
    assert not ps.issue(0)  # No take/reissue on the same registered edge.
    ps.step()
    ps.release()
    with pytest.raises(RuntimeError):
        ps.issue(0)
    assert ds.issue(0, None)
    assert not ds.issue(0, None)
    ds.abort()
    ds.abort()
    with pytest.raises(RuntimeError):
        ds.issue(0, None)
    assert d.count == 0 and p.count == 1


@pytest.mark.parametrize("period", [25, 40, 50])
@pytest.mark.parametrize("age", [0, 1, 2, 3])
@pytest.mark.parametrize("blocked", [False, True])
def test_bounded_two_bank_calendar_and_unarbitrated_shared_bank_counterexample(
    period, age, blocked
):
    row, _ = run_node_calendar(period, age, blocked)
    assert max(row["peak_det_PI_error_messages"]) <= row["capacity_each"] == age + 2
    assert row["max_PI_count_ahead_of_detector"] <= age + 1
    assert row["exclusive_two_bank_DSP"] == 8
    assert not row["old4096_node_budget_passed"]
    assert not row["whole_receiver16MSps"] and row["complete_latency_cycles"] is None
    if blocked or period == 25:
        assert row["drop"] is not None and row["joined"] < 320
    else:
        assert row["joined"] == row["accepted"] == 320 and row["drop"] is None
    if period == 40 and not blocked:
        collisions = row["shared4DSP_or_tree_conflicts"]
        assert any(c["stage"] == "4DSP_capture" for c in collisions)
        assert any(c["stage"] == "2x80bit_pair_add" for c in collisions)
        assert any(c["stage"] == "80bit_final_add" for c in collisions)
