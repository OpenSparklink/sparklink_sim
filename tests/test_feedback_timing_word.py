"""Independent FIFO order/PI Fraction rounding;canonical detector is a shared reference."""

from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import (
    KI_Q48,
    KP_Q48,
    LOWER_Q32,
    ONE,
    UPPER_Q32,
    TimingWordFormat,
    WordTimingState,
)
from nearlink_sdr.sim.feedback_timing_word import FeedbackAgeTimingState
from nearlink_sdr.sim.timing_word_flow import TimingWordService


def words(seed, count):
    random = np.random.default_rng(seed)
    mask = (1 << 22) - 1
    for n in range(count):
        r, i = map(int, random.integers(-(1 << 20), 1 << 20, 2))
        if n % 11 == 0:
            r = i = 0
        exponent = (-60, -15, 0, 30, 100)[(n // 7) % 5]
        yield (r & mask) | ((i & mask) << 22) | ((exponent & 255) << 44)


@pytest.mark.parametrize("age", [0, 1, 2, 3])
@pytest.mark.parametrize("fmt", [TimingWordFormat(30, 24), TimingWordFormat(30, 32),
                                 TimingWordFormat(38, 32)])
@pytest.mark.parametrize("hold", [32, 62, 126])
def test_fraction_PI_and_independent_fixed_FIFO_order(age, fmt, hold):
    actual = FeedbackAgeTimingState((1 << 50, -40), 4337, hold, fmt, error_age=age)
    detector = WordTimingState((1 << 50, -40), 4337, hold, fmt)
    data = list(words(2026112097 + age + hold, 320))
    midpoint_words = list(words(2026112073 + hold, 320))
    pending, omega, position, rate_limits = [0] * age, 4 * ONE, 4337 << 29, 0
    for n, current in enumerate(data):
        midpoint = midpoint_words[n] if n >= hold else None
        detector.advance_words(current, midpoint)
        measured = detector.error_word
        used = measured if age == 0 else (pending[0] if midpoint is not None else 0)
        shift = 48 + fmt.error_bits - 32
        ki, kp = (round(Fraction(c * used, 1 << shift)) for c in (KI_Q48, KP_Q48))
        proposed = omega + ki
        omega = max(LOWER_Q32, min(UPPER_Q32, proposed))
        rate_limits += proposed != omega
        position += omega + kp
        previous = actual.state.next_q32
        actual.advance_words(current, midpoint)
        if midpoint is not None and age:
            pending = [*pending[1:], measured]
        state = actual.state
        assert state.next_q32 == position and state.previous_q32 == previous
        assert state.omega_q32 == omega and state.rate_limit_events == rate_limits
        assert actual.pending_errors == pending and actual.last_applied_error_word == used
        for field in ("power_word", "previous_word", "error_word", "count", "power_floor_events",
                      "power_alignment_erased", "ted_alignment_erased", "error_clip_events"):
            assert getattr(state, field) == getattr(detector, field)
        if age == 0:
            assert state.snapshot() == detector.snapshot()
        assert len(actual.pending_errors) == age
    cost = actual.estimate()
    assert cost["additional_functional_persistent_bits"] <= 160
    assert cost["pipeline_context_bits"] is None and not cost["executable_cycle_service"]
    assert cost["measured_II"] is None and not cost["full_receiver16MSps"]


@pytest.mark.parametrize("invalid", [True, -1, 4, 1.0, np.int64(1)])
def test_invalid_compiled_age_rejected(invalid):
    with pytest.raises(ValueError, match="explicit compiled"):
        FeedbackAgeTimingState((1 << 50, -40), 4337, 32, error_age=invalid)


def test_zero_fill_hold_and_release_do_not_apply_future_error():
    state = FeedbackAgeTimingState((1 << 50, -40), 4337, 32, error_age=3)
    for word in words(713, 32):
        state.advance_words(word, None)
        assert state.pending_errors == [0, 0, 0]
    state.advance_words(next(words(717, 1)), next(words(719, 1)))
    assert state.last_applied_error_word == 0
    before = deepcopy(state.state.snapshot())
    state.release()
    state.release()
    assert state.pending_errors == [0, 0, 0] and state.state.snapshot() == before
    with pytest.raises(RuntimeError, match="cannot resume"):
        state.advance_words(0, 0)
    assert state.state.snapshot() == before


@pytest.mark.parametrize("fault", ["count", "NCO", "FIFO_length", "FIFO_range", "FIFO_type"])
def test_failed_admission_has_no_partial_FIFO_state_change(fault):
    state = FeedbackAgeTimingState((1 << 50, -40), 4337, 32, error_age=1)
    for word in words(1791, 33):
        state.advance_words(word, None if state.state.count < 32 else word)
    if fault == "count":
        state.state.count = (1 << 64) - 1
    elif fault == "NCO":
        state.state.next_q32 = (1 << 96) - 1
    elif fault == "FIFO_length":
        state.pending_errors.append(0)
    elif fault == "FIFO_range":
        state.pending_errors[0] = 1 << 35
    else:
        state.pending_errors[0] = True
    before = deepcopy(state.snapshot())
    with pytest.raises((OverflowError, RuntimeError, ValueError)):
        state.advance_words(next(words(1793, 1)), next(words(1797, 1)))
    assert state.snapshot() == before


def test_functional_model_cannot_masquerade_as_existing_cycle_state():
    state = FeedbackAgeTimingState((1 << 50, -40), 4337, 32, error_age=1)
    with pytest.raises(ValueError):
        TimingWordService(state)
