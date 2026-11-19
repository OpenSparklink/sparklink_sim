"""Actual DSP tiles/divider bits,held output,abort and finite context continuity."""

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.timing_word_flow import TimingWordService
from tests.test_sync_fit_word import packed


@pytest.mark.parametrize(
    "fmt", [TimingWordFormat(), TimingWordFormat(30, 32), TimingWordFormat(38, 32)]
)
def test_actual_value_cycle_service_matches_functional_state_for_all_word_ranges(fmt):
    state = WordTimingState((1 << 53, -53), 57, 32, fmt)
    reference = WordTimingState((1 << 53, -53), 57, 32, fmt)
    service = TimingWordService(state)
    rng = np.random.default_rng(1061121)
    latency = set()
    for k in range(100):
        e = [-128, -20, -19, 127][(k // 8) % 4]
        r, i = (int(v) for v in rng.integers(-(2**21), 2**21, 2))
        current = packed(r, i, e)
        midpoint = None if k < 32 else packed(int(rng.integers(-(2**21), 2**21)), 17, e)
        before = state.snapshot()
        assert service.issue(current, midpoint)
        assert not service.issue(current, midpoint)
        while not service.ready:
            service.step()
            if not service.ready:
                assert state.snapshot() == before  # Commit only after complete finite arithmetic.
        expected = reference.advance_words(current, midpoint)
        latency.add(service.ready_cycle - service.first_cycle)
        for _ in range(3):
            service.step()
            assert service.result == expected
        assert service.take() == expected and service.take() is None
        assert not service.input_ready
        service.step()
    assert 18 in latency
    assert max(latency) == fmt.division_bits + 40
    assert service.stats["divider_steps"] > 0
    assert service.stats["native_products"] >= 4 * 100
    assert service.stats["output_stall_edges"] == 300
    assert service.estimate()["DSP25x18"] == 4
    assert service.estimate()["state_register_bits_upper"] <= 4096
    assert service.estimate()["extra_raw_RAM_bits"] == 0
    assert not service.estimate()["intermodule_DSP_sharing_credited"]


@pytest.mark.parametrize("edges", [0, 1, 6, 13, 28])
def test_abort_invalid_input_and_backpressure_never_manufacture_timing_result(edges):
    s = TimingWordService(WordTimingState((1 << 53, -53), 8, 32))
    assert s.take() is None
    old = s.state.snapshot()
    with pytest.raises(ValueError):
        s.issue(True, None)
    assert s.input_ready and s.state.snapshot() == old
    assert s.issue(packed(1000000, 100, -20), None)
    for _ in range(min(edges, 10)):
        s.step()
    s.abort()
    for _ in range(20):
        s.step()
    assert s.take() is None and not s.ready
    assert s.state.snapshot() == old
    with pytest.raises(RuntimeError):
        s.issue(0, None)
    s.abort()
    assert s.stats["aborted"] == 1


def test_bad_context_and_counter_exhaustion_are_explicit():
    with pytest.raises(ValueError):
        TimingWordService(None)
    s = TimingWordService(WordTimingState((1 << 53, -53), 0, 32))
    s.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        s.step()


@pytest.mark.parametrize("fmt", [TimingWordFormat(30, 32), TimingWordFormat(38, 32)])
def test_actual_cycle_path_handles_small_mantissa_and_zero_scale_counterexamples(fmt):
    for zero in [False, True]:
        initial = (1 << 53, -59 if zero else -53)
        w = packed(1 << 20, 0, -23) if zero else packed(1, 0, 0)
        words = [w] * 31 + [0 if zero else packed(3, 0, -1), w]
        service = TimingWordService(WordTimingState(initial, 0, 32, fmt))
        reference = WordTimingState(initial, 0, 32, fmt)
        for k, current in enumerate(words):
            mid = None if k < 32 else w
            assert service.issue(current, mid)
            while not service.ready:
                service.step()
            assert service.take() == reference.advance_words(current, mid)
            service.step()
