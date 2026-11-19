"""Exact finite recurrence,real registered cache,bounded lookahead and DSP handoff."""

import numpy as np
import pytest

from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT
from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.hold_timing_flow import HoldTimingPipeline, PipelinedPacketTimingService
from nearlink_sdr.sim.initial_interpolant_flow import InitialInterpolantPort
from tests.test_sync_fit_word import packed

FMT = TimingWordFormat(30, 32)


def pipeline(words, *, initial=(1 << 53, -53), composite=False):
    n = len(words)
    port = InitialInterpolantPort(INPUT_FORMAT, n)
    for i, word in enumerate(words):
        assert port.write(1 + 4 * i, 24576, word, i)
    state = WordTimingState(initial, 11, n, FMT)
    cls = PipelinedPacketTimingService if composite else HoldTimingPipeline
    return cls(state, port, n), port


@pytest.mark.parametrize("n", [32, 62, 126])
@pytest.mark.parametrize("kind", ["tiny", "zeros", "floor", "wide", "steps"])
def test_every_committed_state_and_word_matches_original_with_bounded_lookahead(n, kind):
    rng = np.random.default_rng(1081123)
    if kind == "tiny":
        words = [packed(1 + i % 3, -i % 7, i % 2 - 2) for i in range(n)]
    elif kind == "zeros":
        words = [0 if i % 3 else packed(1 << 20, 0, -23) for i in range(n)]
    elif kind == "floor":
        words = [packed(i % 3, 0, -80) for i in range(n)]
    else:
        words = [
            packed(
                int(rng.integers(-(2**21), 2**21)),
                int(rng.integers(-(2**21), 2**21)),
                [-128, -19, 127][i % 3] if kind == "wide" else (-90 if i < n // 2 else 50),
            )
            for i in range(n)
        ]
    initial = (1 << 53, -180 if kind == "floor" else -53)
    flow, port = pipeline(words, initial=initial)
    reference = WordTimingState(initial, 11, n, FMT)
    accepted, edges = 0, []
    while not flow.drained:
        old = flow.state.snapshot()
        before = flow.peek()
        got = flow.take()
        if got is not None:
            assert before["index"] == accepted and before["word"] == words[accepted]
            assert got == reference.advance_words(words[accepted], None)
            assert flow.take() is None
            accepted += 1
            edges.append(flow.cycle)
        else:
            assert flow.state.snapshot() == old
        flow.step()
        assert flow.cycle < 1000 and flow.requested - accepted <= 4
    assert accepted == n and flow.stats["native_products"] == 4 * n
    assert flow.stats["cache_requests"] == flow.stats["cache_returns"] == n
    assert flow.stats["power_updates"] == n and port.cache.state == "released"
    assert set(np.diff(edges)) == {6} and edges[-1] == 6 * n + 7
    assert flow.estimate()["additional_state_bits_upper"] <= 4096
    flow.release()
    assert flow.take() is None
    with pytest.raises(RuntimeError, match="Closed"):
        flow.release()


def test_stalled_sink_is_bounded_and_drains_every_original_value_once():
    words = [packed(1 << 19, i, -18) for i in range(32)]
    flow, _ = pipeline(words)
    before = flow.state.snapshot()
    for _ in range(200):
        flow.step()
        assert flow.state.snapshot() == before
        assert flow.requested <= 4 and len(flow.outputs) <= 2
    assert len(flow.outputs) == 2 and flow.stats["output_stall_edges"]
    with pytest.raises(RuntimeError, match="drain"):
        flow.release()
    reference = WordTimingState((1 << 53, -53), 11, 32, FMT)
    while not flow.drained:
        k = reference.count
        if flow.ready and flow.cycle % 11 == 0:
            assert flow.take() == reference.advance_words(words[k], None)
        flow.step()
    assert flow.state.snapshot() == reference.snapshot()


@pytest.mark.parametrize("edges", [0, 2, 7, 12, 20])
def test_abort_discards_all_reserved_work_and_never_commits_it(edges):
    flow, port = pipeline([packed(1001, -53, -13)] * 32)
    old = flow.state.snapshot()
    for _ in range(edges):
        flow.step()
    flow.abort()
    flow.abort()
    for _ in range(10):
        flow.step()
    assert flow.state.snapshot() == old and flow.take() is None
    assert flow.stats["aborted"] == 1 and port.pending is None
    assert port.cache.state == "released" and flow.mode == "aborted"


def test_shared_DSP_transition_requires_drain_and_preserves_actual_data_feedback():
    words = [packed(100000 + i * 113, -i * 67, -20) for i in range(32)]
    flow, _ = pipeline(words, composite=True)
    reference = WordTimingState((1 << 53, -53), 11, 32, FMT)
    with pytest.raises(RuntimeError, match="drain"):
        flow.issue(words[0], words[0])
    while reference.count < 32:
        got = flow.take()
        if got is not None:
            assert got == reference.advance_words(words[reference.count], None)
            if reference.count == 32:
                assert not flow.issue(words[0], words[0])
                assert flow.normal is None
        flow.step()
    assert flow.normal is None and flow.hold.drained
    current, mid = packed(300000, -12000, -20), packed(250000, 300, -20)
    assert flow.issue(current, mid)
    while flow.normal is not None and not flow.normal.ready:
        flow.step()
    assert flow.take() == reference.advance_words(current, mid)
    assert flow.hold.mode == "released" and flow.normal.stats["divider_steps"] == 64
    assert flow.estimate()["DSP25x18"] == 4
    assert not flow.estimate()["old4096_state_budget_passed"]
    flow.step()
    flow.release()
    assert flow.mode == "released"


def test_geometry_state_clock_and_counter_errors_fail_explicitly():
    flow, port = pipeline([packed(100, -1, -5)] * 32)
    with pytest.raises(ValueError):
        HoldTimingPipeline(None, port)
    other = WordTimingState((1 << 53, -53), 12, 32, FMT)
    with pytest.raises(ValueError, match="matching"):
        HoldTimingPipeline(other, port)
    flow.step()
    with pytest.raises(ValueError):
        HoldTimingPipeline(flow.state, port)
    while not flow.ready:
        flow.step()
    flow.state.omega_q32 += 1
    with pytest.raises(RuntimeError, match="changed"):
        flow.take()
    flow.state.omega_q32 -= 1
    flow.stats["accepted"] = 2**64 - 1
    old, queued = flow.state.snapshot(), len(flow.outputs)
    with pytest.raises(OverflowError):
        flow.take()
    assert flow.state.snapshot() == old and len(flow.outputs) == queued


def test_registered_preparation_costs_an_edge_and_take_uses_that_actual_result(monkeypatch):
    word = packed(773001, -8312, -20)
    flow, _ = pipeline([word] * 32)
    old = flow.state.snapshot()
    while not flow.outputs:
        flow.step()
    assert not flow.ready and flow.state.snapshot() == old
    ready_edge = flow.cycle + 1
    flow.step()
    assert flow.ready and flow.commit_ready == ready_edge
    assert flow.state.snapshot() == old

    def forbidden(*_a, **_kw):
        raise AssertionError("Taking a registered result cannot recompute timing")

    monkeypatch.setattr(flow.state, "commit", forbidden)
    expected = WordTimingState((1 << 53, -53), 11, 32, FMT).advance_words(word, None)
    peek = flow.peek()
    peek["word"] = 0  # Public observations cannot steer the actual queued word.
    assert flow.take() == expected
    assert flow.state.previous_word == word
