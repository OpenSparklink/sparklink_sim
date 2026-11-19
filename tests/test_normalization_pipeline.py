"""Independent value oracle, finite credits and real shared-bank arbitration."""

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat, WordNormalizationState
from nearlink_sdr.sim.normalization_pipeline import PipelinedNormalizationService
from tests.normalization_reference import raw_word
from tests.test_normalization_word import fit


def run(words, *, mode="linear", period=7, blocked=(), prepare=False):
    f = fit((1 << 40) - 713, -(1 << 32) + 103, (1234567, -31))
    state = WordNormalizationState(f, NormalizationFormat(mode))
    oracle = WordNormalizationState(f, NormalizationFormat(mode))
    service = PipelinedNormalizationService(state)
    if prepare:
        for _ in range(33):
            service.step()
    issued = 0
    records = []
    last_issue = -period
    stalled = None
    while len(records) < len(words):
        c = service.cycle
        if issued < len(words) and c - last_issue >= period and service.can_issue:
            assert service.issue(words[issued])
            issued += 1
            last_issue = c
        before = state.snapshot()
        if service.peek() is not None:
            if c in blocked:
                if stalled is None:
                    stalled = (service.peek(), state.snapshot())
                assert (service.peek(), state.snapshot()) == stalled
            else:
                result = service.take()
                assert result == oracle.advance_word(words[len(records)])
                assert state.snapshot() == oracle.snapshot()
                records.append((service.first_cycle, service.ready_cycle, c, result))
                stalled = None
        service.step()
        if c in blocked:
            # First inverse preparation is independent of output transfers.
            assert state.count == before["count"] and state.phase == before["phase_turn40"]
        assert service.owned <= 4
        assert sum(x is not None for x in service.slots) == service.owned
        assert c < 50_000
    assert state.snapshot() == oracle.snapshot()
    service.release()
    return service, records


@pytest.mark.parametrize("mode", ["nearest", "linear"])
@pytest.mark.parametrize("period", [1, 6, 7, 8])
def test_arbitration_extreme_words_zero_and_stalled_ordered_results(mode, period):
    rng = np.random.default_rng(1122213)
    words = [0, raw_word(0, 0, 127), raw_word(-(1 << 21), 7, -128),
             raw_word(19, (1 << 21) - 1, 127)]
    words += [raw_word(*map(int, rng.integers(-(1 << 21), 1 << 21, 2)),
                       int(rng.integers(-128, 128))) for _ in range(92)]
    service, records = run(words, mode=mode, period=period, blocked=range(47, 148))
    assert service.peak_owned == 4
    assert service.pipeline_stats["predictions"] == len(words)
    assert service.stats["native_products"] == len(words) * (10 if mode == "linear" else 8)
    assert service.stats["rom_reads"] == len(words) * (4 if mode == "linear" else 2)
    assert service.stats["divider_steps"] == 30
    assert service.stats["completed"] == service.stats["accepted"] == len(words)
    assert [r[3]["index"] for r in records] == list(range(len(words)))
    assert not service.estimate()["actual_cache_broadcast_or_timing_join_included"]
    assert not service.estimate()["complete_receiver16MSps"]


def test_phase_prediction_uses_real_edge_and_main_owner_stays_uncommitted():
    state = WordNormalizationState(fit(1717, -36363), NormalizationFormat())
    service = PipelinedNormalizationService(state)
    assert service.issue(raw_word(117, -213, 4))
    assert service.private_phase == state.phase == 1717
    assert service.private_index == state.count == 0
    assert service.slots[0]["stage"] == "predict"
    service.step()
    assert service.private_phase == (1717 - 36363) % (1 << 40)
    assert service.private_index == 1 and state.count == 0 and state.phase == 1717
    while service.peek() is None:
        service.step()
    snapshot = state.snapshot()
    for _ in range(51):
        service.step()
        assert state.snapshot() == snapshot
    service.peek()["stats"]["accepted"] = 999
    assert service.validate_take() and service.take()["stats"]["accepted"] == 1
    assert service.take() is None


@pytest.mark.parametrize("edge", [0, 4, 17, 34, 44, 55])
def test_full_credits_abort_and_invalid_input_do_not_retire_contexts(edge):
    state = WordNormalizationState(fit(), NormalizationFormat())
    service = PipelinedNormalizationService(state)
    for i in range(4):
        assert service.issue(raw_word(11 + i, 3, -7))
        assert not service.issue(0)
        service.step()
    assert not service.issue(0) and service.owned == 4
    before = (state.snapshot(), service.owned, service.stats.copy())
    with pytest.raises(ValueError):
        service.issue(True)
    assert (state.snapshot(), service.owned, service.stats) == before
    for _ in range(edge):
        service.step()
    service.abort()
    service.abort()
    assert service.mode == "aborted" and service.owned == 0
    assert service.slots == [None] * 4 and service.peek() is None
    assert state.count == 0 and state.phase == 0
    assert service.stats["aborted"] == 1 and service.pipeline_stats["aborted_contexts"] == 4
    with pytest.raises(RuntimeError):
        service.issue(0)
    with pytest.raises(RuntimeError):
        service.release()


def test_preflight_counter_exhaustion_leaves_held_output_and_owner_unchanged():
    state = WordNormalizationState(fit(), NormalizationFormat())
    service = PipelinedNormalizationService(state)
    with pytest.raises(RuntimeError):
        service.release()
    assert service.issue(0)
    while service.peek() is None:
        service.step()
    service.stats["completed"] = (1 << 64) - 1
    before = (service.peek(), state.snapshot(), service.owned)
    with pytest.raises(OverflowError):
        service.take()
    assert (service.peek(), state.snapshot(), service.owned) == before
    service.stats["completed"] = 0
    state.phase = 1
    with pytest.raises(RuntimeError):
        service.validate_take()
    state.phase = 0
    assert service.take() is not None
    service.stats["accepted"] = 1 << 17
    assert not service.can_issue and not service.issue(0)
    service.cycle = (1 << 64) - 1
    with pytest.raises(OverflowError):
        service.step()


def test_prepared_no_stall_interval7_is_measured_not_inherited_from_power_II6():
    words = [raw_word(17 + i, -31, -7) for i in range(128)]
    service, records = run(words, prepare=True)
    issued, ready = np.array([(r[0], r[1]) for r in records]).T
    np.testing.assert_array_equal(np.diff(issued), 7)
    np.testing.assert_array_equal(np.diff(ready), 7)
    assert set(ready - issued) == {25}  # Includes newly registered prediction edge.
    cost = service.estimate()
    assert cost["state_register_bits_upper"] == sum(cost["state_parts"].values())
    assert cost["state_register_bits_upper"] <= 8192
    assert cost["coefficient_ROM_read_ports"] == 2
    assert cost["additional_DSP25x18"] == 4
    assert cost["wide_macro_resources_increased_vs_serial"]


def test_drained_bank_transfer_retains_original24edge_data_and_one_owner():
    f = fit(1717, -36363)
    state = WordNormalizationState(f, NormalizationFormat())
    oracle = WordNormalizationState(f, NormalizationFormat())
    service = PipelinedNormalizationService(state)
    with pytest.raises(RuntimeError):
        service.enter_serial()
    word = raw_word(117, -213, 4)
    assert service.issue(word)
    with pytest.raises(RuntimeError):
        service.enter_serial()
    while service.peek() is None:
        service.step()
    assert service.take() == oracle.advance_word(word)
    service.enter_serial()
    with pytest.raises(RuntimeError):
        service.enter_serial()
    for k in range(3):
        service.step()
        start = service.cycle
        assert service.issue(word)
        for _ in range(24):
            service.step()
        assert service.ready_cycle == start + 24
        assert service.validate_take()
        assert service.take() == oracle.advance_word(word)
        assert state.snapshot() == oracle.snapshot()
        assert service.private_index == 1  # Dormant predictor does not add data work.
        assert state.count == k + 2
    service.step()
    assert service.issue(word)
    service.abort()
    assert service.pipeline_stats["aborted_contexts"] == 1
