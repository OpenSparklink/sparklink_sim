"""Actual restoring divider,registered ROM/DSP and one-owner normalization checks."""

import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat, WordNormalizationState
from nearlink_sdr.sim.normalization_word_flow import SymbolNormalizationService
from tests.normalization_reference import raw_word
from tests.test_normalization_word import fit


@pytest.mark.parametrize("mode,latency,products,reads", [
    ("nearest", 18, 8, 2), ("linear", 24, 10, 4),
])
@pytest.mark.parametrize("gain", [(1, -512), ((1 << 54) - 1, 511), (1234567, -31)])
def test_actual_words_states_and_declared_serial_edges(mode, latency, products, reads, gain):
    f = fit((1 << 40) - 713, -(1 << 32) + 103, gain)
    functional = WordNormalizationState(f, NormalizationFormat(mode))
    actual = WordNormalizationState(f, NormalizationFormat(mode))
    service = SymbolNormalizationService(actual)
    for _ in range(33):
        service.step()
    assert service.preparation_ready_cycle == 33
    assert service.stats["divider_steps"] == 30
    for k, word in enumerate([0, raw_word(0, 0, 127), raw_word(-(1 << 21), 7, -128),
                              raw_word(19, (1 << 21) - 1, 127), raw_word(3, -11, -3)]):
        start = service.cycle
        before = actual.snapshot()
        assert service.issue(word)
        assert not service.issue(word)
        for _ in range(latency - 1):
            service.step()
            assert service.peek() is None
        assert actual.snapshot() == before
        service.step()
        assert service.cycle - start == latency and service.peek() is not None
        assert actual.snapshot() == before
        result = service.take()
        assert result == functional.advance_word(word)
        assert actual.snapshot() == functional.snapshot()
        assert not service.issue(word), "One real handoff edge precedes the next input"
        service.step()
        assert service.stats["native_products"] == (k + 1) * products
        assert service.stats["rom_reads"] == (k + 1) * reads
    service.release()
    with pytest.raises(RuntimeError):
        service.issue(0)
    with pytest.raises(RuntimeError):
        service.release()


def test_gain_startup_overlap_stall_and_registered_acceptance_do_not_recompute(monkeypatch):
    state = WordNormalizationState(fit(19191, 373737), NormalizationFormat())
    service = SymbolNormalizationService(state)
    word = raw_word(117, -213, 4)
    assert service.issue(word)
    while service.peek() is None:
        service.step()
    assert service.cycle == 46  # Includes real gain startup, not33+24 or zero.
    owner = state.snapshot()
    for _ in range(89):
        service.step()
        assert state.snapshot() == owner and not service.can_issue
    exported = service.peek()
    exported["stats"]["accepted"] = 91
    exported["normalized_word"] = 0
    monkeypatch.setattr(state, "prepare_result", lambda *_: pytest.fail("accept recomputed NCO"))
    result = service.take()
    assert result["normalized_word"] != 0 and state.count == state.stats["accepted"] == 1
    assert service.stats["output_stall_edges"] == 89


@pytest.mark.parametrize("edges", [0, 4, 17, 34, 44])
def test_abort_preserves_owner_and_closes_one_bounded_context(edges):
    state = WordNormalizationState(fit(), NormalizationFormat())
    service = SymbolNormalizationService(state)
    assert service.issue(raw_word(11, 3, -7))
    for _ in range(edges):
        service.step()
    service.abort()
    assert service.mode == "aborted" and service.peek() is None and service.job is None
    assert state.count == 0 and state.phase == 0
    assert service.stats["aborted"] == 1
    service.abort()
    assert service.stats["aborted"] == 1


def test_counter_exhaustion_and_invalid_requests_cannot_consume_registered_output():
    state = WordNormalizationState(fit(), NormalizationFormat())
    service = SymbolNormalizationService(state)
    before = state.snapshot()
    with pytest.raises(ValueError):
        service.issue(True)
    assert service.job is None and state.snapshot() == before
    assert service.issue(raw_word(1, 0, 0))
    while service.peek() is None:
        service.step()
    expected = service.peek()
    service.stats["completed"] = (1 << 64) - 1
    before = state.snapshot()
    with pytest.raises(OverflowError):
        service.take()
    assert service.peek() == expected and state.snapshot() == before
    service.stats["completed"] = 0
    assert service.take() == expected
    service.cycle = (1 << 64) - 1
    before = state.snapshot()
    with pytest.raises(OverflowError):
        service.step()
    assert state.snapshot() == before
    assert service.estimate()["state_register_bits_upper"] == 3469
    assert not service.estimate()["complete_receiver16MSps"]
