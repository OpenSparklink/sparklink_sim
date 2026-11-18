"""Actual integer values traverse bounded fit stages;no prerecorded answer schedule."""

import numpy as np
import pytest

from nearlink_sdr.phy.sync_fit_word import SyncFitFormat, WordSyncFit, compile_sync_phases
from nearlink_sdr.sim.sync_fit_flow import SyncFitWordFlow
from tests.test_selected_rx import ordinary
from tests.test_sync_fit_word import independent_input, packed


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("peak", [30, 34])
@pytest.mark.parametrize("gaps", [False, True])
def test_value_pipeline_matches_independent_word_model_with_bounded_credits(ft, peak, gaps):
    codes = compile_sync_phases(ordinary(ft, 8, 0).sync_symbols())
    n = len(codes)
    t = np.arange(n)
    rng = np.random.default_rng(949121 + ft)
    values = (0.8 + rng.random(n)) * np.exp(1j * (np.array(codes) * np.pi / 8 + 0.17 + 0.013 * t))
    words, _ = independent_input(values)
    fmt = SyncFitFormat(peak)
    reference = WordSyncFit(codes, 123456789, fmt)
    for w in words:
        reference.feed_word(w)
    expected = reference.finish()
    flow = SyncFitWordFlow(codes, 123456789, fmt)
    offered = 0
    accepted = []
    while not flow.ready:
        if offered < n and (not gaps or flow.cycle % 7 == 0) and flow.issue(words[offered]):
            accepted.append(flow.cycle)
            offered += 1
        flow.advance()
        assert flow.cycle < 40 * n + 1200
        assert (flow.job is not None) == (flow.mode == "word")
    assert offered == n and flow.result == expected
    assert flow.ready_cycle == flow.cycle
    if not gaps:
        assert accepted == list(range(0, 40 * n, 40))
        assert 40 * n < flow.cycle < 40 * n + 300
    assert flow.stats["native_products"] == 4 * n + 18
    assert flow.stats["cordic_iterations"] == 32 * n and flow.stats["divider_steps"] == 128
    for _ in range(23):
        assert not flow.issue(words[0])
        flow.advance()
        assert flow.result == expected
    assert flow.stats["output_stall_cycles"] == 23
    assert flow.take() == expected and flow.take() is None
    cost = flow.estimate()
    assert cost["register_state_upper_bits"] <= 4096
    assert cost["logical_template_and_constant_bits"] <= 2560
    assert cost["additional_DSP25x18"] == 1 and cost["workspaces_mutually_exclusive"]
    assert (
        not cost["full_ADC_or_RX16MSps"] and cost["known_sync_hold_PI_head_decoder_cycles"] is None
    )


def test_flow_full_exponent_range_zero_and_abort_are_not_hidden():
    words = [packed(131071, -12345, int(e)) for e in np.resize([-128, 0, 127], 32)]
    reference = WordSyncFit([0] * 32, 0)
    flow = SyncFitWordFlow([0] * 32, 0)
    offered = 0
    while not flow.ready:
        if offered < len(words) and flow.issue(words[offered]):
            reference.feed_word(words[offered])
            offered += 1
        flow.advance()
    assert flow.take() == reference.finish()
    assert reference.power.erased and reference.magnitude.erased
    flow = SyncFitWordFlow([0] * 32, 0)
    assert flow.issue(0)
    for _ in range(7):
        flow.advance()
    detail = flow.abort()
    assert detail["owned_input"] == 1
    assert not flow.ready and flow.take() is None and flow.abort() is None
    with pytest.raises(RuntimeError):
        flow.advance()
    with pytest.raises(RuntimeError):
        flow.issue(0)


def test_context_and_counter_overflow_cannot_turn_into_success():
    flow = SyncFitWordFlow([0] * 32, 0)
    for w in [-1, 2**52, True]:
        with pytest.raises(ValueError):
            flow.issue(w)
    flow.cycle = 2**63 - 1
    with pytest.raises(OverflowError):
        flow.advance()
    flow = SyncFitWordFlow([0] * 32, 0)
    flow.stats["accepted_words"] = 2**64 - 1
    with pytest.raises(OverflowError):
        flow.issue(0)
