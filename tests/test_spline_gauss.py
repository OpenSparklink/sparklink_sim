"""Conjugate Gauss identities on actual typed words,including minimum negatives."""

from itertools import product

import numpy as np
import pytest

from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_transport import CaptureRequest
from nearlink_sdr.sim.score_word_runtime import ScoreWordRuntime
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SplineScoreFlow,
    SyncSignTemplate,
    _gauss_difference,
    _monitor_tags,
)
from nearlink_sdr.sim.word_arithmetic import WordFabric, WordFabricAssumptions
from tests.test_spline_score import inputs, oracle


def test_real_word_fabric_conjugate_identity_extremes_and_exhaustive_small_domain():
    fabric = WordFabric()
    issued = []

    def completed(op, value, overflow):
        assert not overflow

    runtime = ScoreWordRuntime(
        fabric, fabric.tick, lambda op, _: issued.append(op), completed,
        _monitor_tags(True, False, True),
    )
    rng = np.random.default_rng(421029)
    cases = [*product((-2**21, 2**21 - 1), repeat=4), *product(range(-2, 2), repeat=4)]
    cases += [tuple(map(int, row)) for row in rng.integers(-2**21, 2**21, (64, 4))]
    for a, b, c, d in cases:
        generator = _gauss_difference((a, b), (c, d))
        start = len(issued)
        op = next(generator)
        while True:
            value = runtime.execute(op)
            try:
                op = generator.send(value)
            except StopIteration as result:
                assert result.value == (a * c + b * d, b * c - a * d)
                break
        assert sum(op["code"] == "tile" for op in issued[start:]) == 6
        assert sum(op["tag"] in ("gauss_cd", "gauss_ab", "gauss_ba")
                   for op in issued[start:]) == 3
    assert not any(fabric.busy) and not any(fabric.mailboxes)


@pytest.mark.parametrize("fabric", [None, 1, 2, 4])
@pytest.mark.parametrize("reuse_power", [False, True])
def test_all_candidate_words_scores_ties_and_fraction_reference_exact(fabric, reuse_power):
    words = inputs(48)
    template = SyncSignTemplate(((1, 1), (-1, 1), (1, -1), (-1, -1)))
    a = SplineScoreAssumptions(48, reuse_symbol_power=reuse_power)
    arms = []
    for gauss in (False, True):
        trace = []

        def sink(event):
            if event["event"] in ("sample", "score", "selected_sample"):
                trace.append({k: v for k, v in event.items() if k != "cycle"})

        flow = SplineScoreFlow(
            a, gauss_difference=gauss,
            word_fabric=None if fabric is None else WordFabricAssumptions(fabric),
        )
        result = flow.run(words, template, (2, 8), sink=sink)
        assert result["numerical_pass"] and not result["resource_drop"]
        arms.append((result, trace))
    (old, before), (new, after) = arms
    assert old["best"] == new["best"] == oracle(words, template, (2, 8))[1]
    assert before == after and old["scoring_reads"] == new["scoring_reads"]
    for key in ("prepared_source_words", "prepared_second_words", "common_exponent"):
        assert old[key] == new[key]
    assert new["operation_counts"]["tile"] < old["operation_counts"]["tile"]


@pytest.mark.parametrize("gauss", [False, True])
def test_zero_threshold_future_wait_and_empty_keep_existing_semantics(gauss):
    flow = SplineScoreFlow(
        SplineScoreAssumptions(32, reuse_symbol_power=True, defer_all_future=True),
        word_fabric=WordFabricAssumptions(2), gauss_difference=gauss,
    )
    words, template = [(0, 0, 0)] * 32, SyncSignTemplate(((0, 1),) * 3)
    result = flow.run(words, template, (2,), threshold=(0, 1))
    assert result["best"]["threshold_pass"] and result["best"]["zero_energy"]
    wait = flow.run(words, template, (2,), lookahead_ticks=32768)
    assert wait["geometry_wait"] and wait["numerical_pass"] is None
    assert wait["scoring_cycles"] < result["scoring_cycles"]
    empty = flow.run(words, template, ())
    assert empty["best"] is None and empty["numerical_pass"] is None
    interrupted = flow.run(words, template, (2,), max_cycles=500)
    assert interrupted["resource_drop"] and interrupted["best"] is None
    assert interrupted["numerical_pass"] is None


def test_explicit_mode_and_added_live_intermediates_are_charged():
    a = SplineScoreAssumptions(632, reuse_symbol_power=True, defer_all_future=True)
    for fabric, contexts in ((None, 1), (WordFabricAssumptions(2), 9)):
        old = SplineScoreFlow(a, word_fabric=fabric).estimate()
        new = SplineScoreFlow(a, word_fabric=fabric, gauss_difference=True).estimate()
        assert "gauss_difference" not in old
        extra = contexts * 207
        assert new["additional_query_state_upper_bits"] - old[
            "additional_query_state_upper_bits"
        ] == extra
        assert new["gauss_difference"]["incremental_register_bits"] == extra <= 2048
        assert new["shared_DSP25x18_count"] == old["shared_DSP25x18_count"] == 1
        assert new["shared_preparation_banks"] == old["shared_preparation_banks"]
        assert new["peak_bit_monitor_slots"] <= 80
        if fabric is not None:
            assert new["additional_word_and_context_bits"] == 22799 <= 24576
    with pytest.raises(ValueError, match="Gauss"):
        SplineScoreFlow(gauss_difference=1)


def test_existing_capture_worker_explicit_port_uses_same_source_image():
    words = inputs(256)
    # A literal actual FT2 template,not an arbitrary assumed sync alphabet.
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    symbols = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols()
    template = SyncSignTemplate.from_symbols(symbols)
    source = [pack_filter_word(*word) for word in words]
    request = CaptureRequest(256, 0, (2,), 8, 128)
    event = dict(geometry=request, window_samples=256, after_samples=256, path=0)
    outputs = []
    for gauss in (False, True):
        worker = CaptureFineWorker(
            template, 256, confirmation_required=True, reuse_symbol_power=True,
            word_fabric=WordFabricAssumptions(2), gauss_difference=gauss,
        )
        outputs.append(worker(event, source))
    assert outputs[0].disposition == outputs[1].disposition
    assert outputs[0].start_tick == outputs[1].start_tick
