"""Actual shared-bank fine scoring, finite contexts and no numerical shortcut."""

import pytest

from nearlink_sdr.sim.score_word_runtime import ScoreWordRuntime, score_word_command
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabric, WordFabricAssumptions
from tests.test_spline_score import inputs, oracle


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_real_fine_words_all_scores_selection_and_normalized_ranking(lanes):
    words, tpl = inputs(40), SyncSignTemplate(((1, 1), (-1, 1), (1, -1), (-1, -1)))
    a = SplineScoreAssumptions(40, reuse_symbol_power=True, defer_all_future=True)
    traces = []
    results = []
    for fabric in (None, WordFabricAssumptions(lanes)):
        events = []
        result = SplineScoreFlow(a, word_fabric=fabric).run(words, tpl, [2, 7], sink=events.append)
        results.append(result)
        traces.append(
            [{k: v for k, v in e.items() if k != "cycle"} for e in events
             if e["event"] in ("sample", "score", "selected_sample")]
        )
        # Every actual memory issue remains port-limited; no extra Y/M copy.
        ports = [(e["cycle"], e["bank"]) for e in events if e["event"] == "read_issue"]
        assert len(ports) == len(set(ports))
    old, new = results
    assert old["best"] == new["best"] == oracle(words, tpl, [2, 7])[1]
    assert traces[0] == traces[1] and old["operation_counts"] == new["operation_counts"]
    for key in ("prepared_source_words", "prepared_second_words", "observed_node_bits"):
        assert old[key] == new[key]
    assert new["word_context_peak"] == 9 and new["word_fabric_idle"]
    assert new["word_fabric_statistics"]["issued"] == new["word_fabric_statistics"]["taken"]
    assert new["numerical_pass"] and not new["whole_rx_verified"]
    cost = SplineScoreFlow(a, word_fabric=WordFabricAssumptions(lanes)).estimate()
    assert cost["additional_word_and_context_bits"] <= 24576
    assert cost["legacy_scratch_not_subtracted"] and cost["full_window_copies_added"] == 0


def test_geometry_wait_empty_zero_and_finite_cycle_drop():
    f = SplineScoreFlow(
        SplineScoreAssumptions(32, reuse_symbol_power=True, defer_all_future=True),
        word_fabric=WordFabricAssumptions(2),
    )
    words, tpl = [(0, 0, 0)] * 32, SyncSignTemplate(((0, 1),) * 3)
    wait = f.run(words, tpl, [2], lookahead_ticks=32768)
    assert wait["geometry_wait"] and wait["numerical_pass"] is None and wait["scoring_reads"] == 0
    empty = f.run(words, tpl, [])
    assert empty["best"] is None and empty["numerical_pass"] is None
    good = f.run(words, tpl, [2], threshold=(0, 1), output_stalls=((7000, 7100),))
    assert good["best"]["threshold_pass"] and good["best"]["zero_energy"]
    drop = f.run(words, tpl, [2], max_cycles=500)
    assert drop["resource_drop"] and drop["best"] is None and drop["numerical_pass"] is None
    for config in (True, WordFabricAssumptions(contexts=8)):
        with pytest.raises(ValueError):
            SplineScoreFlow(word_fabric=config)
    with pytest.raises(ValueError):
        SplineScoreFlow(SplineScoreAssumptions(reuse_stencil=False), word_fabric=WordFabricAssumptions())


def test_static_signed_formats_and_runtime_overflow_are_not_success():
    base = dict(code="abs", a=-7, b=0, bits=22, signed=False, tag="difference_rr.abs_a")
    assert score_word_command(base, 0).a_signed
    with pytest.raises(ValueError):
        score_word_command(base | dict(tag="rank_left.abs_a"), 0)
    f = WordFabric()
    emitted = []

    def completed(command, value, overflow):
        emitted.append((value, overflow))
        if overflow:
            raise OverflowError("kept")

    runtime = ScoreWordRuntime(f, f.tick, lambda *a: None, completed, {"test"})
    with pytest.raises(ValueError):
        runtime.fork(tuple(range(10)))
    with pytest.raises(OverflowError, match="kept"):
        runtime.execute(dict(code="add", a=7, b=1, bits=4, signed=True, tag="test"))
    assert emitted == [(8, True)]
