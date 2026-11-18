"""Full-growth identity against direct difference-square and Fraction references."""

import json
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SplineScoreFlow,
    SyncSignTemplate,
    _symbol_power,
    spline_score_cost,
)
from tests.test_spline_score import oracle


def node_cases(validation=False):
    """New node vectors vary geometry,template,signal construction and exponents.

    These are prepared-node experiments,not newly claimed RF reception trials.
    """
    rng = np.random.default_rng(730219 if validation else 730117)
    for k in range(8):
        ft = (2, 3, 4)[(k + int(validation)) % 3]
        n = ({2: 272, 3: 392, 4: 640} if validation else {2: 256, 3: 376, 4: 632})[ft]
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=8,
            symbol_rate_mhz=4,
            pid=4 if validation else 1,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
        )
        tpl = SyncSignTemplate.from_symbols(PSKReceiveProfile.ordinary(cfg).sync_symbols())
        x = np.arange(n, dtype=np.int64)
        r, i = rng.integers(-60000, 60001, size=(2, n))
        exponent = np.full(n, -6 if validation else -10, dtype=np.int64)
        if k == 0:
            r[:] = i[:] = 0
        elif k == 1:
            r[:] = 57777 if validation else -41319
            i[:] = -11223 if validation else 13337
        elif k == 2:
            r = ((x * (701 if validation else 503)) % 100003) - 50001
            i = np.roll(r, 7 if validation else 3)
        elif k == 3:
            r[:] = i[:] = 0
            r[n // 3 : n // 3 + (9 if validation else 5)] = 131000
            i[n // 2] = -130000
        elif k == 4:
            r = np.where(x % (3 if validation else 2), 119999, -120000)
            i = np.roll(r, 1)
        elif k == 5:
            exponent = np.where(x < n // (3 if validation else 2), -31, 7)
        elif k == 6:
            r = np.rint(35000 * np.cos(x * (0.107 if validation else 0.071))).astype(np.int64)
            i = np.rint(35000 * np.sin(x * (0.107 if validation else 0.071))).astype(np.int64)
        else:
            exponent = rng.integers(-40, 11, n)
        words = [(int(a), int(b), int(e)) for a, b, e in zip(r, i, exponent, strict=True)]
        yield (
            dict(id=k, frame_type=ft, n=n, pattern=k, validation=validation),
            words,
            tpl,
            ((4, 12) if k % 2 else (6,)),
            dict(minimum_tick=(25 if validation else 9), lookahead_ticks=160 if k % 3 == 0 else 0),
        )


@pytest.mark.parametrize("k", [0, 1, 3, 5])
def test_all_scores_samples_and_peak_selection_match_independent_fraction(k):
    meta, words, tpl, peaks, kw = list(node_cases())[k]
    _ref, best, samples = oracle(words, tpl, peaks, kw["minimum_tick"], kw["lookahead_ticks"])
    baseline = SplineScoreFlow(SplineScoreAssumptions(meta["n"])).run(words, tpl, peaks, **kw)
    events = []
    wanted = {"sample", "score", "selected_sample"}
    flow = SplineScoreFlow(SplineScoreAssumptions(meta["n"], reuse_symbol_power=True))
    reused = flow.run(
        words, tpl, peaks, **kw, sink=lambda e: events.append(e) if e["event"] in wanted else None
    )
    assert reused["best"] == baseline["best"] == best
    assert reused["prepared_source_words"] == baseline["prepared_source_words"]
    assert reused["prepared_second_words"] == baseline["prepared_second_words"]
    assert [e["word"] for e in events if e["event"] == "selected_sample"] == samples[
        best["start_tick"]
    ]
    for event in events:
        if event["event"] == "score":
            y = samples[event["start_tick"]]
            # Independent direct expansion retains the original two squared differences.
            energy = sum(
                (a * r + b * i) ** 2 + (b * r - a * i) ** 2
                for (a, b), (r, i) in zip(y[1:], y[:-1], strict=True)
            )
            assert energy == event["energy"]
    assert (
        reused["cycles"] < baseline["cycles"]
        and reused["scoring_reads"] == baseline["scoring_reads"]
    )
    assert set(reused["declared_node_bits"]) <= flow.monitor_tags
    assert reused["operation_counts"]["tile"] < baseline["operation_counts"]["tile"]


def test_symbol_power_signed22_extremes_fit_literal_primitives_and_unsigned44():
    # Test the actual generator,including every finite intermediate bound.
    for word in [(-(2**21), -(2**21)), (2**21 - 1, -(2**21)), (0, 0), (-1, 1)]:
        program = _symbol_power(word)
        op = next(program)
        while True:
            a, b = op["a"], op["b"]
            if op["code"] == "tile":
                assert 0 <= a < 2**24 and 0 <= b < 2**17
                result = a * b
            else:
                assert op["code"] in ("abs", "add", "mov", "neg")
                result = {"abs": abs(a), "add": a + b, "mov": a, "neg": -a}[op["code"]]
            bits = op["bits"]
            assert (
                -(2 ** (bits - 1)) <= result < 2 ** (bits - 1)
                if op["signed"]
                else 0 <= result < 2**bits
            )
            try:
                op = program.send(result)
            except StopIteration as done:
                assert done.value == word[0] ** 2 + word[1] ** 2 < 2**44
                break


def test_complete_monitor_reservations_and_power_state_are_charged():
    old, new = (
        spline_score_cost(SplineScoreAssumptions(reuse_symbol_power=v)) for v in (False, True)
    )
    assert old["peak_bit_monitor_slots"] == 66 and new["peak_bit_monitor_slots"] == 71
    assert old["additional_query_state_upper_bits"] == 6657
    assert (
        new["additional_query_state_upper_bits"] - old["additional_query_state_upper_bits"] == 444
    )
    assert new["peak_bit_monitor_slots"] <= new["peak_bit_monitor_reserved_slots"] == 80
    assert new["peak_bit_monitor_slot_bits"] == 8
    assert new["shared_DSP25x18_count"] == old["shared_DSP25x18_count"] == 1
    assert old["shared_preparation_banks"] == new["shared_preparation_banks"]
    with pytest.raises(ValueError):
        SplineScoreAssumptions(reuse_symbol_power=1)


@pytest.mark.parametrize("aligned", [False, True])
def test_power_reuse_keeps_preparation_rejection_and_zero_resource_horizon(aligned):
    frozen = json.loads((Path(__file__).parent / "data/spline_score_saturation.json").read_text())
    raw = [tuple(w) for w in frozen["words"]]
    words = [w[:2] for w in raw] if aligned else raw
    kw = dict(exponent=0) if aligned else {}
    tpl = SyncSignTemplate(((1, 0),))
    old = SplineScoreFlow(SplineScoreAssumptions(8)).run(words, tpl, [1], **kw)
    new = SplineScoreFlow(SplineScoreAssumptions(8, reuse_symbol_power=True))
    assert new.run(words, tpl, [1], **kw) == old
    assert old["numerical_rejection"] and not old["resource_drop"]
    assert new.run(words, tpl, [1], max_cycles=1, **kw)["resource_drop"]


def test_output_stalls_future_wait_ties_and_early_abort_remain_explicit():
    words, tpl = [(91, -47, 0)] * 32, SyncSignTemplate(((1, 0),) * 3)
    flow = SplineScoreFlow(SplineScoreAssumptions(32, reuse_symbol_power=True))
    events = []
    result = flow.run(words, tpl, [2, 7], lookahead_ticks=240, threshold=(1, 1), sink=events.append)
    assert (
        result["best"]["threshold_pass"] and result["best"]["disposition"] == "maximum_needs_future"
    )
    first = next(e["cycle"] for e in events if e["event"] == "selected_sample")
    blocked = flow.run(
        words,
        tpl,
        [2, 7],
        lookahead_ticks=240,
        threshold=(1, 1),
        output_stalls=((first, first + 10),),
    )
    assert blocked["best"] == result["best"] and blocked["cycles"] == result["cycles"] + 10
    aborted = flow.run(words, tpl, [2, 7], max_cycles=first)
    assert aborted["resource_drop"] and aborted["best"] is None


def test_node_audit_preserves_numerical_rejection_without_calling_it_checked(tmp_path, monkeypatch):
    import importlib

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    driver = importlib.import_module("model_spline_power")
    frozen = json.loads((Path(__file__).parent / "data/spline_score_saturation.json").read_text())
    words = [tuple(w) for w in frozen["words"]]
    row, results = driver.run_node(
        tmp_path, "reject", words, SyncSignTemplate(((1, 0),)), [1], {}, independent=True
    )
    assert row["numerical_rejection"]
    assert not row["independent_fraction_and_difference_square_checked"]
    assert results["baseline"] == results["reused"]
    assert results["reused"]["numerical_pass"] is False
