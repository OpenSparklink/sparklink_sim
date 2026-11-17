"""Independent rational score, exact words, finite-bank reuse and resource drops."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.spline_fixed import FixedUniformCubicEighth, SplineFixedFormat
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SplineScoreFlow,
    SyncSignTemplate,
    spline_score_cost,
)


def inputs(n=40):
    rng = np.random.default_rng(670117)
    return [
        (int(r), int(i), int(e))
        for r, i, e in zip(
            rng.integers(-(2**18), 2**18, n),
            rng.integers(-(2**18), 2**18, n),
            rng.integers(-10, 10, n),
            strict=True,
        )
    ]


def oracle(words, template, peaks, minimum=0, lookahead=0, threshold=(17, 20)):
    """Direct Fraction spline weights and arbitrary-precision rational scores."""
    ref = FixedUniformCubicEighth.from_rrc_words(
        *zip(*words, strict=True), SplineFixedFormat(18, 14)
    )
    candidates, samples = [], {}
    count = len(template.words) + 1
    for start in sorted(
        {8 * p + d for p in peaks for d in range(-4, 5)}, key=lambda t: (t % 32, t // 32)
    ):
        if start < minimum or start + 32 * (count - 1) >= 8 * (len(words) - 1):
            continue
        y = []
        for k in range(count):
            j, m = divmod(start + 32 * k, 8)
            if m == 0:
                word = ref.source_words[j]
            else:
                z = Fraction(m, 8)
                weights = (1 - z, z, ((1 - z) ** 3 - (1 - z)) / 6, (z**3 - z) / 6)
                values = (
                    ref.source_words[j],
                    ref.source_words[j + 1],
                    ref.second_words[j],
                    ref.second_words[j + 1],
                )
                word = tuple(
                    round(sum(w * v[c] for w, v in zip(weights, values, strict=True)))
                    for c in (0, 1)
                )
            y.append(word)
        cr = ci = energy = 0
        for (a, b), (r, i), (sr, si) in zip(y[1:], y[:-1], template.words, strict=True):
            dr, di = a * r + b * i, b * r - a * i
            cr += dr * sr + di * si
            ci += di * sr - dr * si
            energy += dr * dr + di * di
        num, den = cr * cr + ci * ci, energy * template.energy
        candidates.append(
            dict(
                start_tick=start,
                correlation=(cr, ci),
                numerator=num,
                energy=energy,
                denominator=den,
                zero_energy=not energy,
                threshold_pass=Fraction(num, den) >= Fraction(*threshold) ** 2
                if den
                else threshold[0] == 0,
                disposition="maximum_needs_future"
                if start + 32 * (count - 1) + lookahead > 8 * (len(words) - 1)
                else "ready",
            )
        )
        samples[start] = y
    best = (
        max(
            candidates,
            key=lambda r: (
                Fraction(r["numerator"], r["denominator"]) if r["denominator"] else Fraction(0)
            ),
        )
        if candidates
        else None
    )
    return ref, best, samples


@pytest.mark.parametrize("reuse,read", [(False, 1), (True, 1), (True, 2)])
def test_literal_words_scores_rank_and_actual_bank_replies(reuse, read):
    words = inputs()
    tpl = SyncSignTemplate(((1, 1), (-1, 1), (1, -1), (-1, -1)))
    expected, best, samples = oracle(words, tpl, [2, 7])
    events = []
    result = SplineScoreFlow(SplineScoreAssumptions(40, reuse, read)).run(
        words, tpl, [2, 7], sink=events.append
    )
    assert result["best"] == best and not result["aborted"]
    assert result["prepared_source_words"] == expected.source_words
    assert result["prepared_second_words"] == expected.second_words
    actual = [e["word"] for e in events if e["event"] == "selected_sample"]
    assert actual == samples[best["start_tick"]]
    assert result["cache_peak"] <= (6 if reuse else 4) and result["pending_read_peak"] <= 4
    assert result["selected_vectors_stored"] == 0 and not result["whole_rx_verified"]
    issues = {}
    for e in events:
        if e["event"] == "read_issue":
            key = e["cycle"], e["bank"]
            assert key not in issues
            issues[key] = True
    assert all(
        e["a"] < 2**24 and e["b"] < 2**17
        for e in events
        if e["event"] == "op_issue" and e["code"] == "tile"
    )


def test_reuse_only_changes_memory_work_and_cycles_not_scoring():
    tpl, words = SyncSignTemplate(((0, 1), (0, -1), (0, 1))), inputs(32)
    old = SplineScoreFlow(SplineScoreAssumptions(32, False)).run(words, tpl, [2])
    new = SplineScoreFlow(SplineScoreAssumptions(32)).run(words, tpl, [2])
    assert old["best"] == new["best"] and old["operation_counts"] == new["operation_counts"]
    assert new["scoring_reads"] == 40 < old["scoring_reads"] == 148
    assert new["scoring_cycles"] < old["scoring_cycles"]
    # All products were25x18-sized primitives, not a free full-width host product.
    assert new["operation_counts"]["tile"] > 1000


def test_zero_energy_tie_threshold_and_future_wait_without_score_retry():
    words = [(0, 0, 0)] * 32
    tpl = SyncSignTemplate(((0, 1),) * 3)
    result = SplineScoreFlow(SplineScoreAssumptions(32)).run(
        words, tpl, [2, 7], lookahead_ticks=240
    )
    assert result["best"]["zero_energy"] and not result["best"]["threshold_pass"]
    assert (
        result["best"]["start_tick"]
        == oracle(words, tpl, [2, 7], lookahead=240)[1]["start_tick"]
        == 12
        and result["best"]["disposition"] == "maximum_needs_future"
    )
    assert SplineScoreFlow().run(words, tpl, [], threshold=(0, 1))["best"] is None
    assert SplineScoreFlow().run(words, tpl, [2], threshold=(0, 1))["best"]["threshold_pass"]


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_actual_profile_sync_alphabet_and_frozen_reference_float_score(ft):
    cfg = TxConfig(
        frame_type=ft,
        ctrl_bits_len=28 if ft == 2 else 27,
        head_llid=None if ft == 2 else 0x234567,
        mcs_index=8,
        symbol_rate_mhz=4,
    )
    ref = PSKReceiveProfile.ordinary(cfg).sync_symbols()
    tpl = SyncSignTemplate.from_symbols(ref)
    assert tpl.energy == (2 if ft == 2 else 1) * (len(ref) - 1)
    words = inputs(640)
    result = SplineScoreFlow().run(words, tpl, [2])
    fixed = FixedUniformCubicEighth.from_rrc_words(
        *zip(*words, strict=True), SplineFixedFormat(18, 14)
    )
    best = result["best"]
    y = fixed(best["start_tick"] / 8 + 4 * np.arange(len(ref)))
    dy, dr = y[1:] * y[:-1].conj(), ref[1:] * ref[:-1].conj()
    value = abs(np.sum(dy * dr.conj())) ** 2 / (np.sum(abs(dy) ** 2) * np.sum(abs(dr) ** 2))
    assert abs(value - best["numerator"] / best["denominator"]) < 2e-15


def test_explicit_resource_drop_and_validation_reject_without_fallback():
    f = SplineScoreFlow(SplineScoreAssumptions(32))
    words, tpl = inputs(32), SyncSignTemplate(((0, 1),) * 3)
    assert f.run(words, tpl, [2], max_cycles=1)["reason"] == "preparation_cycle_limit"
    result = f.run(words, tpl, [2], max_cycles=250)
    assert result["aborted"] and result["reason"] == "cycle_limit" and result["best"] is None
    for kwargs in (
        dict(peaks=[1, 2]),
        dict(peaks=[-1]),
        dict(peaks=[True]),
        dict(threshold=(2, 1)),
        dict(threshold=(0, 0)),
        dict(threshold=[1, 2]),
        dict(max_cycles=True),
        dict(minimum_tick=-1),
        dict(lookahead_ticks=10**100),
    ):
        args = dict(peaks=[2], **{k: v for k, v in kwargs.items() if k != "peaks"})
        args["peaks"] = kwargs.get("peaks", [2])
        with pytest.raises(ValueError):
            f.run(words, tpl, **args)
    with pytest.raises(ValueError):
        f.run([(True, 0, 0)] * 4, tpl, [2])
    with pytest.raises(ValueError):
        f.run(words, None, [2])
    with pytest.raises(ValueError):
        SplineScoreFlow(True)
    for kwargs in (
        dict(capacity=641),
        dict(reuse_stencil=1),
        dict(read_latency=3),
        dict(multiplier_latency=True),
        dict(alu_segment_bits=64),
    ):
        with pytest.raises(ValueError):
            SplineScoreAssumptions(**kwargs)
    for template in ((), ((0, 0),), ((1, 1), (0, 1)), ((True, 0),)):
        with pytest.raises(ValueError):
            SyncSignTemplate(template)
    for symbols in ([], [0, 0], [1, 2], [1, np.nan], [1, np.exp(0.1j)]):
        with pytest.raises(ValueError):
            SyncSignTemplate.from_symbols(symbols)
    assert spline_score_cost()["additional_query_state_upper_bits"] > 1000
    with pytest.raises(ValueError):
        spline_score_cost(True)


def test_aligned_boundary_output_stall_and_exact_threshold_boundary():
    raw, tpl = inputs(32), SyncSignTemplate(((0, 1),) * 3)
    source = FixedUniformCubicEighth.from_rrc_words(
        *zip(*raw, strict=True), SplineFixedFormat(18, 14)
    )
    events = []
    flow = SplineScoreFlow(SplineScoreAssumptions(32))
    base = flow.run(source.source_words, tpl, [2], exponent=source.exponent, sink=events.append)
    first = next(e["cycle"] for e in events if e["event"] == "selected_sample")
    blocked = flow.run(
        source.source_words,
        tpl,
        [2],
        exponent=source.exponent,
        output_stalls=((first, first + 20),),
    )
    assert blocked["best"] == base["best"] and blocked["cycles"] == base["cycles"] + 20
    assert blocked["output_blocked_cycles"] == 20
    stopped = flow.run(
        source.source_words,
        tpl,
        [2],
        exponent=source.exponent,
        output_stalls=((first, first + 20),),
        max_cycles=first + 5,
    )
    assert stopped["aborted"]
    # Constant nonzero symbols with constant differential template have Q=1 exactly.
    constant = [(100, 0, 0)] * 32
    one = SyncSignTemplate(((1, 0),) * 3)
    exact = flow.run(constant, one, [2], threshold=(1, 1))
    assert exact["best"]["numerator"] == exact["best"]["denominator"]
    assert exact["best"]["threshold_pass"]
    for kwargs in (
        dict(exponent=True),
        dict(output_stalls=None),
        dict(output_stalls=[(3, 1)]),
        dict(output_stalls=[(0, 4), (3, 5)]),
    ):
        with pytest.raises(ValueError):
            flow.run(source.source_words, tpl, [2], **kwargs)


@pytest.mark.parametrize("aligned", [False, True])
def test_preparation_saturation_rejects_before_any_score_or_selected_symbol(aligned):
    frozen = json.loads((Path(__file__).parent / "data/spline_score_saturation.json").read_text())
    raw = tuple(tuple(w) for w in frozen["words"])
    words = tuple(w[:2] for w in raw) if aligned else raw
    events = []
    flow = SplineScoreFlow(SplineScoreAssumptions(8))
    result = flow.run(
        words, SyncSignTemplate(((1, 0),)), [1], exponent=0 if aligned else None, sink=events.append
    )
    assert result["aborted"] and result["reason"] == "preparation_saturated"
    assert result["numerical_rejection"] and result["numerical_pass"] is False
    assert not result["resource_drop"] and result["best"] is None
    assert result["preparation_arithmetic"]["saturations"] == frozen["reference_saturations"]
    assert result["preparation_arithmetic"]["saturation_by_node"] == {"second_rhs": 2}
    assert result["scoring_reads"] == result["scoring_cycles"] == result["scores_evaluated"] == 0
    assert [e["event"] for e in events] == ["numerical_reject"]
    # Keep neighboring no-overflow inputs usable; do not reject every window.
    safe = tuple((w[0] // 2, w[1] // 2) for w in raw)
    accepted = flow.run(safe, SyncSignTemplate(((1, 0),)), [1], exponent=1)
    assert not accepted["aborted"] and accepted["numerical_pass"]
    assert accepted["preparation_arithmetic"]["saturations"] == 0 and accepted["best"] is not None
