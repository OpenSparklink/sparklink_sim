"""Independent rational Q12 oracle,registered credit and bounded serial formats."""

from collections import deque
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat
from nearlink_sdr.sim.rank_serial import Q12Serial
from nearlink_sdr.sim.scaled_native import FoldResult, Q12Pipeline, ScorePipeline


@pytest.mark.parametrize("terms,coefficient", [(31, 62), (61, 61), (125, 125)])
@pytest.mark.parametrize("digit_bits", [1, 2])
def test_serial_exact_ratio_boundaries_and_full_width_ports(terms, coefficient, digit_bits):
    widths = FixedBankFormat(10).widths(terms, coefficient, 0.6)
    node = Q12Serial(widths, terms, coefficient, digit_bits=digit_bits)
    rng = np.random.default_rng(821537 + terms)
    maximum = (1 << widths["window_energy"]) - 1
    energies = [0, 1, 2, maximum, maximum // 2]
    energies += [int(v) for v in rng.integers(1, maximum, 43)]
    count, subtracts = 0, 0
    for energy in energies:
        denominator = energy * coefficient
        numerators = [0, 1, denominator, denominator + 1, (1 << widths["numerator"]) - 1]
        for q in [1, 7, 2047, 2048, 4095]:
            boundary = (q * denominator) // 4096
            numerators += [max(0, boundary - 1), boundary, boundary + 1]
        for numerator in numerators:
            source = FoldResult(count, 0, 0, energy, count % 127)
            assert node.offer(source, numerator, denominator, bool(count % 2))
            before = deepcopy((node.cell, node.phase, node.digits, node.stats))
            assert not node.offer(source, numerator, denominator, False)
            assert (node.cell, node.phase, node.digits, node.stats) == before
            for edge in range(node.iterations + 1):
                assert not node.will_emit
                assert node.step() is None
                assert node.will_emit == (edge == node.iterations)
                if node.cell["mode"] == 0 and node.phase == 2:
                    assert 0 <= node.cell["remainder"] < denominator
            out = node.step()
            expected = (
                0 if not denominator else min(4096, int(Fraction(numerator << 12, denominator)))
            )
            assert out["rank"] == expected
            assert out["source"] == source and out["hit"] == bool(count % 2)
            assert (out["numerator"], out["denominator"]) == (numerator, denominator)
            assert node.idle and not node.will_emit and node.step() is None
            count += 1
    assert node.stats["accepted"] == node.stats["completed"] == count
    subtracts = node.stats["subtracts"]
    assert 0 < subtracts <= count * node.iterations
    cost = node.estimate()
    assert cost["research_register_cap_pass"] and cost["total_register_bits"] <= 640
    assert cost["DSP"] == cost["new_RAM"] == cost["pipeline_register_bits"] == 0
    assert cost["contexts"] == 1 and cost["initiation_interval"] == (14 if digit_bits == 1 else 8)


@pytest.mark.parametrize("digit_bits", [1, 2])
def test_actual_wide_score_producer_keeps_order_and_matches_pipeline(digit_bits):
    widths = FixedBankFormat(10).widths(125, 125, 0.6)
    sources = [
        FoldResult(i, (-1) ** i * (17 + i), 13 + i, (i * 10091) % 70013, i % 7) for i in range(37)
    ]
    outputs = []
    for node in [Q12Pipeline(widths, 125, 125), Q12Serial(widths, 125, 125, digit_bits=digit_bits)]:
        score, incoming, values = ScorePipeline(widths, 125), deque(sources), []
        for _cycle in range(3000):
            result = node.step()
            if result is not None:
                values.append(result)
            score.step(incoming, node)
            if not incoming and score.idle and node.idle:
                break
        assert len(values) == len(sources)
        assert [row["source"].tag for row in values] == list(range(len(sources)))
        assert score.native.stats["products"] == 8 * len(sources)
        outputs.append(values)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("digit_bits", [1, 2])
@pytest.mark.parametrize("fault", ["source", "tag", "real", "energy", "N", "D", "zero", "hit"])
def test_bad_port_or_producer_context_does_not_consume_credit(digit_bits, fault):
    widths = FixedBankFormat(10).widths(31, 62, 0.6)
    node = Q12Serial(widths, 31, 62, digit_bits=digit_bits)
    source, n, d, hit = FoldResult(0, 1, -1, 1, 0), 2, 62, False
    if fault == "source":
        source = object()
    elif fault == "tag":
        source = FoldResult(-1, 1, -1, 1, 0)
    elif fault == "real":
        source = FoldResult(0, 1 << widths["correlation"], -1, 1, 0)
    elif fault == "energy":
        source = FoldResult(0, 1, -1, -1, 0)
    elif fault == "N":
        n = 1 << widths["numerator"]
    elif fault == "D":
        d = 1 << node.d
    elif fault == "zero":
        d = 0
    else:
        hit = 1
    before = deepcopy((node.cell, node.phase, node.digits, node.stats))
    with pytest.raises((ValueError, OverflowError)):
        node.offer(source, n, d, hit)
    assert (node.cell, node.phase, node.digits, node.stats) == before
    assert node.idle


@pytest.mark.parametrize("digits", [0, 3, True, 1.0])
def test_only_compiled_digit_structures_are_accepted(digits):
    with pytest.raises(ValueError, match="compiled"):
        Q12Serial(FixedBankFormat(10).widths(31, 62, 0.6), 31, 62, digit_bits=digits)


@pytest.mark.parametrize("bad", ["terms", "coefficient", "widths", "missing"])
def test_other_compile_formats_need_separate_validation(bad):
    terms, coefficient = (
        (7, 62) if bad == "terms" else (31, 93) if bad == "coefficient" else (31, 62)
    )
    widths = (
        None
        if bad == "missing"
        else FixedBankFormat(12 if bad == "widths" else 10).widths(31, 62, 0.6)
    )
    with pytest.raises(ValueError, match="compiled"):
        Q12Serial(widths, terms, coefficient)


def test_nonzero_digit_statistic_overflow_preserves_the_owned_context():
    node = Q12Serial(FixedBankFormat(10).widths(31, 62, 0.6), 31, 62)
    assert node.offer(FoldResult(0, 7, 3, 1, 0), 58, 62, True)
    node.step()
    node.stats["subtracts"] = (1 << 64) - 1
    before = deepcopy((node.cell, node.phase, node.digits, node.stats))
    with pytest.raises(OverflowError):
        node.step()
    assert (node.cell, node.phase, node.digits, node.stats) == before


def test_compiled_format_is_latched_and_statistics_never_wrap_or_half_consume():
    widths = FixedBankFormat(10).widths(31, 62, 0.6)
    node = Q12Serial(widths, 31, 62)
    widths["numerator"] = 1
    assert node.w["numerator"] > 1
    with pytest.raises(TypeError):
        node.w["numerator"] = 1
    source = FoldResult(0, 1, 0, 1, 0)
    node.stats["accepted"] = (1 << 64) - 1
    with pytest.raises(OverflowError):
        node.offer(source, 1, 62, False)
    assert node.idle
    node.stats["accepted"] = 0
    assert node.offer(source, 1, 62, False)
    for _ in range(7):
        node.step()
    assert node.will_emit
    before = deepcopy((node.cell, node.phase, node.digits))
    node.stats["completed"] = (1 << 64) - 1
    with pytest.raises(OverflowError):
        node.step()
    assert (node.cell, node.phase, node.digits) == before
