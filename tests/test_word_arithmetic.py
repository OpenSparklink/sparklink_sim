"""Independent integer/Fraction references, real carry steps and finite ports."""

from fractions import Fraction
from random import Random

import pytest

from nearlink_sdr.sim.word_arithmetic import (
    WordCommand,
    WordFabric,
    WordFabricAssumptions,
    word_fabric_cost,
)


def command(code, a, b=0, width=208, signed=True):
    return WordCommand(code, a, b, width, width, signed, signed, width, signed)


def execute(c):
    f = WordFabric()
    assert f.issue(c, 0, 0)
    while True:
        f.tick()
        reply = f.take(0)
        if reply is not None:
            return f, reply


def test_random_and_adversarial_operators_against_wide_independent_reference():
    rng = Random(810117)
    for width in (1, 2, 17, 22, 32, 33, 45, 98, 107, 206, 208):
        for signed in (False, True):
            lo, hi = -(1 << (width - 1)) * signed, 1 << (width - signed)
            values = [lo, hi - 1, 0, *[rng.randrange(lo, hi) for _ in range(8)]]
            for a, b in zip(values, reversed(values), strict=True):
                for code, expected in (
                    ("add", a + b),
                    ("sub", a - b),
                    ("abs", abs(a)),
                    ("neg", -a),
                    ("mov", a),
                ):
                    f, reply = execute(command(code, a, b, width, signed))
                    assert reply.value == expected
                    assert reply.overflow == (not lo <= expected < hi)
                    assert f.idle and f.stats["taken"] == 1
                c = WordCommand("cmp", a, b, width, width, signed, signed, 2, True)
                _, reply = execute(c)
                assert reply.value == (a > b) - (a < b) and not reply.overflow
    # Mixed signed/unsigned and narrower output expose assumptions hidden by
    # result-width-only latency; these inputs need carry into extra sign bits.
    for a, b in ((2**208 - 1, -(2**207)), (0, -1), (2**207, 2**207 - 1)):
        for code in ("add", "sub", "cmp"):
            c = WordCommand(code, a, b, 208, 208, False, True, 3, True)
            _, reply = execute(c)
            expected = a + b if code == "add" else a - b if code == "sub" else (a > b) - (a < b)
            assert reply.value == expected and reply.overflow == (not -4 <= expected < 4)


def test_rne_fixed_four_steps_independent_fraction_ties_extremes_and_overflow():
    rng = Random(810211)
    values = [-(2**32), 2**32 - 1]
    values += [q * 1024 + d for q in (-2049, -2, -1, 0, 1, 2, 2048) for d in (511, 512, 513)]
    values += [rng.randrange(-(2**32), 2**32) for _ in range(160)]
    for value in values:
        c = WordCommand("rne", value, 10, 33, 4, True, False, 22, True)
        f, reply = execute(c)
        assert reply.value == round(Fraction(value, 1024))
        assert reply.overflow == (not -(2**21) <= reply.value < 2**21)
        assert reply.completed_cycle == 4 and f.stats["alu_steps"] == 4


def test_each_tick_advances_only_one_segment_and_no_answer_at_issue():
    c = command("add", 2**160 - 1, 1)
    f = WordFabric()
    assert f.issue(c, 0, 0)
    job = f.lanes[0]
    assert job.result == 0 and f.mailboxes[0] is None
    f.tick()
    assert job.cursor == 1 and job.result == 0 and job.carry == 1
    assert f.take(0) is None
    while f.stats["completed"] == 0:
        f.tick()
    assert f.cycle == 7 and f.take(0).value == 2**160
    assert not f.issue(command("mov", 1), 0, 1)  # No same-edge dependent forwarding.
    f.tick()
    assert f.issue(command("mov", 1), 0, 1)


def test_native_dsp_ii1_three_stages_and_single_output_port_with_stalls():
    f = WordFabric(WordFabricAssumptions(4))
    tile = WordCommand("tile", 2**24 - 1, 2**17 - 1, 24, 17, False, False, 43, True)
    assert f.issue(tile, 0, 0)
    assert f.dsp[0][1] is None
    f.tick()
    assert f.dsp[1][1] == tile.a * tile.b and f.take(0) is None
    assert f.issue(tile, 1, 0)
    assert not f.issue(tile, 2, 0)  # Global one-issue port even with idle units.
    f.tick()
    assert f.issue(command("mov", 7), 2, 0)
    f.tick()
    assert f.mailboxes[0].completed_cycle == f.mailboxes[2].completed_cycle == 3
    assert f.take(0).value == tile.a * tile.b
    assert f.take(2) is None
    for _ in range(12):
        f.tick()
    assert f.take(2).value == 7 and f.take(1) is None
    f.tick()
    assert f.take(1).value == tile.a * tile.b
    assert f.idle and f.stats == dict(issued=3, completed=3, taken=3, alu_steps=1)


def test_mailbox_backpressure_tokens_flush_and_clock_bounds():
    f = WordFabric()
    c = command("mov", 5)
    assert f.issue(c, 0, 0)
    f.tick()
    assert not f.issue(c, 0, 1)
    with pytest.raises(ValueError):
        f.issue(c, 0, 0)
    for _ in range(10):
        f.tick()
    assert f.take(0).value == 5 and f.issue(c, 0, 1)
    f.flush()
    assert f.idle and not any(f.mailboxes) and f.next_token == [0] * 9
    f.tick()
    assert not any(f.mailboxes) and f.issue(c, 0, 0)
    f.flush()
    f.next_token[0] = 65535
    assert f.issue(c, 0, 65535)
    f.tick()
    f.take(0)
    with pytest.raises(ValueError, match="token"):
        f.issue(c, 0, 65536)
    f = WordFabric(initial_cycle=2**63 - 1)
    with pytest.raises(OverflowError, match="clock"):
        f.tick()
    f = WordFabric()
    f.stats["alu_steps"] = 2**64 - 1
    f.issue(c, 0, 0)
    with pytest.raises(OverflowError, match="counter"):
        f.tick()


def test_compile_validation_and_explicit_cost():
    for a in (WordFabricAssumptions(1), WordFabricAssumptions(2), WordFabricAssumptions(4)):
        cost = word_fabric_cost(a)
        assert cost["state_bits"] <= 8192 and cost["DSP25x18"] == 1
        assert cost["issue_ports"] == cost["result_take_ports"] == 1
        assert not cost["synthesis_or_timing_evidence"]
    assert word_fabric_cost(WordFabricAssumptions(4))["state_bits"] == 6472
    for kw in (dict(alu_lanes=True), dict(alu_lanes=3), dict(contexts=10), dict(contexts=True)):
        with pytest.raises(ValueError):
            WordFabricAssumptions(**kw)
    valid = vars(command("mov", 0))
    for changes in (
        dict(code="divide"),
        dict(a=True),
        dict(a=2**207),
        dict(a_bits=True),
        dict(bits=209),
        dict(signed=1),
        dict(tag=128),
        dict(code="tile"),
        dict(code="rne"),
        dict(code="cmp", bits=1),
    ):
        with pytest.raises(ValueError):
            WordCommand(**(valid | changes))
    for kw in (dict(assumptions=True), dict(initial_cycle=True), dict(sink=1)):
        with pytest.raises(ValueError):
            WordFabric(**kw)
    for context in (-1, True, 9):
        with pytest.raises(ValueError):
            WordFabric().take(context)
    with pytest.raises(ValueError):
        WordFabric().issue(None, 0, 0)
    with pytest.raises(ValueError):
        word_fabric_cost(True)
