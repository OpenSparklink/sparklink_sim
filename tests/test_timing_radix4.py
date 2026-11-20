"""Exact radix4 quotient, actual registered edges and unchanged packet owners."""

import importlib
import json
from fractions import Fraction
from functools import partial
from random import Random

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay
from nearlink_sdr.sim.timing_word_flow import TimingWordService
from tests.test_finite_timing_replay import FMT
from tests.test_head_word_replay import run
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_sync_fit_word import packed

RADIX4 = partial(HeadWordsFrameReplay, timing_divider="radix4-prefix")


def ratio(fmt, n, d, negative, mode):
    service = TimingWordService(
        WordTimingState((1 << 53, -53), 0, 32, fmt), divider=mode
    )
    service.num, service.den, service.negative = n, d, negative
    service.quotient = service.remainder = service.div_index = 0
    service.mode, service.job = "prefix_check", dict(error=0)
    while service.mode != "round_ratio":
        service.step()
    service.step()
    return service


@pytest.mark.parametrize("fmt", [TimingWordFormat(), FMT, TimingWordFormat(38, 32)])
def test_fraction_oracle_ties_maximum_dynamic_range_and_registered_savings(fmt):
    rng = Random(11201017)  # Unit development vectors,not the later validation seed.
    for d in [1, 2, 3, (1 << 49) - 1] + [rng.randrange(1, 1 << 49) for _ in range(50)]:
        maximum = min((1 << fmt.division_bits) - 1, d << (fmt.error_bits + 1))
        for n in [0, 1, d // 2, d + d // 2, maximum - 1, maximum, rng.randrange(maximum + 1)]:
            if not 0 <= n <= maximum:
                continue
            for negative in (False, True):
                a = ratio(fmt, n, d, negative, "bounded-prefix")
                b = ratio(fmt, n, d, negative, "radix4-prefix")
                assert a.job["error"] == b.job["error"] == round(Fraction(-n if negative else n, d))
                assert (a.quotient, a.remainder) == (b.quotient, b.remainder) == divmod(n, d)
                assert b.stats["divider_steps"] == (fmt.error_bits + 2) // 2
                assert a.cycle - b.cycle == (fmt.error_bits + 2) // 2 - 1
                assert b.three_den == 3 * d < (1 << 51)


@pytest.mark.parametrize("fmt", [TimingWordFormat(), FMT, TimingWordFormat(38, 32)])
def test_real_TED_and_feedback_commits_are_exact_and_backpressure_is_held(fmt):
    a, b = [TimingWordService(WordTimingState((1 << 53, -53), 41, 32, fmt), divider=m)
            for m in ("bounded-prefix", "radix4-prefix")]
    rng = np.random.default_rng(11201019)
    savings = set()
    for k in range(96):
        # First exercise real,non-clipped ratios at steady power;then strong/weak
        # extremes. Changing exponent each symbol alone only exercised bypasses.
        exponent = -21 if k < 80 else [-128, -23, 0, 127][k % 4]
        r, i = map(int, rng.integers(-(1 << 21), 1 << 21, 2))
        word = packed(r, i, exponent)
        midpoint = None if k < 32 else packed(int(rng.integers(-(1 << 21), 1 << 21)),
                                             19, exponent)
        before = b.state.snapshot()
        for s in (a, b):
            assert s.issue(word, midpoint)
            assert not s.issue(word, midpoint)
            while not s.ready:
                s.step()
                if s is b and not s.ready:
                    assert s.state.snapshot() == before
        savings.add((a.ready_cycle - a.first_cycle) - (b.ready_cycle - b.first_cycle))
        result = b.result.copy()
        for _ in range(3):
            b.step()
            assert b.result == result and not b.input_ready
        assert a.take() == b.take() and a.state.snapshot() == b.state.snapshot()
        a.step()
        b.step()
    assert savings == {0, (fmt.error_bits + 2) // 2 - 1}
    old, new = a.estimate(), b.estimate()
    assert new["state_register_bits_upper"] - old["state_register_bits_upper"] == 51
    assert new["DSP25x18"] == old["DSP25x18"] == 4
    assert new["extra_raw_RAM_bits"] == new["additional_source_RAM_ports"] == 0
    assert new["divider"]["parallel_comparators"] == 3
    assert new["divider"]["remainder_comparison_bits"] == 51
    assert not new["synthesized"] and new["complete_logic_LUTs_or_timing"] is None


@pytest.mark.parametrize("mode", ["radix4_prepare", "radix4_divide", "round_ratio"])
def test_abort_never_commits_unfinished_ratio_or_resumes(mode):
    s = TimingWordService(WordTimingState((1 << 53, -53), 0, 32, FMT),
                          divider="radix4-prefix")
    s.num, s.den, s.negative = 1234, 579, False
    s.quotient = s.remainder = s.div_index = 0
    s.job, s.mode = dict(error=0), "prefix_check"
    before = s.state.snapshot()
    while s.mode != mode:
        s.step()
    s.abort()
    for _ in range(4):
        s.step()
    assert s.take() is None and s.state.snapshot() == before
    with pytest.raises(RuntimeError, match="Closed packet"):
        s.issue(packed(17, 31, 0), None)


@pytest.mark.parametrize("bad", ["zero", "prefix", "counter"])
def test_invalid_prefix_never_enters_radix4_or_partially_updates_state(bad):
    s = TimingWordService(WordTimingState((1 << 53, -53), 0, 32, FMT),
                          divider="radix4-prefix")
    s.num, s.den, s.negative = 1, 3, False
    s.quotient = s.remainder = s.div_index = 0
    s.mode = "prefix_check"
    if bad == "zero":
        s.den = 0
    elif bad == "prefix":
        s.num = 3 << 34
    else:
        s.stats["prefix_checks"] = (1 << 64) - 1
    before = s.state.snapshot(), s.remainder, s.div_index, s.mode
    with pytest.raises((ArithmeticError, OverflowError)):
        s.step()
    assert (s.state.snapshot(), s.remainder, s.div_index, s.mode) == before


@pytest.mark.parametrize("ft,mcs,pilot,damage,status", [
    (2, 8, 8, None, "valid"), (3, 12, 4, None, "valid"), (4, 10, 16, None, "valid"),
    (3, 8, 0, "bad_head", "bad_header_crc"),
    (4, 12, 0, "bad_data", "bad_data_crc"),
])
def test_actual_word_head_frame_payload_and_trajectories_do_not_change(
        ft, mcs, pilot, damage, status):
    p = ordinary(ft, mcs, pilot)
    values, expected, selection = selected_input(
        p, length=19, **({damage: True} if damage else {})
    )
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    before, a, _ = run(p, words, selection, HeadWordsFrameReplay)
    after, b, barriers = run(p, words, selection, RADIX4, (1, 7, 31))
    assert before["frame"] == after["frame"] and after["frame"]["status"] == status
    for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(before[key], after[key])
    assert a.consumer.word_timing.snapshot() == b.consumer.word_timing.snapshot()
    assert a.consumer.normalization.snapshot() == b.consumer.normalization.snapshot()
    normal = b.timing_service.normal
    checks = normal.stats["prefix_checks"] if normal else 0
    assert a.cycle - b.cycle == 16 * checks
    packets = b.take_mac_result().packets
    assert [x.mac_payload for x in packets] == ([expected] if damage is None else [])
    assert b.cycle_evidence_valid == (not barriers)
    assert b.estimate()["payload_cycles"] is None  # Functional payload has no cycle credit.


def test_validation_rejects_changed_source_before_creating_evidence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_timing_radix4")
    frozen = tmp_path / "freeze.json"
    frozen.write_text(json.dumps(dict(source={"wrong": "source"},
                                     policy=json.loads(model.POLICY.read_text()))))
    out = tmp_path / "validation"
    with pytest.raises(AssertionError):
        model.run(out, freeze=frozen)
    assert not out.exists()


def test_reproduction_never_overwrites_previous_results(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_timing_radix4")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"original failure")
    with pytest.raises(ValueError, match="fresh evidence"):
        model.run(tmp_path)
    assert sentinel.read_bytes() == b"original failure"
