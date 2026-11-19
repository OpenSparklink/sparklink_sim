"""Exact quotient bound, independent RNE oracle and real frame/EOF ownership."""

from fractions import Fraction
from functools import partial
from random import Random

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.overlapped_normalization_replay import OverlappedNormalizationFrameReplay
from nearlink_sdr.sim.timing_word_flow import TimingWordService
from tests.test_finite_timing_replay import FMT, value_trace
from tests.test_normalized_word_replay import run
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_sync_fit_word import packed

PREFIX = partial(OverlappedNormalizationFrameReplay, timing_divider="bounded-prefix")


def ratio_service(fmt, numerator, denominator, negative, divider):
    s = TimingWordService(WordTimingState((1 << 53, -53), 0, 32, fmt), divider=divider)
    s.num, s.den, s.negative = numerator, denominator, negative
    s.quotient = s.remainder = s.div_index = 0
    s.mode = "prefix_check" if divider == "bounded-prefix" else "divide"
    s.job = dict(error=0)
    while s.mode != "round_ratio":
        s.step()
    s.step()
    return s


@pytest.mark.parametrize("fmt", [TimingWordFormat(), FMT, TimingWordFormat(38, 32)])
def test_zero_quotient_prefix_matches_independent_Fraction_RNE_including_ties(fmt):
    rng = Random(1133981)
    divisors = [1, 2, 3, (1 << 29) - 1, 1 << 29, (1 << 30) - 1,
                (1 << 48) - 1, 1 << 48, (1 << 49) - 1]
    for d in divisors + [rng.randrange(1, 1 << 49) for _ in range(160)]:
        maximum = min((1 << fmt.division_bits) - 1, d << (fmt.error_bits + 1))
        values = [0, 1, maximum, max(0, maximum - 1), rng.randrange(maximum + 1)]
        if d % 2 == 0:
            values += [d // 2, d + d // 2, 2 * d + d // 2]
        for n in values:
            if n > maximum:
                continue
            negative = bool(rng.getrandbits(1))
            a = ratio_service(fmt, n, d, negative, "full")
            b = ratio_service(fmt, n, d, negative, "bounded-prefix")
            assert a.job["error"] == b.job["error"] == round(Fraction(-n if negative else n, d))
            assert a.remainder == b.remainder and a.quotient == b.quotient
            assert a.stats["divider_steps"] == fmt.division_bits
            assert b.stats["divider_steps"] == fmt.error_bits + 2
            assert b.stats["prefix_checks"] == 1
            assert a.cycle - b.cycle == fmt.division_bits - fmt.error_bits - 3


@pytest.mark.parametrize("fmt", [TimingWordFormat(), FMT, TimingWordFormat(38, 32)])
def test_real_TED_clipping_bypass_power_state_and_delayed_commit_remain_identical(fmt):
    old = WordTimingState((1 << 53, -53), 41, 32, fmt)
    new = WordTimingState((1 << 53, -53), 41, 32, fmt)
    a, b = TimingWordService(old), TimingWordService(new, divider="bounded-prefix")
    rng = np.random.default_rng(1133987)
    differences = set()
    for k in range(130):
        e = [-128, -23, 0, 127][k % 4]
        r, i = map(int, rng.integers(-(1 << 21), 1 << 21, 2))
        word = packed(r, i, e)
        mid = None if k < 32 else packed(int(rng.integers(-(1 << 21), 1 << 21)), 19, e)
        initial = new.snapshot()
        for s in [a, b]:
            assert s.issue(word, mid)
            while not s.ready:
                s.step()
                if s is b and not s.ready:
                    assert new.snapshot() == initial
        assert a.take() == b.take() and old.snapshot() == new.snapshot()
        differences.add((a.ready_cycle - a.first_cycle) - (b.ready_cycle - b.first_cycle))
        a.step()
        b.step()
    assert differences == {0, fmt.division_bits - fmt.error_bits - 3}
    assert new.error_clip_events > 0
    assert b.stats["prefix_checks"] > 0
    cost, original = b.estimate(), a.estimate()
    assert cost["state_register_bits_upper"] - original["state_register_bits_upper"] == 64
    assert cost["DSP25x18"] == original["DSP25x18"] == 4
    assert cost["extra_raw_RAM_bits"] == cost["additional_source_RAM_ports"] == 0
    assert cost["divider"]["registered_prefix_check_edges"] == 1


@pytest.mark.parametrize("bad", ["prefix", "zero_denominator", "counter"])
def test_prefix_invariant_or_counter_failure_never_commits_partial_state(bad):
    s = TimingWordService(WordTimingState((1 << 53, -53), 0, 32, FMT),
                          divider="bounded-prefix")
    s.num, s.den, s.negative = 1, 3, False
    s.quotient = s.remainder = s.div_index = 0
    s.mode = "prefix_check"
    if bad == "prefix":
        s.num = 3 << 34
    elif bad == "zero_denominator":
        s.den = 0
    else:
        s.stats["prefix_checks"] = (1 << 64) - 1
    before = s.state.snapshot(), s.remainder, s.div_index, s.mode
    with pytest.raises((ArithmeticError, OverflowError)):
        s.step()
    assert (s.state.snapshot(), s.remainder, s.div_index, s.mode) == before
    s.abort()
    assert s.take() is None


@pytest.mark.parametrize("ft,mcs,pilot,damage,status", [
    (2, 8, 0, None, "valid"), (3, 12, 16, None, "valid"), (4, 10, 4, None, "valid"),
    (3, 8, 0, "bad_head", "bad_header_crc"), (4, 12, 0, "bad_data", "bad_data_crc"),
    (4, 12, 0, "EOF", "truncated_data"),
])
def test_same_selected_frame_words_trajectory_CRC_and_MAC_across_arbitrary_blocks(
        ft, mcs, pilot, damage, status):
    p = ordinary(ft, mcs, pilot)
    iq, payload, selection = selected_input(
        p, length=49, **({damage: True} if damage in ("bad_head", "bad_data") else {}))
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    if damage == "EOF":
        words = words[:selection["start_tick"] // 8 + 4 * 448]
    a, oa, sa = run(p, words, selection, flow_type=OverlappedNormalizationFrameReplay)
    b, ob, sb = run(p, words, selection, (1, 17, 3, 31), flow_type=PREFIX)
    assert a.event["frame"] == b.event["frame"] and b.event["frame"]["status"] == status
    assert value_trace(oa) == value_trace(ob)
    assert [r["values"] for r in sa] == [r["values"] for r in sb]
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(a.event[key], b.event[key])
    assert a.consumer.word_timing.snapshot() == b.consumer.word_timing.snapshot()
    assert a.consumer.normalization.snapshot() == b.consumer.normalization.snapshot()
    normal = b.timing_service.normal
    checks = normal.stats["prefix_checks"] if normal else 0
    assert a.cycle - b.cycle == 29 * checks
    if damage is None:
        assert b.take_mac_result().packets[0].mac_payload == payload
        assert checks > 0
    else:
        assert not b.take_mac_result().packets
    assert b.event["finite_timing_services"]["complete_latency_cycles"] is None


def test_bad_compiled_mode_leaves_prepared_fit_owner_unclaimed():
    p = ordinary()
    iq, _, selection = selected_input(p, length=11)
    words = quantize_input(np.r_[iq, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    prepared, _ = prepare(p, words, selection)
    with pytest.raises(ValueError, match="divider"):
        OverlappedNormalizationFrameReplay(p, prepared, timing_divider="unbounded",
            normalization_format=NormalizationFormat(), word_timing_format=FMT, **selection)
    assert not prepared.fit.output_taken
