"""Actual query/source/word ownership,aged errors,finite head and explicit resource failures."""

from copy import deepcopy
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat, WordNormalizationState
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.feedback_timing_word import FeedbackAgeTimingState
from nearlink_sdr.sim.retained_stencil_replay import RetainedStencilFrameReplay
from nearlink_sdr.sim.split_sampling_replay import SplitSamplingFrameReplay
from tests.test_head_word_replay import run
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input

FORMAT = TimingWordFormat(30, 32)
FIELDS = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")
TRACES = ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error")


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 19), (3, 12, 8, 47), (4, 10, 4, 23)])
@pytest.mark.parametrize("age", [0, 1, 2, 3])
def test_actual_sampling_and_tagged_join_match_functional_words(ft, mcs, pilot, length, age):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    words, raw, services = quantize_input(values), [], []
    event, flow, barriers = run(profile, words, selection, partial(
        SplitSamplingFrameReplay, error_age=age, observer=raw.append,
        service_observer=services.append))
    reference = FeedbackAgeTimingState(flow.initial_fit["power"], selection["start_tick"],
                                       len(profile.sync_symbols()), FORMAT, error_age=age)
    normalization = WordNormalizationState(flow.initial_fit, NormalizationFormat())
    samples = {(r["symbol"], r["role"]): r for r in raw}
    joined = [r["values"] for r in services if r["kind"] == "joined"]
    assert len(joined) == len(event["symbols"])
    for n, record in enumerate(joined):
        for role, query in enumerate(reference.state.queries):
            if query is not None:
                assert {k: samples[n, role][k] for k in ("cell", "phase")} == query
        current = samples[n, 0]["word"]
        middle = samples[n, 1]["word"] if n >= reference.state.hold_symbols else None
        reference.advance_words(current, middle)
        expected = reference.state.snapshot()
        assert all(record["detector"][k] == v for k, v in expected.items()
                   if k in record["detector"])
        assert all(record["PI"][k] == v for k, v in expected.items() if k in record["PI"])
        assert record["PI"]["applied_error_word"] == reference.last_applied_error_word
        assert record["normalization"]["normalized_word"] == (
            normalization.advance_word(current)["normalized_word"])
    assert flow.consumer.word_timing.snapshot() == reference.state.snapshot()
    assert event["frame"]["payload"] == payload
    assert event["frame"]["header_crc"] and event["frame"]["data_crc"]
    assert event["frame"]["mcs"] == mcs and barriers == ["payload"]
    assert flow.query_count == flow.sample_index == len(joined)
    assert flow.head_owner.stats["header_issues"] == 1
    assert event["split_sampling"]["max_PI_lead"] == flow.stats["max_PI_lead"]
    assert flow.stats["max_PI_lead"] <= age + 1
    assert max(flow.peak_messages) <= flow.capacity
    assert all(not q for q in (flow.detector_messages, flow.pi_messages,
                              flow.error_messages, flow.sample_messages))
    assert len(flow.take_mac_result().packets) == 1 and not flow.take_mac_result().packets
    assert flow.release_source() is flow.history and flow.release_source() is None
    if age == 0:
        before, old, _ = run(profile, words, selection, RetainedStencilFrameReplay)
        assert all(before["frame"].get(k) == event["frame"].get(k) for k in FIELDS)
        for key in TRACES:
            np.testing.assert_array_equal(before[key], event[key])
        assert old.consumer.word_timing.snapshot() == flow.consumer.word_timing.snapshot()


@pytest.mark.parametrize("age", [1, 3])
@pytest.mark.parametrize("damage,status", [
    ({"bad_head": True}, "bad_header_crc"), ({"bad_data": True}, "bad_data_crc"),
    ({"empty": True}, "invalid_length"), ({"mcs": 15}, "unsupported_mcs"),
])
def test_actual_head_is_a_gate_not_an_answer_selector(age, damage, status):
    p = ordinary(2 if damage.get("empty") else 3, 8, 0)
    values, _, selection = selected_input(p, length=19, **damage)
    event, flow, barriers = run(p, quantize_input(values), selection, partial(
        SplitSamplingFrameReplay, error_age=age))
    assert event["frame"]["status"] == status
    assert event["frame"]["payload"] is None and not flow.take_mac_result().packets
    assert flow.head_owner.stats["header_issues"] == 1
    if status != "bad_data_crc":
        assert flow.query_count == flow.consumer._head_required
        assert flow.consumer.frame_end_symbol is None and not barriers
    else:
        assert barriers == ["payload"]


@pytest.mark.parametrize("age", [0, 1, 2, 3])
@pytest.mark.parametrize("symbols,status", [(40, "truncated_header"), (120, "truncated_data")])
def test_eof_drains_admitted_words_and_never_fabricates_future(age, symbols, status):
    p = ordinary(2, 8, 0)
    values, _, selection = selected_input(p, length=255)
    cut = selection["start_tick"] // 8 + 4 * symbols
    event, flow, barriers = run(p, quantize_input(values)[:cut], selection, partial(
        SplitSamplingFrameReplay, error_age=age))
    assert event["frame"]["status"] == status and not barriers
    assert event["frame"]["data_crc"] is None and not flow.take_mac_result().packets
    assert flow.history.end == cut and flow.history.pending is None and not flow.kernel.live
    assert flow.sample_index == flow.stats["joined_symbols"]
    assert flow.consumer.word_timing.count == flow.sample_index
    assert not flow.write_input(0) and flow.cache.cache.state == "released"


@pytest.mark.parametrize("pattern", [(1,), (32,), (7, 1, 23, 4)])
@pytest.mark.parametrize("context", ["LLID", "seed"])
def test_real_wrong_context_and_suffix_preservation_with_input_partitions(pattern, context):
    p = ordinary(3 if context == "LLID" else 2, 12, 8)
    values, _, selection = selected_input(p, length=19)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    p = replace(p, config=replace(p.config, **{
        "head_llid" if context == "LLID" else "head_crc_seed": 7 if context == "LLID" else 0}))
    event, flow, barriers = run(p, words, selection, partial(
        SplitSamplingFrameReplay, error_age=2), pattern)
    assert event["frame"]["status"] == "bad_header_crc" and not barriers
    assert flow.head_owner.header.stats["field_bits"] == 0
    assert flow.query_count == flow.consumer._head_required
    assert flow.history.end < len(words) and not flow.take_mac_result().packets
    for index in range(flow.history.head, flow.history.end):
        assert flow.history.words[index % 512] == words[index]


@pytest.mark.parametrize("change", [dict(error_age=True), dict(error_age=-1), dict(error_age=4),
                                    dict(word_timing_format=TimingWordFormat(30, 24)),
                                    dict(normalization_format=None), dict(reuse_stencil=False),
                                    dict(service_observer=1)])
def test_invalid_mode_does_not_take_prepared_source(change):
    p = ordinary()
    values, _, selection = selected_input(p, length=3)
    prepared, _ = prepare(p, quantize_input(values), selection)
    options = dict(error_age=1, normalization_format=NormalizationFormat(),
                   word_timing_format=FORMAT, reuse_stencil=True) | change
    with pytest.raises(ValueError):
        SplitSamplingFrameReplay(p, prepared, **options, **selection)
    assert prepared.ready and not prepared.fit.output_taken


def test_continuous_input_overflow_is_recorded_separately_from_CRC():
    p = ordinary(4, 12, 8)
    values, _, selection = selected_input(p, length=255)
    words = quantize_input(np.r_[values, np.zeros(2000)])
    selection["sample_limit"] = len(words)
    prepared, pos = prepare(p, words, selection)
    flow = SplitSamplingFrameReplay(p, prepared, error_age=1,
                                   normalization_format=NormalizationFormat(),
                                   word_timing_format=FORMAT, reuse_stencil=True, **selection)
    origin = flow.cycle
    while not flow.done:
        if (pos < len(words)
                and flow.cycle >= origin + (pos - prepared.stencil.activation_end) * 12.5):
            used = flow.write_input(words[pos])
            pos += int(used)
        flow.step()  # ADC keeps arriving,including during head/payload backpressure.
        assert flow.cycle - origin < 100000
    assert flow.event["frame"]["status"] == "history_overflow"
    assert flow.stats["input_overflow"] == 1 and not flow.take_mac_result().packets
    assert flow.history.end - flow.history.head <= 512
    cost = flow.estimate()
    assert sum(cost["dedicated_DSP_banks"].values()) == 18
    assert cost["payload_cycles"] is None and not cost["whole_receiver16MSps"]
    assert not cost["old2048_controller_budget_passed"]


@pytest.mark.parametrize("fault", ["counter", "past_end", "negative"])
def test_query_preflight_preserves_credit_history_and_stencil_on_invalid_reply(fault):
    p = ordinary(2, 8, 0)
    values, _, selection = selected_input(p, length=19)
    words = quantize_input(values)
    prepared, pos = prepare(p, words, selection)
    flow = SplitSamplingFrameReplay(p, prepared, error_age=1,
                                   normalization_format=NormalizationFormat(),
                                   word_timing_format=FORMAT, reuse_stencil=True, **selection)
    while not (flow.mode == "nco_query" and flow.query_service.ready
               and flow.sample_index > len(p.sync_symbols())):
        if flow.wants_replay_input:
            pos += flow.feed_replay_words(words[pos:pos + 1])
        flow.step()
        assert not flow.done and flow.pending_service is None
    if fault == "counter":
        flow.stats["query_replies"] = (1 << 64) - 1
    else:
        current, middle = flow.query_service.result["queries"]
        middle = middle | dict(cell=(1 << 64) - 2 if fault == "past_end" else 0)
        flow.query_service.result["queries"] = (current, middle)

    def snapshot():
        return deepcopy(dict(query_mode=flow.query_service.mode, query=flow.query_service.result,
                             head=flow.history.head, end=flow.history.end,
                             staged=flow.staged, start=flow.stage_start,
                             words=flow.words, queries=flow.queries, future=flow.future_end))

    before = snapshot()
    with pytest.raises(OverflowError):
        flow.step()
    assert snapshot() == before
