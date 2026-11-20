"""Actual source reuse,finite ownership/rejection and unchanged frame/word trajectories."""

from copy import deepcopy
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.power_ted_replay import PowerTEDFrameReplay
from nearlink_sdr.sim.retained_stencil_replay import RetainedStencilFrameReplay, _overlap_words
from tests.stencil_reference import MAX_START, address_vectors, prefix_oracle
from tests.test_data_joint_replay import TRACE, at_data_issue
from tests.test_head_word_replay import run
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def ready_query():
    flow = at_data_issue(RetainedStencilFrameReplay)
    while flow.mode != "nco_query":
        assert not flow.done and flow.pending_service is None
        flow.step()
    while not flow.query_service.ready:
        flow.step()
    assert len(flow.staged) == 4
    return flow


def snapshot(flow):
    return deepcopy(dict(
        mode=flow.mode, staged=flow.staged, start=flow.stage_start, words=flow.words,
        queries=flow.queries, query_index=flow.query_index, future_end=flow.future_end,
        query_mode=flow.query_service.mode, query_result=flow.query_service.result,
        source_head=flow.history.head, source_end=flow.history.end,
        timing=flow.consumer.word_timing.snapshot(),
        normalization=flow.consumer.normalization.snapshot(), stats=flow.stats,
    ))


def test_new_development_addresses_match_independent_source_dictionary_prefix():
    for previous, following in address_vectors("development"):
        try:
            expected = prefix_oracle(previous, following)
        except OverflowError:
            with pytest.raises(OverflowError):
                _overlap_words(previous, following)
        else:
            assert _overlap_words(previous, following) == expected


@pytest.mark.parametrize("previous,following", [(True, 1), (0, False), (1.0, 2), (0, "2")])
def test_native_address_types_are_required(previous, following):
    with pytest.raises(ValueError, match="native"):
        _overlap_words(previous, following)


@pytest.mark.parametrize("previous,following", [(-1, 0), (0, -1), (MAX_START + 1, 0),
                                               (0, MAX_START + 1), (1 << 64, 0)])
def test_four_word_end_does_not_wrap(previous, following):
    with pytest.raises(OverflowError):
        _overlap_words(previous, following)


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 23), (3, 12, 4, 41), (4, 10, 8, 17)])
@pytest.mark.parametrize("layout", ["dense", "depth-parity"])
@pytest.mark.parametrize("divider", ["bounded-prefix", "radix4-prefix"])
def test_complete_frame_and_sampled_words_match_existing_owner(ft, mcs, pilot, length,
                                                            layout, divider):
    p = ordinary(ft, mcs, pilot)
    values, expected, selection = selected_input(p, length=length)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    before, old, _ = run(p, words, selection, partial(
        PowerTEDFrameReplay, head_layout=layout, timing_divider=divider))
    after, new, barriers = run(p, words, selection, partial(
        RetainedStencilFrameReplay, head_layout=layout, timing_divider=divider), (1, 7, 31))
    assert before["frame"] == after["frame"] and after["frame"]["payload"] == expected
    for key in TRACE:
        np.testing.assert_array_equal(before[key], after[key])
    assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
    assert old.consumer.normalization.snapshot() == new.consumer.normalization.snapshot()
    assert new.history.head == old.history.head and new.history.end == old.history.end
    for i in range(new.history.head, new.history.end):
        assert new.history.words[i % 512] == words[i]
    assert new.stats["cross_symbol_words_reused"] > 0
    assert new.stats["raw_reads"] < old.stats["raw_reads"] and new.cycle <= old.cycle
    assert new.estimate()["new_controller_state_bits"] == (
        old.estimate()["new_controller_state_bits"] + 64)
    assert new.estimate()["borrowed_interpolator"] == old.estimate()["borrowed_interpolator"]
    assert new.estimate()["borrowed_history"] == old.estimate()["borrowed_history"]
    assert new.staged == [] and not new.kernel.live and new.history.pending is None
    assert barriers == ["payload"] and not new.cycle_evidence_valid
    assert [x.mac_payload for x in new.take_mac_result().packets] == [expected]


@pytest.mark.parametrize("damage,status", [({"bad_head": True}, "bad_header_crc"),
                                          ({"bad_data": True}, "bad_data_crc"),
                                          ({"empty": True}, "invalid_length")])
def test_failed_real_crc_and_length_do_not_deliver(damage, status):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, **damage)
    before, old, _ = run(p, quantize_input(values), selection, PowerTEDFrameReplay)
    after, new, _ = run(p, quantize_input(values), selection, RetainedStencilFrameReplay)
    assert before["frame"] == after["frame"] and after["frame"]["status"] == status
    assert not new.take_mac_result().packets and not old.take_mac_result().packets
    assert new.staged == [] and not new.write_input(0)


def test_wrong_crc_context_does_not_deliver():
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p)
    p = replace(p, config=replace(p.config, head_crc_seed=0))
    event, flow, _ = run(p, quantize_input(values), selection, RetainedStencilFrameReplay)
    assert event["frame"]["status"] == "bad_header_crc"
    assert event["frame"]["data_crc"] is None and not flow.take_mac_result().packets


@pytest.mark.parametrize("after_sync", [8, 80])
def test_eof_clears_staging_without_float_header_fallback(monkeypatch, after_sync):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, length=99)
    cut = selection["start_tick"] // 8 + 4 * (len(p.sync_symbols()) + after_sync)
    selection = selection | dict(sample_limit=cut)

    def forbidden(*args, **kwargs):
        pytest.fail("EOF cannot re-decode a floating header")

    monkeypatch.setattr("nearlink_sdr.phy.selected_rx.decode_psk_frame", forbidden)
    event, flow, _ = run(p, quantize_input(values)[:cut], selection, RetainedStencilFrameReplay)
    assert event["frame"]["status"] == ("truncated_header" if after_sync == 8 else "truncated_data")
    assert event["frame"]["header_crc"] is (None if after_sync == 8 else True)
    assert event["frame"]["data_crc"] is None and not flow.take_mac_result().packets
    assert flow.staged == [] and flow.history.pending is None and not flow.kernel.live


@pytest.mark.parametrize("fault", ["counter", "count", "midpoint", "negative", "past_end"])
def test_guard_rejects_before_query_take_source_retire_or_staging_mutation(fault):
    flow = ready_query()
    if fault == "counter":
        flow.stats["cross_symbol_words_reused"] = (1 << 64) - 1
    elif fault == "count":
        flow.query_service.result["count"] += 1
    elif fault == "midpoint":
        current, _ = flow.query_service.result["queries"]
        flow.query_service.result["queries"] = (current, None)
    else:
        flow.query_service.result["queries"][1]["cell"] = (
            0 if fault == "negative" else (1 << 64) - 1
        )
    before = snapshot(flow)
    with pytest.raises((OverflowError, RuntimeError)):
        flow.step()
    assert before == snapshot(flow) and flow.query_service.ready
    assert flow.abort()["frame"]["data_crc"] is None and not flow.take_mac_result().packets


def test_cache_miss_preserves_full_counter_and_uses_normal_source_path():
    flow = ready_query()
    flow.staged = []  # Legal empty optional cache;do not use a CRC-driven fallback.
    flow.stats["cross_symbol_words_reused"] = (1 << 64) - 1
    flow.step()
    assert flow.mode == "queries" and flow.staged == []
    assert flow.stats["cross_symbol_words_reused"] == (1 << 64) - 1
    assert flow.abort()["frame"]["data_crc"] is None


@pytest.mark.parametrize("field,value", [("reuse", False), ("sps", 2), ("rate", 1)])
def test_unsupported_configuration_rejects_before_taking_prepared_owner(field, value):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, length=19)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    prepared, _ = prepare(p, words, selection)
    if field == "sps":
        p = replace(p, config=replace(p.config, sps=value))
    elif field == "rate":
        p = replace(p, config=replace(p.config, symbol_rate_mhz=value))
    with pytest.raises(ValueError):
        RetainedStencilFrameReplay(
            p, prepared, reuse_stencil=value if field == "reuse" else True,
            normalization_format=NormalizationFormat(),
            word_timing_format=TimingWordFormat(30, 32), **selection)
    assert prepared.ready and prepared.fit.ready


def test_default_replay_still_clears_completed_staging_and_does_not_charge_new_counter():
    old, new = at_data_issue(PowerTEDFrameReplay), at_data_issue(RetainedStencilFrameReplay)
    assert old.staged == [] and len(new.staged) == 4
    assert "cross_symbol_words_reused" not in old.stats
    assert new.estimate()["retained_stencil_logic_sharing_credit"] is False
    assert new.estimate()["complete_latency_cycles"] is None
