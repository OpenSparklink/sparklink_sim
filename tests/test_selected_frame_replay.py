"""Actual prepared cache/source ownership,shared frame gate and nonzero-service barriers."""

import math
from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.sync_fit_word import TURN, SyncFitFormat, compile_sync_phases
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_sync_fit import consume_fit
from tests.test_selected_word import quantize_input


def prepare(profile, words, selection, *, slack=0):
    cell = selection["start_tick"] // 8
    phase = (selection["start_tick"] % 8) * 8192
    origin = cell - 1
    source = ReceiveHistoryPort(first_index=origin)
    service = InitialSyncFitService(
        source,
        compile_sync_phases(profile.sync_symbols()),
        round(selection["delta"] * TURN / math.tau),
        cell,
        phase,
        fmt=SyncFitFormat(30),
    )
    pos = origin
    ceiling = min(len(words), service.stencil.activation_end + slack)
    while not service.ready:
        c = service.cycle
        if pos < ceiling and c == math.ceil((pos - origin) * 200_000_000 / 16_000_000):
            assert source.write(pos, words[pos], c)
            pos += 1
        service.step()
        assert c < 20000, "Initialization must reach explicit readiness"
    return service, pos


def replay(profile, words, selection, *, reuse=False, chunks=(32,), slack=0):
    service, pos = prepare(profile, words, selection, slack=slack)
    flow = SelectedFrameReplay(profile, service, reuse_stencil=reuse, **selection)
    j = 0
    while not flow.done:
        if flow.pending_service is not None:
            flow.resolve_functional()
        else:
            if flow.wants_replay_input:
                if pos < len(words):
                    used = flow.feed_replay_words(words[pos : pos + int(chunks[j % len(chunks)])])
                    assert used > 0
                    pos += used
                    j += 1
                else:
                    flow.end_input()
            flow.step()
        assert flow.cycle < 2000000
    return flow.event, flow, pos


def same_values(a, b):
    for key in ["status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload"]:
        assert a["frame"].get(key) == b["frame"].get(key), key
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(a[key], b[key])
    assert a["timing_loop"] == b["timing_loop"]


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 19), (3, 12, 8, 47), (4, 10, 4, 23)])
def test_ready_handoff_and_overlap_reuse_match_original_word_receiver(ft, mcs, pilot, length):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    baseline, _ = consume_fit(profile, words, selection, [32])
    reference, old, _ = replay(profile, words, selection)
    candidate, new, _ = replay(profile, words, selection, reuse=True)
    same_values(baseline, reference)
    same_values(reference, candidate)
    assert candidate["frame"]["payload"] == payload
    assert new.stats["cache_reads"] == len(profile.sync_symbols())
    assert new.stats["data_queries"] == old.stats["data_queries"]
    assert new.stats["raw_reads"] + new.stats["overlap_words_reused"] == old.stats["raw_reads"]
    assert new.stats["overlap_words_reused"] > 0
    assert (
        new.history is new.prepared.stencil.history
        and new.kernel is new.prepared.stencil.interpolator
    )
    assert new.consumer.reference_memory()["private_raw_samples"] == 0
    assert new.estimate()["new_controller_state_bits"] <= 2048
    assert new.estimate()["source_staging_words"] == 4 and new.estimate()["extra_DSP"] == 0
    assert not candidate["replay_transport"]["cycle_evidence_valid"]
    assert candidate["replay_transport"]["complete_latency_cycles"] is None
    assert len(candidate["symbols"]) == new.consumer.frame_end_symbol
    wrapped = MacStreamReceiver._wrap(None, [candidate])
    assert len(wrapped.packets) == 1 and wrapped.packets[0].mac_payload == payload
    assert new.history.pending is None and new.kernel.live == 0 and new.cache.pending is None
    with pytest.raises(RuntimeError):
        new.prepared.step()
    with pytest.raises(RuntimeError):
        new.prepared.abort()
    with pytest.raises(RuntimeError):
        new.prepared.end_input()
    with pytest.raises(ValueError):
        SelectedFrameReplay(profile, new.prepared, **selection)


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_bad_header_payload_and_length_never_choose_another_answer(damage, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, **damage)
    words = quantize_input(values)
    baseline, _ = consume_fit(profile, words, selection, [32])
    event, flow, _ = replay(profile, words, selection, reuse=True)
    same_values(baseline, event)
    assert event["frame"]["status"] == status and event["frame"]["payload"] is None
    assert not MacStreamReceiver._wrap(None, [event]).packets
    if status != "bad_data_crc":
        assert flow.consumer.frame_end_symbol is None


def test_buffered_unrelated_suffix_is_returned_without_copy_or_silent_discard():
    profile = ordinary(2, 8, 0)
    values, payload, selection = selected_input(profile, length=1)
    words = quantize_input(np.r_[values, np.zeros(3000)])
    selection["sample_limit"] = len(words)
    baseline, _ = consume_fit(profile, words, selection, [32])
    service, pos = prepare(profile, words, selection, slack=400)
    target = pos + 256
    origin = selection["start_tick"] // 8 - 1
    while pos < target:
        c = service.cycle
        if c == math.ceil((pos - origin) * 200_000_000 / 16_000_000):
            assert service.stencil.history.write(pos, words[pos], c)
            pos += 1
        service.step()
    flow = SelectedFrameReplay(profile, service, reuse_stencil=True, **selection)
    while not flow.done:
        if flow.pending_service is not None:
            flow.resolve_functional()
        else:
            assert not flow.wants_replay_input
            flow.step()
    event = flow.event
    same_values(baseline, event)
    assert event["frame"]["payload"] == payload and flow.history.end == pos
    suffix = event["replay_transport"]["unconsumed_source_suffix"]
    assert suffix > 128 and suffix == flow.history.live
    assert event["delivered_after_filtered_samples"] > baseline["delivered_after_filtered_samples"]
    for k in range(flow.history.head, flow.history.end):
        assert flow.history.words[k % 512] == words[k]
    assert flow.history.head == flow.required_end
    assert flow.feed_replay_words([2**46]) == 0


def test_bus_edges_cannot_implicitly_execute_unknown_arithmetic_or_hide_live_overflow():
    profile = ordinary(4, 8, 0)
    values, _, selection = selected_input(profile, length=255)
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    service, pos = prepare(profile, words, selection)
    flow = SelectedFrameReplay(profile, service, **selection)
    while flow.pending_service is None:
        flow.step()
    assert (
        flow.pending_service["kind"] == "symbol"
        and flow.pending_service["arithmetic_cycles"] is None
    )
    for _ in range(17):
        flow.step()
    assert flow.consumer._active["count"] == 0 and flow.cycle_evidence_valid
    while not flow.done:
        flow.write_input(words[pos])
        pos += 1
        flow.step()
    assert flow.event["frame"]["status"] == "history_overflow"
    assert flow.stats["input_overflow"] == 1 and flow.stats["functional_resolutions"] == 0
    assert flow.consumer.last_event["sync_symbols_consumed"] == 0
    assert flow.history.live <= 512 and flow.cache.cache.state == "released"
    assert not MacStreamReceiver._wrap(None, [flow.event]).packets


@pytest.mark.parametrize(
    "count,status", [(127, "truncated_header"), (126 + 256 + 63 + 3, "truncated_data")]
)
def test_eof_drains_observed_queries_and_never_manufactures_future(count, status):
    profile = ordinary(4, 8, 0)
    values, _, selection = selected_input(profile, length=255)
    words = quantize_input(values)
    cut = selection["start_tick"] // 8 + 4 * count
    baseline, _ = consume_fit(profile, words[:cut], selection, [32])
    event, flow, pos = replay(profile, words[:cut], selection, reuse=True)
    same_values(baseline, event)
    assert event["frame"]["status"] == status and event["frame"]["payload"] is None
    assert pos == cut and flow.eof


def test_wrong_seed_illegal_length_and_arbitrary_input_blocks_use_the_same_real_gate():
    profile = ordinary(3, 12, 8)
    values, _, selection = selected_input(profile, length=79)
    words = quantize_input(values)
    for bad in [
        replace(profile, max_payload_bytes=31),
        replace(profile, config=replace(profile.config, head_llid=7)),
    ]:
        event, _, _ = replay(bad, words, selection, reuse=True)
        assert event["frame"]["status"] in ("bad_header_crc", "invalid_length")
        assert event["frame"]["payload"] is None
    a, _, _ = replay(profile, words, selection, reuse=True, chunks=[1])
    b, _, _ = replay(profile, words, selection, reuse=True, chunks=[7, 31, 3, 19])
    same_values(a, b)
    assert a["replay_transport"] == b["replay_transport"]


def test_terminal_MAC_and_source_ownership_transfer_once_without_extra_tail():
    profile = ordinary(2, 8, 0)
    values, payload, selection = selected_input(profile, length=7)
    words = quantize_input(np.r_[values, np.zeros(500)])
    selection["sample_limit"] = len(words)
    service, _ = prepare(profile, words, selection, slack=128)
    unclaimed = SelectedFrameReplay(profile, service, reuse_stencil=True, **selection)
    with pytest.raises(RuntimeError):
        unclaimed.take_mac_result()
    with pytest.raises(RuntimeError):
        unclaimed.release_source()
    unclaimed.abort()
    assert (
        unclaimed.take_mac_result().packets == []
        and unclaimed.take_mac_result().diagnostics["events"] == []
    )
    source = unclaimed.release_source()
    assert source is unclaimed.history and unclaimed.release_source() is None
    _event, flow, _ = replay(profile, words, selection, reuse=True, slack=128)
    assert flow.take_mac_result().packets[0].mac_payload == payload
    assert not flow.take_mac_result().packets
    assert flow.release_source() is flow.history
    assert flow.release_source() is None


def test_context_mismatch_and_input_guards_do_not_consume_prepared_credit():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=3)
    words = quantize_input(values)
    service, _ = prepare(profile, words, selection)
    for changed in [
        dict(start_tick=selection["start_tick"] + 1),
        dict(delta=selection["delta"] + 0.1),
    ]:
        with pytest.raises(ValueError):
            SelectedFrameReplay(profile, service, **(selection | changed))
        assert service.ready and not service.fit.output_taken
    with pytest.raises(ValueError):
        SelectedFrameReplay(profile, service, reuse_stencil=1, **selection)
    flow = SelectedFrameReplay(profile, service, **selection)
    for bad in [[], [0] * 33, np.zeros((2, 2))]:
        with pytest.raises(ValueError):
            flow.feed_replay_words(bad)
    with pytest.raises(RuntimeError):
        flow.resolve_functional()
    with pytest.raises(ValueError):
        flow.abort("success")
    flow.end_input()
    with pytest.raises(RuntimeError):
        flow.write_input(1)


def test_valid_B1_CRC_with_unsupported_MCS_is_not_delivered():
    profile = ordinary(3, 8, 0)
    values, _, selection = selected_input(profile, mcs=15, length=19)
    words = quantize_input(values)
    event, flow, _ = replay(profile, words, selection, reuse=True)
    assert event["frame"]["status"] == "unsupported_mcs" and event["frame"]["header_crc"] is True
    assert flow.consumer.frame_end_symbol is None and not flow.take_mac_result().packets


def test_unverified_header_blocks_payload_queries_until_real_parse_completes():
    profile = ordinary(3, 8, 0)
    values, _, selection = selected_input(profile, length=31)
    words = quantize_input(values)
    service, pos = prepare(profile, words, selection)
    flow = SelectedFrameReplay(profile, service, reuse_stencil=True, **selection)
    while flow.pending_service is None or flow.pending_service["kind"] != "header":
        if flow.pending_service is not None:
            flow.resolve_functional()
        else:
            if flow.wants_replay_input:
                pos += flow.feed_replay_words(words[pos : pos + 1])
            flow.step()
    assert flow.consumer.frame_end_symbol is None
    before = flow.stats.copy()
    for _ in range(19):
        flow.step()
    assert flow.stats["raw_reads"] == before["raw_reads"]
    assert flow.stats["data_queries"] == before["data_queries"]
    flow.resolve_functional()
    assert flow.consumer.header_result["header_crc"] is True
    assert flow.consumer.frame_end_symbol > flow.consumer._active["count"]
    assert flow.stats["header_operations"] == 1
    flow.abort()
    assert not flow.take_mac_result().packets
