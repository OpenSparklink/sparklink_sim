"""Causal captured-window ownership,finite setup and continuous-source exits."""

import math
import random
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from functools import partial

import numpy as np
import pytest

from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.sync_fit_word import GAIN_FLOOR
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.initialized_frame_replay import (
    InitializedFrameReplay,
    TimingContextInstallService,
)
from nearlink_sdr.sim.packed_head_replay import PackedHeadSamplingReplay
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from nearlink_sdr.sim.split_timing_flow import split_timing_states
from tests.test_b1_wire_length import b1_header_wire
from tests.test_b1_wire_length import profile as b1_profile
from tests.test_head_word_replay import run
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_strict_receive import iq_record

CASES = [(2, 8, 0, 19), (3, 12, 8, 47), (4, 10, 4, 23)]


def boot(profile, words, selection, *, clock=200, slack=0):
    first = selection["start_tick"] // 8 - 1
    end = first + 1 + 4 * (len(profile.sync_symbols()) - 1) + 3 + bool(selection["start_tick"] % 8)
    history = ReceiveHistoryPort(1024, first_index=first)
    for i in range(first, end + slack):
        assert history.write(i, words[i], math.ceil((i - first) * clock / 16))
    flow = InitializedFrameReplay(profile, history, start_cycle=history.cycle, **selection)
    return flow, end + slack, first


def clocked(profile, words, selection, *, clock=200, blocked=False, batches=(1,), eof=False):
    flow, pos, first = boot(profile, words, selection, clock=clock)
    if eof:
        flow.end_input()
    groups, group = 0, 0
    while not flow.done:
        # Batches change only caller control boundaries,never the one-word/edge calendar.
        for _ in range(batches[group % len(batches)]):
            if flow.done:
                break
            if flow.pending_service is not None and not blocked:
                flow.resolve_functional()
                continue
            if not eof and pos < len(words) and flow.cycle >= math.ceil((pos - first) * clock / 16):
                used = flow.write_input(words[pos])
                pos += int(used)
            flow.step()
            assert flow.cycle < 200000
        group += 1
        groups += 1
    return flow, groups


def independent_power(mantissa, exponent):
    shift = mantissa.bit_length() - 30
    if shift > 0:
        q, r = divmod(mantissa, 1 << shift)
        q += 2 * r > 1 << shift or (2 * r == 1 << shift and q & 1)
    else:
        q = mantissa << -shift
    if q == 1 << 30:
        q >>= 1
        shift += 1
    value = Fraction(q) * Fraction(2) ** (exponent + shift)
    floor = Fraction(GAIN_FLOOR[0]) * Fraction(2) ** GAIN_FLOOR[1]
    if value < floor:
        return independent_power(*GAIN_FLOOR)
    return q, exponent + shift


@pytest.mark.parametrize("hold", [32, 62, 126])
def test_actual_three_edges_and_independent_RNE_initial_power(hold):
    rng = random.Random(272721 + hold)
    powers = [
        (1, -300),
        (2**30 - 1, 0),
        (2**30, 0),
        (2**54 - 1, 40),
        (2**32 + 2, -90),
        (2**32 + 6, -90),
        (1, 250),
    ]
    powers += [(rng.randrange(1, 2**54), rng.randrange(-200, 200)) for _ in range(50)]
    for i, power in enumerate(powers):
        fit = dict(valid=True, symbols=hold, power=power)
        untouched = deepcopy(fit)
        service = TimingContextInstallService(start_cycle=17)
        assert service.issue(fit, 8 * (i + 1) + i % 8, hold)
        assert service.fit is fit and not service.issue(fit, 8, hold)
        for stage in ["floor", "install", "ready"]:
            service.step()
            assert service.mode == stage
        assert service.cycle == 20 and service.ready and fit == untouched
        detector, pi = service.result
        assert detector.power_word == independent_power(*power)
        a, b = split_timing_states(power, 8 * (i + 1) + i % 8, hold, TimingWordFormat(30, 32))
        assert detector.snapshot() == a.snapshot() and pi.snapshot() == b.snapshot()
        saved = detector.snapshot(), pi.snapshot()
        for _ in range(5):
            service.step()
        assert (detector.snapshot(), pi.snapshot()) == saved
        assert service.take() == (detector, pi) and service.take() is None
        assert service.stats == dict(accepted=1, completed=1, held_edges=5)
        assert service.estimate()["DSP"] == service.estimate()["new_RAM"] == 0


@pytest.mark.parametrize("cut", [0, 1, 2, 3])
def test_setup_abort_has_no_stale_or_second_packet_receipt(cut):
    flow = TimingContextInstallService()
    fit = dict(valid=True, symbols=32, power=(2**40, -40))
    flow.issue(fit, 13, 32)
    for _ in range(cut):
        flow.step()
    flow.abort()
    assert flow.take() is None and not flow.ready and flow.result is flow.fit is None
    assert not flow.issue(fit, 13, 32)
    with pytest.raises(RuntimeError, match="Aborted"):
        flow.step()


@pytest.mark.parametrize("bad", [None, (0, 0), (1, 512), (2**54, 0), (True, 0)])
def test_invalid_initial_power_is_not_accepted_as_a_successful_fit(bad):
    flow = TimingContextInstallService()
    with pytest.raises(ValueError):
        flow.issue(dict(valid=True, symbols=32, power=bad), 9, 32)
    assert flow.mode == "idle" and flow.stats["accepted"] == 0


@pytest.mark.parametrize("ft,mcs,pilot,length", CASES)
def test_continuous_initialization_through_real_head_matches_selected_reference(
    ft, mcs, pilot, length
):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    old, _, _ = run(profile, words, selection, partial(PackedHeadSamplingReplay, error_age=1))
    new, _ = clocked(profile, words, selection)
    assert new.event["frame"] == old["frame"] and new.event["frame"]["payload"] == payload
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(new.event[key], old[key])
    assert new.transfer_cycle - new.fit_ready_cycle == 3
    assert new.prepared.fit.output_taken and new.install.mode == "taken"
    assert new.frame.head_owner is new.head_owner
    assert new.frame.kernel is new.prepared.stencil.interpolator
    assert new.frame.history is new.history and new.history.capacity == 1024
    assert new.history.stats["peak_live"] < 1024 and not new.stats["input_overflow"]
    assert new.frame.stats["functional_resolutions"] == 1  # Payload only;no zero-cycle credit.
    assert new.take_mac_result().packets[0].mac_payload == payload
    assert (
        not new.take_mac_result().packets
        and not new.event["replay_transport"]["cycle_evidence_valid"]
    )
    assert new.release_source() is new.history and new.release_source() is None
    cost = new.estimate()
    assert cost["borrowed_history"]["bram36"] == 2
    assert cost["borrowed_history"]["global_reads_per_clock"] == 1
    assert cost["head_reserved_through_init"]["BRAM36_sites_upper"] == (2 if ft == 2 else 3)
    assert cost["RRC_capture_and_payload_cycles"] is None and not cost["full_receiver16MSps"]
    assert cost["static_DSP_no_unproved_sharing"] == 19 and not cost["old18DSP_budget_passed"]
    assert cost["source_cache_head_BRAM36_minimum"] == (5 if ft == 2 else 6)


@pytest.mark.parametrize("ft,mcs,pilot,length", CASES)
def test_arbitrary_caller_batch_boundaries_preserve_exact_source_and_sampling_calendar(
    ft, mcs, pilot, length
):
    profile = ordinary(ft, mcs, pilot)
    values, _, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    reference, _ = clocked(profile, words, selection)
    rng = random.Random(272731 + ft)
    for batches in [(7, 31, 1), tuple(rng.randrange(1, 80) for _ in range(13))]:
        trial, _ = clocked(profile, words, selection, batches=batches)
        assert trial.cycle == reference.cycle and trial.history.stats == reference.history.stats
        assert trial.event["frame"] == reference.event["frame"]
        for key in [
            "symbols",
            "symbol_centers_filtered",
            "timing_rate_ppm",
            "timing_detector_error",
        ]:
            np.testing.assert_array_equal(trial.event[key], reference.event[key])


@pytest.mark.parametrize("ft,mcs,pilot,length", CASES)
def test_blocked_payload_records_bounded_resource_loss_instead_of_fake_delivery(
    ft, mcs, pilot, length
):
    profile = ordinary(ft, mcs, pilot)
    values, _, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(8192)])
    selection["sample_limit"] = len(words)
    flow, _ = clocked(profile, words, selection, blocked=True)
    assert flow.event["frame"]["status"] == "history_overflow"
    assert flow.frame.consumer.head_receipt_taken
    assert flow.history.stats["peak_live"] == 1024 and not flow.take_mac_result().packets
    assert not flow.write_input(0)


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_EOF_during_actual_initialization_drains_then_reports_truncated_header(ft):
    p = ordinary(ft, 8, 0)
    values, _, selection = selected_input(p, length=11)
    words = quantize_input(values)
    flow, _, _ = boot(p, words, selection)
    flow.end_input()
    with pytest.raises(RuntimeError, match="EOF"):
        flow.write_input(0)
    while not flow.done:
        flow.step()
    assert flow.event["frame"]["status"] == "truncated_header"
    assert not flow.take_mac_result().packets and flow.history.pending is None


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
        ({"mcs": 15}, "unsupported_mcs"),
    ],
)
def test_selected_bad_head_length_MCS_or_data_does_not_retry(damage, status):
    p = ordinary(2 if damage.get("empty") else 3, 12, 8)
    values, _, selection = selected_input(p, length=17, **damage)
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    flow, _ = clocked(p, words, selection)
    assert flow.event["frame"]["status"] == status and not flow.take_mac_result().packets
    assert flow.frame.head_owner.stats["header_issues"] == 1
    if status != "bad_data_crc":
        assert flow.frame.query_count == flow.frame.consumer._head_required
        assert flow.frame.consumer.frame_end_symbol is None


@pytest.mark.parametrize(
    "case", ["no_sync_future", "expired", "pending", "wrong_capacity", "old_clock"]
)
def test_bad_captured_window_is_rejected_before_ownership_or_source_changes(case):
    p = ordinary()
    values, _, selection = selected_input(p)
    words = quantize_input(values)
    _, end, first = boot(p, words, selection)
    h = ReceiveHistoryPort(1024, first_index=first)
    for i in range(first, end):
        assert h.write(i, words[i], math.ceil((i - first) * 200 / 16))
    if case == "no_sync_future":
        h.end -= 1
    elif case == "expired":
        h.retire_before(first + 1, h.cycle)
    elif case == "pending":
        h.request(first, h.cycle)
    elif case == "wrong_capacity":
        h = ReceiveHistoryPort(512, first_index=first)
    before = deepcopy(h.__dict__)
    with pytest.raises(ValueError):
        InitializedFrameReplay(
            p, h, start_cycle=h.cycle - 1 if case == "old_clock" else max(0, h.cycle), **selection
        )
    assert h.__dict__ == before


@pytest.mark.parametrize("phase", ["initial_fit", "install"])
def test_initial_abort_and_overload_record_discarded_unverified_window(phase):
    p = ordinary()
    values, _, selection = selected_input(p)
    words = quantize_input(np.r_[values, np.zeros(8192)])
    selection["sample_limit"] = len(words)
    flow, pos, _ = boot(p, words, selection)
    if phase == "install":
        while flow.phase != phase:
            flow.step()
        start = flow.fit_ready_cycle - (1024 - flow.history.live) + 1
        flow, pos, _ = boot(p, words, selection)
        while flow.phase != phase:
            if flow.cycle >= start:
                assert flow.write_input(words[pos])
                pos += 1
            flow.step()
    else:
        flow.step()  # The captured preload already used the constructor's first edge.
    while not flow.done:
        used = flow.write_input(words[pos])  # Sustained overload,one word each edge.
        pos += int(used)
        flow.step()
    assert flow.event["frame"]["status"] == "history_overflow"
    assert flow.stats["input_overflow"] == 1 and flow.stats["initial_discarded_words"] == 1024
    assert flow.history.pending is None and not flow.history.live
    assert not flow.take_mac_result().packets
    assert flow.release_source() is flow.history


def test_zero_sync_fit_is_rejected_before_install_or_head_CRC():
    p = ordinary()
    values, _, selection = selected_input(p)
    words = quantize_input(np.zeros_like(values))
    flow, _, _ = boot(p, words, selection)
    while not flow.done:
        flow.step()
    assert flow.event["frame"]["status"] == "invalid_initial_fit"
    assert flow.install is None and flow.frame is None
    assert flow.event["frame"]["header_crc"] is None and not flow.take_mac_result().packets


@pytest.mark.parametrize("value", [True, 1.5])
def test_service_clock_is_an_integer(value):
    with pytest.raises(ValueError):
        TimingContextInstallService(start_cycle=value)


@pytest.mark.parametrize("ft", [3, 4])
def test_actual_CRC_valid_B1_length_smaller_than_data_CRC_is_rejected(ft):
    p = b1_profile(ft)
    symbols, _ = b1_header_wire(p, 1)
    iq, _, _, _ = iq_record(symbols, cfo=19000, ppm=-25, offset=0.413)
    values = np.convolve(iq, rrc_filter(0.4, 4))
    found = search_psk_sync(values, 4, p.sync_symbols(), 8)
    assert found is not None
    index = found["offset"]
    selection = dict(
        start_tick=round(8 * (found["phase"] + 4 * index)),
        delta=found["delta"],
        differential_score=found["score"],
        coherent_threshold=None,
        selected_sync=found["phase_symbols"][index : index + len(p.sync_symbols())],
    )
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    f, _ = clocked(p, words, selection)
    assert f.event["frame"]["status"] == "invalid_length"
    assert f.event["frame"]["header_crc"] is True and f.event["frame"]["data_crc"] is None
    assert f.frame.consumer.frame_end_symbol is None and not f.take_mac_result().packets


@pytest.mark.parametrize(
    "field,status", [("head_llid", "bad_header_crc"), ("crc_seed", "bad_data_crc")]
)
def test_wrong_actual_CRC_context_has_one_answer_and_no_delivery(field, status):
    p = ordinary(3, 12, 8)
    values, _, selection = selected_input(p, length=21)
    p = replace(p, config=replace(p.config, **{field: getattr(p.config, field) ^ 1}))
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    f, _ = clocked(p, words, selection)
    assert f.event["frame"]["status"] == status and not f.take_mac_result().packets
    assert f.frame.head_owner.stats["header_issues"] == 1


def test_duplicate_window_claim_and_premature_return_are_rejected():
    p = ordinary()
    values, _, selection = selected_input(p)
    words = quantize_input(values)
    f, _, _ = boot(p, words, selection)
    with pytest.raises(ValueError, match="exclusive"):
        InitializedFrameReplay(p, f.history, start_cycle=f.cycle, **selection)
    with pytest.raises(RuntimeError, match="Active"):
        f.release_source()
    with pytest.raises(RuntimeError, match="terminal"):
        f.take_mac_result()
    with pytest.raises(RuntimeError, match="actual frame"):
        f.resolve_functional()
    with pytest.raises(ValueError, match="explicit"):
        f.abort("silent_drop")
    f.abort()
    assert f.history._initialized_owner() is f
    assert f.release_source() is f.history and f.history._initialized_owner is None


def test_EOF_after_verified_head_keeps_data_unverified_and_drains_finite_states():
    p = ordinary(3, 12, 8)
    values, _, selection = selected_input(p, length=251)
    words = quantize_input(values)
    f, pos, first = boot(p, words, selection)
    while f.frame is None or not f.frame.consumer.head_receipt_taken:
        if f.cycle >= math.ceil((pos - first) * 200 / 16):
            assert f.write_input(words[pos])
            pos += 1
        f.step()
    assert not f.done
    f.end_input()
    while not f.done:
        f.step()
    assert f.event["frame"]["status"] == "truncated_data"
    assert f.event["frame"]["header_crc"] is True and f.event["frame"]["data_crc"] is None
    assert not f.take_mac_result().packets
    assert f.frame.kernel.live == 0 and not f.frame.error_messages


def test_extreme_power_exponent_overflow_is_explicit_without_saturation():
    f = TimingContextInstallService()
    assert f.issue(dict(valid=True, symbols=32, power=(2**53, 511)), 9, 32)
    with pytest.raises(OverflowError):
        f.step()
    assert not f.ready and f.result is None and f.cycle == 0


def test_default_selected_replay_keeps_its512_word_guard():
    p = ordinary()
    values, _, selection = selected_input(p)
    words = quantize_input(values)
    probe, _, _ = boot(p, words, selection)
    prepared = probe.prepared
    while not prepared.ready:
        prepared.step()
    assert isinstance(prepared, InitialSyncFitService)
    before = prepared.fit.result.copy()
    with pytest.raises(ValueError, match="Prepared geometry"):
        PackedHeadSamplingReplay(
            p,
            prepared,
            error_age=1,
            normalization_format=NormalizationFormat(),
            word_timing_format=TimingWordFormat(30, 32),
            reuse_stencil=True,
            **selection,
        )
    assert not prepared.fit.output_taken and prepared.fit.result == before


def test_bank_return_allows_next_profile_without_clearing_unrelated_suffix():
    p, q = ordinary(2, 8, 0), ordinary(3, 12, 8)
    a, payload_a, sa = selected_input(p, length=5)
    b, payload_b, sb = selected_input(q, length=11)
    offset = len(a) + 47
    words = quantize_input(np.r_[a, np.zeros(47), b, np.zeros(4096)])
    sa["sample_limit"] = len(words)
    sb["sample_limit"] = len(words)
    sb["start_tick"] += 8 * offset
    f, _, first = boot(p, words, sa)

    def drain(flow):
        while not flow.done:
            if flow.pending_service is not None:
                flow.resolve_functional()
                continue
            pos = flow.history.end
            if flow.cycle >= math.ceil((pos - first) * 200 / 16):
                assert flow.write_input(words[pos])
            flow.step()

    drain(f)
    assert f.take_mac_result().packets[0].mac_payload == payload_a
    h = f.release_source()
    prefix = [h.words[i % 1024] for i in range(h.head, h.end)]
    assert prefix and h._initialized_owner is None
    second_end = (
        sb["start_tick"] // 8 + 4 * (len(q.sync_symbols()) - 1) + 3 + bool(sb["start_tick"] % 8)
    )
    cycle = f.cycle
    lookback = max(64, len(q.sync_symbols()) + 32) * 4 + 4 * 32
    while h.end < second_end:
        pos = h.end
        cycle = max(cycle, math.ceil((pos - first) * 200 / 16))
        if cycle == h.last_write_cycle:
            cycle += 1
        # Inactive upstream search owns retention,with the existing fixed lookback.
        # No next-packet start or payload is used to decide what can expire.
        h.retire_before(max(h.head, h.end - lookback), cycle)
        assert h.write(pos, words[pos], cycle)
    assert all(h.words[i % 1024] == words[i] for i in range(h.head, h.end))
    g = InitializedFrameReplay(q, h, start_cycle=cycle, **sb)
    drain(g)
    assert g.take_mac_result().packets[0].mac_payload == payload_b
    assert g.frame.head_owner is not f.frame.head_owner and g.install.stats["completed"] == 1
    assert g.release_source() is h and h._initialized_owner is None


@pytest.mark.parametrize("ft,mcs,pilot,length", CASES)
@pytest.mark.parametrize("change_phase", ["initial_fit", "install", "frame"])
def test_caller_next_packet_config_cannot_change_current_owner(
    ft, mcs, pilot, length, change_phase
):
    profile = ordinary(ft, mcs, pilot)
    captured_profile = replace(profile)
    values, payload, selection = selected_input(profile, length=length)
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection["sample_limit"] = len(words)
    reference, _, _ = run(
        captured_profile, words, selection, partial(PackedHeadSamplingReplay, error_age=1)
    )
    flow, pos, first = boot(profile, words, selection)
    changed = False
    while not flow.done:
        if not changed and flow.phase == change_phase:
            # The caller reuses its negotiated config for a later packet.
            # Current packet CRC/length/mapping context must already be latched.
            profile.config.crc_seed ^= 1
            changed = True
        if flow.pending_service is not None:
            flow.resolve_functional()
            continue
        if pos < len(words) and flow.cycle >= math.ceil((pos - first) * 200 / 16):
            pos += int(flow.write_input(words[pos]))
        flow.step()
        assert flow.cycle < 200000
    assert changed and flow.profile is not profile
    assert flow.profile.config is not profile.config
    assert flow.profile == captured_profile == flow.head_owner.header.profile
    assert flow.event["frame"] == reference["frame"]
    assert flow.event["frame"]["payload"] == payload
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(flow.event[key], reference[key])
    assert flow.take_mac_result().packets[0].mac_payload == payload
    flow.release_source()
    # A fresh owner must latch the NEW config,not a global or previous snapshot.
    later, _, _ = boot(profile, words, selection)
    assert later.profile == profile and later.profile != captured_profile
    assert later.profile.config is not profile.config
    later.abort()
    later.release_source()
