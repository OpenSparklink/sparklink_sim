"""Actual selected words→finite init/loops/head and explicit remaining payload barrier."""

from functools import partial
from unittest.mock import patch

import pytest

from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay
from tests.captured_frame_reference import delivered, receive
from tests.initialized_capture_reference import captured, drive
from tests.test_coherent_captured_frame import profile_case


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_actual_unknown_start_source_to_existing_finite_frame_and_MAC(ft):
    profile, iq, quiet, expected = profile_case(ft, size=23, pilots=0 if ft == 2 else 8)
    factory = partial(InitializedCapturedFrameReplay, payload_mode="functional")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", factory):
        result, frames, reads, _, source, _ = receive(
            profile, iq, quiet, retire_before_fine=True)
    assert delivered(frames) == [expected]
    event = next(e for e, mac in frames if mac.packets)
    assert event["frame"]["header_crc"] is event["frame"]["data_crc"] is True
    assert event["frame"]["length_ok"] and event["frame"]["mcs"] == profile.config.mcs_index
    detail = event["initialized_capture_bridge"]
    assert detail["statistics"]["functional_payloads"] == 1
    assert detail["peak_private_words"] <= 1024
    assert not detail["complete_fixed_or_cycles"] and detail["payload_cycles"] is None
    assert event["complete_latency_cycles"] is None
    assert result["peak_source_operations_per_primitive"] <= 2
    assert source.done and len(reads) == result["frame_prefix_reads"] + result["frame_source_reads"]


@pytest.fixture(scope="module")
def selected():
    return captured(size=23)


def test_stop_at_actual_payload_barrier_preserves_head_without_fake_CRC_or_delivery(selected):
    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    event, mac = drive(owner, selection, edge, words)
    assert event["frame"]["status"] == "payload_service_unmodeled"
    assert event["frame"]["header_crc"] is event["frame"]["length_ok"] is True
    assert event["frame"]["data_crc"] is event["frame"]["payload"] is None
    assert not mac.packets and owner.bridge_stats["functional_payloads"] == 0
    assert owner.bridge_stats["unmodeled_payload_stops"] == 1
    assert owner.take() is None
    owner.release()
    assert owner.history is owner.flow is owner.selection is None


@pytest.mark.parametrize("mode", [None, True, "free", 0])
def test_payload_mode_must_be_explicit_before_any_owner_is_created(mode):
    profile, *_ = profile_case()
    with pytest.raises(ValueError, match="payload mode"):
        InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode=mode)


def test_constructor_bad_profile_is_rejected_without_consuming_a_selection():
    with pytest.raises(ValueError, match="PSK profile"):
        InitializedCapturedFrameReplay(None, coherent_threshold=.45, payload_mode="stop")


def test_bad_context_and_raw_admission_do_not_partially_advance_source(selected):
    from copy import deepcopy

    profile, words, _, selection, edge = selected
    limited = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                            payload_mode="stop", sample_limit=4)
    with pytest.raises(ValueError, match="sync future"):
        limited.begin(selection, edge)
    assert not limited.active and limited.cycle == -1
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    owner.begin(selection, edge)
    with pytest.raises(RuntimeError, match="taken"):
        owner.release()
    while not owner.needs_word:
        owner.tick(owner.cycle + 1)
    before = owner.index, deepcopy(owner.stats), owner.pending, owner.history.end
    for index, word in [(owner.source_index + 1, words[owner.source_index]),
                        (owner.source_index, 1 << 46), (owner.source_index, True)]:
        with pytest.raises(ValueError):
            owner.issue_read(index, word, owner.cycle)
        assert (owner.index, owner.stats, owner.pending, owner.history.end) == before
    owner.issue_read(owner.source_index, words[owner.source_index], owner.cycle)
    with pytest.raises(RuntimeError, match="Drain"):
        owner.end_input(owner.cycle)
    owner.abort("cancelled", owner.cycle + 1)
    event, mac = owner.take()
    assert not mac.packets and event["frame"]["status"] == "cancelled"
    owner.release()


@pytest.mark.parametrize("phase", ["confirmation", "prefill", "initial_fit", "frame"])
def test_cancellation_has_no_late_word_head_MAC_or_private_owner_leak(selected, phase):
    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    owner.begin(selection, edge)
    while True:
        matched = (owner.phase == phase or (owner.flow is not None
                                           and owner.flow.phase == phase))
        if matched:
            break
        owner.tick(owner.cycle + 1)
        if owner.needs_word:
            owner.issue_read(owner.source_index, words[owner.source_index], owner.cycle)
        assert owner.cycle < edge + 100000
    owner.abort("source_epoch_aborted", owner.cycle + 1)
    event, mac = owner.take()
    assert event["frame"]["status"] == "source_epoch_aborted" and not mac.packets
    assert event["frame"]["payload"] is None
    assert owner.pending is None and not owner.needs_word
    if owner.flow is not None:
        assert owner.flow.done
    owner.release()
    assert owner.flow is owner.history is None


@pytest.mark.parametrize("case,status", [("bad_head", "bad_header_crc"),
                                         ("bad_data", "bad_data_crc")])
def test_real_CRC_failures_remain_failures_without_alternate_selection(case, status):
    profile, words, _, selection, edge = captured(size=31, **{case: True})
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    event, mac = drive(owner, selection, edge, words)
    assert event["frame"]["status"] == status and not mac.packets
    assert event["selection_id"] == selection.ident
    if case == "bad_head":
        assert event["frame"]["header_crc"] is False
        assert event["frame"]["data_crc"] is None
        assert owner.bridge_stats["functional_payloads"] == 0
        assert event["copied_source_end"] <= owner.origin + owner.flow.frame.future_end
    else:
        assert event["frame"]["header_crc"] is True and event["frame"]["data_crc"] is False


@pytest.mark.parametrize("when,status", [("confirmation", "truncated_header"),
                                         ("prefill", "truncated_header"),
                                         ("header", "truncated_data")])
def test_EOF_drains_actual_services_without_padding_or_success(selected, when, status):
    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    sent = False

    def eof(node):
        nonlocal sent
        match = node.phase == when or (when == "header" and node.header is not None)
        if match and not sent and node.pending is None:
            node.end_input(node.cycle)
            sent = True

    event, mac = drive(owner, selection, edge, words, after_tick=eof)
    assert sent and not mac.packets and event["frame"]["status"] == status
    assert event["frame"]["data_crc"] is None
    if when == "header":
        assert event["frame"]["header_crc"] is True


def test_terminal_must_be_taken_before_private_history_can_be_released(selected):
    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    event, _ = drive(owner, selection, edge, words)
    # drive took it;restore only the test's external-consumption flag to exercise the guard.
    owner._taken = False
    token = owner.history._initialized_owner
    with pytest.raises(RuntimeError, match="taken"):
        owner.release()
    assert owner.history._initialized_owner is token and owner.event is event
    owner._taken = True
    owner.release()


def test_prefetch_bound_is_below_three_minimum_steps_and_cost_is_not_a_full_receiver(selected):
    from fractions import Fraction

    from nearlink_sdr.phy.timing_word import KP_Q48, LOWER_Q32

    max_kp = round(Fraction(KP_Q48 * (2 << 32), 1 << 48))
    assert 3 * (LOWER_Q32 - max_kp) > 9 * 2**32
    profile, *_ = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    cost = owner.estimate()
    assert cost["new_private_source_BRAM36"] == 2
    assert cost["new_bridge_control_bits"] <= 2048
    assert cost["actual_source_read_ports_added"] == 0
    assert cost["register_bits_upper"] is cost["known_register_bits_lower"] is None
    assert cost["known_register_reservation_subtotal"] > 0
    assert (cost["initialized_static_DSP"] == 19
            and not cost["coherent_and_initial_fit_DSP_sharing_credit"])
    assert cost["payload_cycles"] is cost["complete_latency_cycles"] is None


@pytest.mark.parametrize("policy,status", [("length", "invalid_length"),
                                           ("MCS", "unsupported_mcs")])
def test_valid_CRC_does_not_authorize_illegal_length_or_MCS(selected, policy, status):
    from dataclasses import replace

    profile, words, _, selection, edge = selected
    profile = (replace(profile, max_payload_bytes=16) if policy == "length"
               else replace(profile, allowed_mcs=(12,)))
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    event, mac = drive(owner, selection, edge, words)
    assert event["frame"]["status"] == status
    assert event["frame"]["header_crc"] is True and not mac.packets
    assert event["frame"]["data_crc"] is event["frame"]["payload"] is None
    assert owner.bridge_stats["functional_payloads"] == 0


def test_zero_energy_selected_cache_rejects_without_allocating_a_tail_or_decoding(selected):
    from dataclasses import replace

    profile, _, _, selection, edge = selected
    selection = replace(selection, words=((0, 0),) * 32, correlation=(0, 0),
                        numerator=0, energy=0)
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    event, mac = drive(owner, selection, edge, [])
    assert event["frame"]["status"] == "sync_rejected" and not mac.packets
    assert event["frame"]["header_crc"] is event["frame"]["data_crc"] is None
    assert owner.stats["coherent_accepted"] == 0
    assert owner.history is owner.flow is None
    assert owner.bridge_stats["copied_words"] == 0


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_initial_failure_releases_but_does_not_consume_the_prefetched_suffix(ft):
    """Inject inconsistent raw/cache ports to exercise cleanup,not RF performance."""
    from dataclasses import replace

    from nearlink_sdr.sim.head_prefetched_capture import HeadPrefetchedCaptureReplay

    profile, words, _, selection, edge = captured(ft, size=23, pilots=8)
    owner = HeadPrefetchedCaptureReplay(profile, coherent_threshold=.45, payload_mode="stop")
    processed = []

    def observe(node):
        if node.flow is not None and not node.flow.done:
            processed.append(node.origin + node.history.head)

    # The accepted cache is kept;zero raw port words force an invalid finite fit.
    event, mac = drive(owner, selection, edge, [0] * len(words), after_tick=observe)
    assert event["frame"]["status"] == "invalid_initial_fit" and not mac.packets
    assert owner.flow.frame is None and owner.history.live == 0
    assert owner.flow.stats["initial_discarded_words"] > 0
    assert event["consumed_source_end"] == processed[-1]
    assert event["consumed_source_end"] == owner.origin + owner.flow.consumed_history_end
    assert event["copied_source_end"] > event["consumed_source_end"]
    assert event["frame"]["header_crc"] is event["frame"]["data_crc"] is None
    frontier = owner.retired_through
    owner.release()

    # A descriptor inside that unused suffix must survive the retirement gate.
    # Only the ownership gate is tested;this translated cache is not a new RF trial.
    shift = frontier + 2 - selection.source_start_tick // 8
    later = replace(selection, ident=selection.ident + 1,
                    source_end=selection.source_end + shift)
    owner.begin(later, owner.cycle + 1)
    assert owner.active and not owner.done and owner.retired_through == frontier
    owner.abort("cancelled", owner.cycle + 1)
    assert not owner.take()[1].packets
    assert owner.retired_through == frontier
    owner.release()


@pytest.mark.parametrize("mode", ["functional", "stop"])
def test_downstream_pause_holds_unknown_payload_and_never_becomes_free_cycles(selected, mode):
    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode=mode)
    owner.begin(selection, edge)
    while owner.flow is None or owner.flow.pending_service is None:
        owner.tick(owner.cycle + 1)
        if owner.needs_word:
            owner.issue_read(owner.source_index, words[owner.source_index], owner.cycle)
        assert owner.cycle < edge + 100000
    end, functional = owner.history.end, owner.stats["functional_resolutions"]
    for _ in range(37):
        owner.tick(owner.cycle + 1, blocked=True)
        assert owner.take() is None and owner.history.end == end
        assert owner.stats["functional_resolutions"] == functional
    owner.tick(owner.cycle + 1)
    assert owner.done
    event, mac = owner.take()
    assert bool(mac.packets) == (mode == "functional")
    assert event["complete_latency_cycles"] is None


def test_reused_owner_resets_packet_nodes_but_retains_consumed_global_source_frontier(selected):
    from dataclasses import replace

    profile, words, expected, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    _, first = drive(owner, selection, edge, words)
    assert [p.mac_payload for p in first.packets] == [expected]
    frontier = owner.retired_through
    owner.release()
    shift = 8192
    following = replace(selection, ident=selection.ident + 1,
                        source_end=selection.source_end + shift,
                        available_cycle=owner.cycle + 17)
    event, second = drive(owner, following, following.available_cycle, [0] * shift + words)
    assert [p.mac_payload for p in second.packets] == [expected]
    assert owner.retired_through > frontier and owner.bridge_stats["initialized"] == 2
    assert event["selection_id"] == following.ident
    assert owner.word_stats == dict(issued=2, completed=2, cancelled=0)
    owner.release()


def test_existing_finite_owner_exactly_replays_actual_copy_edges_and_all_input_accounting(selected):
    import numpy as np

    from tests.initialized_capture_reference import existing_at_copy_edges

    profile, words, _, selection, edge = selected
    reads = []
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional", raw_observer=reads.append)
    event, _ = drive(owner, selection, edge, words)
    expected = existing_at_copy_edges(profile, selection, reads, owner.cycle)
    assert event["frame"] == expected.event["frame"]
    for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"]:
        np.testing.assert_array_equal(event[key], expected.event[key])
    assert owner.flow.stats == expected.stats
    assert owner.flow.frame.stats == expected.frame.stats
    assert (owner.flow.frame.consumer.detector.snapshot()
            == expected.frame.consumer.detector.snapshot())
    assert owner.flow.frame.consumer.pi.snapshot() == expected.frame.consumer.pi.snapshot()
    assert owner.flow.stats["input_words"] + owner.flow.frame.stats["input_words"] > 0
    assert owner.history.head == expected.history.head and owner.history.end == expected.history.end
    assert len(reads) == owner.history.stats["accepted_words"]


def test_same_age_float_reference_checks_real_numeric_budget_separate_from_algorithm_age(selected):
    import numpy as np

    from tests.test_feedback_timing_replay import run

    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    event, _ = drive(owner, selection, edge, words)
    origin = selection.source_end - selection.window
    local = words[origin:]
    floating = run(profile, local, dict(selection.reference_selection(), sample_limit=len(local),
                                       coherent_threshold=None), age=1, finite=False).event
    assert event["frame"]["payload"] == floating["frame"]["payload"]
    assert event["frame"]["header_crc"] == floating["frame"]["header_crc"]
    assert event["frame"]["data_crc"] == floating["frame"]["data_crc"]
    for key, limit in [("symbol_centers_filtered", .001), ("timing_rate_ppm", .1)]:
        assert np.max(abs(event[key] - floating[key])) <= limit
    difference = abs(event["symbols"] - floating["symbols"])
    scale = max(float(np.mean(abs(floating["symbols"])**2))**.5, 1e-30)
    assert float(np.mean((difference / scale)**2))**.5 <= .0001
    assert float(np.max(difference / scale)) <= .001


def test_valid_header_releases_at_its_real_tail_without_reading_unrelated_capture(selected):
    profile, words, expected, selection, edge = selected
    copies = []
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional", raw_observer=copies.append)
    event, mac = drive(owner, selection, edge, words+[0]*40000)
    assert [p.mac_payload for p in mac.packets] == [expected]
    origin = selection.source_end-selection.window
    assert event["copied_source_end"] <= origin+owner.flow.frame.future_end
    assert len(copies) == copies[-1]["index"]-copies[0]["index"]+1
    assert copies[-1]["index"] < len(words)
    assert event["consumed_source_end"] <= event["copied_source_end"]
    assert owner.stats["registered_reads"] == owner.stats["read_returns"] == len(copies)


def test_actual_word_receipt_crosses_legacy_85percent_guard_without_a_second_gate(selected):
    from dataclasses import replace
    from math import ceil

    from nearlink_sdr.sim.initialized_frame_replay import InitializedFrameReplay
    from tests.initialized_capture_reference import same_age_confirmed_float

    profile, words, expected, selection, edge = selected
    # Synthetic descriptor guard test,not a claim about RF reception at this score.
    selection = replace(
        selection, energy=ceil(selection.numerator/(.84**2*selection.template_energy)))
    assert selection.reference_selection()["differential_score"] < .85
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                          payload_mode="functional")
    event, mac = drive(owner, selection, edge, words)
    assert [p.mac_payload for p in mac.packets] == [expected]
    receipt = owner.word_receipt
    assert receipt.accepted and owner.flow._upstream_word_receipt is receipt
    assert owner.flow.frame.consumer._upstream_word_receipt is receipt
    assert event["coherent_word_service"]["word_decision"] is True
    fields = dict(selection.reference_selection(), sample_limit=2**20, coherent_threshold=None)
    with pytest.raises(ValueError, match="already accepted"):
        InitializedFrameReplay(profile, owner.history, start_cycle=owner.cycle, **fields)
    origin = selection.source_end-selection.window
    floating = same_age_confirmed_float(profile, words[origin:], fields, receipt)
    assert floating.event["frame"]["payload"] == expected
    assert floating.event["frame"]["data_crc"] is True


@pytest.mark.parametrize("change", ["bool", "future", "template", "position", "score",
                                  "samples", "second_gate"])
def test_receipt_cannot_be_a_success_flag_or_another_transaction(selected, change):
    from dataclasses import replace

    import numpy as np

    from nearlink_sdr.sim.initialized_frame_replay import InitializedFrameReplay

    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    drive(owner, selection, edge, words)
    receipt = owner.word_receipt
    fields = dict(selection.reference_selection(), start_cycle=owner.cycle)
    if change == "bool":
        receipt = True
    elif change == "future":
        fields["start_cycle"] = receipt.available_cycle-1
    elif change == "template":
        receipt = replace(receipt, phase_codes=tuple((v+1) % 16 for v in receipt.phase_codes))
    elif change == "position":
        fields["start_tick"] += 1
    elif change == "score":
        fields["differential_score"] -= .001
    elif change == "samples":
        fields["selected_sync"] = np.zeros_like(fields["selected_sync"])
    elif change == "second_gate":
        fields["threshold"] = .45
    with pytest.raises(ValueError, match="confirmation"):
        InitializedFrameReplay.checked_word_confirmation(receipt, profile, **fields)


@pytest.mark.parametrize("capacity", [None, 512])
def test_legacy_selected_constructor_initial_view_belongs_to_its_actual_history(selected, capacity):
    from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver

    profile, _, _, selection, _ = selected
    owner = SelectedPSKReceiver(profile, **selection.reference_selection(), sample_limit=4096,
                               coherent_threshold=None, timing_backend="python",
                               history_capacity=capacity)
    assert owner._filtered_values.base is owner._retained
    assert len(owner._filtered_values) == owner.input_samples == 0
    assert len(owner._retained) == (4096 if capacity is None else 2*capacity)


def test_new_cost_never_inherits_zero_peak_or_an_absent_float_raw_mirror(selected):
    profile, words, _, selection, edge = selected
    owner = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    drive(owner, selection, edge, words)
    actual = owner.flow.frame.consumer
    assert actual.reference_memory()["persistent_numpy_storage_bytes"] > 0
    assert len(actual._retained) == 0 and owner.history.capacity == 1024
    cost = owner.estimate()
    assert cost["actual_peak_reference_bytes"] is None and cost["reference_peak_not_measured"]
    assert cost["functional_raw_history_samples"] == cost["functional_host_mirror_copies"] == 0
    assert not cost["old_float_raw_consumer_constructed"]
    assert cost["private_history"]["capacity_words"] == 1024
    assert cost["private_history"]["bram36"] == 2
    assert cost["known_register_bits_lower"] is cost["register_bits_upper"] is None
    assert cost["known_register_reservation_subtotal"] == 10726
