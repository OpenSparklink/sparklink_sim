"""Bounded init prefetch changes availability,never the selected answer or head gate."""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import (
    KI_Q48,
    KP_Q48,
    LOWER_Q32,
    TimingWordFormat,
    WordTimingState,
    rne_shift,
)
from nearlink_sdr.sim.head_prefetched_capture import (
    MINIMUM_STEP_Q32,
    HeadPrefetchedCaptureReplay,
)
from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay
from tests.captured_frame_reference import delivered, receive
from tests.initialized_capture_reference import captured, drive
from tests.test_coherent_captured_frame import profile_case


@pytest.fixture(scope="module", params=[2, 3, 4])
def actual_selection(request):
    return captured(request.param, size=23, pilots=8)


def test_prefetch_during_initialization_preserves_actual_frame_and_all_sample_indices(
    actual_selection,
):
    profile, words, expected, selection, edge = actual_selection
    old = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                         payload_mode="functional")
    reference, _ = drive(old, selection, edge, words)
    reads, initial = [], []
    node = HeadPrefetchedCaptureReplay(profile, coherent_threshold=.45,
                                       payload_mode="functional", raw_observer=reads.append)

    def observe(owner):
        if owner.flow is not None and owner.flow.frame is None and owner.needs_word:
            initial.append(owner.source_index)
            assert owner.history.end < owner.head_floor_end
            assert owner.cycle >= owner.head_floor_available_cycle
            assert owner.flow.head_owner.end == owner._compiled_head_end

    event, mac = drive(node, selection, edge, words, after_tick=observe)
    assert initial and delivered([(event, mac)]) == [expected]
    assert event["frame"] == reference["frame"]
    for k in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(event[k], reference[k])
    indices = [r["index"] for r in reads]
    assert indices == list(range(indices[0], indices[-1] + 1))
    assert all(r["word"] == words[r["index"]] for r in reads)
    assert node.history.stats["peak_live"] <= 1024
    assert not node.flow.stats["input_overflow"]
    assert event["complete_latency_cycles"] is None


@pytest.mark.parametrize("phase", range(8))
def test_guaranteed_prefix_survives_actual_extreme_PI_commits(phase):
    # Exercise the real finite NCO/clamp,not a second closed-form copy of the bound.
    profile, *_ = profile_case(4, pilots=8)
    node = HeadPrefetchedCaptureReplay(profile, coherent_threshold=.45, payload_mode="stop")
    start = 8 * 37 + phase
    state = WordTimingState((1, 0), start, 126, TimingWordFormat(30, 32))
    for k in range(node._compiled_head_end - 1):
        error = 0 if k < 126 else -(2 << 32)
        ki, kp = (rne_shift(c * error, 48) for c in (KI_Q48, KP_Q48))
        before = state.next_q32
        state.commit(0, state.power_word, error, False, 0, 0, False, ki, kp)
        assert state.next_q32 - before >= MINIMUM_STEP_Q32
    assert 37 + node._head_offsets[phase] <= state.required_end
    assert state.omega_q32 >= LOWER_Q32 and state.rate_limit_events > 0


@pytest.mark.parametrize("damage,status", [("bad_head", "bad_header_crc"),
                                          ("bad_data", "bad_data_crc")])
def test_prefetch_never_uses_a_CRC_to_retry_or_read_past_a_bad_head(damage, status):
    profile, words, _, selection, edge = captured(size=17, **{damage: True})
    node = HeadPrefetchedCaptureReplay(profile, coherent_threshold=.45,
                                       payload_mode="functional")
    event, mac = drive(node, selection, edge, words)
    assert event["selection_id"] == selection.ident and event["frame"]["status"] == status
    assert not mac.packets and not event["bounded_head_prefetch"]["uses_received_length_or_CRC"]
    if damage == "bad_head":
        assert node.source_index <= node.origin + node.flow.frame.future_end
        assert node.bridge_stats["functional_payloads"] == 0
    node.release()
    assert node.head_floor_end == node.head_floor_available_cycle == 0
    assert node.history is node.flow is None


def test_actual_unknown_start_arbitrary_RRC_blocks_still_deliver():
    profile, iq, quiet, expected = profile_case(3, size=37, pilots=4)
    factory = partial(HeadPrefetchedCaptureReplay, payload_mode="functional")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", factory):
        run = receive(profile, iq, quiet, groups=(1, 97, 13, 256), retire_before_fine=True)
    assert delivered(run[1]) == [expected]
    assert run[0]["peak_source_operations_per_primitive"] <= 2


def test_head_prefetch_cost_has_no_new_history_port_or_unmodeled_free_datapath():
    profile, *_ = profile_case(4)
    old = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    new = HeadPrefetchedCaptureReplay(profile, coherent_threshold=.45, payload_mode="stop")
    a, b = old.estimate(), new.estimate()
    assert b["known_register_reservation_subtotal"] - a[
        "known_register_reservation_subtotal"] == 84
    assert b["new_bridge_control_bits"] == a["new_bridge_control_bits"] + 84 <= 2048
    assert b["new_private_source_BRAM36"] == a["new_private_source_BRAM36"] == 2
    assert b["actual_source_read_ports_added"] == a["actual_source_read_ports_added"] == 0
    assert b["bounded_head_prefetch"]["coefficient_bits"] == 88
    assert b["complete_latency_cycles"] is b["payload_cycles"] is None
    assert b["actual_peak_reference_bytes"] is b["known_register_bits_lower"] is None
    assert not b["full_fixed_or16MSps"] and not b["actual_synthesis"]
