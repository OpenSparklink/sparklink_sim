"""Legal consumption,not a new sync/timing answer;independent tail-envelope checks."""

from copy import deepcopy
from itertools import pairwise

import numpy as np
import pytest

from nearlink_sdr.sim.head_prefetched_capture import MINIMUM_STEP_Q32
from nearlink_sdr.sim.moving_tail_horizon import MovingFrameTailHorizon, MovingTailCaptureReplay
from nearlink_sdr.sim.private_prefetch import ContinuousPrivatePrefetch
from tests.captured_frame_reference import delivered
from tests.initialized_capture_reference import captured, drive


def finish(node):
    for _ in range(19):
        node.advance(node.cycle + 1)


def test_actual_old_eligibility_witness_becomes_available_after19_edges_not_at_issue():
    node = MovingFrameTailHorizon(2**20)
    assert node.issue(21731731511482, 1239, 1515)
    for edge in range(1, 19):
        node.advance(edge)
        assert node.published_boundary == 0
    node.advance(19)
    assert node.published_boundary == 6148 > 6080  # Actual old static stop,not truth.
    assert node.issue(21731731511482 + MINIMUM_STEP_Q32 + 2**32, 1240, 1515)
    assert node.published_boundary == 6148
    finish(node)
    assert node.published_boundary == 6149
    assert node.stats == dict(accepted=2, completed=2, cancelled=0)


@pytest.mark.parametrize("phase", range(8))
def test_moving_envelope_is_below_independent_actual_extreme_PI_tail(phase):
    from nearlink_sdr.phy.timing_word import (
        KI_Q48,
        KP_Q48,
        TimingWordFormat,
        WordTimingState,
        rne_shift,
    )

    count, n = 0, 1000
    state = WordTimingState((1, 0), 8 * 37 + phase, 32, TimingWordFormat(30, 32))
    node = MovingFrameTailHorizon(2**20)
    bounds, positions = [], []
    for index in range(n):
        positions.append(state.next_q32)
        if index % 101 == 0 or index == n - 1:
            assert node.issue(state.next_q32, state.count, n)
            position = state.next_q32 + (n - 1 - state.count) * MINIMUM_STEP_Q32
            finish(node)
            assert node.published_boundary == (position + 2**32 - 1) // 2**32 + 3
            bounds.append(node.published_boundary)
            count += 1
        error = 0 if state.count < 32 else ((-1 if index % 5 else 1) * (2 << 32))
        ki, kp = (rne_shift(c * error, 48) for c in (KI_Q48, KP_Q48))
        state.commit(0, state.power_word, error, False, 0, 0, False, ki, kp)
    actual_final_end = (positions[-1] + 2**32 - 1) // 2**32 + 3
    assert all(a <= b for a, b in pairwise(bounds))
    assert all(bound <= actual_final_end for bound in bounds)
    assert bounds[-1] == actual_final_end and node.stats["completed"] == count
    with pytest.raises(ValueError):
        node.issue(state.next_q32, n, n)  # Extrapolated q[N] is outside the actual frame.


@pytest.mark.parametrize("position,count,end", [
    (True, 0, 10), (-1, 0, 10), (((2**20 + 8) << 32) + 1, 0, 10),
    (0, False, 10), (0, -1, 10), (0, 10, 10), (0, 0, 2**17),
])
def test_bad_snapshot_cannot_mutate_or_claim_a_request(position, count, end):
    node = MovingFrameTailHorizon(1024)
    before = deepcopy(vars(node))
    with pytest.raises(ValueError):
        node.issue(position, count, end)
    assert vars(node) == before


def test_one_outstanding_request_skips_intermediate_updates_without_a_queue():
    node = MovingFrameTailHorizon(400)
    assert node.issue(100 << 32, 0, 100)
    assert not node.issue(104 << 32, 1, 100)
    with pytest.raises(ValueError, match="every real edge"):
        node.advance(2)
    finish(node)
    assert node.published_boundary == 400  # Existing local sample cap.
    assert not node.issue(100 << 32, 0, 100)
    assert node.issue(140 << 32, 10, 100)
    node.advance(node.cycle + 1)
    node.close(node.cycle + 1)
    assert not node.ready and node.stats["cancelled"] == 1
    assert node.published_boundary == 400
    node.release()
    assert node.last_issued_count == node.published_boundary == 0
    assert node.issue(0, 0, 1)
    finish(node)
    assert node.published_boundary == 3


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_actual_selected_words_loop_head_end_and_MAC_are_unchanged(ft):
    profile, words, expected, selection, edge = captured(ft, size=23, pilots=8)
    old = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="functional")
    before, _ = drive(old, selection, edge, words)
    reads = []
    node = MovingTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional",
                                   raw_observer=reads.append)
    event, mac = drive(node, selection, edge, words)
    assert delivered([(event, mac)]) == [expected] and event["frame"] == before["frame"]
    for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(event[key], before[key])
    indices = [r["index"] for r in reads]
    assert indices == list(range(indices[0], indices[-1] + 1))
    assert all(r["word"] == words[r["index"]] for r in reads)
    assert node.horizon.stats["completed"] > 1
    assert node.history.stats["peak_live"] <= 1024
    assert event["copied_source_end"] >= event["consumed_source_end"]
    assert event["bounded_private_prefetch"]["horizon_limit"] <= node.sample_limit
    assert node.horizon.last_issued_count < event["frame"]["frame_end_symbol"]
    node.release()
    assert node.horizon.published_boundary == 0


@pytest.mark.parametrize("damage,status", [("bad_head", "bad_header_crc"),
                                          ("bad_data", "bad_data_crc")])
def test_actual_failed_CRC_never_becomes_an_alternate_sync_or_MAC_answer(damage, status):
    profile, words, _, selection, edge = captured(3, size=17, **{damage: True})
    node = MovingTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, selection, edge, words)
    assert event["frame"]["status"] == status and not mac.packets
    assert event["selection_id"] == selection.ident and node.stats["accepted"] == 1
    if damage == "bad_head":
        assert node.horizon.stats["accepted"] == 0


def test_cost_keeps_original_state_and_ports_and_bills_the_two_added_registers():
    profile, *_ = captured(3, size=17)
    old = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="stop")
    new = MovingTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="stop")
    a, b = old.estimate(), new.estimate()
    assert MovingFrameTailHorizon.estimate()["state_bits"] == 444 <= 512
    assert b["new_bridge_control_bits"] == a["new_bridge_control_bits"] + 38 <= 2048
    assert b["known_register_reservation_subtotal"] == a[
        "known_register_reservation_subtotal"] + 38
    assert b["actual_source_read_ports_added"] == 0
    assert b["new_private_source_BRAM36"] == a["new_private_source_BRAM36"] == 2
    assert b["initialized_static_DSP"] == a["initialized_static_DSP"] == 19
    assert b["complete_latency_cycles"] is b["payload_cycles"] is None
    assert not b["full_fixed_or16MSps"] and not b["actual_synthesis"]


def test_EOF_cannot_manufacture_the_unreceived_head_or_leave_a_stale_horizon():
    profile, words, _, selection, edge = captured(4, size=37)
    node = MovingTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, selection, edge, words, raw_end=selection.source_end)
    assert event["frame"]["status"] == "truncated_header" and not mac.packets
    assert event["frame"]["data_crc"] is None and node.source_index == selection.source_end
    assert node.horizon.stats["accepted"] == 0 and not node.needs_word
