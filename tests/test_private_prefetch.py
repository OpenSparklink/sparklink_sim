"""Finite horizon arithmetic and literal borrowed-bank ownership,not shared RF truth."""

from copy import deepcopy
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.sim.head_prefetched_capture import MINIMUM_STEP_Q32
from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay
from nearlink_sdr.sim.private_prefetch import ContinuousPrivatePrefetch, FrameTailHorizon
from tests.captured_frame_reference import delivered, receive
from tests.initialized_capture_reference import captured, drive
from tests.test_coherent_captured_frame import profile_case


@pytest.mark.parametrize("phase", range(8))
@pytest.mark.parametrize("symbols", [1, 32, 445, 65536, 131071])
def test_serial_integer_horizon_equals_independent_product_and_has_real_latency(phase, symbols):
    node = FrameTailHorizon(2**20)
    tick = 8 * 37 + phase
    assert node.issue(tick, symbols)
    assert not node.issue(tick, symbols)
    for edge in range(1, 19):
        node.advance(edge)
        assert not node.ready
    node.advance(19)
    position = (tick << 29) + (symbols - 1) * MINIMUM_STEP_Q32
    expected = min(2**20, (position + (2**32 - 1)) // 2**32 + 3)
    assert node.ready and node.boundary == expected
    snapshot = deepcopy(node.stats), node.boundary
    node.advance(97)
    assert snapshot == (node.stats, node.boundary)
    node.close(98)
    node.release()
    assert node.mode == "idle" and node.boundary == 0
    assert node.stats == dict(accepted=1, completed=1, cancelled=0)


@pytest.mark.parametrize("tick,n", [(True, 32), (1., 32), (-1, 32), (2**23, 32),
                                   (8, False), (8, 0), (8, 2**17)])
def test_invalid_horizon_context_cannot_partially_claim_credit(tick, n):
    node = FrameTailHorizon(4096)
    before = deepcopy(vars(node))
    with pytest.raises(ValueError):
        node.issue(tick, n)
    assert vars(node) == before


def test_cancelled_serial_product_cannot_emit_a_stale_frame_boundary():
    node = FrameTailHorizon(4096)
    assert node.issue(8 * 1000000, 131071)
    node.advance(1)
    with pytest.raises(ValueError, match="every real edge"):
        node.advance(3)
    node.close(2)
    assert node.mode == "closed" and not node.ready
    node.advance(1000)
    with pytest.raises(RuntimeError):
        FrameTailHorizon(4096).release()
    node.release()
    assert node.issue(9, 32)
    for edge in range(1001, 1020):
        node.advance(edge)
    assert node.ready and node.stats == dict(accepted=2, completed=1, cancelled=1)


def test_static_horizon_is_below_actual_extreme_finite_PI_sample_requirement():
    from nearlink_sdr.phy.timing_word import (
        KI_Q48,
        KP_Q48,
        TimingWordFormat,
        WordTimingState,
        rne_shift,
    )

    for phase in range(8):
        state = WordTimingState((1, 0), 8 * 37 + phase, 32, TimingWordFormat(30, 32))
        node = FrameTailHorizon(2**20)
        assert node.issue(8 * 37 + phase, 1000)
        for _ in range(999):
            error = 0 if state.count < 32 else -(2 << 32)
            ki, kp = (rne_shift(c * error, 48) for c in (KI_Q48, KP_Q48))
            state.commit(0, state.power_word, error, False, 0, 0, False, ki, kp)
        for edge in range(1, 20):
            node.advance(edge)
        assert node.boundary <= state.required_end
        assert state.rate_limit_events > 0


@pytest.fixture(scope="module", params=[2, 3, 4])
def selection(request):
    return captured(request.param, size=23, pilots=8)


def test_prefetch_uses_same_words_loop_and_actual_head_without_overconsuming(selection):
    profile, words, expected, selected, edge = selection
    reference = InitializedCapturedFrameReplay(profile, coherent_threshold=.45,
                                               payload_mode="functional")
    old, _ = drive(reference, selected, edge, words)
    reads = []
    node = ContinuousPrivatePrefetch(profile, coherent_threshold=.45,
                                     payload_mode="functional", raw_observer=reads.append)
    event, mac = drive(node, selected, edge, words)
    assert delivered([(event, mac)]) == [expected] and event["frame"] == old["frame"]
    for field in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(event[field], old[field])
    assert node.horizon.stats["accepted"] == node.horizon.stats["completed"] == 1
    assert node.history.stats["peak_live"] <= 1024
    assert event["copied_source_end"] >= event["consumed_source_end"]
    assert not node.flow.stats["input_overflow"]
    indices = [r["index"] for r in reads]
    assert indices == list(range(indices[0], indices[-1] + 1))
    assert all(r["word"] == words[r["index"]] for r in reads)
    node.release()
    assert node.history is node.flow is None and node.horizon.mode == "idle"


@pytest.mark.parametrize("damage,status", [("bad_head", "bad_header_crc"),
                                          ("bad_data", "bad_data_crc")])
def test_provisional_suffix_never_changes_CRC_answer_or_becomes_a_delivered_packet(damage, status):
    profile, words, _, selected, edge = captured(3, size=17, **{damage: True})
    node = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, selected, edge, words)
    assert event["frame"]["status"] == status and not mac.packets
    assert event["selection_id"] == selected.ident and node.stats["accepted"] == 1
    suffix = event["bounded_private_prefetch"]["provisional_copied_suffix_words"]
    assert 0 <= suffix <= 1024
    if damage == "bad_head":
        assert node.horizon.stats["accepted"] == node.bridge_stats["functional_payloads"] == 0
        assert event["frame"]["data_crc"] is None


@pytest.mark.parametrize("gate,status", [("length", "invalid_length"),
                                        ("MCS", "unsupported_mcs")])
def test_valid_head_CRC_does_not_authorize_an_illegal_frame_horizon(gate, status):
    from dataclasses import replace

    profile, words, _, selected, edge = captured(3, size=23)
    profile = (replace(profile, max_payload_bytes=16) if gate == "length"
               else replace(profile, allowed_mcs=(8,)))
    node = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, selected, edge, words)
    assert event["frame"]["status"] == status and not mac.packets
    assert event["frame"]["header_crc"] is True and event["frame"]["data_crc"] is None
    assert node.horizon.stats["accepted"] == node.bridge_stats["functional_payloads"] == 0


def test_initial_fault_never_retires_the_provisional_suffix_as_consumed(selection):
    profile, words, _, selected, edge = selection
    node = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="stop")
    processed = []

    def observe(owner):
        if owner.flow is not None and not owner.flow.done:
            processed.append(owner.origin + owner.history.head)

    event, mac = drive(node, selected, edge, [0] * len(words), after_tick=observe)
    assert event["frame"]["status"] == "invalid_initial_fit" and not mac.packets
    assert event["consumed_source_end"] == processed[-1]
    assert event["copied_source_end"] > event["consumed_source_end"]
    assert node.horizon.stats["accepted"] == 0 and node.history.live == 0


def test_actual_EOF_during_prefetch_drains_present_words_without_padding(selection):
    profile, words, _, selected, edge = selection
    node = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="functional")
    # Full selected sync is present;no head or payload future is manufactured.
    end = selected.source_end
    event, mac = drive(node, selected, edge, words, raw_end=end)
    assert event["frame"]["status"] == "truncated_header" and not mac.packets
    assert event["frame"]["data_crc"] is None and node.source_index == end
    assert node.pending is None and not node.needs_word


def test_unknown_start_actual_RRC_chunking_and_bounded_ports_remain_compatible():
    profile, iq, quiet, expected = profile_case(3, size=37, pilots=4)
    factory = partial(ContinuousPrivatePrefetch, payload_mode="functional")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", factory):
        run = receive(profile, iq, quiet, groups=(1, 97, 13, 256), retire_before_fine=True)
    assert delivered(run[1]) == [expected]
    assert run[0]["peak_source_operations_per_primitive"] <= 2
    assert all(e["initialized_capture_bridge"]["peak_private_words"] <= 1024 for e, _ in run[1])


def test_horizon_cost_includes_all_state_and_preserves_existing_resource_failures():
    profile, *_ = profile_case(4)
    old = InitializedCapturedFrameReplay(profile, coherent_threshold=.45, payload_mode="stop")
    new = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="stop")
    a, b = old.estimate(), new.estimate()
    assert FrameTailHorizon.estimate()["state_bits"] == 406
    assert b["new_bridge_control_bits"] == a["new_bridge_control_bits"] + 406 <= 2048
    assert b["known_register_reservation_subtotal"] == a[
        "known_register_reservation_subtotal"] + 406
    assert b["new_private_source_BRAM36"] == a["new_private_source_BRAM36"] == 2
    assert b["actual_source_read_ports_added"] == 0 and b["initialized_static_DSP"] == 19
    assert b["payload_cycles"] is b["complete_latency_cycles"] is None
    assert not b["full_fixed_or16MSps"] and not b["actual_synthesis"]
