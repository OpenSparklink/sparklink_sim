"""Independent envelope products and actual bounded ownership/CRC regressions."""

from copy import deepcopy
from dataclasses import replace
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.sim.legal_tail_ceiling import (
    MAXIMUM_STEP_Q32,
    LegalTailCeiling,
    UpperTailCaptureReplay,
)
from nearlink_sdr.sim.private_prefetch import ContinuousPrivatePrefetch
from tests.captured_frame_reference import delivered, receive
from tests.initialized_capture_reference import captured, drive
from tests.test_coherent_captured_frame import profile_case


@pytest.mark.parametrize("phase", range(8))
@pytest.mark.parametrize("symbols", [1, 32, 445, 65536, 131071])
def test_upper_product_is_independent_exact_and_needs19_real_edges(phase, symbols):
    node = LegalTailCeiling(2**20)
    tick = 8 * 37 + phase
    assert node.issue(tick, symbols) and not node.issue(tick, symbols)
    for edge in range(1, 19):
        node.advance(edge)
        assert not node.ready
    node.advance(19)
    position = (tick << 29) + (symbols - 1) * 17416609614
    expected = min(2**20, (position + 2**32 - 1) // 2**32 + 3)
    assert node.ready and node.boundary == expected
    node.close(20)
    node.release()
    assert node.mode == "idle" and node.shifted == node.boundary == 0


@pytest.mark.parametrize("phase", range(8))
def test_true_extreme_PI_queries_stay_below_upper_fence_with_unchanged_gains(phase):
    from nearlink_sdr.phy.timing_word import (
        KI_Q48,
        KP_Q48,
        TimingWordFormat,
        WordTimingState,
        rne_shift,
    )

    state = WordTimingState((1, 0), 8 * 37 + phase, 32, TimingWordFormat(30, 32))
    node = LegalTailCeiling(2**20)
    assert node.issue(8 * 37 + phase, 1000)
    for edge in range(1, 20):
        node.advance(edge)
    observed = []
    for index in range(1000):
        observed.append(state.required_end)
        error = 0 if state.count < 32 else (2 << 32) * (-1 if index % 7 == 0 else 1)
        ki, kp = (rne_shift(c * error, 48) for c in (KI_Q48, KP_Q48))
        state.commit(0, state.power_word, error, False, 0, 0, False, ki, kp)
    assert all(value <= node.boundary for value in observed)
    assert state.rate_limit_events > 0


def test_width_extremes_and_cancel_are_bounded_without_a_stale_next_packet():
    node = LegalTailCeiling(2**20)
    assert node.issue(2**23 - 1, 2**17 - 1)
    for edge in range(1, 20):
        node.advance(edge)
    assert node.boundary == 2**20
    assert node.accumulator.bit_length() <= 53 and node.shifted.bit_length() <= 52
    node.close(20)
    node.release()
    assert node.issue(8, 64)
    node.advance(21)
    node.close(22)
    assert node.stats == dict(accepted=2, completed=1, cancelled=1)
    node.release()
    assert node.issue(9, 32) and node.shifted == MAXIMUM_STEP_Q32


@pytest.mark.parametrize("tick,count", [(True, 1), (-1, 1), (2**23, 32),
                                       (8, False), (8, 0), (8, 2**17)])
def test_invalid_context_never_partially_claims_a_fence(tick, count):
    node = LegalTailCeiling(4096)
    before = deepcopy(vars(node))
    with pytest.raises(ValueError):
        node.issue(tick, count)
    assert vars(node) == before


@pytest.fixture(scope="module", params=[2, 3, 4])
def selection(request):
    return captured(request.param, size=31, pilots=8)


def test_same_actual_selected_words_loop_and_MAC_with_only_copy_fence_changed(selection):
    profile, words, expected, chosen, edge = selection
    baseline = ContinuousPrivatePrefetch(profile, coherent_threshold=.45,
                                         payload_mode="functional")
    old, _ = drive(baseline, chosen, edge, words)
    reads = []
    node = UpperTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional",
                                  raw_observer=reads.append)
    event, mac = drive(node, chosen, edge, words)
    assert delivered([(event, mac)]) == [expected] and event["frame"] == old["frame"]
    for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(event[key], old[key])
    assert node.horizon.stats["accepted"] == node.horizon.stats["completed"] == 1
    assert node.history.stats["peak_live"] <= 1024 and not node.flow.stats["input_overflow"]
    assert event["copied_source_end"] >= event["consumed_source_end"]
    indices = [row["index"] for row in reads]
    assert indices == list(range(indices[0], indices[-1] + 1))
    assert all(row["word"] == words[row["index"]] for row in reads)
    node.release()
    assert node.history is node.flow is None and node.horizon.mode == "idle"


@pytest.mark.parametrize("damage,status", [("bad_head", "bad_header_crc"),
                                          ("bad_data", "bad_data_crc")])
def test_bad_real_CRC_is_not_an_answer_retry_or_delivered_payload(damage, status):
    profile, words, _, chosen, edge = captured(3, size=17, **{damage: True})
    node = UpperTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, chosen, edge, words)
    assert event["frame"]["status"] == status and not mac.packets
    assert event["selection_id"] == chosen.ident and node.stats["accepted"] == 1
    if damage == "bad_head":
        assert node.horizon.stats["accepted"] == node.bridge_stats["functional_payloads"] == 0
        assert event["frame"]["data_crc"] is None


def test_EOF_keeps_a_truncated_header_without_manufacturing_upper_fence_input():
    profile, words, _, chosen, edge = captured(4, size=37)
    node = UpperTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, chosen, edge, words, raw_end=chosen.source_end)
    assert event["frame"]["status"] == "truncated_header" and not mac.packets
    assert event["frame"]["data_crc"] is None and node.source_index == chosen.source_end
    assert node.horizon.stats["accepted"] == 0 and not node.needs_word


@pytest.mark.parametrize("gate,status", [("length", "invalid_length"),
                                        ("mcs", "unsupported_mcs")])
def test_head_CRC_does_not_authorize_an_illegal_length_or_MCS_fence(gate, status):
    profile, words, _, chosen, edge = captured(3, size=23)
    profile = (replace(profile, max_payload_bytes=16) if gate == "length"
               else replace(profile, allowed_mcs=(8,)))
    node = UpperTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="functional")
    event, mac = drive(node, chosen, edge, words)
    assert event["frame"]["status"] == status and not mac.packets
    assert event["frame"]["header_crc"] is True and event["frame"]["data_crc"] is None
    assert node.horizon.stats["accepted"] == node.bridge_stats["functional_payloads"] == 0


def test_unknown_start_arbitrary_block_boundaries_still_use_real_bounded_ports():
    profile, iq, quiet, expected = profile_case(3, size=37, pilots=4)
    factory = partial(UpperTailCaptureReplay, payload_mode="functional")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", factory):
        run = receive(profile, iq, quiet, groups=(1, 97, 13, 256), retire_before_fine=True)
    assert delivered(run[1]) == [expected]
    assert run[0]["peak_source_operations_per_primitive"] <= 2
    assert all(e["initialized_capture_bridge"]["peak_private_words"] <= 1024 for e, _ in run[1])


def test_cost_reuses_static_hardware_without_hiding_old_resource_failures(selection):
    profile, *_ = selection
    old = ContinuousPrivatePrefetch(profile, coherent_threshold=.45, payload_mode="stop")
    new = UpperTailCaptureReplay(profile, coherent_threshold=.45, payload_mode="stop")
    a, b = old.estimate(), new.estimate()
    assert LegalTailCeiling.estimate()["state_bits"] == 406
    assert b["new_bridge_control_bits"] == a["new_bridge_control_bits"] == 1034
    assert b["known_register_reservation_subtotal"] == a["known_register_reservation_subtotal"]
    assert b["new_private_source_BRAM36"] == a["new_private_source_BRAM36"] == 2
    assert b["actual_source_read_ports_added"] == 0 and b["initialized_static_DSP"] == 19
    assert b["complete_latency_cycles"] is b["payload_cycles"] is None
    assert not b["full_fixed_or16MSps"] and not b["actual_synthesis"]
