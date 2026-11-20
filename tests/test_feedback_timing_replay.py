"""Same-age closed sampling and real CRC;no truth or expected bytes in selection."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.timing import GardnerPIState, PSKTimingConfig
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.feedback_timing_replay import FeedbackAgeFrameReplay, _FloatAgeState
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input

FORMAT = TimingWordFormat(30, 32)
FIELDS = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")
TRACES = ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error")


def run(profile, words, selection, *, age, finite, chunks=(32,), legacy=False):
    prepared, pos = prepare(profile, words, selection)
    options = dict(selection, reuse_stencil=True)
    if finite:
        options["word_timing_format"] = FORMAT
    if legacy:
        flow = SelectedFrameReplay(profile, prepared, **options)
    else:
        flow = FeedbackAgeFrameReplay(profile, prepared, error_age=age, finite=finite, **options)
    count = 0
    while not flow.done:
        if flow.pending_service is not None:
            flow.resolve_functional()
        else:
            if flow.wants_replay_input:
                if pos == len(words):
                    flow.end_input()
                else:
                    used = flow.feed_replay_words(words[pos:pos + chunks[count % len(chunks)]])
                    assert used == 1  # Actual single source port,not a hidden block append.
                    pos += used
                    count += 1
            flow.step()
        assert flow.cycle < 2_000_000
    return flow


@pytest.mark.parametrize("age", [0, 1, 2, 3])
@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 7), (3, 12, 8, 11), (4, 12, 4, 13)])
def test_closed_sampling_real_frame_and_same_age_frozen_budgets(age, ft, mcs, pilot, length):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    words = quantize_input(values)
    floating = run(profile, words, selection, age=age, finite=False)
    finite = run(profile, words, selection, age=age, finite=True)
    a, b = floating.event, finite.event
    assert all(a["frame"].get(k) == b["frame"].get(k) for k in FIELDS)
    assert b["frame"]["payload"] == payload and b["frame"]["mcs"] == mcs
    assert b["frame"]["header_crc"] and b["frame"]["data_crc"]
    for key, limit in [("symbol_centers_filtered", .001), ("timing_rate_ppm", .1)]:
        assert np.max(abs(a[key] - b[key])) <= limit
    scale = max(float(np.mean(abs(a["symbols"]) ** 2)) ** .5, 1e-30)
    difference = abs(a["symbols"] - b["symbols"]) / scale
    assert float(np.mean(difference ** 2)) ** .5 <= .0001
    assert float(np.max(difference)) <= .001
    assert not b["replay_transport"]["cycle_evidence_valid"]
    assert b["replay_transport"]["complete_latency_cycles"] is None
    assert b["feedback_after_release"]["closed"]
    assert b["feedback_after_release"]["pending_errors"] == [0] * age
    assert floating.consumer.feedback.closed
    assert floating.consumer.feedback.pending_errors == [0.] * age
    assert finite.kernel is finite.prepared.stencil.interpolator
    assert finite.history.capacity == 512 and finite.cache.cache.state == "released"
    assert finite.take_mac_result().packets[0].mac_payload == payload
    assert not finite.take_mac_result().packets
    with pytest.raises(RuntimeError, match="old cycle service"):
        finite.consumer.claim_timing_state()
    if age == 0:
        for candidate in (floating, finite):
            old = run(profile, words, selection, age=0, finite=candidate.finite, legacy=True)
            assert all(old.event["frame"].get(k) == candidate.event["frame"].get(k) for k in FIELDS)
            for key in TRACES:
                np.testing.assert_array_equal(old.event[key], candidate.event[key])


@pytest.mark.parametrize("damage,status", [
    ({"bad_head": True}, "bad_header_crc"),
    ({"bad_data": True}, "bad_data_crc"),
    ({"empty": True}, "invalid_length"),
    ({"mcs": 15}, "unsupported_mcs"),
])
def test_actual_bad_frames_do_not_retry_or_deliver(damage, status):
    profile = ordinary(2 if damage.get("empty") else 3, 8, 0)
    values, _, selection = selected_input(profile, length=19, **damage)
    words = quantize_input(values)
    for finite in (False, True):
        flow = run(profile, words, selection, age=3, finite=finite)
        assert flow.event["frame"]["status"] == status
        assert flow.event["frame"]["payload"] is None and not flow.take_mac_result().packets
        assert flow.consumer.feedback.closed
        if status != "bad_data_crc":
            assert flow.consumer.frame_end_symbol is None


@pytest.mark.parametrize("cut_symbols,status", [(127, "truncated_header"), (450, "truncated_data")])
def test_eof_and_packet_reset_preserve_explicit_failures(cut_symbols, status):
    profile = ordinary(4, 8, 0)
    values, _, selection = selected_input(profile, length=255)
    words = quantize_input(values)
    cut = selection["start_tick"] // 8 + 4 * cut_symbols
    for finite in (False, True):
        flow = run(profile, words[:cut], selection, age=2, finite=finite)
        assert flow.event["frame"]["status"] == status and flow.consumer.feedback.closed
        assert not flow.take_mac_result().packets
        node = flow.consumer.feedback
        snapshot = node.snapshot if finite else lambda: deepcopy(node.__dict__)
        before = snapshot()
        flow.consumer.finish()
        assert snapshot() == before


def test_wrong_seed_legal_length_and_input_partition_use_the_same_gate():
    profile = ordinary(3, 12, 8)
    values, _, selection = selected_input(profile, length=31)
    words = quantize_input(values)
    for bad in (replace(profile, max_payload_bytes=7),
                replace(profile, config=replace(profile.config, head_llid=7))):
        result = run(bad, words, selection, age=1, finite=True)
        assert result.event["frame"]["status"] in ("invalid_length", "bad_header_crc")
        assert not result.take_mac_result().packets
    a = run(profile, words, selection, age=1, finite=True, chunks=(1,))
    b = run(profile, words, selection, age=1, finite=True, chunks=(7, 31, 3, 19))
    for key in TRACES:
        np.testing.assert_array_equal(a.event[key], b.event[key])
    assert a.event["frame"] == b.event["frame"]
    assert a.event["feedback_before_release"] == b.event["feedback_before_release"]
    assert a.event["replay_transport"] == b.event["replay_transport"]
    assert a.history.head == b.history.head and a.history.end == b.history.end


@pytest.mark.parametrize("change", [
    dict(error_age=True), dict(error_age=-1), dict(error_age=4), dict(error_age=1.),
    dict(finite=1), dict(finite=True, word_timing_format=None),
    dict(finite=False, word_timing_format=FORMAT),
])
def test_invalid_experiment_cannot_consume_prepared_owner(change):
    profile = ordinary()
    values, _, selection = selected_input(profile, length=3)
    words = quantize_input(values)
    prepared, _ = prepare(profile, words, selection)
    options = dict(error_age=1, finite=True, word_timing_format=FORMAT) | change
    with pytest.raises(ValueError):
        FeedbackAgeFrameReplay(profile, prepared, **selection, **options)
    assert prepared.ready and not prepared.fit.output_taken


@pytest.mark.parametrize("age", [0, 1, 2, 3])
def test_float_FIFO_PI_is_independent_of_immediate_parent_PI(age):
    config = PSKTimingConfig()
    state = _FloatAgeState(4, 1., config, error_age=age)
    detector = GardnerPIState(4, 1., config)
    queue, omega, limits = [0.] * age, 4., 0
    random = np.random.default_rng(71324)
    previous = 1 + 0j
    for n in range(320):
        current, middle = random.normal(size=(2, 2)) @ np.array([1, 1j])
        midpoint = None if n < 32 else middle
        _, measured = detector.advance(current, previous, midpoint)
        used = measured if age == 0 else (queue[0] if midpoint is not None else 0.)
        proposed = omega + state.ki * used
        limits += proposed < state.lower or proposed > state.upper
        omega = max(state.lower, min(state.upper, proposed))
        step, error = state.advance(current, previous, midpoint)
        assert step == omega + state.kp * used and error == measured
        assert state.omega == omega and state.rate_limit_events == limits
        if midpoint is not None and age:
            queue = [*queue[1:], measured]
        assert state.pending_errors == queue
        previous = current
    state.release()
    assert state.pending_errors == [0.] * age
    with pytest.raises(RuntimeError, match="cannot resume"):
        state.advance(current, previous, midpoint)
