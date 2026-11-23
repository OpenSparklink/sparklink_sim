"""Finite signed primitives,exact schedule variants and raw-to-MAC regressions."""

import math
from collections import Counter
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.phy.sync_fit_word import compile_sync_phases
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from nearlink_sdr.sim.coherent_captured_frame import WordConfirmedCapturedFrameReplay
from nearlink_sdr.sim.coherent_interleaved import (
    InterleavedCoherentWordService,
    ScheduledCoherentCapturedFrameReplay,
    SignedCoherentWordService,
    _ArithmeticContext,
)
from nearlink_sdr.sim.coherent_word_flow import CoherentWordService
from nearlink_sdr.sim.word_arithmetic import WordCommand, WordFabric, WordFabricAssumptions
from tests.captured_frame_reference import delivered, fixture, receive
from tests.test_coherent_captured_frame import profile_case, selected_case
from tests.test_coherent_word_flow import integer_reference, source


def execute(cls, selected, codes, *, sink=None):
    node = cls(selected, codes, sink=sink)
    while not node.ready:
        node.step()
    return node, node.take()


def receipt_words(receipt):
    return {k: v for k, v in vars(receipt).items() if k != "available_cycle"}


def test_signed_native_extremes_independent_integer_reference_and_actual_three_edges():
    values = [(-(2**21), -(2**17)), (2**21 - 1, 2**17 - 1),
              (-(2**21), 2**17 - 1), (2**21 - 1, -(2**17)), (0, -1), (-7, -13)]
    rng = np.random.default_rng(2931103)
    values += list(zip(map(int, rng.integers(-(2**21), 2**21, 37)),
                       map(int, rng.integers(-(2**17), 2**17, 37)), strict=True))
    for a, b in values:
        f = WordFabric()
        assert f.issue(WordCommand("tile", a, b, 22, 18, True, True, 43, True), 0, 0)
        assert f.dsp[0][1] is None
        f.tick()
        assert f.dsp[1][1] == a * b and f.take(0) is None
        f.tick()
        assert f.take(0) is None
        f.tick()
        reply = f.take(0)
        assert reply.value == a * b and not reply.overflow and reply.completed_cycle == 3


def test_mixed_native_formats_share_one_pipeline_and_held_mailboxes_cancel_cleanly():
    f = WordFabric(WordFabricAssumptions(2, 4))
    commands = [WordCommand("tile", -7129, 65536, 22, 18, True, True, 43, True),
                WordCommand("tile", 2**24 - 1, 2**17 - 1, 24, 17, False, False, 43, True)]
    for j, c in enumerate(commands):
        assert f.issue(c, j, 0)
        assert not f.issue(commands[0], 3, 0)
        f.tick()
    for _ in range(4):
        f.tick()
    assert f.take(0).value == commands[0].a * commands[0].b
    assert f.take(1) is None
    f.tick()
    assert f.take(1).value == commands[1].a * commands[1].b and f.idle
    f.tick()
    assert f.issue(commands[0], 2, 0)
    f.flush()
    for _ in range(4):
        f.tick()
    assert f.idle and f.take(2) is None and f.next_token == [0] * 4


@pytest.mark.parametrize("shape", [(23, 18, True, True), (22, 19, True, True),
                                   (22, 18, False, True), (24, 17, True, True)])
def test_uncompiled_native_widths_or_signedness_still_rejected(shape):
    with pytest.raises(ValueError, match="Native tile"):
        WordCommand("tile", 0, 0, *shape, 43, True)


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("omega", [-math.pi, -0.731, 0, 0.481, math.pi - 1e-9])
def test_original_receipt_and_independent_ROM_integer_reference_exact(ft, omega):
    profile, *_ = fixture(ft)
    codes = compile_sync_phases(profile.sync_symbols())
    rng = np.random.default_rng(493341 + ft)
    words = tuple(map(tuple, rng.integers(-(2**21), 2**21, (len(codes), 2)).tolist()))
    c = tuple(round(x * 2**49) for x in (math.cos(omega), math.sin(omega)))
    selected = source(words, c, exponent=-19)
    original, a = execute(CoherentWordService, selected, codes)
    for cls in [SignedCoherentWordService, InterleavedCoherentWordService]:
        node, b = execute(cls, selected, codes)
        assert receipt_words(a) == receipt_words(b)
        assert b.selection is selected
        assert (b.correlation, b.signal_energy, b.numerator, b.denominator) == (
            integer_reference(selected, codes, b.step_word20)
        )
        assert node.stats["native_products"] == original.stats["native_products"] == 8 * len(codes) + 12
        assert node.cycle < original.cycle and node.fabric.idle
        assert node.fabric.stats["taken"] == node.fabric.stats["issued"]


def test_actual_single_issue_take_two_ALU_and_four_pending_bound():
    selected = source(tuple(((-1)**k * (913147 + k), 543117 - k) for k in range(126)))
    events = []
    node = InterleavedCoherentWordService(selected, tuple(k % 16 for k in range(126)),
                                         sink=events.append)
    concurrent = 0
    while not node.ready:
        before = node.fabric.stats["alu_steps"]
        node.step()
        assert node.fabric.stats["alu_steps"] - before <= 2
        assert sum(node.fabric.busy) <= 4 and len(node._workers) <= 4
        concurrent = max(concurrent, sum(node.fabric.busy))
    for kind in ["word_issue", "word_take"]:
        counts = Counter(e["cycle"] for e in events if e["event"] == kind)
        assert max(counts.values()) == 1
    assert concurrent >= 3 and not node._workers
    assert len([e for e in events if e["event"] == "coherent_cache_read"]) == 126
    assert node.take().selection is selected


@pytest.mark.parametrize("cut", [1, 36, 48, 73, 121])
def test_cancel_during_shared_native_and_ALU_workers_has_no_stale_reply(cut):
    node = InterleavedCoherentWordService(source(((913147, -531123),) * 32), (0,) * 32)
    for _ in range(cut):
        node.step()
    old = node.cycle
    node.abort()
    assert node.cycle == old + 1 and node.fabric.idle and not node._workers
    assert node.mode == "aborted" and node.take() is None
    assert not any(node.fabric.mailboxes) and not any(node.fabric.dsp)
    with pytest.raises(RuntimeError, match="Retired"):
        node.step()


def test_exact_squared_threshold_zero_energy_and_long_completed_output_hold():
    imag = [4, -4] * 8 + [3, -3] * 5 + [1, -1] * 3
    chosen = source(tuple((3, i) for i in imag))
    node = InterleavedCoherentWordService(chosen, (0,) * 32)
    while not node.ready:
        node.step()
    ready = node.ready_cycle
    for _ in range(32769):
        node.step()
    r = node.take()
    assert r.comparison == 0 and r.accepted and r.quality == .45
    assert r.available_cycle == ready and node.stats["output_hold_edges"] == 32769
    _, zero = execute(InterleavedCoherentWordService, source(((0, 0),) * 32), (0,) * 32)
    assert not zero.accepted and zero.signal_energy == zero.denominator == 0


def test_compiled_pending_word_bound_and_explicit_cost_without_free_ports_or_DSP():
    node = InterleavedCoherentWordService(source(((1, 0),) * 32), (0,) * 32)
    worker = _ArithmeticContext(node, 0)
    with pytest.raises(ValueError, match="110"):
        next(worker._command("mov", 0, ab=111, bits=111))
    for cls, state in [(SignedCoherentWordService, 4973), (InterleavedCoherentWordService, 7873)]:
        cost = cls.estimate()
        assert cost["register_bits_upper"] == state
        assert state == cost["fabric"]["state_bits"] + sum(cost["register_fields"].values())
        assert cost["DSP25x18"] == 1 and cost["new_IQ_RAM"] == 0
        assert cost["coefficient_ROM_read_ports"] == 2 and cost["cache_read_ports"] == 1
        assert cost["signed_input_mux2_bits"] == 43
    cost = InterleavedCoherentWordService.estimate()
    assert cost["issue_ports"] == cost["result_take_ports"] == 1
    assert cost["register_fields"]["pending_commands"] == 4 * 275
    assert cost["register_fields"]["scheduler_control"] == 128
    assert cost["register_bits_upper"] + 194 <= 8192
    assert not cost["actual_synthesis"] and not cost["continuous16MSps"]


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_actual_raw_to_MAC_and_symbol_arrays_preserved_with_scheduled_gate(ft):
    profile, iq, quiet, expected = profile_case(ft, size=23)
    outputs = []
    for cls, options in [(WordConfirmedCapturedFrameReplay, {}),
                         (ScheduledCoherentCapturedFrameReplay, dict(structure="serial-signed22x18")),
                         (ScheduledCoherentCapturedFrameReplay,
                          dict(structure="interleaved-signed22x18"))]:
        with patch("tests.captured_frame_reference.CapturedFrameReplay", partial(cls, **options)):
            outputs.append(receive(profile, iq, quiet, retire_before_fine=True))
    before = next(e for e, mac in outputs[0][1] if mac.packets)
    for output in outputs:
        assert delivered(output[1]) == [expected]
        event = next(e for e, mac in output[1] if mac.packets)
        for key in ["symbols", "symbol_centers_filtered"]:
            assert event[key].dtype == before[key].dtype and event[key].tobytes() == before[key].tobytes()
        assert event["frame"]["header_crc"] and event["frame"]["data_crc"]
        assert event["frame"]["length_ok"] and event["frame"]["mcs"] == profile.config.mcs_index
        assert output[0]["peak_owned"] <= 3 and output[0]["peak_source_operations_per_primitive"] <= 2
        assert event["complete_latency_cycles"] is None
    assert outputs[2][0]["frame_replay_cost"]["word_gate_control_node_bits"] == 8067


@pytest.mark.parametrize("bad_head,bad_data,status", [
    (True, False, "bad_header_crc"), (False, True, "bad_data_crc"),
])
def test_interleaved_real_bad_CRC_is_not_retried_or_delivered(bad_head, bad_data, status):
    profile, iq, quiet, _ = profile_case(2, size=31, bad_head=bad_head, bad_data=bad_data)
    owner = partial(ScheduledCoherentCapturedFrameReplay, structure="interleaved-signed22x18")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", owner):
        result = receive(profile, iq, quiet, retire_before_fine=True)
    assert not delivered(result[1])
    event = next(e for e, _ in result[1] if e["frame"]["status"] == status)
    assert event["coherent_word_service"]["word_decision"]
    assert event["frame"]["header_crc"] is (not bad_head)
    assert event["frame"]["data_crc"] is (False if bad_data else None)


def test_source_fault_cancels_actual_parallel_workers_before_any_header(monkeypatch):
    profile, iq, quiet, _ = profile_case(2, size=31)
    fault = {}
    original_begin = ScheduledCoherentCapturedFrameReplay.begin
    original_advance = CoarseCaptureSource.advance

    def begin(owner, selected, cycle):
        original_begin(owner, selected, cycle)
        if owner._word_waiting:
            fault["edge"] = cycle + 48

    def advance(source, cycle):
        if cycle == fault.get("edge"):
            source.flow.abort()
        return original_advance(source, cycle)

    monkeypatch.setattr(ScheduledCoherentCapturedFrameReplay, "begin", begin)
    monkeypatch.setattr(CoarseCaptureSource, "advance", advance)
    owner = partial(ScheduledCoherentCapturedFrameReplay, structure="interleaved-signed22x18")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", owner):
        result = receive(profile, iq, quiet, retire_before_fine=True)
    assert result[4].fault_cycle == fault["edge"] and result[4].done
    assert result[0]["input_samples"] == len(iq) and not delivered(result[1])
    event = next(e for e, _ in result[1] if e["frame"]["status"] == "source_epoch_aborted")
    service = event["coherent_word_service"]
    assert service["statistics"]["native_products"] > 0
    assert service["service_state"] == "aborted" and not service["actual_finite_values"]
    assert not result[3].active


def test_held_parallel_gate_EOF_and_new_packet_reset_keep_finite_ownership():
    profile, selected, cycle = selected_case()
    owner = ScheduledCoherentCapturedFrameReplay(
        profile, structure="interleaved-signed22x18", coherent_threshold=.45,
    )
    owner.begin(selected, cycle)
    with pytest.raises(RuntimeError, match="confirmation"):
        owner.end_input(cycle)
    while not owner.word_service.ready:
        owner.tick(owner.cycle + 1, blocked=True)
        assert not owner.needs_word and owner.rx is None
    ready = owner.word_service.ready_cycle
    for _ in range(17):
        owner.tick(owner.cycle + 1, blocked=True)
    assert owner.word_service.ready_cycle == ready and not owner.needs_word
    owner.tick(owner.cycle + 1)
    owner.abort("cancelled", owner.cycle + 1)
    event, mac = owner.take()
    assert not mac.packets and event["coherent_word_service"]["computed_edges"] == ready - cycle
    owner.release()
    assert owner.word_service is owner.word_receipt is None and not owner.active


def test_uncompiled_schedule_is_rejected_before_profile_or_input_mutation():
    profile, *_ = profile_case(2)
    for mode in [None, True, [], "unbounded"]:
        with pytest.raises(ValueError, match="compiled"):
            ScheduledCoherentCapturedFrameReplay(profile, structure=mode, coherent_threshold=.45)
