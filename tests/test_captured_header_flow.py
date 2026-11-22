"""Actual head receipts,finite availability,raw continuity and explicit incomplete cycles."""

import math
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scripts import model_captured_header

from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.captured_header_flow import ClockedHeaderCapturedFrameReplay
from nearlink_sdr.sim.fine_selection import FineSelection
from tests.captured_frame_reference import delivered, fixture, receive
from tests.test_captured_frame_owner import selected
from tests.test_selected_rx import ordinary, selected_input


def case(ft=2, *, damage=None, change=None):
    p = replace(
        ordinary(ft),
        length_contract="standard",
        allowed_mcs=(8,),
        head_backend="word-dense" if ft == 2 else "word-depth-parity",
    )
    values, payload, answer = selected_input(p, **(damage or {}))
    origin = answer["start_tick"] // 8 - 32
    window = {2: 256, 3: 376, 4: 632}[ft]
    exponent = (
        math.frexp(
            max(abs(z.real) for z in answer["selected_sync"])
            + max(abs(z.imag) for z in answer["selected_sync"])
        )[1]
        - 20
    )
    words = tuple(
        (round(math.ldexp(z.real, -exponent)), round(math.ldexp(z.imag, -exponent)))
        for z in answer["selected_sync"]
    )
    angle = answer["delta"]
    correlation = (round(math.cos(angle) * 10**9), round(math.sin(angle) * 10**9))
    numerator = sum(x * x for x in correlation)
    template_energy = len(words) - 1
    choice = FineSelection(
        0,
        origin + window,
        window,
        10,
        answer["start_tick"] - 8 * origin,
        exponent,
        words,
        correlation,
        numerator,
        (numerator + template_energy - 1) // template_energy,
        template_energy,
    )
    if change:
        p = change(p)
    return p, values, payload, choice


def raw_word(value):
    exponent = math.frexp(max(abs(value.real), abs(value.imag)))[1] - 18
    return pack_filter_word(
        round(math.ldexp(value.real, -exponent)), round(math.ldexp(value.imag, -exponent)), exponent
    )


def advance(owner, values, *, to_header=False):
    """Pauseable selected-port unit driver,not a continuous ADC calendar."""
    while not owner.done:
        if owner.header_waiting:
            if to_header:
                return
            owner.tick(owner._head_ready)
        elif owner.source_index < len(values):
            owner.issue_read(owner.source_index, raw_word(values[owner.source_index]), owner.cycle)
            owner.tick(owner.cycle + 1)
        else:
            owner.end_input(owner.cycle)


def begin(ft=2, **options):
    profile, values, payload, choice = case(ft, **options)
    owner = ClockedHeaderCapturedFrameReplay(profile, coherent_threshold=0.45)
    owner.begin(choice, 10)
    advance(owner, values, to_header=True)
    assert owner.header_waiting and owner.header is None and not owner.done
    return owner, values, payload


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_actual_receipt_blocks_raw_length_and_delivery_until_positive_service_finishes(ft):
    owner, values, payload = begin(ft)
    index, retired = owner.source_index, owner.retired_through
    assert not owner.needs_word and owner.pending is None
    assert owner.stats["header_valid"] == owner.stats["header_failed"] == 0
    assert owner._head_edges > 0 and owner._head_ready - owner._head_issue == owner._head_edges
    owner.tick(owner._head_ready - 1)
    assert owner.source_index == index and owner.retired_through == retired
    assert owner.header is None and owner.take() is None
    with pytest.raises(ValueError, match="next"):
        owner.issue_read(index, raw_word(values[index]), owner.cycle)
    with pytest.raises(RuntimeError, match="header service"):
        owner.end_input(owner.cycle)
    owner.tick(owner._head_ready, blocked=True)
    assert owner.header is None and not owner.needs_word
    owner.tick(owner._head_ready + 3)
    assert owner.needs_word and owner.header["header_crc"] and owner.header["length_ok"]
    assert owner.header["frame_end_symbol"] == owner.rx.frame_end_symbol
    assert owner.source_index == index and owner.stats["header_valid"] == 1
    advance(owner, values)
    event, mac = owner.take()
    assert [p.mac_payload for p in mac.packets] == [payload]
    assert event["frame"]["data_crc"] is True
    clock = event["header_service_calendar"]
    assert clock["available_cycle"] == clock["nominal_ready_cycle"] + 3
    assert clock["head_demap_cycles"] is event["complete_latency_cycles"] is None
    assert not event["complete_cycle_evidence_valid"]
    assert owner.header_clock_stats == dict(issued=1, completed=1, cancelled=0)
    owner.release()
    assert not owner.header_waiting and owner._head_ready is None


@pytest.mark.parametrize(
    "options,status,crc",
    [
        (dict(damage=dict(bad_head=True)), "bad_header_crc", False),
        (
            dict(change=lambda p: replace(p, config=replace(p.config, head_crc_seed=0x2D9))),
            "bad_header_crc",
            False,
        ),
        (dict(change=lambda p: replace(p, max_payload_bytes=16)), "invalid_length", True),
        (dict(change=lambda p: replace(p, allowed_mcs=(12,))), "unsupported_mcs", True),
    ],
)
def test_bad_head_and_policy_exit_wait_for_actual_service_without_payload_or_retry(
    options, status, crc
):
    owner, _, _ = begin(**options)
    consumed = owner.index
    assert owner._held_terminal is not None and owner.header is None and owner.event is None
    owner.tick(owner._head_ready - 1)
    assert not owner.done
    owner.tick(owner._head_ready)
    event, mac = owner.take()
    assert event["frame"]["status"] == status and event["frame"]["header_crc"] is crc
    assert event["frame"]["data_crc"] is event["frame"]["payload"] is None
    assert not mac.packets and owner.index == consumed and not owner.needs_word
    assert owner.stats["header_failed"] == 1 and owner.header_clock_stats["completed"] == 1
    assert owner.take() is None


@pytest.mark.parametrize("reason", ["source_epoch_aborted", "tail_history_expired", "cancelled"])
def test_abort_discards_pending_good_or_bad_head_without_early_CRC_delivery(reason):
    owner, _, _ = begin(damage=dict(bad_head=True))
    owner.abort(reason, owner.cycle + 1)
    event, mac = owner.take()
    assert not owner.header_waiting and owner.header is None and not mac.packets
    assert event["frame"]["status"] == reason and event["frame"]["header_crc"] is None
    assert event["header_service_calendar"]["available_cycle"] is None
    assert owner.header_clock_stats == dict(issued=1, completed=0, cancelled=1)


def test_pending_clock_readiness_abort_and_counter_errors_are_not_hidden():
    owner, _, _ = begin()
    for cycle, blocked in [(owner.cycle - 1, False), (owner.cycle, 1)]:
        with pytest.raises(ValueError):
            owner.tick(cycle, blocked=blocked)
    with pytest.raises(ValueError, match="abort"):
        owner.abort("retry", owner.cycle)
    assert owner.header_waiting
    owner.header_clock_stats["completed"] = 2**64 - 1
    with pytest.raises(OverflowError, match="statistic"):
        owner.tick(owner._head_ready)
    assert owner.header_waiting and owner.header is None


def test_EOF_after_completed_head_never_fabricates_missing_payload():
    owner, _, _ = begin()
    index = owner.index
    owner.tick(owner._head_ready)
    owner.end_input(owner.cycle)
    event, mac = owner.take()
    assert event["frame"]["status"] == "truncated_data" and not mac.packets
    assert event["frame"]["header_crc"] is True and event["frame"]["data_crc"] is None
    assert owner.index == index


def test_reference_backend_has_no_invented_head_cycles_and_cost_charges_existing_decoder():
    p, *_ = fixture()
    with pytest.raises(ValueError, match="word-head"):
        ClockedHeaderCapturedFrameReplay(p, coherent_threshold=0.45)
    p = replace(p, head_backend="word-dense")
    owner = ClockedHeaderCapturedFrameReplay(p, coherent_threshold=0.45)
    cost = owner.estimate()
    assert cost["header_calendar_register_bits"] == 397 <= cost["header_calendar_research_cap_bits"]
    assert cost["combined_owner_register_bits"] == 1090 + 397
    assert cost["register_bits_upper"] == cost["combined_owner_register_bits"]
    assert sum(value for value in cost["register_fields"].values() if type(value) is int) == 1487
    assert cost["header_service"]["logical_total_bits"] > 0
    assert cost["header_service"]["BRAM36_sites_upper"] > 0
    assert cost["header_service"]["demap_cycles"] is cost["complete_latency_cycles"] is None
    assert not cost["full_fixed_or16MSps"] and not cost["actual_synthesis"]
    owner.begin(selected(p), 10)
    owner.end_input(10)
    assert owner.event["frame"]["status"] == "truncated_sync"
    assert owner.event["header_service_calendar"]["service_edges"] == 0
    assert owner.event["header_service_calendar"]["available_cycle"] is None


@pytest.mark.parametrize(
    "edges,extra,exception",
    [
        (0, {}, RuntimeError),
        (-1, {}, RuntimeError),
        (True, {}, RuntimeError),
        (10001, {}, RuntimeError),
        (1, {"data_crc": True}, RuntimeError),
        (1, {"payload": b"unearned"}, RuntimeError),
    ],
)
def test_malformed_internal_receipt_is_not_accepted_as_cycle_evidence(edges, extra, exception):
    p, *_ = fixture()
    owner = ClockedHeaderCapturedFrameReplay(
        replace(p, head_backend="word-dense"), coherent_threshold=0.45
    )
    owner.begin(selected(p), 10)
    owner.rx = SimpleNamespace(
        header_result=dict(head_word_model=dict(service_edges=edges), **extra)
    )
    with pytest.raises(exception):
        owner._accept_reference_results([])
    assert not owner.header_waiting and owner.header is None


def test_duplicate_receipt_clock_overflow_and_statistics_cannot_create_a_second_answer():
    owner, _, _ = begin()
    with pytest.raises(RuntimeError, match="only one"):
        owner._accept_reference_results([])
    owner._head_issue = None
    owner.cycle = 2**63 - 2
    with pytest.raises(OverflowError, match="availability"):
        owner._accept_reference_results([])
    owner.cycle = 10
    owner.header_clock_stats["issued"] = 2**64 - 1
    with pytest.raises(OverflowError, match="statistic"):
        owner._accept_reference_results([])
    assert owner.header is None


@pytest.mark.parametrize("mode", ["hash", "empty", "missing-id", "duplicate-id"])
def test_reproduction_rejects_changed_index_and_empty_or_ambiguous_scope(
    tmp_path, monkeypatch, mode
):
    manifest = tmp_path / "inputs.json"
    manifest.write_text("[]" if mode == "empty" else '[{"id":0}]')
    sha = model_captured_header.reference.shared.sha(manifest)
    args = [
        "model_captured_header",
        "--input-dir",
        str(tmp_path),
        "--manifest-sha256",
        "changed" if mode == "hash" else sha,
        "--output",
        str(tmp_path / "out"),
    ]
    if mode in ("missing-id", "duplicate-id"):
        args += ["--case-id", "1"] if mode == "missing-id" else ["--case-id", "0"] * 2
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as failed:
        model_captured_header.main()
    assert failed.value.code == 2 and not (tmp_path / "out").exists()


def test_report_real_header_CRC_is_separate_from_length_MCS_and_data_CRC():
    frames = [
        (
            {"frame": dict(header_crc=True, length_ok=False, data_crc=None)},
            SimpleNamespace(packets=[]),
        ),
        (
            {"frame": dict(header_crc=True, length_ok=True, mcs=12, data_crc=False)},
            SimpleNamespace(packets=[]),
        ),
        (
            {"frame": dict(header_crc=False, length_ok=None, data_crc=None)},
            SimpleNamespace(packets=[]),
        ),
    ]
    result = model_captured_header.gate_counts((None, frames), (8,))
    assert result == dict(
        head_CRC_pass=2,
        head_CRC_fail=1,
        length_pass=1,
        length_fail=1,
        permitted_MCS=0,
        data_CRC_pass=0,
        data_CRC_fail=1,
        MAC_packets=0,
    )


@pytest.mark.parametrize("ft", [2, 3])
def test_unknown_start_actual_IQ_uses_bounded_shared_source_and_identical_surviving_trace(ft):
    p, iq, quiet, payload, _, _ = fixture(ft, pilots=8)
    p = replace(p, head_backend="word-dense" if ft == 2 else "word-depth-parity")
    old = receive(p, iq, quiet, retire_before_fine=True)
    with patch(
        "tests.captured_frame_reference.CapturedFrameReplay", ClockedHeaderCapturedFrameReplay
    ):
        new = receive(p, iq, quiet, retire_before_fine=True, groups=(1, 31, 127))
    assert delivered(old[1]) == delivered(new[1]) == [payload]
    before = next(e for e, m in old[1] if m.packets)
    after = next(e for e, m in new[1] if m.packets)
    assert before["frame"] == after["frame"]
    for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(before[key], after[key])
    calendar = after["header_service_calendar"]
    used = [r for r in new[2] if r["id"] == after["selection_id"]]
    head = next(r for r in used if r["returned_cycle"] == calendar["issue_cycle"])
    following = next(r for r in used if r["index"] == head["index"] + 1)
    assert following["returned_cycle"] >= calendar["nominal_ready_cycle"] + 1
    assert [r["index"] for r in used] == list(
        range(after["source_window_origin"], after["consumed_source_end"])
    )
    assert new[0]["input_samples"] == len(iq) and new[4].done
    assert new[0]["peak_source_operations_per_primitive"] <= 2
    assert new[0]["complete_latency_cycles"] is None
    assert new[3].estimate()["additional_raw_RAM_DSP_ports"] == 0
