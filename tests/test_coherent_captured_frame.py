"""Actual chosen fine cache→finite gate→real head/data/MAC and finite fault exits."""

from dataclasses import replace
from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.sim.captured_header_flow import ClockedHeaderCapturedFrameReplay
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from nearlink_sdr.sim.coherent_captured_frame import WordConfirmedCapturedFrameReplay
from tests.captured_frame_reference import delivered, fixture, receive
from tests.test_coherent_word_flow import source


def profile_case(ft=2, **options):
    profile, iq, quiet, expected, *_ = fixture(ft, **options)
    profile = replace(profile, head_backend="word-dense" if ft == 2 else "word-depth-parity")
    return profile, iq, quiet, expected


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_raw_RRC_fine_selection_word_function_and_clocked_gate_preserve_real_packet(ft):
    profile, iq, quiet, expected = profile_case(ft, size=23)
    modes = [ClockedHeaderCapturedFrameReplay,
             partial(WordConfirmedCapturedFrameReplay, clocked=False),
             partial(WordConfirmedCapturedFrameReplay, clocked=True)]
    outputs = []
    for owner in modes:
        with patch("tests.captured_frame_reference.CapturedFrameReplay", owner):
            outputs.append(receive(profile, iq, quiet, retire_before_fine=True))
    for result in outputs:
        assert delivered(result[1]) == [expected]
        assert result[0]["input_samples"] == len(iq) and result[4].done
        assert result[0]["live_source_statistics"] == outputs[0][0]["live_source_statistics"]
        assert result[0]["peak_owned"] <= 3
        assert result[0]["peak_source_operations_per_primitive"] <= 2
        event = next(e for e, mac in result[1] if mac.packets)
        assert event["frame"]["header_crc"] is event["frame"]["data_crc"] is True
        assert event["frame"]["length_ok"] is True
        assert event["frame"]["mcs"] == profile.config.mcs_index
        assert event["header_service_calendar"]["service_edges"] > 0
        assert event["complete_latency_cycles"] is None
    for result, clocked in zip(outputs[1:], [False, True], strict=True):
        event = next(e for e, mac in result[1] if mac.packets)
        row = event["coherent_word_service"]
        assert row["actual_finite_values"] and row["finite_computation_completed"]
        assert row["clocked"] is clocked and row["word_decision"] is True
        assert row["computed_edges"] == row["ready_cycle"] - row["issued_cycle"] > 0
        assert row["statistics"]["native_products"] == 8 * len(profile.sync_symbols()) + 12
        assert row["fabric_statistics"]["issued"] == row["fabric_statistics"]["taken"]
        base = next(e for e, mac in outputs[0][1] if mac.packets)
        arrays = {k for k, v in base.items() if isinstance(v, np.ndarray)}
        assert {"symbols", "symbol_centers_filtered"} <= arrays
        assert arrays == {k for k, v in event.items() if isinstance(v, np.ndarray)}
        for key in arrays:
            assert base[key].dtype == event[key].dtype and base[key].tobytes() == event[key].tobytes()


def selected_case():
    # Use a real fine descriptor obtained by the source in the test,not truth.
    profile, iq, quiet, _ = profile_case(2, size=31)
    selections = []
    original = ClockedHeaderCapturedFrameReplay.begin

    def begin(owner, selected, cycle):
        selections.append((selected, cycle))
        return original(owner, selected, cycle)

    with (
        patch("tests.captured_frame_reference.CapturedFrameReplay", ClockedHeaderCapturedFrameReplay),
        patch.object(ClockedHeaderCapturedFrameReplay, "begin", begin),
    ):
        receive(profile, iq, quiet, retire_before_fine=True)
    return profile, *selections[0]


def test_pending_and_held_gate_never_reads_raw_or_publishes_confirmation_early():
    profile, selected, cycle = selected_case()
    owner = WordConfirmedCapturedFrameReplay(profile, coherent_threshold=.45)
    owner.begin(selected, cycle)
    assert owner.active and owner.rx is None and not owner.needs_word
    with pytest.raises(RuntimeError, match="confirmation"):
        owner.end_input(cycle)
    with pytest.raises(ValueError, match="registered edge"):
        owner.abort("cancelled", cycle)
    while not owner.word_service.ready:
        owner.tick(owner.cycle + 1, blocked=True)
        assert owner.rx is None and not owner.needs_word and owner.header is None
    ready = owner.word_service.ready_cycle
    for _ in range(71):
        owner.tick(owner.cycle + 1, blocked=True)
    assert owner.word_receipt is None and owner.word_service.ready_cycle == ready
    owner.tick(owner.cycle + 1)
    assert owner.needs_word and owner.word_stats["completed"] == 1
    owner.abort("cancelled", owner.cycle + 1)
    event, mac = owner.take()
    row = event["coherent_word_service"]
    assert row["computed_edges"] == ready - cycle
    assert row["elapsed_owned_edges"] == row["computed_edges"] + 72
    assert not mac.packets and owner.stats["registered_reads"] == 0
    owner.release()
    assert not owner.active and owner.word_receipt is owner.word_service is None


def test_finite_receipt_is_the_only_confirmation_decision_not_a_second_float_gate(monkeypatch):
    profile, iq, quiet, expected = profile_case(2, size=31)
    monkeypatch.setattr(
        "nearlink_sdr.phy.selected_rx.coherent_sync_confirmation",
        lambda *a, **k: pytest.fail("Word gate must not call floating confirmation again"),
    )
    with patch("tests.captured_frame_reference.CapturedFrameReplay", WordConfirmedCapturedFrameReplay):
        result = receive(profile, iq, quiet, retire_before_fine=True)
    assert delivered(result[1]) == [expected]


@pytest.mark.parametrize("bad_head,bad_data,status", [
    (True, False, "bad_header_crc"), (False, True, "bad_data_crc"),
])
def test_real_CRC_failures_remain_separate_from_successful_word_confirmation(bad_head, bad_data, status):
    profile, iq, quiet, _ = profile_case(2, size=31, bad_head=bad_head, bad_data=bad_data)
    with patch("tests.captured_frame_reference.CapturedFrameReplay", WordConfirmedCapturedFrameReplay):
        result = receive(profile, iq, quiet, retire_before_fine=True)
    assert not delivered(result[1])
    row = next(e for e, _ in result[1] if e["frame"]["status"] == status)
    assert row["coherent_word_service"]["word_decision"] is True
    assert row["frame"]["header_crc"] is (not bad_head)
    assert row["frame"]["data_crc"] is (False if bad_data else None)


def test_source_epoch_failure_during_word_service_has_no_stale_gate_or_MAC(monkeypatch):
    profile, iq, quiet, _ = profile_case(2, size=31)
    fault = {}
    original_begin = WordConfirmedCapturedFrameReplay.begin
    original_advance = CoarseCaptureSource.advance

    def begin(owner, selected, cycle):
        original_begin(owner, selected, cycle)
        if owner._word_waiting:
            fault["edge"] = cycle + 3

    def advance(source, cycle):
        if cycle == fault.get("edge"):
            source.flow.abort()
        return original_advance(source, cycle)

    monkeypatch.setattr(WordConfirmedCapturedFrameReplay, "begin", begin)
    monkeypatch.setattr(CoarseCaptureSource, "advance", advance)
    with patch("tests.captured_frame_reference.CapturedFrameReplay", WordConfirmedCapturedFrameReplay):
        result = receive(profile, iq, quiet, retire_before_fine=True)
    assert result[4].fault_cycle == fault["edge"] and result[4].done
    assert result[0]["input_samples"] == len(iq) and not delivered(result[1])
    assert result[0]["source_epoch_drops"] > 0 and not result[3].active
    event = next(e for e, _ in result[1] if e["frame"]["status"] == "source_epoch_aborted")
    assert not event["coherent_word_service"]["actual_finite_values"]
    assert event["coherent_word_service"]["service_state"] == "aborted"
    assert event["frame"]["header_crc"] is event["frame"]["data_crc"] is None


def test_zero_energy_rejects_before_any_header_raw_word_or_fixed_success_marker():
    profile, *_ = profile_case(2)
    selected = source(((0, 0),) * 32)
    owner = WordConfirmedCapturedFrameReplay(profile, coherent_threshold=.45)
    owner.begin(selected, selected.available_cycle)
    while not owner.done:
        owner.tick(owner.cycle + 1)
    event, mac = owner.take()
    assert event["frame"]["status"] == "sync_confirmation_rejected" and not mac.packets
    assert event["coherent_word_service"]["word_decision"] is False
    assert owner.stats["registered_reads"] == owner.stats["coherent_accepted"] == 0


def test_cost_totals_include_existing_head_service_new_word_state_ROM_and_control():
    profile, *_ = profile_case(2)
    owner = WordConfirmedCapturedFrameReplay(profile, coherent_threshold=.45)
    cost = owner.estimate()
    assert sum(v for v in cost["register_fields"].values() if type(v) is int) == (
        cost["register_bits_upper"]
    ) == cost["combined_owner_register_bits"] == 6554 + 94
    assert cost["word_gate_control_node_bits"] == 5067 + 94 <= (
        cost["word_gate_control_research_cap_bits"]
    )
    assert cost["coherent_word_service"]["coefficient_RAM18"] == 1
    assert cost["coherent_word_service"]["DSP25x18"] == 1
    assert cost["remaining_bootstrap_carrier_demap_payload_cycles"] is None
    assert not cost["complete_cycle_evidence_valid"]
    for options in [dict(coherent_threshold=.44), dict(coherent_threshold=.45, clocked=1)]:
        with pytest.raises(ValueError):
            WordConfirmedCapturedFrameReplay(profile, **options)
