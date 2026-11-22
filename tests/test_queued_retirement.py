"""Ready-slot lifetime,registered exclusion and source faults on actual RX words."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.captured_header_flow import ClockedHeaderCapturedFrameReplay
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from tests.captured_frame_reference import delivered, receive


@pytest.fixture(scope="module")
def fixed():
    root = Path(__file__).parent / "data/captured-header-calendar"
    row = json.loads((root / "inputs.json").read_text())[0]
    raw = (root / row["file"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row["SHA256"]
    profile = replace(
        PSKReceiveProfile.ordinary(TxConfig(**row["config"])),
        head_backend=row["head_backend"],
        allowed_mcs=(row["config"]["mcs_index"],),
    )
    with np.load(root / row["file"], allow_pickle=False) as data:
        iq, quiet = data["iq"], data["quiet"]
    return profile, iq, quiet, [bytes.fromhex(x) for x in row["expected"]]


def run(fixed, **options):
    checks = []
    with patch("tests.captured_frame_reference.CapturedFrameReplay", ClockedHeaderCapturedFrameReplay):
        result = receive(
            *fixed[:3], retire_before_fine=True, retirement_observer=checks.append, **options
        )
    return result, checks


@pytest.fixture(scope="module")
def paired(fixed):
    return run(fixed), run(fixed, retire_queued=True)


def test_immutable_queued_windows_free_before_next_packet_without_CRC_selection(fixed, paired):
    (old, _), (new, checks) = paired
    assert delivered(old[1]) == fixed[3][:1]  # Retained original head-wait resource failure.
    assert delivered(new[1]) == fixed[3]
    assert new[0]["queued_retired"] == 3 and new[0]["queue_drops"] < old[0]["queue_drops"]
    assert new[0]["input_samples"] == old[0]["input_samples"] == len(fixed[1])
    assert new[0]["live_source_statistics"] == old[0]["live_source_statistics"]
    assert new[0]["live_coarse_statistics"] == old[0]["live_coarse_statistics"]
    assert new[0]["requests"] == sum(
        new[0].get(k, 0) for k in (
            "completed", "queued_retired", "pre_fine_retired", "queue_drops", "history_drops",
            "startup_drops", "source_epoch_drops",
        )
    )
    pending = None
    for row in checks:
        assert not any("crc" in k.lower() or "payload" in k.lower() for k in row)
        if row["event"] == "queued_check":
            assert pending is None and row["frame_active"]
            assert row["copied_cycle"] < row["cycle"]
            assert row["predicate_frontier"] <= row["retired_through"]
            assert row["exclude"] == (row["source_end"] <= row["predicate_frontier"])
            pending = row
        elif row["event"] in ("queued_kept", "queued_retired"):
            assert pending is not None and row["id"] == pending["id"]
            assert row["cycle"] == pending["cycle"] + 1
            assert row["exclude"] is pending["exclude"]
            assert (row["event"] == "queued_retired") is row["exclude"]
            pending = None
    assert pending is None
    issue_cycles = {r["cycle"] for r in checks if r["event"] == "queued_check"}
    assert issue_cycles.isdisjoint(r["cycle"] for r in checks if r["event"] == "check")
    assert new[0]["peak_owned"] <= 3 and new[0]["peak_source_operations_per_primitive"] <= 2
    assert new[0]["complete_latency_cycles"] is None and not new[0]["whole_system_accepted"]
    # Recovered later packet does not change the original surviving packet's sampling.
    a, b = old[1][0][0], new[1][0][0]
    for key in a.keys() & b.keys():
        if isinstance(a[key], np.ndarray):
            assert a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()


@pytest.mark.parametrize("groups", [(1, 31, 127), (73, 5, 239, 11, 2, 157)])
def test_chunk_boundaries_leave_words_checks_and_MAC_identical(fixed, paired, groups):
    new, checks = paired[1]
    other, other_checks = run(fixed, retire_queued=True, groups=groups)
    assert other[0] == new[0] and other[2] == new[2] and other_checks == checks
    assert delivered(other[1]) == delivered(new[1])


def test_blocked_MAC_still_allows_registered_release_and_unstoppable_input(fixed, paired):
    _, checks = paired[1]
    edge = next(r["cycle"] for r in checks if r["event"] == "queued_retired")
    result, observed = run(fixed, retire_queued=True, stalls=[(edge, edge + 71)])
    assert any(r["event"] == "queued_retired" and r["cycle"] == edge for r in observed)
    assert result[0]["input_samples"] == len(fixed[1]) and result[4].done
    assert not result[0]["adc_backpressure"] and not result[3].active
    assert result[0]["peak_owned"] <= 3


def test_source_epoch_fault_cancels_borrowed_head_before_registered_reply(fixed, monkeypatch):
    original, fault, checks = CoarseCaptureSource.advance, {}, []

    def abort(source, cycle):
        if cycle == fault.get("cycle"):
            source.flow.abort()
        return original(source, cycle)

    def observe(row):
        checks.append(row)
        if row["event"] == "queued_check" and not fault:
            fault["cycle"] = row["cycle"] + 1

    monkeypatch.setattr(CoarseCaptureSource, "advance", abort)
    with patch("tests.captured_frame_reference.CapturedFrameReplay", ClockedHeaderCapturedFrameReplay):
        result, frames, _, owner, source, outcomes = receive(
            *fixed[:3], retire_before_fine=True, retire_queued=True,
            retirement_observer=observe,
        )
    assert source.fault_cycle == fault["cycle"] and source.done and not owner.active
    assert checks[-1]["event"] == "queued_check"
    assert result["input_samples"] == len(fixed[1]) and not delivered(frames)
    assert result["source_epoch_drops"] > 0 and result["queued_retired"] == 0
    assert len({r["id"] for r in outcomes}) == len(outcomes)
    assert any(r["status"] == "source_epoch_aborted" for r in outcomes)


def test_incremental_cost_includes_polling_mux_counter_and_state_without_free_RAM():
    transport = CaptureTransport(CaptureTransportAssumptions(paths=1, history=4096))
    base = transport.estimate(measured_worker=True, released_requests=True, retire_before_fine=True)
    new = transport.estimate(
        measured_worker=True, released_requests=True, retire_before_fine=True, retire_queued=True
    )
    cost = new["queued_retirement_gate"]
    assert new["register_reservation_total_bits"] - base["register_reservation_total_bits"] == 66
    assert cost["register_bits"] == 66 <= cost["research_cap_bits"] == 128
    assert cost["combined_retirement_state_bits"] == 132
    assert cost["comparator_bits"] == cost["additional_operand_mux2_bits"] == 63
    assert cost["comparator_shared_with_disjoint_pre_fine"]
    assert cost["comparisons_per_active_ready_clock_upper"] == cost["check_II"] == 1
    assert cost["extra_IQ_RAM_DSP_or_sample_ports"] == 0
    assert new["transport_bram36"] == base["transport_bram36"]
    assert cost["achieved_Fmax"] is None and not new["full_chain_verified"]


@pytest.mark.parametrize("choice", [1, None, "yes", True])
def test_unowned_or_nonexplicit_ready_retirement_is_rejected(choice):
    with pytest.raises(ValueError, match="retirement"):
        CaptureTransport().estimate(retire_queued=choice)
    with pytest.raises(ValueError, match="retirement"):
        CaptureTransport().simulate([[0, 0]], (), retire_queued=choice)


def test_disabled_flag_leaves_original_result_schema_and_decisions(paired):
    result, checks = paired[0]
    assert "queued_retired" not in result[0] and "queued_retirement_gate_cost" not in result[0]
    assert not any(r["event"].startswith("queued_") for r in checks)
