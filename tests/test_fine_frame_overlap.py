"""Actual RX words through the bounded fine/frame overlap scheduler."""

import pytest

from nearlink_sdr.sim.capture_fine import CaptureFineWorker
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.legal_tail_ceiling import UpperTailCaptureReplay
from tests.captured_frame_reference import delivered
from tests.test_queued_retirement import fixed as fixed
from tests.test_queued_tail_composition import run


@pytest.fixture(scope="module")
def paired(fixed):
    return run(fixed, True), run(fixed, True, overlap_fine=True)


def test_overlap_uses_the_same_fine_receipts_and_exclusive_owned_slots(fixed, paired):
    (baseline, _), (candidate, checks) = paired
    stats, events, _, owner, source, outcomes = candidate
    assert delivered(baseline[1]) and delivered(events) == delivered(baseline[1])
    assert [e["selection_id"] for e, _ in events] == sorted(e["selection_id"] for e, _ in events)
    assert stats["fine_frame_overlap_cycles"] > 0 and stats["selected_result_wait_cycles"] > 0
    assert stats["input_samples"] == len(fixed[1]) and source.done and not owner.active
    assert stats["peak_owned"] <= 3 and stats["peak_source_operations_per_primitive"] <= 2
    pred = [x["cycle"] for x in checks if x["event"] in ("check", "queued_check")]
    assert len(pred) == len(set(pred))
    assert any(x["event"] == "fine_start" and x["frame_active"] for x in checks)
    assert len(outcomes) == len({e["id"] for e in outcomes}) == stats["requests"]
    first_old, first_new = baseline[1][0][0], events[0][0]
    assert first_old["upstream_selection"] == first_new["upstream_selection"]
    assert first_old["symbol_centers_filtered"].tobytes() == (
        first_new["symbol_centers_filtered"].tobytes()
    )
    assert stats["complete_latency_cycles"] is None and not stats["whole_system_accepted"]


def test_one_cache_credit_holds_under_backpressure_and_drains_at_eof(fixed, paired):
    (baseline, _), (candidate, _) = paired
    events = candidate[1]
    release = events[0][0]["initialized_capture_bridge"]["actual_terminal_cycle"]
    stalled, _ = run(fixed, True, overlap_fine=True, stalls=[(release - 23, release + 73)])
    assert delivered(stalled[1]) == delivered(baseline[1])
    assert stalled[0]["drain_cycle"] >= release + 73
    assert stalled[0]["input_samples"] == len(fixed[1]) and not stalled[0]["adc_backpressure"]
    assert not stalled[3].active and stalled[0]["pending_descriptors"] == 0
    assert stalled[0]["selected_result_wait_cycles"] >= candidate[0]["selected_result_wait_cycles"]


def test_new_fine_service_cannot_reuse_the_selected_cache_during_initialization(fixed, monkeypatch):
    owners, concurrent = [], []
    init = UpperTailCaptureReplay.__init__
    service = CaptureFineWorker.__call__

    def capture_owner(owner, *args, **kwargs):
        init(owner, *args, **kwargs)
        owners.append(owner)

    def inspect_credit(worker, descriptor, words):
        owner = owners[-1]
        if owner.active:
            assert owner.flow is not None and owner.flow.frame is not None
            assert owner.flow.transfer_cycle <= descriptor["service_start_cycle"]
            concurrent.append(descriptor["id"])
        return service(worker, descriptor, words)

    monkeypatch.setattr(UpperTailCaptureReplay, "__init__", capture_owner)
    monkeypatch.setattr(CaptureFineWorker, "__call__", inspect_credit)
    run(fixed, True, overlap_fine=True)
    assert concurrent


def test_fault_cancels_both_owners_without_reusing_a_selection(fixed, paired):
    (_, _), (candidate, checks) = paired
    cycle = next(x["cycle"] for x in checks if x["event"] == "fine_start" and x["frame_active"])
    # Inject an invalid RRC pulse during the measured concurrent ownership interval.
    start = next(x for x in candidate[5] if x["fine_started_cycle"] == cycle)
    end = cycle + start["fine_cycles"]
    from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions

    flow = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=200_000_000))
    pulse = next(x["sample"] for x in flow.simulate(len(fixed[1]), trace=True)["trace"]
                 if x["event"] == "output" and cycle < x["cycle"] < end)
    fault, _ = run(fixed, True, overlap_fine=True, defect=pulse)
    assert not delivered(fault[1]) and not fault[3].active and fault[4].done
    cancelled = [x for x in fault[5] if x["status"] == "source_epoch_aborted"]
    assert len(cancelled) == fault[0]["source_epoch_drops"] >= 2
    assert len(cancelled) == len({x["id"] for x in cancelled})
    assert fault[0]["input_samples"] == len(fixed[1])


def test_overlap_estimate_requires_queued_owner_and_charges_second_metadata_read():
    transport = CaptureTransport(CaptureTransportAssumptions(paths=1))
    with pytest.raises(ValueError, match="queued-retirement"):
        transport.estimate(overlap_fine=True)
    with pytest.raises(ValueError, match="queued-retirement"):
        transport.simulate([], (), overlap_fine=True)
    common = dict(measured_worker=True, released_requests=True,
                  retire_before_fine=True, retire_queued=True)
    before, after = transport.estimate(**common), transport.estimate(**common, overlap_fine=True)
    cost = after["fine_frame_overlap"]
    key = "register_reservation_total_bits"
    assert after[key] - before[key] == 133
    assert cost["extra_descriptor_read_mux_bits_upper"] == before["descriptor_reserved_bits_each"]
    assert cost["selection_credits"] == 1 and cost["extra_IQ_RAM_DSP_or_sample_ports"] == 0
    assert cost["slot_reuse_waits_for_installed_frame"]
    assert after["transport_bram36"] == before["transport_bram36"]
