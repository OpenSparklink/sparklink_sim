"""Whole consumed window retirement after the actual fine service has completed."""

import pytest

from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from tests.captured_frame_reference import delivered
from tests.test_queued_retirement import fixed as fixed
from tests.test_queued_tail_composition import run


@pytest.fixture(scope="module")
def comparison(fixed):
    before = run(fixed, True, overlap_fine=True)
    after = run(fixed, True, overlap_fine=True, retire_completed_fine=True)
    return before, after


def test_completed_fine_uses_one_registered_compare_and_charges_real_fine_work(comparison):
    (baseline, _), (new, checks) = comparison
    assert delivered(baseline[1]) and delivered(baseline[1]) == delivered(new[1])
    assert new[0]["input_samples"] == baseline[0]["input_samples"]
    assert new[0]["peak_owned"] <= 3 and new[0]["peak_source_operations_per_primitive"] <= 2
    assert len(new[5]) == new[0]["requests"] == len({x["id"] for x in new[5]})
    issue = [x["cycle"] for x in checks
             if x["event"] in ("check", "queued_check", "completed_fine_check")]
    assert len(issue) == len(set(issue))
    retired = [x for x in new[5] if x["status"] == "retired_after_fine"]
    assert len(retired) == 1
    result = retired[0]
    before_release = next(x for x in checks if x["event"] == "completed_fine_check"
                          and x["id"] == result["id"] and x["exclude"])
    assert before_release["cycle"] + 1 == result["cycle"]
    assert before_release["frame_active"] and before_release["cycle"] % 2 == 0
    assert result["after_samples"] <= before_release["predicate_frontier"]
    assert result["fine_started_cycle"] + result["fine_cycles"] <= before_release["cycle"]
    assert not any(e["selection_id"] == result["id"] for e, _ in new[1])
    assert new[1][0][0]["symbol_centers_filtered"].tobytes() == (
        baseline[1][0][0]["symbol_centers_filtered"].tobytes()
    )


def test_completed_fine_under_sink_backpressure_keeps_unique_results(fixed, comparison):
    (_, _), (unblocked, _) = comparison
    end = unblocked[1][0][0]["initialized_capture_bridge"]["actual_terminal_cycle"]
    stalled, _ = run(fixed, True, overlap_fine=True, retire_completed_fine=True,
                     stalls=[(end - 11, end + 51)])
    assert delivered(stalled[1]) == delivered(unblocked[1])
    assert stalled[0]["drain_cycle"] > end + 51 and not stalled[0]["adc_backpressure"]
    assert stalled[0]["peak_owned"] <= 3 and not stalled[3].active
    assert stalled[0]["pending_descriptors"] == 0
    assert len(stalled[5]) == len({x["id"] for x in stalled[5]}) == stalled[0]["requests"]


def test_completed_fine_reuses_comparison_register_and_capacity():
    t = CaptureTransport(CaptureTransportAssumptions(paths=1))
    with pytest.raises(ValueError, match="overlapping ownership"):
        t.estimate(retire_completed_fine=True)
    with pytest.raises(ValueError, match="overlapping ownership"):
        t.simulate([], (), retire_completed_fine=True)
    options = dict(measured_worker=True, released_requests=True, retire_before_fine=True,
                   retire_queued=True, overlap_fine=True)
    base = t.estimate(**options)
    revised = t.estimate(**options, retire_completed_fine=True)
    key = "register_reservation_total_bits"
    assert revised[key] - base[key] == 1
    assert revised["transport_bram36"] == base["transport_bram36"]
    cost = revised["completed_fine_retirement"]
    assert cost["comparator_input_mux_bits_upper"] == 64
    assert cost["extra_IQ_RAM_DSP_or_sample_ports"] == 0
    assert cost["no_new_telemetry_counter_or_pending_descriptor"]
