"""Conservative registered pre-fine retirement,not CRC-selected synchronization."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from tests.captured_frame_reference import delivered, fixture, receive


def compare_packets_and_trace(old, new):
    assert delivered(old[1]) == delivered(new[1])
    a = [(e, m) for e, m in old[1] if m.packets]
    b = [(e, m) for e, m in new[1] if m.packets]
    assert len(a) == len(b)
    for (x, _), (y, _) in zip(a, b, strict=True):
        assert x["frame"] == y["frame"]
        for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm"):
            np.testing.assert_array_equal(x[key], y[key])


def test_registered_gate_removes_old_fine_work_without_new_words_or_CRC_choices():
    p, iq, quiet, payload, _, _ = fixture()
    old = receive(p, iq, quiet)
    checks = []
    new = receive(p, iq, quiet, retire_before_fine=True, retirement_observer=checks.append)
    result, frames, reads, owner, source, outcomes = new
    compare_packets_and_trace(old, new)
    assert delivered(frames) == [payload] and source.done and not owner.active
    assert result["pre_fine_retired"] == 2
    assert result["completed"] == old[0]["completed"] - 2
    assert result["requests"] == sum(
        result[k]
        for k in (
            "completed",
            "pre_fine_retired",
            "queue_drops",
            "history_drops",
            "startup_drops",
            "source_epoch_drops",
        )
    )
    assert sum(r["fine_cycles"] or 0 for r in outcomes) < sum(r["fine_cycles"] or 0 for r in old[5])
    assert result["live_source_statistics"] == old[0]["live_source_statistics"]
    assert [(r["id"], r["index"], r["word"]) for r in reads] == [
        (r["id"], r["index"], r["word"]) for r in old[2]
    ]
    by_id = {r["id"]: r for r in checks if r["event"] == "check"}
    for r in checks:
        assert not r["frame_active"]
        if r["event"] in ("retired", "fine_start"):
            before = by_id[r["id"]]
            assert r["cycle"] == before["cycle"] + 1
            assert before["retired_through"] == r["retired_through"]
        if r["event"] == "fine_available":
            assert r["retired_through"] == by_id[r["id"]]["retired_through"]
    for r in outcomes:
        if r["status"] == "retired_before_fine":
            assert r["fine_cycles"] is r["disposition"] is r["start_tick"] is None
            assert by_id[r["id"]]["exclude"] is True
            assert r["cycle"] == by_id[r["id"]]["cycle"] + 1
    assert result["peak_source_operations_per_primitive"] <= 2
    assert result["retirement_gate_cost"]["register_bits"] == 66
    assert result["complete_latency_cycles"] is None and not result["whole_system_accepted"]


@pytest.mark.parametrize("bad_head,bad_data", [(True, False), (False, True)])
def test_failed_CRC_remains_failed_and_retirement_is_based_on_consumed_source(bad_head, bad_data):
    p, iq, quiet, _, _, _ = fixture(bad_head=bad_head, bad_data=bad_data)
    checks = []
    new = receive(p, iq, quiet, retire_before_fine=True, retirement_observer=checks.append)
    assert delivered(new[1]) == []
    row = next(e["frame"] for e, _ in new[1])
    assert row["header_crc"] is (not bad_head)
    if bad_data:
        assert row["data_crc"] is False
    assert new[0]["pre_fine_retired"] > 0
    for r in checks:
        if r["event"] == "check":
            assert r["exclude"] == (r["source_end"] <= r["retired_through"])
            assert not any("crc" in k.lower() or "payload" in k.lower() for k in r)


def test_host_chunks_preserve_gate_cycles_words_and_sampling_with_fixed_gate():
    p, iq, quiet, _, _, _ = fixture(2, size=43, pilots=8)
    first = receive(p, iq, quiet, retire_before_fine=True)
    groups = tuple(np.random.default_rng(145817).integers(1, 140, 17))
    other = receive(p, iq, quiet, groups=groups, retire_before_fine=True)
    assert first[0] == other[0] and first[2] == other[2]

    def metadata(events):
        return [
            {k: v for k, v in e.items() if k not in ("strict_frame", "MAC_packets")} for e in events
        ]

    assert metadata(first[5]) == metadata(other[5])
    for (a, _), (b, _) in zip(first[1], other[1], strict=True):
        assert a["frame"] == b["frame"]
        for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm"):
            np.testing.assert_array_equal(a[key], b[key])
    compare_packets_and_trace(first, other)


def test_resource_fault_during_live_frame_still_cancels_and_consumes_every_input():
    p, iq, quiet, _, _, _ = fixture(size=713)
    result, frames, reads, owner, source, _ = receive(
        p, iq, quiet, defect=3200, retire_before_fine=True
    )
    assert delivered(frames) == [] and source.done and not owner.active
    assert result["input_samples"] == len(iq) and result["source_epoch_drops"] > 0
    assert all(r["returned_cycle"] < source.fault_cycle for r in reads)
    assert any(e["frame"]["status"] == "source_epoch_aborted" for e, _ in frames)


def test_source_abort_on_registered_check_return_does_not_start_fine_or_deliver(monkeypatch):
    p, iq, quiet, _, _, _ = fixture(size=713)
    fault, checks, fine = {}, [], []
    original = CoarseCaptureSource.advance

    def external_abort(source, cycle):
        if cycle == fault.get("cycle"):
            source.flow.abort()  # Explicit testbench source-epoch fault,not a receiver choice.
        return original(source, cycle)

    def observe(row):
        checks.append(row)
        if row["event"] == "check":
            fault["cycle"] = row["cycle"] + 1

    monkeypatch.setattr(CoarseCaptureSource, "advance", external_abort)
    result, frames, reads, owner, source, outcomes = receive(
        p,
        iq,
        quiet,
        retire_before_fine=True,
        retirement_observer=observe,
        fine_observer=lambda *args: fine.append(args),
    )
    assert source.done and source.fault_cycle == fault["cycle"]
    assert len(checks) == 1 and not fine and not frames and not reads and not owner.active
    assert result["input_samples"] == len(iq) and result["pre_fine_retired"] == 0
    assert result["source_epoch_drops"] > 0 and result["completed"] == 0
    assert any(r["status"] == "source_epoch_aborted" for r in outcomes)


def test_incremental_cost_has_no_free_counter_comparator_or_registered_decision():
    t = CaptureTransport(CaptureTransportAssumptions(paths=1, history=4096))
    base = t.estimate(measured_worker=True, released_requests=True)
    new = t.estimate(measured_worker=True, released_requests=True, retire_before_fine=True)
    assert new["register_reservation_total_bits"] - base["register_reservation_total_bits"] == 66
    assert new["transport_bram36"] == base["transport_bram36"]
    assert new["retirement_gate"]["unsigned_comparator_bits"] == 63
    assert new["retirement_gate"]["registered_check_cycles"] == 1
    assert new["retirement_gate"]["achieved_Fmax"] is None
    assert new["extra_dsp"] == 0 and not new["full_chain_verified"]


@pytest.mark.parametrize("choice", [1, None, "yes"])
def test_nonexplicit_or_unowned_retirement_cannot_change_default_transport(choice):
    with pytest.raises(ValueError, match="retirement"):
        CaptureTransport().simulate([[0, 0]], (), retire_before_fine=choice)
    with pytest.raises(ValueError, match="retirement"):
        CaptureTransport().estimate(retire_before_fine=choice)


@pytest.mark.parametrize(
    "options",
    [
        dict(retire_before_fine=True),
        dict(retirement_observer=lambda _: None),
        dict(retirement_observer=3),
    ],
)
def test_missing_actual_frame_owner_or_disabled_observer_is_an_error(options):
    with pytest.raises(ValueError, match="retirement"):
        CaptureTransport().simulate([[0, 0]], (), **options)


def test_gate_cost_rejects_unavailable_or_incompatible_compile_context():
    a = CaptureTransportAssumptions(paths=1, history=4096)
    for options in (dict(), dict(measured_worker=True), dict(released_requests=True)):
        with pytest.raises(ValueError, match="retirement"):
            CaptureTransport(a).estimate(retire_before_fine=True, **options)
    for changes in (dict(paths=2), dict(mode="sdp1"), dict(word_bits=45)):
        with pytest.raises(ValueError, match="retirement"):
            CaptureTransport(replace(a, **changes)).estimate(
                measured_worker=True, released_requests=True, retire_before_fine=True
            )


@pytest.mark.parametrize("window", [256, 376, 632])
def test_whole_window_bound_includes_fractional_ticks_and_preserves_straddling_uncertainty(window):
    for origin in (0, 17, 2**49 - window):
        end = origin + window
        all_starts = [8 * origin + tick for tick in range(8, 8 * window)]
        assert max(all_starts) < 8 * end
        for frontier in (end, end + 1, end + 157):
            assert all(tick < 8 * frontier for tick in all_starts)
        # A smaller frontier does not prove exclusion for the full port domain.
        assert not all(tick < 8 * (end - 1) for tick in all_starts)
