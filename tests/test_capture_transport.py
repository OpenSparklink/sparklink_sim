"""Independent absolute-index oracle, wrap/EOF/expiry and finite-port overloads."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions


def source(n):
    rng = np.random.default_rng(610117)
    values = rng.integers(0, 2**46, (n, 2), dtype=np.uint64)
    values[:4] = [[0, 2**46-1], [1, 2**45], [2**38, 127], [255, 65535]]
    return values


def verify(words, **options):
    requests = options.pop("requests", [(i, i % 2) for i in range(256, len(words)+1, 64)])
    a = options.pop("assumptions", CaptureTransportAssumptions())
    events = []

    def sink(event, window):
        if window is not None:
            end, path = event["after_samples"], event["path"]
            assert window == words[end-256:end, path].tolist()
        events.append(event)

    result = CaptureTransport(a).simulate(words, requests, sink=sink, **options)
    assert result["input_samples"] == len(words)
    assert result["input_writes"] == len(words)*a.paths
    assert (result["completed"] + result["queue_drops"] + result["history_drops"]
            + result["startup_drops"]) == len(events)
    assert result["peak_owned"] <= a.owned and result["pending_descriptors"] == 0
    assert result["peak_source_operations_per_primitive"] <= 2
    assert result["peak_destination_writes_per_bank"] <= (2 if a.mode == "tdp2" else 1)
    assert not result["adc_backpressure"] and not result["whole_system_accepted"]
    return result, events


@pytest.mark.parametrize("mode", ["sdp1", "tdp2", "copied8"])
@pytest.mark.parametrize("clock", [100_000_000, 160_000_000, 200_000_000])
def test_every_accepted_window_matches_independent_source_through_wrap_and_eof(mode, clock):
    a = CaptureTransportAssumptions(mode=mode, clock_hz=clock, fine_service_cycles=182)
    out, events = verify(source(2049), assumptions=a)
    assert out["completed"] == len(events) and not out["history_drops"]
    assert out["source_reads"] == out["destination_writes"] == len(events)*256
    assert out["min_read_expiry_slack_samples"] >= 0


def test_two_port_time_sharing_trades_ram_for_continuous_copy_bandwidth():
    words = source(4097)
    requests = [(i, (i//32) % 2) for i in range(256, len(words)+1, 32)]
    a = CaptureTransportAssumptions(clock_hz=100_000_000, fine_service_cycles=193)
    two, _ = verify(words, requests=requests, assumptions=a)
    one, _ = verify(words, requests=requests, assumptions=replace(a, mode="sdp1"))
    assert two["completed"] == len(requests) and not two["queue_drops"]
    assert one["queue_drops"] > 0
    assert CaptureTransport(a).estimate()["transport_bram36"] == 10
    assert CaptureTransport(replace(a, mode="sdp1")).estimate()["transport_bram36"] == 5


def test_result_stall_never_stalls_input_or_reuses_owned_windows():
    out, events = verify(source(2048), stalls=[(0, 20000)])
    assert out["completed_while_blocked"] > 0 and out["queue_drops"] > 0
    assert out["peak_owned"] == 3
    assert any(e["status"] == "owned_queue_full" for e in events)


def test_overwritten_history_drops_explicitly_instead_of_reading_wrong_generation():
    a = CaptureTransportAssumptions(mode="sdp1", history=256)
    out, events = verify(source(512), assumptions=a, requests=[(256, 0)]*3)
    assert out["history_drops"] == 2 and out["completed"] == 1
    assert [e["status"] for e in events].count("source_expired") == 2


def test_startup_short_history_and_no_candidates_still_charge_continuous_input():
    a = CaptureTransportAssumptions(paths=1)
    out, _ = verify(source(257)[:, :1], assumptions=a, requests=[(1, 0), (257, 0)])
    assert out["startup_drops"] == 1 and out["completed"] == 1
    no_requests, _ = verify(source(4), requests=[])
    assert no_requests["requests"] == 0 and no_requests["input_writes"] == 8


def test_slow_fine_workload_is_not_hidden_by_faster_copy_ports():
    out, _ = verify(source(4096), requests=[(i, 0) for i in range(256, 4097, 32)],
                    assumptions=CaptureTransportAssumptions(fine_service_cycles=1000))
    assert out["queue_drops"] > 0 and not out["fine_memory_ports_verified"]


def test_layout_is_not_logical_bits_divided_by_bram_capacity():
    small = CaptureTransport().estimate()
    copied = CaptureTransport(replace(CaptureTransportAssumptions(), mode="copied8")).estimate()
    assert small["logical_sample_bits"] == copied["logical_sample_bits"] == 70656
    assert small["source_bram36"] == 4 and small["destination_bram36"] == 6
    assert copied["transport_bram36"] == 40
    assert copied["input_writes_broadcast_per_sample"] == 16
    assert small["register_reservation_total_bits"] > 3*256
    assert sum(small["descriptor_fields_bits"].values()) <= small["descriptor_reserved_bits_each"]
    assert small["descriptor_reserved_bits_each"] > 256
    assert not small["other_source_readers_permitted"]
    assert small["lut_ff_synthesis"] is None


@pytest.mark.parametrize("changes", [dict(mode="unknown"), dict(input_hz=201_000_000),
    dict(word_bits=73), dict(owned=0), dict(owned=17), dict(paths=3), dict(history=128),
    dict(window=128), dict(clock_hz=True), dict(fine_service_cycles=2**31)])
def test_invalid_assumptions_do_not_become_free_resources(changes):
    with pytest.raises(ValueError):
        CaptureTransportAssumptions(**changes)


@pytest.mark.parametrize("requests", [[(257, 0)], [(1, 0), (0, 0)], [(256, 2)],
    [(True, 0)], [(256, True)], [(256, 0), (255, 0)]])
def test_bad_requests_fail_explicitly(requests):
    with pytest.raises(ValueError):
        CaptureTransport().simulate(source(256), requests)


@pytest.mark.parametrize("stalls", [None, [(-1, 2)], [(2, 2)], [(2, 5), (4, 6)],
                                    [(True, 2)], [(0, 2**63)], [(0, 1)]*1025])
def test_bad_external_stalls_do_not_silently_change_service(stalls):
    with pytest.raises(ValueError):
        CaptureTransport().simulate(source(4), [], stalls=stalls)


@pytest.mark.parametrize("words", [[], [[0]], [[-1, 0]], [[2**46, 0]], [[True, 0]], [[1., 0]]])
def test_bad_words_never_wrap_or_get_silently_quantized(words):
    with pytest.raises(ValueError):
        CaptureTransport().simulate(words, [])


def test_wrong_model_type_is_rejected():
    with pytest.raises(ValueError):
        CaptureTransport({})


def test_timestamp_and_access_counter_horizon_fails_before_simulation():
    with pytest.raises(OverflowError, match="horizon"):
        CaptureTransport().simulate(source(4), [], stalls=[(0, 2**60)])


def test_reproducer_preserves_explicit_old_resource_failure_and_real_word_control(
    tmp_path, monkeypatch
):
    import importlib
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    driver = importlib.import_module("model_capture_transport")
    out = tmp_path/"model"
    assert driver.run(out, 512) == 1
    report = json.loads((out/"model.json").read_text())
    assert len(report["scenarios"]) == 36 and not report["whole_system_accepted"]
    assert report["original_capture_budget_still_failed"]
    assert report["rrc_word_control"]["additional_word_errors"] == 0
    assert not report["rrc_word_control"]["upstream_rrc_numeric_gate"]
    for row in report["scenarios"]:
        assert sum(row["fine_stages"].values()) == row["optimistic_fine_service_cycles"]
        if row["workload"] == "old_fine_budget" and row["clock_hz"] == 200_000_000:
            assert row["optimistic_fine_service_cycles"] == 469
    with pytest.raises(ValueError):
        driver.run(tmp_path/"invalid", 128)
