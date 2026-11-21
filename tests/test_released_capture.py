"""A delayed candidate owns its original source window,never a later one."""

from dataclasses import asdict

import pytest

from nearlink_sdr.sim.capture_transport import (
    CaptureRequest,
    CaptureTransport,
    CaptureTransportAssumptions,
    FineService,
    ReleasedCaptureRequest,
)


class Source:
    """Generate only the next arriving word;no capture-sized input oracle."""

    def __init__(self, count=512, paths=1):
        self.count, self.paths, self.seen = count, paths, 0

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        assert index == self.seen
        self.seen += 1
        return tuple(index + 65536 * path for path in range(self.paths))


def run(requests, *, count=512, paths=1, **kwargs):
    """Evidence copies leave the hardware model;one fixed immutable window."""
    records = []

    def sink(event, data):
        records.append((dict(event), None if data is None else tuple(data)))

    model = CaptureTransport(
        CaptureTransportAssumptions(paths=paths, owned=1, fine_service_cycles=1)
    )
    source = Source(count, paths)
    result = model.simulate(source, requests, released_requests=True, sink=sink, **kwargs)
    assert source.seen == count == result["input_samples"]
    assert result["completed"] + result["queue_drops"] + result["history_drops"] + result[
        "startup_drops"
    ] == result["requests"]
    return result, records


def test_delayed_candidate_preserves_window_and_cannot_admit_early():
    # The two naive old bindings admitted at3188 or shifted all256 words by11.
    request = ReleasedCaptureRequest(256, 0, (2,), available_cycle=3325)
    result, [(event, words)] = run([request])
    assert result["completed"] == 1
    assert event["offered_cycle"] == 3325 and event["after_samples"] == 256
    assert words == tuple(range(256))
    assert result["pending_request_capacity"] == 1


@pytest.mark.parametrize("available", [3188, 3325, 4000])
def test_release_and_source_readiness_are_both_required(available):
    _, [(event, words)] = run(
        [ReleasedCaptureRequest(256, 0, (2,), available_cycle=available)]
    )
    assert event["offered_cycle"] == max(3188, available)
    assert words == tuple(range(256))


def test_late_request_after_eof_uses_retained_source_without_padding():
    result, [(event, words)] = run(
        [ReleasedCaptureRequest(400, 0, (2,), available_cycle=6500)]
    )
    assert result["completed"] == 1 and event["offered_cycle"] == 6500
    assert words == tuple(range(144, 400))


def test_expired_late_window_is_not_replaced_by_available_recent_samples():
    result, [(event, words)] = run(
        [ReleasedCaptureRequest(256, 0, (2,), available_cycle=6000)]
    )
    assert result["history_drops"] == 1 and result["completed"] == 0
    assert event["status"] == "source_expired" and words is None
    assert result["source_state_words"] == 384 and result["owned_state_words"] == 256


def test_late_request_keeps_path_and_base_fine_geometry_without_timestamp_copy():
    observed = []

    def fine(event, words):
        assert type(event["geometry"]) is CaptureRequest
        assert asdict(event["geometry"]) == asdict(CaptureRequest(256, 1, (2,), 8, 128))
        assert words == list(range(65536, 65536 + 256))
        observed.append(event)
        return FineService(9, "no_candidate")

    result, _ = run(
        [ReleasedCaptureRequest(256, 1, (2,), 8, 128, available_cycle=3325)],
        paths=2,
        fine_worker=fine,
    )
    assert len(observed) == 1 and result["completed"] == 1
    assert result["worker_results"]["no_candidate"] == 1


def test_one_owned_credit_drops_later_request_without_stopping_source():
    result, records = run(
        [
            ReleasedCaptureRequest(256, 0, (2,), available_cycle=3325),
            ReleasedCaptureRequest(267, 0, (2,), available_cycle=3326),
        ]
    )
    assert result["completed"] == result["queue_drops"] == 1
    assert {event["status"] for event, _ in records} == {"completed", "owned_queue_full"}
    assert next(words for event, words in records if event["status"] == "completed") == tuple(
        range(256)
    )


@pytest.mark.parametrize("window", [256, 376, 632])
@pytest.mark.parametrize("clock", [100_000_000, 160_000_000, 200_000_000])
def test_each_geometry_and_clock_preserves_source_ids_with_late_arrival(window, clock):
    model = CaptureTransport(
        CaptureTransportAssumptions(
            paths=1, owned=1, history=window + 128, window=window, clock_hz=clock
        )
    )
    source = Source(window + 1024)
    end, records = window + 32, []
    available = ((end - 1) * clock + 15_999_999) // 16_000_000 + 91

    def sink(event, words):
        records.append((dict(event), None if words is None else tuple(words)))

    result = model.simulate(
        source,
        [ReleasedCaptureRequest(end, 0, (2,), available_cycle=available)],
        released_requests=True,
        sink=sink,
    )
    assert source.seen == len(source) and result["completed"] == 1
    [(event, words)] = records
    assert event["offered_cycle"] == available and words == tuple(range(32, end))
    assert result["peak_source_operations_per_primitive"] <= 2
    assert result["peak_destination_writes_per_bank"] <= 2


@pytest.mark.parametrize("value", [True, -1, 1.5, 2**60, "123"])
def test_bad_release_format_rejected(value):
    with pytest.raises(ValueError, match="availability"):
        ReleasedCaptureRequest(256, 0, (2,), available_cycle=value)


def test_inherited_bad_source_geometry_rejected():
    with pytest.raises(ValueError, match="geometry"):
        ReleasedCaptureRequest(0, 0, (2,), available_cycle=0)


def test_release_order_and_finite_horizon_cannot_be_hidden_by_retained_samples():
    for requests in (
        [
            ReleasedCaptureRequest(256, 0, (2,), available_cycle=3325),
            ReleasedCaptureRequest(267, 0, (2,), available_cycle=3324),
        ],
        [ReleasedCaptureRequest(256, 0, (2,), available_cycle=2**59)],
        [ReleasedCaptureRequest(256, 0, (2,), available_cycle=0)],
        [
            ReleasedCaptureRequest(256, 0, (2,), available_cycle=3325),
            ReleasedCaptureRequest(267, 0, (2,), available_cycle=3325),
        ],
    ):
        with pytest.raises(ValueError, match="ordered release times"):
            run(requests)


def test_explicit_port_required_and_modes_cannot_be_mixed():
    request = ReleasedCaptureRequest(256, 0, (2,), available_cycle=3325)
    with pytest.raises(ValueError, match="matching port"):
        CaptureTransport().simulate(Source(paths=2), [request])
    with pytest.raises(ValueError, match="matching port"):
        run([CaptureRequest(256, 0, (2,))])
    with pytest.raises(ValueError, match="mode"):
        CaptureTransport().simulate(Source(paths=2), [], released_requests=1)


def test_only_registered_descriptor_port_added_and_legacy_cost_schema_retained():
    model = CaptureTransport()
    old = model.estimate(measured_worker=True)
    new = model.estimate(measured_worker=True, released_requests=True)
    assert "released_request_port" not in old
    assert new["released_request_port"]["register_bits"] == 242 <= 256
    assert new["register_reservation_total_bits"] - old["register_reservation_total_bits"] == 242
    assert new["logical_sample_bits"] == old["logical_sample_bits"]
    assert new["transport_bram36"] == old["transport_bram36"] == 10
    assert new["extra_dsp"] == old["extra_dsp"] == 0
    for key in ("max_source_reads_per_cycle", "max_destination_writes_per_cycle"):
        assert new[key] == old[key]
    with pytest.raises(ValueError, match="mode"):
        model.estimate(released_requests=1)


def test_empty_opt_in_stream_still_consumes_adc_and_does_not_manufacture_request():
    result, records = run([])
    assert result["requests"] == 0 and records == []


def test_source_startup_is_explicit_even_when_candidate_timestamp_is_legal():
    result, [(event, words)] = run(
        [ReleasedCaptureRequest(128, 0, (2,), available_cycle=1700)]
    )
    assert result["startup_drops"] == 1 and words is None
    assert event["status"] == "incomplete_history"


