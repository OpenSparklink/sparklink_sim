"""Independent absolute-word oracle, finite leases, stalls and raw ownership."""

from itertools import pairwise

import pytest

from nearlink_sdr.sim.capture_tail import CaptureTailReplay, TailReplayAssumptions
from nearlink_sdr.sim.capture_transport import (
    CaptureRequest,
    CaptureTransport,
    CaptureTransportAssumptions,
    FineService,
)
from tests.test_capture_fine import CausalSource, word_at


@pytest.mark.parametrize("ii", [1, 4, 8])
def test_real_raw_slot_tail_relay_values_and_one_credit_backpressure(ii):
    source, emitted, events = CausalSource(1537), [], []
    a = CaptureTransportAssumptions(history=384)
    tail = CaptureTailReplay(TailReplayAssumptions(4096, 3, 2, ii), sink=emitted.append)
    requests = [CaptureRequest(t, j % 2, (2,)) for j, t in enumerate((256, 288, 320, 800))]
    result = CaptureTransport(a).simulate(
        source,
        requests,
        tail_replay=tail,
        fine_worker=lambda e, w: FineService(500, "confirm_pending", 16),
        stalls=((1400, 1473), (2100, 2200)),
        sink=lambda e, w: events.append(e),
    )
    assert source.seen == result["input_samples"] == len(source)
    assert not result["adc_backpressure"] and result["fine_and_replay_serialized"]
    assert result["completed"] + result["queue_drops"] + result["tail_drops"] == len(requests)
    assert not result["tail_drops"] and not any(tail.leases)
    for event in events:
        words = [x for x in emitted if x["id"] == event["id"]]
        if event["status"] != "completed":
            assert not words
            continue
        end = event["tail"]["next_live_index"]
        assert event["tail"]["replay_done"] and not event["tail"]["tail_overflow"]
        assert [x["index"] for x in words] == list(range(event["after_samples"] - 256, end))
        assert [x["value"] for x in words] == [
            word_at(k, event["path"]) for k in range(event["after_samples"] - 256, end)
        ]
        assert all(b["cycle"] - a["cycle"] >= ii for a, b in pairwise(words))
        assert not event["complete_frame_or_MAC_verified"]
    assert result["tail_cost"]["tail_bram36"] == 24
    assert result["tail_cost"]["register_bits"] <= 4096
    assert result["raw_snapshot_held_through_replay"]
    assert result["source_ring_check_excludes_tail"]
    assert not result["retention_is_necessary_only"]
    assert not any("retention" in k for k in result["worker_results"])
    assert all(e["retention_necessary_pass"] is None for e in events)


@pytest.mark.parametrize("blocked", [False, True])
def test_fine_overflow_sticky_no_early_cancel_credit_or_CRC_success(blocked):
    source, outputs, events = CausalSource(2048), [], []
    tail = CaptureTailReplay(TailReplayAssumptions(16), sink=outputs.append)
    result = CaptureTransport().simulate(
        source,
        [CaptureRequest(256, 0, (2,))],
        tail_replay=tail,
        fine_worker=lambda e, w: FineService(5000, "ready", 8),
        stalls=((0, 10000),) if blocked else (),
        sink=lambda e, w: events.append(e),
    )
    assert source.seen == 2048 and result["completed"] == 0 and result["tail_drops"] == 1
    assert not outputs and result["tail_statistics"]["overflow_leases"] == 1
    e = events[0]
    assert e["status"] == "tail_capacity_exceeded" and e["tail"]["overflow_index"] == 272
    assert e["cycle"] >= e["fine_started_cycle"] + e["fine_cycles"]
    assert e["tail"]["next_live_index"] is None and not e["complete_frame_or_MAC_verified"]


def test_registered_read_same_address_deferred_and_reused_slot_never_reads_old_epoch():
    out = []
    tail = CaptureTailReplay(TailReplayAssumptions(8, 1, 1, 1), sink=out.append)
    tail.input(0, [word_at(0, 0)], 0)
    tail.admit(0, 0, 1, 0)
    tail.begin(0, [word_at(0, 0)], 0)
    for cycle in range(8):
        if 1 <= cycle < 4:
            tail.input(cycle, [word_at(cycle, 0)], cycle)
        tail.tick(cycle)
        if tail.done(0):
            break
    assert tail.report(0)["next_live_index"] == 4
    assert [x["value"] for x in out] == [word_at(j, 0) for j in range(4)]
    assert tail.stats["same_address_deferrals"] == 1
    assert out[0]["cycle"] == 1  # No same-cycle RAM response.
    tail.input(4, [word_at(4, 0)], cycle + 1)
    tail.tick(cycle + 1)
    assert tail.report(0)["next_live_index"] == 4  # Stable while result is held.
    tail.close(0)
    tail.admit(0, 1, 5, 0)
    tail.begin(0, [word_at(4, 0)], cycle + 2)
    for c in range(cycle + 2, cycle + 6):
        tail.tick(c)
        if tail.done(0):
            break
    assert [x["index"] for x in out if x["id"] == 1] == [4]
    tail.close(0)
    assert not any(tail.leases) and tail.pending is None


def test_backpressure_overflows_during_replay_and_preserves_partial_output_as_failure():
    events, words = [], []
    tail = CaptureTailReplay(TailReplayAssumptions(512, consumer_ii=4), sink=words.append)
    r = CaptureTransport().simulate(
        CausalSource(3000),
        [CaptureRequest(256, 0, (2,))],
        tail_replay=tail,
        fine_worker=lambda e, w: FineService(50, "ready", 16),
        stalls=((3700, 20000),),
        sink=lambda e, w: events.append(e),
    )
    assert words and r["tail_drops"] == 1 and r["completed"] == 0
    assert all(x["cycle"] < 3700 for x in words)
    assert events[0]["tail"]["tail_overflow"] and not events[0]["tail"]["replay_done"]


def test_nonconfirming_fine_releases_without_replay_even_if_tail_full():
    words, events = [], []
    tail = CaptureTailReplay(TailReplayAssumptions(8), sink=words.append)
    r = CaptureTransport().simulate(
        CausalSource(1000),
        [CaptureRequest(256, 0, (2,))],
        tail_replay=tail,
        fine_worker=lambda e, w: FineService(500, "geometry_wait"),
        sink=lambda e, w: events.append(e),
    )
    assert r["completed"] == 1 and not r["tail_drops"] and not words
    assert events[0]["tail"]["tail_overflow"] and not any(tail.leases)


def test_bad_configuration_clock_and_word_contracts_fail_explicitly():
    for kw in (
        dict(capacity=7),
        dict(capacity=12),
        dict(capacity=8192),
        dict(owned=17),
        dict(paths=True),
        dict(consumer_ii=0),
    ):
        with pytest.raises(ValueError):
            TailReplayAssumptions(**kw)
    with pytest.raises(ValueError):
        CaptureTailReplay(True)
    with pytest.raises(ValueError):
        CaptureTailReplay(sink=True)
    t = CaptureTailReplay()
    for index, row, cycle in (
        (1, [0, 0], 0),
        (0, [True, 0], 0),
        (0, [1.1, 0], 0),
        (0, [2**46, 0], 0),
        (0, [0], 0),
    ):
        with pytest.raises(ValueError):
            t.input(index, row, cycle)
    t.input(0, [0, 1], 0)
    with pytest.raises(ValueError):
        t.input(1, [0, 0], 0)
    with pytest.raises(ValueError):
        t.admit(0, 0, 2, 0)
    t.admit(0, 0, 1, 0)
    with pytest.raises(ValueError):
        t.admit(0, 1, 1, 0)
    with pytest.raises(ValueError):
        t.begin(1, [0], 0)
    with pytest.raises(ValueError):
        t.begin(0, [], 0)
    t.begin(0, [2**46], 0)
    with pytest.raises(ValueError):
        t.tick(0)
    t.close(0)
    with pytest.raises(ValueError):
        t.tick(0)
    for options in (
        dict(tail_replay=True),
        dict(tail_replay=CaptureTailReplay()),
        dict(tail_replay=t, fine_worker=lambda e, w: FineService(1, "no_candidate")),
    ):
        with pytest.raises(ValueError):
            CaptureTransport().simulate(CausalSource(256), [], **options)
