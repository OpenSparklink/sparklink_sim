"""Port, loss and value witnesses for the bounded B1 ingress RAM slots."""

import numpy as np
import pytest

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.polar import PolarDecoder
from nearlink_sdr.sim.payload_polar_flow import (
    B1PayloadArrival,
    decode_payload_polar_word,
    payload_polar_calendar,
)
from nearlink_sdr.sim.payload_polar_ingress import replay_payload_polar_ingress


def _work(frames, geometry, lanes=2, fraction=0):
    words = [word for f in frames for block in b1_code_block_plan(
        f.payload_with_crc_bits, f.rate, standard=True,
    ) for word in block.codewords]
    rng = np.random.default_rng(114)
    signal, expected, service, load = [], [], {}, {}
    for n, k in words:
        vals = rng.normal(0, 5, size=n)
        vals[1:4] = 70.1, -82, .25
        signal.extend(vals.tolist())
        expected.append(tuple(PolarDecoder(n, k).decode(
            np.rint(np.clip(vals, -64, 64) * (1 << fraction)),
        )))
        if (n, k) not in service:
            _, r = decode_payload_polar_word(vals, n, k, geometry=geometry,
                                              lanes=lanes, fraction_bits=fraction)
            service[n, k] = r["cycles"]
            load[n, k] = r["input_load_cycles"]
            assert r["input_last_read_cycle"] <= load[n, k] - 4
    return signal, expected, service, load


@pytest.mark.parametrize("geometry,lanes,fraction", [
    ("indexed", 2, 0), ("ordered", 1, 1), ("threshold", 2, 1),
])
def test_early_retirement_uses_same_ram_during_prior_decode(geometry, lanes, fraction):
    frames = [B1PayloadArrival(0, 464, "3/4", 3)]
    signal, expected, service, load = _work(frames, geometry, lanes, fraction)
    events = []
    result = list(replay_payload_polar_ingress(
        frames, iter(signal), word_cycles=service, input_load_cycles=load, geometry=geometry,
        lanes=lanes, fraction_bits=fraction, clock_hz=200_000_000, observer=events.append,
    ))
    assert [r["info_bits"] for r in result] == expected
    assert [r["ingress_slot"] for r in result] == [0, 1, 0]
    assert result[0]["ingress_release"] == result[0]["load_complete"]
    assert result[0]["input_last_read"] < result[0]["ingress_release"] < result[2]["first"]
    assert result[2]["first"] < result[0]["decoded"]
    assert sum(e["reused_during_decode"] for e in events if e["kind"] == "reserve") == 1
    cross = [e for e in events if e["kind"] == "write" and e["edge"] == result[1]["first"]]
    assert {x["slot"] for x in cross} == {0, 1}
    assert sum(e["count"] for e in cross) == 3
    assert sum(e["kind"] == "read" for e in events) == len(signal)


def test_first_beat_can_write_at_exact_retirement_edge():
    first = B1PayloadArrival(0, 336, "3/4", 3)
    _, _, cycles, load = _work([first], "threshold")
    first_result, = payload_polar_calendar([first], word_cycles=cycles, input_load_cycles=load,
                                           clock_hz=200_000_000)
    second = B1PayloadArrival(first_result["ingress_release"], 40, "3/4", 3)
    signal, expected, cycles, load = _work([first, second], "threshold")
    result = list(replay_payload_polar_ingress(
        [first, second], signal, word_cycles=cycles, input_load_cycles=load,
        clock_hz=200_000_000, geometry="threshold",
    ))
    assert [r["info_bits"] for r in result] == expected
    assert [r["ingress_slot"] for r in result] == [0, 0]
    assert result[1]["first"] == result[0]["ingress_release"]
    assert result[1]["first"] < result[0]["decoded"]


def test_failed_suffix_does_not_cancel_inflight_word_and_resets_at_next_frame():
    frames = [B1PayloadArrival(0, 640, "3/4", 3), B1PayloadArrival(150_000, 40, "3/4", 3)]
    signal, expected, service, _ = _work(frames, "threshold")
    events = []
    results = list(replay_payload_polar_ingress(
        frames, signal, word_cycles=service, geometry="threshold", clock_hz=200_000_000,
        stall_until=100_000, discard_failed_suffix=True, observer=events.append,
    ))
    assert [x["info_bits"] for x in results] == [expected[0], expected[1], expected[-1]]
    assert [x["frame"] for x in results] == [0, 0, 1]
    assert results[0]["output_start"] == 100_000
    assert [x["kind"] for x in events if x["kind"] in ("resource_drop", "incomplete_suffix")] == [
        "resource_drop", "incomplete_suffix",
    ]
    drop, = [x for x in events if x["kind"] == "resource_drop"]
    assert drop["first_frame_loss"]
    assert not any(x["kind"]=="write" and x["frame"]==0 and x["edge"]>=drop["edge"]
                   for x in events if x["word"]>1)


def test_ingress_detects_size_values_and_forged_service_metadata():
    frames = [B1PayloadArrival(0, 40, "3/4", 3)]
    signal, _, service, load = _work(frames, "indexed")
    kwargs = dict(word_cycles=service, clock_hz=200_000_000)
    events = []
    with pytest.raises(ValueError, match="Truncated"):
        list(replay_payload_polar_ingress(frames, signal[:-1], observer=events.append, **kwargs))
    assert events[-1]["kind"] == "truncated"
    with pytest.raises(ValueError, match="Trailing"):
        list(replay_payload_polar_ingress(frames, [*signal, None], **kwargs))
    with pytest.raises(ValueError, match="finite"):
        list(replay_payload_polar_ingress(frames, [float("nan"), *signal[1:]], **kwargs))
    n_k = (64, 44)
    with pytest.raises(ValueError, match="retirement disagrees"):
        list(replay_payload_polar_ingress(frames, signal,
                                          input_load_cycles={n_k: load[n_k] + 1}, **kwargs))
    with pytest.raises(RuntimeError, match="ownership retired"):
        list(replay_payload_polar_ingress(frames, signal, input_load_cycles={n_k: 1}, **kwargs))
    with pytest.raises(ValueError, match="Decode cycles disagree"):
        list(replay_payload_polar_ingress(frames, signal, clock_hz=200_000_000,
                                          word_cycles={n_k: service[n_k] + 1}))
    with pytest.raises(ValueError, match="within measured"):
        list(replay_payload_polar_ingress(frames, signal, input_load_cycles={n_k: 0}, **kwargs))
