"""Per-stage integer feedback and bounded queues against the independent word API."""

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_fixed import FixedCarrierLoop
from nearlink_sdr.sim.carrier_fixed_flow import (
    CarrierWordEngine,
    carrier_word_cost,
    simulate_carrier_words,
)


def test_per_clock_values_and_feedback_commit_match_word_node():
    engine = CarrierWordEngine("QPSK", 3)
    with pytest.raises(RuntimeError):
        engine.tick()
    with pytest.raises(RuntimeError):
        engine.take()
    engine.begin(137, 311)
    with pytest.raises(RuntimeError):
        engine.begin(1, 1)
    events = [engine.tick() for _ in range(41)]
    assert [e["cycle"] for e in events] == list(range(1, 42))
    assert events[11]["angle"] is not None and events[12]["x"] is not None
    assert events[30]["error"] is None and events[31]["error"] is not None
    assert not any(e["done"] for e in events[:-1]) and events[-1]["done"]
    with pytest.raises(RuntimeError):
        engine.tick()
    actual = engine.take()
    word = FixedCarrierLoop("QPSK", 3)
    expected = word.feed_words([137], [311])
    for key, other in [
        ("real", "real"),
        ("imag", "imag"),
        ("phase", "phase_word"),
        ("frequency", "frequency_word"),
        ("error", "error_word"),
        ("held", "held"),
    ]:
        assert actual[key] == expected[other][0]
    assert engine.node.finish() == word.finish()
    with pytest.raises(RuntimeError):
        engine.begin(1, 1)


def test_fast_stream_fields_stalls_and_all_owned_words_are_bounded():
    rng = np.random.default_rng(730121)
    real = rng.integers(-2000, 2001, 512)
    imag = rng.integers(-2000, 2001, 512)
    report = simulate_carrier_words(real, imag, modulation="8PSK", start=3, header_symbols=64)
    assert report["input_sequence_intact"] and not report["resource_packet_invalid"]
    assert report["max_latency"] == 42 and report["runtime_owned_word_bound"] == 9
    assert report["peak_input"] <= 4 and report["peak_output"] <= 4
    node = FixedCarrierLoop("QPSK", 3)
    parts = []
    for i in range(512):
        if i == 64:
            node.switch_modulation("8PSK")
        parts.append(node.feed_words(real[i : i + 1], imag[i : i + 1]))
    for i, item in enumerate(report["emitted"]):
        assert item["tag"] == i and item["emitted_cycle"] >= item["completed_cycle"]
        for key, other in [
            ("real", "real"),
            ("imag", "imag"),
            ("phase", "phase_word"),
            ("frequency", "frequency_word"),
            ("error", "error_word"),
            ("held", "held"),
        ]:
            assert item["result"][key] == parts[i][other][0]
    assert report["final_state"] == node.finish()
    blocked = simulate_carrier_words(real, imag, stalls=((0, 1000),))
    assert blocked["resource_packet_invalid"] and not blocked["input_sequence_intact"]
    assert len(blocked["emitted"]) == 0 and sorted(x["tag"] for x in blocked["dropped"]) == list(
        range(512)
    )


@pytest.mark.parametrize("clock", [100_000_000, 160_000_000])
def test_sustained_overload_is_packet_invalid_not_silent_throttling(clock):
    r = simulate_carrier_words([1024] * 512, [0] * 512, clock_hz=clock)
    assert r["resource_packet_invalid"] and any(
        x["reason"] == "input_FIFO_full" for x in r["dropped"]
    )
    tags = [x["tag"] for x in r["emitted"]] + [x["tag"] for x in r["dropped"]]
    assert sorted(tags) == list(range(512)) and r["peak_input"] <= 4 and r["peak_output"] <= 4
    assert not r["whole_rx16msps_verified"]


def test_saturation_is_numeric_and_validation_rejects_bad_schedules():
    e = CarrierWordEngine("QPSK", 0)
    e.node.phase = -(2**29)
    e.begin(131071, 131071)
    for _ in range(41):
        e.tick()
    e.take()
    assert e.node.stats["output_saturations"] > 0
    quiet = simulate_carrier_words([0], [0])
    assert quiet["max_latency"] == 42
    for a, b, opts in (
        ([], [], {}),
        ([1], [1, 2], {}),
        ([1], [1], dict(header_symbols=1)),
        ([1], [1], dict(symbol_hz=300_000_000)),
        ([1], [1], dict(stalls=((10, 5),))),
    ):
        with pytest.raises(ValueError):
            simulate_carrier_words(a, b, **opts)
    cost = carrier_word_cost()
    assert cost["logical_bits"] == sum(cost["state"].values())
    assert cost["DSP25x18_tiles"] == 6 and cost["service_clocks"] == 41
    assert all(v == 19 for v in cost["counter_parallel_adders_bits"])
    assert 2 * 131072 < 2**19 and cost["dedicated_ROM_BRAM18_estimate"] == 1
    assert not cost["actual_synthesis_or_timing"] and not cost["full_receiver_verified"]


def test_burst_budget_and_prospective_eight_entry_queue():
    a, b = [1024] * 512, [0] * 512
    four = simulate_carrier_words(a, b, source_batch=4)
    eight = simulate_carrier_words(a, b, source_batch=8)
    deeper = simulate_carrier_words(a, b, source_batch=8, input_depth=8)
    assert four["input_sequence_intact"] and deeper["input_sequence_intact"]
    assert eight["resource_packet_invalid"] and len(eight["dropped"]) > 0
    assert deeper["peak_input"] <= 8 and deeper["max_latency"] == 322
    assert carrier_word_cost(8)["logical_bits"] - carrier_word_cost(4)["logical_bits"] == 224
    with pytest.raises(ValueError):
        carrier_word_cost(16)
    with pytest.raises(ValueError):
        simulate_carrier_words(a, b, source_batch=2)
