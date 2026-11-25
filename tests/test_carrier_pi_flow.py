"""Registered PI scheduling and exact values against the original B2 node."""

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_fixed import _FREQUENCY_BOUNDS
from nearlink_sdr.sim.carrier_fixed_flow import CarrierWordEngine, carrier_word_cost
from nearlink_sdr.sim.carrier_pi_flow import (
    ParallelPICarrierEngine,
    parallel_pi_carrier_cost,
    simulate_parallel_pi_carrier,
)


@pytest.mark.parametrize("modulation,order", [("BPSK", 2), ("QPSK", 4), ("8PSK", 8)])
@pytest.mark.parametrize("fraction_bits", [10, 12])
def test_parallel_pi_retires_faster_with_identical_prefix_and_feedback(
    modulation, order, fraction_bits,
):
    old = CarrierWordEngine(modulation, 127, fraction_bits=fraction_bits)
    fast = ParallelPICarrierEngine(modulation, 127, fraction_bits=fraction_bits)
    for engine in (old, fast):
        engine.node.phase = -(2**31)
        engine.node.frequency = -_FREQUENCY_BOUNDS[order]
    with pytest.raises(RuntimeError, match="active unfinished"):
        fast.tick()

    for i, q in [(0, 0), (-131072, 131071), (131071, 131071), (712, 221),
                 (1, 0), (-99, -2230), (313, -1544), (-1802, 817)]:
        previous = fast.node.used
        old.begin(i, q)
        fast.begin(i, q)
        for _ in range(33):
            assert old.tick() == fast.tick()
        for _ in range(8):
            old.tick()
        events = [fast.tick() for _ in range(5)]
        assert fast.node.used == previous + 1
        assert [e["cycle"] for e in events] == list(range(34, 39))
        assert not any(e["done"] for e in events[:-1]) and events[-1]["done"]
        with pytest.raises(RuntimeError, match="active unfinished"):
            fast.tick()
        assert old.take() == fast.take()
        assert (old.node.phase, old.node.frequency) == (fast.node.phase, fast.node.frequency)
        assert old.node.stats == fast.node.stats


@pytest.mark.parametrize(
    "clock,batch,depth,stalls,intact",
    [
        (100_000_000, 1, 4, (), False),
        (160_000_000, 1, 4, (), True),
        (160_000_000, 4, 4, (), True),
        (160_000_000, 8, 4, (), False),
        (160_000_000, 8, 8, (), True),
        (160_000_000, 1, 4, ((0, 1000),), False),
    ],
)
def test_bounded_source_and_stall_policy(clock, batch, depth, stalls, intact):
    rng = np.random.default_rng(46508)
    i, q = rng.integers(-1600, 1601, (2, 512))
    r = simulate_parallel_pi_carrier(
        i, q, clock_hz=clock, source_batch=batch, input_depth=depth, stalls=stalls,
        header_symbols=64, header_modulation="BPSK", modulation="8PSK",
    )
    assert r["input_sequence_intact"] is intact
    assert r["resource_packet_invalid"] is (not intact)
    assert r["peak_input"] <= depth and r["peak_output"] <= 4
    assert r["runtime_owned_word_bound"] == depth + 5
    assert not r["whole_rx16msps_verified"]
    tags = [e["tag"] for e in r["emitted"] + r["dropped"]]
    assert sorted(tags) == list(range(512))
    if intact and batch == 1:
        assert r["max_latency"] == 39


def test_pi_resources_reserve_parallel_carry_without_memory_or_dsp_growth():
    nominal = carrier_word_cost()
    pi = parallel_pi_carrier_cost()
    assert pi["service_clocks"] == 38 and pi["selected_output_latency_cycles"] == 39
    assert pi["logical_bits"] == nominal["logical_bits"] + 34
    assert pi["PI_parallel_RNE_units"] == 2 and pi["PI_extra_adder_bits"] == 34
    assert pi["additional_DSP"] == pi["additional_RAMB18"] == pi["additional_memory_ports"] == 0
    assert not pi["actual_synthesis_or_timing"]
