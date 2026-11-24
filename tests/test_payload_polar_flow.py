"""Rate layout, finite arithmetic and admission tests for the soft B1 engine."""

import numpy as np
import pytest

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.polar import PolarDecoder, PolarEncoder
from nearlink_sdr.phy.control_polar_word import ssc_geometry
from nearlink_sdr.sim.payload_polar_flow import (
    B1PayloadArrival,
    decode_payload_polar_word,
    payload_polar_calendar,
    payload_polar_cost,
)


@pytest.mark.parametrize("n,k", [(64, 44), (128, 88), (256, 144), (512, 96), (1024, 896)])
@pytest.mark.parametrize("fraction_bits", [0, 1])
def test_ssc_values_ports_and_lane_invariance(n, k, fraction_bits):
    rng = np.random.default_rng(n + 20 * k)
    info = rng.integers(0, 2, size=k, dtype=np.int8)
    coded = PolarEncoder(n, k).encode(info)
    llr = (1 - 2 * coded) * 4.5 + rng.normal(0, 5, size=n)
    llr[0:4] = [79, -70, 0.25, -0.25]
    llr[4] = 2.5
    scale = 1 << fraction_bits
    expected = PolarDecoder(n, k).decode(np.rint(np.clip(llr, -64, 64) * scale))
    one, serial = decode_payload_polar_word(llr, n, k, lanes=1,
                                            fraction_bits=fraction_bits)
    two, parallel = decode_payload_polar_word(llr, n, k, lanes=2,
                                              fraction_bits=fraction_bits)
    assert np.array_equal(one, expected)
    assert one == two
    assert serial["cycles"] > parallel["cycles"]
    assert serial["operations"] == parallel["operations"]
    assert parallel["clipped_inputs"] == 2
    assert parallel["quantized_zeros"] >= 2
    assert parallel["signed_alpha_bits"] >= parallel["peak_abs_alpha"].bit_length() + 1
    assert max(parallel["peak_ports"].values()) <= 2
    assert sum(parallel["reads"].values()) > n
    assert sum(parallel["writes"].values()) > n
    assert parallel["blocks"] > 0


def test_all_zero_and_llr_accumulator_edge():
    n, k = 1024, 896
    zeros, summary = decode_payload_polar_word(np.zeros(n), n, k)
    maximum, full = decode_payload_polar_word(np.full(n, 64), n, k, fraction_bits=1)
    assert not any(zeros) and not any(maximum)
    assert summary["quantized_zeros"] == n
    assert 128 <= full["peak_abs_alpha"] <= 131072
    assert full["signed_alpha_bits"] == 19
    assert summary["cycles"] == full["cycles"]


@pytest.mark.parametrize(
    "values,n,k,lanes,fraction_bits",
    [
        ([0.0] * 63, 64, 44, 2, 0), ([0.0] * 65, 64, 44, 2, 0),
        ([float("nan")] * 64, 64, 44, 2, 0), ([0.0] * 64, 64, 43, 2, 0),
        ([0.0] * 64, 64, 44, 0, 0), ([0.0] * 64, 64, 44, 2, True),
    ],
)
def test_invalid_word_inputs(values, n, k, lanes, fraction_bits):
    with pytest.raises(ValueError):
        decode_payload_polar_word(values, n, k, lanes=lanes, fraction_bits=fraction_bits)


def test_multiword_symbol_crossing_and_backpressure():
    frame = B1PayloadArrival(0, 464, "3/4", 3)
    words = [word for block in b1_code_block_plan(464, "3/4", standard=True)
             for word in block.codewords]
    assert words == [(512, 352), (128, 88), (64, 44)]
    # Use actual service counts, preserving the two writes into different
    # reservations when a three-LLR beat straddles a word boundary.
    cycles = {word: decode_payload_polar_word([1] * word[0], *word)[1]["cycles"]
              for word in words}
    running = list(payload_polar_calendar([frame], word_cycles=cycles, clock_hz=200_000_000))
    stalled = list(payload_polar_calendar(
        [frame], word_cycles=cycles, clock_hz=200_000_000, stall_until=100_000,
    ))
    assert running[0]["last"] == running[1]["first"]
    assert running[1]["last"] == running[2]["first"]
    assert not running[0]["lost"] and not running[1]["lost"]
    assert running[2]["lost"] and running[2]["peak_ingress"] == 2
    assert stalled[0]["output_start"] == 100_000
    assert stalled[0]["taken"] == 100_176
    assert stalled[1]["launch"] == 100_177
    assert stalled[2]["lost"]


@pytest.mark.parametrize("clock", [100_000_000, 160_000_000, 200_000_000])
def test_released_slot_reused_across_packets(clock):
    frames = [B1PayloadArrival(0, 40, "3/4", 3), B1PayloadArrival(5000, 40, "3/4", 3)]
    cycles = {(64, 44): decode_payload_polar_word([20] * 64, 64, 44)[1]["cycles"]}
    records = list(payload_polar_calendar(frames, word_cycles=cycles, clock_hz=clock))
    assert [row["frame"] for row in records] == [0, 1]
    assert all(row["ingress_slot"] == 0 and not row["lost"] for row in records)
    assert all(row["queue_wait"] == 0 for row in records)


def test_calendar_precondition_gates():
    frame = B1PayloadArrival(0, 40, "3/4", 3)
    with pytest.raises(ValueError, match="non-overlapping"):
        list(payload_polar_calendar([frame, frame], word_cycles={(64, 44): 100},
                                   clock_hz=100_000_000))
    for invalid in (B1PayloadArrival(0, 0, "3/4", 3), B1PayloadArrival(0, 32, "3/4", 1)):
        with pytest.raises(ValueError, match="legal-range"):
            list(payload_polar_calendar([invalid], word_cycles={}, clock_hz=100_000_000))
    for kwargs in ({"clock_hz": 0}, {"clock_hz": 100_000_000, "stall_until": -1}):
        with pytest.raises(ValueError):
            list(payload_polar_calendar([frame], word_cycles={}, **kwargs))
    with pytest.raises(ValueError, match="executed"):
        list(payload_polar_calendar([frame], word_cycles={(64, 44): 0}, clock_hz=100_000_000))


def test_all_resident_roms_and_three_way_ingress_accounted():
    q8 = payload_polar_cost()
    q9 = payload_polar_cost(fraction_bits=1)
    assert q8["shapes"] == q9["shapes"] == 36
    assert q8["DSP"] == q9["DSP"] == 0
    assert q8["alpha_bits"] == 18 < q9["alpha_bits"]
    assert q9["RAMB18_equivalents"] - q8["RAMB18_equivalents"] == 4
    assert sum(name.startswith("('X'") for name in q8["physical_banks"]) == 6
    assert q8["physical_banks"]["('T', 0)"]["depth"] > 8192
    assert q8["physical_banks"]["('L', 0, 0)"]["ports"] == 2
    assert q8["synthesized"] is False and q8["verified_clock_hz"] is None
    with pytest.raises(ValueError):
        ssc_geometry(200, 100)
