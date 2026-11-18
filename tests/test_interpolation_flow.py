"""Actual words traverse seven finite stages;blocked replies cannot be overwritten."""

from itertools import pairwise

import numpy as np
import pytest

from nearlink_sdr.phy.interpolation_word import interpolate_filter_word
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.interpolation_flow import LagrangeInterpolationFlow
from tests.test_interpolation_word import FORMATS


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("blocked", [False, True])
def test_pipeline_matches_integer_nodes_and_holds_only_seven_contexts(fmt, blocked):
    rng = np.random.default_rng(930333)
    inputs = [
        (
            tuple(
                pack_filter_word(int(r), int(i), int(e))
                for r, i, e in zip(
                    rng.integers(-(2**18), 2**18, 4),
                    rng.integers(-(2**18), 2**18, 4),
                    rng.integers(-100, 50, 4),
                    strict=True,
                )
            ),
            int(rng.integers(0, 2**fmt.phase_bits + 1)),
        )
        for _ in range(41)
    ]
    flow = LagrangeInterpolationFlow(fmt)
    expected = [interpolate_filter_word(w, p, fmt) for w, p in inputs]
    offered, outputs = 0, []
    for cycle in range(2000):
        if not (blocked and 35 <= cycle < 230):
            row = flow.take()
            if row is not None:
                assert row["tag"] == len(outputs)
                assert (row["word"], row["info"]) == expected[len(outputs)]
                outputs.append(row)
        if offered < len(inputs) and flow.issue(*inputs[offered], tag=offered):
            offered += 1
        flow.advance()
        assert flow.live <= 7
        if len(outputs) == len(inputs):
            break
    assert len(outputs) == 41 and flow.stats["peak_live"] <= 7
    assert flow.take() is None
    cost = flow.estimate()
    if not blocked:
        assert outputs[0]["emitted_cycle"] == cost["no_stall_latency"]
        assert all(
            b["accepted_cycle"] - a["accepted_cycle"] <= cost["no_stall_II"]
            for a, b in pairwise(outputs)
        )
    else:
        assert flow.stats["backpressure"] > 0
    assert not cost["whole_receiver16msps"] and not cost["synthesis_or_timing"]


def test_input_rejection_and_clock_counter_bounds():
    with pytest.raises(ValueError):
        LagrangeInterpolationFlow(None)
    flow = LagrangeInterpolationFlow(FORMATS[0])
    with pytest.raises(ValueError):
        flow.issue([0] * 4, 0, 65536)
    flow.cycle = 2**63 - 1
    with pytest.raises(OverflowError):
        flow.advance()
    flow.stats["accepted"] = 2**64 - 1
    with pytest.raises(OverflowError):
        flow.issue([0] * 4, 0)
