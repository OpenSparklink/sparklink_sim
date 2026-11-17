"""Queue ownership and numerical output of the isolated eight-phase weight node."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_weight_flow import CarrierWeightFlow
from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights


def test_continuous_weight_source_matches_words_and_eight_cycle_latency():
    words = np.random.default_rng(530121).integers(0, 2**32, 1024).tolist()
    model = CarrierWeightFlow()
    result = model.simulate(words)
    if directory := os.environ.get("SPARKLINK_TEST_DIAGNOSTICS"):
        path = Path(directory) / "carrier-weight-flow.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(estimate=model.estimate(), result=result), indent=2) + "\n")
    node = BoundedCarrierWeights(2, fixed=True)
    reference = np.concatenate([node.advance_words(words[s:s + 32])
                                for s in range(0, len(words), 32)])
    np.testing.assert_array_equal([r["weight"] for r in result["emitted"]], reference)
    assert result["packet_valid"] and result["dropped"] == []
    assert [r["tag"] for r in result["emitted"]] == list(range(len(words)))
    assert {r["latency"] for r in result["emitted"]} == {8}
    assert result["peak_input"] <= 4 and result["peak_output"] <= 4
    assert model.estimate()["continuous_node_necessary_condition"]


def test_blocked_consumer_is_bounded_and_invalidates_packet_on_loss():
    result = CarrierWeightFlow().simulate([65536] * 100, stalls=[(0, 3000)])
    assert not result["packet_valid"] and result["dropped"]
    assert result["peak_input"] == result["peak_output"] == 4
    assert len(result["emitted"]) + len(result["dropped"]) == 100
    assert len({r["tag"] for r in result["emitted"]}) == len(result["emitted"])
    assert result["max_latency"] > 2000


def test_ram_batch_burst_is_not_claimed_to_be_a_continuous_symbol_schedule():
    model = CarrierWeightFlow(symbol_hz=200_000_000)
    result = model.simulate([65536] * 100)
    assert not model.estimate()["continuous_node_necessary_condition"]
    assert result["dropped"] and not result["packet_valid"]
    assert not model.estimate()["whole_system_accepted"]


@pytest.mark.parametrize("arguments", [dict(clock_hz=0), dict(symbol_hz=True),
                                           dict(clock_hz=1, symbol_hz=2)])
def test_invalid_clock_does_not_produce_a_capacity_claim(arguments):
    with pytest.raises(ValueError):
        CarrierWeightFlow(**arguments)


def test_bad_words_stalls_and_exhausted_clock_are_explicit():
    model = CarrierWeightFlow()
    for words in [[], [-1], [2**41], [1.0]]:
        with pytest.raises(ValueError):
            model.simulate(words)
    with pytest.raises(ValueError):
        model.simulate([1], stalls=[(3, 2)])
    with pytest.raises(OverflowError):
        CarrierWeightFlow(clock_hz=2**63, symbol_hz=1).simulate([1, 1])
