"""Independent lifetime/port/finite queue checks for the native product dataflow."""

import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions


def retain(name, model, result):
    """Caller-owned CI traces, not hardware state or an alternate pass rule."""
    directory = os.environ.get("SPARKLINK_TEST_DIAGNOSTICS")
    if directory:
        path = Path(directory) / "rrc-flow"
        path.mkdir(parents=True, exist_ok=True)
        (path / (name + ".json")).write_text(
            json.dumps(dict(assumptions=asdict(model.a), result=result), indent=2) + "\n"
        )


def test_continuous_source_all_products_and_registered_output_no_loss():
    model = PairRRCFlow()
    result = model.simulate(4096, trace=True)
    retain("base", model, result)
    assert result["continuous_input_without_loss"]
    assert result["emitted"] == 4096 and result["ownership_errors"] == 0
    assert result["history_reads"] == 4096 * 81
    assert result["source_history_writes"] == 4096
    assert result["product_reads"] == result["product_writes"] == 4096 * 41
    outputs = [r for r in result["trace"] if r["event"] == "output"]
    assert [r["sample"] for r in outputs] == list(range(4096))
    assert all(r["cycle"] % 12 == 0 for r in outputs)
    assert result["max_latency"] == 60 and outputs[0]["cycle"] == 60
    assert result["max_merge_wide_ops"] == 8  # phase14 + phase1 at13-cycle separation.
    assert result["peak"]["owners"] == 3
    assert (
        result["max_history_reads_per_cycle"],
        result["max_product_reads_per_cycle"],
        result["max_product_writes_per_cycle"],
    ) == (8, 4, 4)
    starts = {r["sample"]: r for r in result["trace"] if r["event"] == "sum_start"}
    for r in result["trace"]:
        if r["event"] == "product_release":
            # 41complex products,4bank reads/cycle; final synchronous read
            # completes11cycles after first read, never at the earlier10th.
            assert r["cycle"] == starts[r["sample"]]["cycle"] + 11


@pytest.mark.parametrize(
    "changes",
    [
        dict(clock_hz=160_000_000),
        dict(arithmetic_ii=2),
        dict(coefficient_ports=1),
        dict(product_read_ports=1),
        dict(product_buffers=2),
        dict(consumer_ii=14),
    ],
)
def test_slow_cases_lose_outputs_without_overwriting_owned_products(changes):
    model = PairRRCFlow(replace(PairRRCFlowAssumptions(), **changes))
    result = model.simulate(4096, trace=True)
    retain(next(iter(changes)), model, result)
    assert not result["continuous_input_without_loss"] and result["dropped"] > 0
    assert result["emitted"] + result["dropped"] == 4096
    assert result["ownership_errors"] == 0
    assert not model.resource_estimate()["capacity_necessary_condition"]
    assert result["peak"]["input"] <= model.a.fifo_depth
    assert result["peak"]["owners"] <= model.a.product_buffers
    assert max(result["peak"][k] for k in ["fir", "sums", "merges"]) <= 2


def test_backpressure_keeps_gaps_and_source_indices_and_drains_at_eof():
    model = PairRRCFlow()
    result = model.simulate(4096, downstream_stalls=[(100, 4100)], trace=True)
    retain("blocked", model, result)
    assert result["dropped"] > 0 and result["invalid"] > 0
    assert result["ownership_errors"] == 0 and result["peak"]["output"] == 4
    assert result["peak"]["ready"] <= 3
    bad = {r["sample"] for r in result["trace"] if r["event"] == "history_gap"}
    out = [r for r in result["trace"] if r["event"] == "output"]
    assert {r["sample"] for r in out if r["invalid"]} == bad
    assert [r["sample"] for r in out] == sorted(r["sample"] for r in out)
    assert len({r["sample"] for r in out}) == len(out)
    assert len(out) + result["dropped"] == 4096


def test_all_owned_data_padding_and_port_costs_are_charged_separately():
    result = PairRRCFlow().resource_estimate()
    state = result["storage_components"]
    assert state["owned_padded_products"] == 3 * 4 * 11 * (2 * 31 + 7)
    assert state["sum_context_partials"] == 2 * 4 * 2 * (37 + 6)
    assert state["output_fifo"] == 4 * 110
    assert result["register_tap_logical_bits"] == sum(state.values())
    assert result["dedicated_raw_1r1w_ram_copy_bits"] - sum(state.values()) == 7 * 128 * 32
    assert result["wide_units_total"] == 43 and result["small_units_total"] == 64
    assert result["leading_detectors_32bit"] == 12
    assert result["capacity_necessary_condition"] and not result["whole_rx_acceptance"]
    assert not result["monitoring_updates_scheduled"]


def test_empty_input_and_reject_unmodeled_geometry_or_unbounded_stalls():
    assert PairRRCFlow().simulate(0)["cycles"] == 0
    for changes in [
        dict(complex_lanes=3),
        dict(history_samples=81),
        dict(product_buffers=1),
        dict(history_read_ports=1),
        dict(arithmetic_ii=0),
        dict(consumer_ii=256),
    ]:
        with pytest.raises(ValueError):
            PairRRCFlowAssumptions(**changes)
    with pytest.raises(ValueError):
        PairRRCFlow(None).simulate(-1)
    with pytest.raises(ValueError):
        PairRRCFlow().simulate(1, downstream_stalls=[(20, 10)])
