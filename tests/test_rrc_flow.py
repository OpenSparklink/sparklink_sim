"""RRC continuous-input operand lifetime, finite queues and resource sensitivities."""

from dataclasses import replace

import pytest

from nearlink_sdr.phy.rrc_flow import RRCFlow, RRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat


def test_module_schedule_covers_every_input_and_operands_at16msps_under_explicit_ports():
    f = FixedRRCFormat(12, 18)
    flow = RRCFlow(f)
    estimate = flow.resource_estimate()
    assert estimate["initiation_interval_cycles"] == 12
    assert estimate["capacity_necessary_condition"]
    assert estimate["operators_within_module_budget"]
    assert estimate["isolated_latency_cycles"] == 36
    assert estimate["arithmetic_operations_peak"] == 46
    assert estimate["wide_operations_peak"] == 53
    result = flow.simulate(4096, trace=True)
    outputs = [r for r in result["trace"] if r["event"] == "output"]
    assert [r["sample"] for r in outputs] == list(range(4096))
    assert result["maximum_latency_cycles"] == 36
    assert result["gain_word_reads"] == 4096 * 81
    assert result["fir_word_reads"] == 4096 * 162
    assert result["continuous_input_without_loss"]
    assert result["peak_queues"]["gain_contexts"] == 2
    assert result["peak_queues"]["fir_contexts"] == 2


@pytest.mark.parametrize(
    "change", [dict(clock_hz=160_000_000), dict(arithmetic_ii=2), dict(coefficient_read_ports=1)]
)
def test_insufficient_clock_arithmetic_or_coefficient_ports_preserve_overload(change):
    flow = RRCFlow(FixedRRCFormat(12, 18), replace(RRCFlowAssumptions(), **change))
    assert not flow.resource_estimate()["capacity_necessary_condition"]
    result = flow.simulate(2048)
    assert not result["continuous_input_without_loss"] and result["drop_new"] > 0
    assert result["outputs"] + result["drop_new"] == 2048
    assert all(v <= 4 for v in result["peak_queues"].values())


def test_downstream_blocking_keeps_fifos_bounded_and_marks_history_discontinuities():
    flow = RRCFlow(FixedRRCFormat(12, 18))
    r = flow.simulate(4096, downstream_stalls=[(100, 4100)], trace=True)
    assert r["drop_new"] > 0 and r["invalid_history_outputs"] > 0
    assert r["outputs"] + r["drop_new"] == 4096
    assert r["valid_outputs"] + r["invalid_history_outputs"] == r["outputs"]
    assert any(v["event"] == "history_gap" for v in r["trace"])
    assert all(v <= 4 for v in r["peak_queues"].values())


def test_port_mux_and_ram_copy_costs_are_not_free_or_whole_receiver_acceptance():
    estimate = RRCFlow(FixedRRCFormat(14, 18)).resource_estimate()
    assert estimate["read_mux_bit_inputs"] > 50000
    assert estimate["dedicated_1r1w_ram_copy_bits"] > estimate["register_tap_logical_bits"]
    flow = RRCFlow(FixedRRCFormat(14, 18), replace(RRCFlowAssumptions(), memory_budget_bits=8192))
    assert not flow.resource_estimate()["register_tap_storage_within_budget"]
    assert flow.simulate(0)["outputs"] == 0
    for change in [dict(history_samples=80), dict(fifo_depth=17), dict(arithmetic_ii=True)]:
        with pytest.raises(ValueError):
            RRCFlowAssumptions(**change)
    with pytest.raises(ValueError):
        RRCFlow(FixedRRCFormat(12, 18), replace(RRCFlowAssumptions(), history_tag_read_ports=7))
    with pytest.raises(ValueError):
        RRCFlow(None)
    for count, stalls in [(-1, []), (1.5, []), (0, [(4, 3)])]:
        with pytest.raises(ValueError):
            flow.simulate(count, downstream_stalls=stalls)
