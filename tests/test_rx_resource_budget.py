"""Partial allocation must expose overload and never impersonate full RX proof."""

import json
import os
from pathlib import Path

import pytest

from nearlink_sdr.phy.rx_resource_budget import RXResourceBudget, receiver_resource_ledger


def test_present_fine_graph_alone_exceeds_shared_budget_before_tone_or_decoder():
    report = receiver_resource_ledger()
    if directory := os.environ.get("SPARKLINK_TEST_DIAGNOSTICS"):
        path = Path(directory) / "rx-resource-budget.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    d = report["dsp"]
    assert d["fine_products_per_second"] > d["available_primitive_products_per_second"]
    assert not d["known_graph_necessary_condition"]
    assert d["known_graph_necessary_units"] == 36
    assert d["remaining_optimistic_fine_products_per_request"] == 2912
    assert report["storage"]["known_partial_within_budget"]
    assert report["storage"]["remaining_before_unallocated_bits"] == 9968
    assert not report["storage"]["width_reservation_verified_for_tone_and_fine"]
    assert report["wide"]["remaining_before_other_blocks"] == 11
    assert not report["incompatible_legacy_cycle_allocation"][
        "compatible_with_selected_scaled10_two_bank_front"
    ]


def test_even_unlimited_estimated_resources_cannot_accept_missing_schedule():
    report = receiver_resource_ledger(
        RXResourceBudget(dsp25x18=1000, wide_units=1000, storage_bits=1_000_000)
    )
    assert report["dsp"]["known_graph_necessary_condition"]
    assert report["storage"]["known_partial_within_budget"]
    assert not report["full_chain_cost_complete"] and not report["whole_rx_acceptance"]
    assert report["whole_rx_continuous_rate_feasible"] is None
    assert not report["synthesis_evidence"]
    assert report["blocks"]["packet_processing"]["storage_bits"] is None


def test_overallocated_rrc_and_memory_remain_explicit_in_restricted_scenario():
    report = receiver_resource_ledger(RXResourceBudget(dsp25x18=4, wide_units=10, storage_bits=1))
    assert report["dsp"]["remaining_optimistic_fine_products_per_request"] == 0
    assert report["storage"]["remaining_before_unallocated_bits"] < 0
    assert not report["storage"]["known_partial_within_budget"]
    assert report["wide"]["remaining_before_other_blocks"] < 0


@pytest.mark.parametrize("options", [dict(dsp25x18=True), dict(clock_hz=0), dict(input_hz=1e6),
                                     dict(input_hz=300_000_000)])
def test_invalid_resource_assumptions_are_not_rounded_into_feasibility(options):
    with pytest.raises(ValueError):
        RXResourceBudget(**options)


def test_unknown_budget_type_is_rejected():
    with pytest.raises(ValueError):
        receiver_resource_ledger({})
