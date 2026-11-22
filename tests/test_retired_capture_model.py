"""Frozen gate adoption rejects numerical,resource and original-success regressions."""

import json
import sys

import pytest
from scripts import model_retired_capture as model


def rows():
    data = []
    for j in range(10):
        arm = dict(
            complete=[0] if j < 6 else [],
            captured_packet_associations_diagnostic=[0] if j < 6 else [],
            false_MAC=0,
            duplicate_MAC=0,
            fine_cycles=100,
            check_cycles=0,
            fine_calls=3,
            pre_fine_retired=0,
            queue_drops=2,
            history_drops=0,
            coherent=int(j < 6),
            head_CRC=int(j < 6),
            data_CRC=int(j < 6),
            BER_errors=0,
            BER_bits=184 if j < 6 else 0,
            source_lossless=True,
            all_pulses_accounted=True,
            max_ports=2,
            max_owned=3,
            known_drain_cycle=1000,
            complete_latency_cycles=None,
        )
        new = arm | dict(fine_cycles=40, check_cycles=3, pre_fine_retired=2, fine_calls=1)
        data.append(
            dict(
                id=j,
                input_SHA256=f"{j + 1:064x}",
                negative=j >= 6,
                samples=8192,
                old=arm,
                new=new,
                original_success_losses=[],
                recoveries=[],
                same_window_fine_integer_exact=True,
                common_fine_windows=1,
                source_descriptors_exact=True,
                original_trajectory_comparisons=[]
                if j >= 6
                else [dict(packet=0, start_abs=0.0, score_abs=0.0, trajectory_max=0.0)],
                registered_frontier_stable=True,
                all_requests_accounted=True,
                all_words_and_chunks_exact=True,
                additional_control_bits=66,
                new_IQ_RAM_DSP_or_ports=0,
                whole_RX_accepted=False,
            )
        )
    return data


def test_policy_JSON_roundtrip_keeps_existing_budgets_and_no_whole_receiver_claim():
    assert json.loads(json.dumps(model.POLICY)) == model.POLICY
    assert model.POLICY["start_samples_budget"] == 0.125
    assert model.POLICY["trajectory_samples_budget"] == 0.25
    assert model.POLICY["score_abs_budget"] == 0.005
    assert model.POLICY["new_default"] is model.POLICY["full_cycle_or_fixed_or16MSps"] is False


@pytest.mark.parametrize("captured", [None, [], [True], [0, 0], [2], [0, 1]])
def test_packet_funnel_cannot_omit_confirmed_delivery_or_fabricate_capture(captured):
    data = rows()
    assert model.qualify(data)
    data[0]["new"]["captured_packet_associations_diagnostic"] = captured
    assert not model.qualify(data)


@pytest.mark.parametrize(
    "edit",
    [
        dict(original_success_losses=[0]),
        dict(same_window_fine_integer_exact=False),
        dict(source_descriptors_exact=False),
        dict(registered_frontier_stable=False),
        dict(all_requests_accounted=False),
        dict(all_words_and_chunks_exact=False),
        dict(additional_control_bits=129),
        dict(additional_control_bits=-1),
        dict(additional_control_bits=True),
        dict(new_IQ_RAM_DSP_or_ports=1),
        dict(whole_RX_accepted=True),
        dict(negative=True),
        dict(samples=0),
        dict(samples=True),
    ],
)
def test_invalid_receiver_evidence_cannot_become_a_passing_gate_report(edit):
    data = rows()
    assert model.qualify(data)
    data[0].update(edit)
    assert not model.qualify(data)


@pytest.mark.parametrize(
    "edit",
    [
        dict(false_MAC=1),
        dict(duplicate_MAC=1),
        dict(fine_cycles=-1),
        dict(check_cycles=True),
        dict(source_lossless=False),
        dict(source_lossless=1),
        dict(all_pulses_accounted=False),
        dict(max_ports=3),
        dict(max_owned=4),
        dict(complete_latency_cycles=0),
        dict(BER_errors=-1),
        dict(data_CRC=2),
    ],
)
def test_false_delivery_or_unbounded_costs_and_invalid_funnel_are_rejected(edit):
    data = rows()
    data[0]["new"].update(edit)
    assert not model.qualify(data)


@pytest.mark.parametrize(
    "edit",
    [
        dict(start_abs=0.125001),
        dict(score_abs=0.005001),
        dict(trajectory_max=0.250001),
        dict(trajectory_max=None),
        dict(start_abs=float("nan")),
        dict(score_abs=float("inf")),
    ],
)
def test_existing_frozen_error_budgets_are_not_relaxed_for_saved_fine_work(edit):
    data = rows()
    data[0]["original_trajectory_comparisons"][0].update(edit)
    assert not model.qualify(data)


def test_saved_cycles_do_not_hide_negative_confirmation_or_request_losses():
    data = rows()
    data[7]["new"]["coherent"] = 1
    assert not model.qualify(data)
    data = rows()
    data[0]["original_trajectory_comparisons"] = []
    assert not model.qualify(data)
    data = rows()
    data[0]["input_SHA256"] = data[1]["input_SHA256"]
    assert not model.qualify(data)
    assert not model.qualify(rows()[:-1])
    data = rows()
    for r in data:
        r["new"]["fine_cycles"] = 100
    assert not model.qualify(data)
    data = rows()
    data[0]["new"]["queue_drops"] = 3
    assert not model.qualify(data)
    data = rows()
    data[0]["new"]["history_drops"] = 1
    assert not model.qualify(data)


def test_validation_needs_frozen_eligible_development_before_any_new_inputs(tmp_path, monkeypatch):
    out = tmp_path / "not-created"
    monkeypatch.setattr(
        sys, "argv", ["model_retired_capture", "--split", "validation", "--output", str(out)]
    )
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not out.exists()
