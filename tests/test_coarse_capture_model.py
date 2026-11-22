"""Strict frozen-policy representation and honest finite-calendar adoption."""

import json
import sys

import pytest
from scripts import model_coarse_capture as model


def rows():
    return [dict(id=k, input_SHA256=f"{k + 1:064x}", frame_type=(2, 3, 4)[k % 3],
                 actual_fine_windows=3, old_new_coarse_decisions_exact=True,
                 all_actual_source_windows_exact=True, all_pulses_and_outcomes_accounted=True,
                 source_fault=0, additional_register_bits=1517,
                 extra_IQ_RAM_DSP_or_sample_ports=0, requests=4,
                 completed_fine=3, resource_queue_drops=1) for k in range(6)]


def test_frozen_policy_roundtrips_through_actual_JSON_without_key_type_changes():
    assert json.loads(json.dumps(model.POLICY)) == model.POLICY
    assert model.POLICY["lanes_by_FT"] == {"2": 8, "3": 8, "4": 16}


@pytest.mark.parametrize("edit", [
    dict(source_fault=1), dict(additional_register_bits=2049),
    dict(additional_register_bits=-1), dict(extra_IQ_RAM_DSP_or_sample_ports=1),
    dict(all_actual_source_windows_exact=False), dict(old_new_coarse_decisions_exact=False),
    dict(requests=-1, completed_fine=-1, resource_queue_drops=0), dict(resource_queue_drops=0),
])
def test_reject_context_corruption_unaccounted_results_and_unapproved_resources(edit):
    data = rows()
    assert model.qualify(data)
    data[1].update(edit)
    assert not model.qualify(data)


def test_missing_frame_actual_work_or_duplicate_inputs_cannot_qualify():
    data = rows()
    for r in data:
        if r["frame_type"] == 4:
            r["actual_fine_windows"] = 0
    assert not model.qualify(data)
    data = rows()
    data[1]["input_SHA256"] = data[0]["input_SHA256"]
    assert not model.qualify(data)
    assert not model.qualify(rows()[:-1])


def test_validation_rejects_missing_freeze_before_any_output_or_input_generation(
    tmp_path, monkeypatch,
):
    out = tmp_path / "not-created"
    monkeypatch.setattr(sys, "argv", ["model_coarse_capture", "--split", "validation",
                                     "--output", str(out)])
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not out.exists()
