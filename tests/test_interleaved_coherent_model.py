"""Prospective structure selection cannot mask numerical,RF or resource failures."""

import copy
import json

import pytest
from scripts import model_finite_coherent as old
from scripts import model_interleaved_coherent as model


def rows():
    stats = dict(complete=[0], false_MAC=0, duplicate_MAC=0, coherent=0,
                 all_pulses_accounted=True, max_ports=2, max_owned=3)
    detail = dict(word_pairs=[dict(integer_receipt_exact=True, actual_edges=60, original_edges=100)],
                  exact_common_fine_values=True, source_values_and_requests_exact=True,
                  same_selected_symbol_arrays_exact=True, original_float_success_losses=[],
                  lost_vs_original_serial=[], gate_control_bits=8067, DSP25x18=1, ALU32=2,
                  max_start_delta_samples=0, max_trajectory_delta_samples=0,
                  resource_exit_deltas=dict(queue_drops=0, history_drops=0, frame_resource_drops=0))
    return [dict(id=j, negative=j >= 6, candidates={c: copy.deepcopy(detail) for c in model.CANDIDATES},
                 modes={k: copy.deepcopy(stats) for k in ("floating", *model.CANDIDATES)})
            for j in range(10)]


def test_exact_node_does_not_override_original_success_or_resource_loss_or_timing():
    label = model.CANDIDATES[1]
    assert model.qualify(rows(), label)
    for key, value in [("original_float_success_losses", [0]), ("gate_control_bits", 8193),
                       ("exact_common_fine_values", False), ("max_start_delta_samples", .125001),
                       ("max_trajectory_delta_samples", .250001)]:
        r = rows()
        r[0]["candidates"][label][key] = value
        assert not model.qualify(r, label)
    r = rows()
    r[0]["candidates"][label]["resource_exit_deltas"]["queue_drops"] = 1
    assert not model.qualify(r, label)
    r = rows()
    r[0]["candidates"][label]["word_pairs"][0]["actual_edges"] = 81
    assert not model.qualify(r, label)


def test_extra_negative_confirmation_or_wrong_word_never_qualifies():
    label = model.CANDIDATES[1]
    r = rows()
    r[7]["modes"][label]["coherent"] = 1
    assert not model.qualify(r, label)
    r = rows()
    r[0]["candidates"][label]["word_pairs"][0]["integer_receipt_exact"] = False
    assert not model.qualify(r, label)


def test_selection_is_frozen_development_cost_rule_not_best_CRC_held_choice():
    r = rows()
    for row in r:
        row["candidates"][model.CANDIDATES[0]]["ALU32"] = 1
    assert model.select(r, model.CANDIDATES) == model.CANDIDATES[0]
    r[0]["candidates"][model.CANDIDATES[0]]["word_pairs"][0]["actual_edges"] = 99
    assert model.select(r, model.CANDIDATES) == model.CANDIDATES[1]
    r[0]["candidates"][model.CANDIDATES[1]]["original_float_success_losses"] = [0]
    assert model.select(r, model.CANDIDATES) is None


def test_new_families_and_configuration_combinations_are_not_only_new_seeds():
    for held in [False, True]:
        for j in range(12):
            assert model.payload(173, j, held) == model.payload(173, j, held)
            assert model.payload(173, j, held) != old.payload(173, j, held)
    for split in ["development", "validation"]:
        assert model.POLICY[split + "_specs"] != old.POLICY[split + "_specs"]


def test_every_declared_recipe_passes_strict_profile_before_generation():
    model.validate_recipes()


def test_illegal_pilot_rejected_before_any_IQ_or_output_directory(tmp_path, monkeypatch):
    policy = copy.deepcopy(model.POLICY)
    policy["development_specs"][2][2] = 32
    monkeypatch.setattr(model, "POLICY", policy)
    monkeypatch.setattr(model.reference, "make_inputs", lambda *a, **kw: pytest.fail("Before recipe"))
    monkeypatch.setattr("sys.argv", ["model_interleaved_coherent", "--output", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="Illegal pilot"):
        model.main()
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("change", ["source", "policy", "rejected"])
def test_bad_development_freeze_never_creates_a_new_held_set(tmp_path, monkeypatch, change):
    frozen = dict(source={"a.py": "original"}, policy=copy.deepcopy(model.POLICY),
                  selected=model.CANDIDATES[1])
    if change == "source":
        frozen["source"] = {"a.py": "changed"}
    elif change == "policy":
        frozen["policy"]["gate_control_cap_bits"] += 1
    else:
        frozen["selected"] = None
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(frozen))
    monkeypatch.setattr(model.reference.shared, "sources", lambda: {"a.py": "original"})
    monkeypatch.setattr(model.reference, "make_inputs", lambda *a, **kw: pytest.fail("Before freeze"))
    monkeypatch.setattr("sys.argv", ["model_interleaved_coherent", "--split", "validation",
                                   "--freeze", str(path), "--output", str(tmp_path / "held")])
    with pytest.raises(SystemExit) as result:
        model.main()
    assert result.value.code == 2 and not (tmp_path / "held").exists()
