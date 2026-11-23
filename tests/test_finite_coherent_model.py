"""Separate numerical acceptance,resource failures,and prospective input freezes."""

import copy
import json

import pytest
from scripts import model_finite_coherent as model
from scripts import model_queued_retirement as old


def rows():
    result = []
    for j in range(10):
        stats = dict(false_MAC=0, duplicate_MAC=0, coherent=0, all_pulses_accounted=True,
                     max_ports=2, max_owned=3)
        result.append(dict(
            id=j, negative=j >= 6, functional_success_losses=[], gate_control_bits=5161,
            common_fine_integer_and_words_exact=True, source_values_and_requests_exact=True,
            functional_survivor_trajectories_exact=True,
            corresponding_accepted_symbol_arrays_exact=True,
            three_chunkings_and_registered_source_words_exact=True,
            floating=copy.deepcopy(stats), functional=copy.deepcopy(stats), clocked=copy.deepcopy(stats),
            quality_pairs=[dict(absolute_error=.001, angle_circular_error_rad=1e-8,
                                step_circular_error_rad=1e-6)] if j < 4 else [],
            clock_resource_success_losses=[], clock_resource_drop_deltas=dict(queue_drops=0),
        ))
    return result


def test_numerical_node_cannot_hide_loss_false_confirmation_or_budget_failure():
    assert model.qualify_node(rows())
    for key, value in [
        ("functional_success_losses", [1]), ("gate_control_bits", 8193),
        ("corresponding_accepted_symbol_arrays_exact", False),
        ("common_fine_integer_and_words_exact", False),
    ]:
        r = rows()
        r[0][key] = value
        assert not model.qualify_node(r)
    for key, value in [("absolute_error", .005001), ("angle_circular_error_rad", -1.001e-6),
                       ("step_circular_error_rad", 3.101e-6)]:
        r = rows()
        r[0]["quality_pairs"][0][key] = value
        assert not model.qualify_node(r)
    r = rows()
    r[6]["functional"]["coherent"] = 1
    assert not model.qualify_node(r)


def test_numeric_pass_does_not_assert_clock_resource_or_whole_receiver_pass():
    r = rows()
    r[0]["clock_resource_success_losses"] = [0]
    r[0]["clock_resource_drop_deltas"]["queue_drops"] = 2
    assert model.qualify_node(r)  # A different gate;resource loss remains explicitly failed.
    assert not all(not x["clock_resource_success_losses"] for x in r)
    assert model.POLICY["numeric_node_and_clock_resource_gates_separate"]


def test_new_payloads_and_recipes_change_more_than_random_seed():
    for held in (False, True):
        for j in range(12):
            data = model.payload(193, j, held)
            assert len(data) == 193 and data == model.payload(193, j, held)
            assert data != old.payload(193, j, held)
    assert model.POLICY["development_specs"] != old.POLICY["development_specs"]
    assert model.POLICY["validation_specs"] != old.POLICY["validation_specs"]


def test_changed_IQ_is_rejected_before_any_constructor_or_receive(tmp_path, monkeypatch):
    (tmp_path / "input.npz").write_bytes(b"other")
    monkeypatch.setattr(model.reference, "run_arm", lambda *a, **k: pytest.fail("Before input SHA"))
    with pytest.raises(ValueError, match="Frozen IQ"):
        model.compare(tmp_path, tmp_path, dict(file="input.npz", SHA256="00"))


@pytest.mark.parametrize("change", ["source", "format", "numerical_rejection"])
def test_invalid_development_freeze_cannot_generate_or_tune_a_holdout(tmp_path, monkeypatch, change):
    src = {"receiver.py": "original"}
    freeze = dict(source=src, policy=copy.deepcopy(model.POLICY), numeric_node_pass=True)
    if change == "source":
        freeze["source"] = {"receiver.py": "changed"}
    elif change == "format":
        freeze["policy"]["comparison_bits"] = 111
    else:
        freeze["numeric_node_pass"] = False
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    monkeypatch.setattr(model.reference.shared, "sources", lambda: src)
    monkeypatch.setattr(model.reference, "make_inputs", lambda *a, **k: pytest.fail("Before freeze"))
    monkeypatch.setattr("sys.argv", ["model_finite_coherent", "--split", "validation",
                                   "--freeze", str(path), "--output", str(tmp_path / "held")])
    with pytest.raises(SystemExit) as e:
        model.main()
    assert e.value.code == 2 and not (tmp_path / "held").exists()
