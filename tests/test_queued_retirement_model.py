"""Prospective freeze and independent acceptance checks for queue-lifetime study."""

import copy
import json

import pytest
from scripts import model_queued_retirement as model
from scripts import model_retired_capture as original


def accepted_rows():
    rows = []
    for j in range(10):
        stats = dict(
            complete=[], false_MAC=0, duplicate_MAC=0, coherent=0,
            all_pulses_accounted=True, max_ports=2, max_owned=3,
        )
        rows.append(dict(
            id=j, negative=j >= 6, baseline=copy.deepcopy(stats), queued=copy.deepcopy(stats),
            original_success_losses=[], additional_control_bits=66,
            same_source_descriptors=True, common_fine_integer_and_words_exact=True,
            survivor_trajectories_exact=True, registered_lifetime_and_accounting_verified=True,
            all_input_words_and_three_host_chunkings_exact=True,
        ))
    return rows


def test_acceptance_does_not_hide_loss_false_MAC_or_resource_violation():
    assert model.qualify(accepted_rows())
    for field, value in [
        ("original_success_losses", [1]), ("additional_control_bits", 129),
        ("common_fine_integer_and_words_exact", False), ("survivor_trajectories_exact", False),
    ]:
        rows = accepted_rows()
        rows[0][field] = value
        assert not model.qualify(rows)
    for field, value in [
        ("false_MAC", 1), ("duplicate_MAC", 1), ("max_ports", 3),
        ("max_owned", 4), ("all_pulses_accounted", False), ("coherent", 1),
    ]:
        rows = accepted_rows()
        rows[6]["queued"][field] = value
        assert not model.qualify(rows)
    assert not model.qualify(accepted_rows()[:-1])


def test_new_payload_families_and_configuration_recipes_are_not_seed_only_changes():
    for held in (False, True):
        for j in range(12):
            data = model.payload(173, j, held)
            assert len(data) == 173 and data != original.payload(173, j, held)
            assert data == model.payload(173, j, held)
    old = {tuple((r[0], r[1], r[2], tuple(r[3]), r[4], r[5]))
           for name in ("development_specs", "validation_specs") for r in original.POLICY[name]}
    for name in ("development_specs", "validation_specs"):
        for r in model.POLICY[name]:
            assert (r[0], r[1], r[2], tuple(r[3]), r[4], r[5]) not in old


def test_changed_IQ_is_rejected_before_receive_or_diagnostic_association(tmp_path, monkeypatch):
    path = tmp_path / "input.npz"
    path.write_bytes(b"changed")
    monkeypatch.setattr(model.reference, "run_arm", lambda *a, **k: pytest.fail("before SHA"))
    with pytest.raises(ValueError, match="Frozen IQ"):
        model.compare(tmp_path, tmp_path, dict(file="input.npz", SHA256="00"))


@pytest.mark.parametrize("change", ["source", "policy", "rejected"])
def test_changed_or_rejected_development_freeze_cannot_generate_validation(
    tmp_path, monkeypatch, change
):
    source = {"receiver.py": "frozen"}
    freeze = dict(source=source, policy=copy.deepcopy(model.POLICY), development_pass=True)
    if change == "source":
        freeze["source"] = {"receiver.py": "other"}
    elif change == "policy":
        freeze["policy"]["owned_windows"] = 4
    else:
        freeze["development_pass"] = False
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze))
    monkeypatch.setattr(model.reference.shared, "sources", lambda: source)
    monkeypatch.setattr(model.reference, "make_inputs", lambda *a, **k: pytest.fail("before freeze"))
    monkeypatch.setattr("sys.argv", [
        "model_queued_retirement", "--output", str(tmp_path / "held"), "--split", "validation",
        "--freeze", str(path),
    ])
    with pytest.raises(SystemExit) as error:
        model.main()
    assert error.value.code == 2 and not (tmp_path / "held").exists()
