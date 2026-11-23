"""Reject recall/resource/held leakage independently of a cheaper tail controller."""

import json
from copy import deepcopy

import pytest
from scripts import model_legal_tail_ceiling as audit

from tests.test_model_private_prefetch import _row


def row():
    value = _row()
    cost = value["comparison"]["cost"]["bounded_private_prefetch"]
    cost["additional_register_reservation_bits"] = cost["horizon"]["state_bits"] = 406
    return value


def test_prior_numerical_rejection_and_recall_loss_still_fail_with_less_work(monkeypatch):
    current, previous = row(), row()
    previous["modes"]["finite"]["fine_cycles"] = 60
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(
        passed=False, reasons=["structural_or_numerical_failure"]))
    assert "structural_or_numerical_failure" in audit.qualify(
        [current], [previous])["reasons"]
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(passed=True, reasons=[]))
    assert audit.qualify([current], [previous])["passed"]
    current["modes"]["finite"]["coherent"] = 0
    assert not audit.qualify([current], [previous])["passed"]


def test_success_loss_ports_and_resource_exits_are_separate_gates(monkeypatch):
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(passed=True, reasons=[]))
    current, previous = row(), row()
    previous["modes"]["finite"]["complete"] = [0, 1]
    current["comparison"]["resource_exit_deltas"]["queue_drops"] = 1
    current["comparison"]["cost"]["bounded_private_prefetch"]["additional_source_read_ports"] = 1
    assert {"static_success_lost", "static_resource_exits_increased",
            "private_prefetch_cost_failed"} <= set(audit.qualify(
                [current], [previous])["reasons"])
    changed = deepcopy(current)
    changed["input_SHA256"] = "substituted"
    with pytest.raises(ValueError, match="SHA"):
        audit.qualify([changed], [previous])


@pytest.mark.parametrize("defect", ["rejected", "same_regression_twice", "wrong_source"])
def test_held_is_not_created_after_a_failed_or_incomplete_freeze(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(audit, "sources", lambda: {})
    paths = []
    roles = ["development", "fixed-regression", "fixed-regression", "fixed-regression"]
    for index, role in enumerate(roles):
        value = dict(source={}, policy=audit.POLICY, role=role,
                     qualification=dict(passed=True), input_manifest_SHA256="fresh")
        if index:
            value["input_manifest_SHA256"] = audit.POLICY["fixed_manifest_SHA256"][index - 1]
        if defect == "rejected" and index == 0:
            value["qualification"]["passed"] = False
        if defect == "wrong_source" and index == 3:
            value["source"] = {"changed": "SHA"}
        if defect == "same_regression_twice" and index == 3:
            value["input_manifest_SHA256"] = audit.POLICY["fixed_manifest_SHA256"][0]
        path = tmp_path / f"freeze{index}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(output), "--split", "validation",
        "--freeze", str(paths[0]), "--regression-freeze", str(paths[1]),
        "--regression-freeze", str(paths[2]), "--regression-freeze", str(paths[3])])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2 and not output.exists()


def test_real_quiet_CLI_keeps_the_structured_failure_and_literal_input_SHA(tmp_path, monkeypatch):
    import numpy as np

    from tests.test_coherent_captured_frame import profile_case

    profile, *_ = profile_case()
    file = tmp_path / "quiet.npz"
    np.savez(file, iq=np.zeros(256, complex), quiet=np.full(1024, .002, complex))
    config = vars(profile.config).copy()
    config.pop("_mcs")
    inputs = tmp_path / "inputs.json"
    inputs.write_text(json.dumps([dict(id=0, file=file.name,
        SHA256=audit.old.reference.shared.sha(file), config=config,
        head_backend=profile.head_backend, expected=[], negative=True,
        truth_diagnostic_only=[])]))
    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(output), "--inputs", str(inputs)])
    assert audit.main() == 1
    result = json.loads((output / "decision.json").read_text())
    assert "incomplete_stream_set" in result["qualification"]["reasons"]
    assert result["input_manifest_SHA256"] == audit.old.reference.shared.sha(inputs)
    assert result["qualification"]["static_resource_exit_deltas"]["queue_drops"] == 0
