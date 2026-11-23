"""Adoption guards on real output schemas and frozen development/held separation."""

import json
from copy import deepcopy

import pytest
from scripts import model_private_prefetch as audit


def _row():
    return dict(id=0, input_SHA256="literal", comparison=dict(cost=dict(
        bounded_private_prefetch=dict(additional_register_reservation_bits=406,
            additional_DSP=0, additional_BRAM36=0, additional_source_read_ports=0,
            horizon=dict(state_bits=406, latency_edges=19))),
        resource_exit_deltas=dict(queue_drops=0, history_drops=0, frame_resource_drops=0)),
        modes=dict(original=dict(complete=[0]), finite=dict(complete=[0], fine_cycles=50,
            coherent=1, captured_packet_associations_diagnostic=[0])))


def test_less_work_cannot_hide_recall_loss_or_previous_numerical_failure(monkeypatch):
    row, baseline = _row(), _row()
    baseline["modes"]["finite"]["fine_cycles"] = 60
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(
        passed=False, reasons=["structural_or_numerical_failure"]))
    result = audit.qualify([row], [baseline])
    assert not result["passed"] and "structural_or_numerical_failure" in result["reasons"]
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(passed=True, reasons=[]))
    assert audit.qualify([row], [baseline])["passed"]
    row["modes"]["finite"]["coherent"] = 0
    assert "no_recovery_or_same_recall_schedule_benefit" in audit.qualify(
        [row], [baseline])["reasons"]


def test_cost_success_resource_and_literal_input_gates_are_independent(monkeypatch):
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(passed=True, reasons=[]))
    row, baseline = _row(), _row()
    baseline["modes"]["finite"]["complete"] = [0, 1]
    row["comparison"]["resource_exit_deltas"]["frame_resource_drops"] = 1
    row["comparison"]["cost"]["bounded_private_prefetch"]["additional_BRAM36"] = 1
    reasons = audit.qualify([row], [baseline])["reasons"]
    assert {"M37_success_lost", "M37_resource_exits_increased",
            "private_prefetch_cost_failed"} <= set(reasons)
    changed = deepcopy(row)
    changed["input_SHA256"] = "other"
    with pytest.raises(ValueError, match="SHA"):
        audit.qualify([changed], [baseline])
    with pytest.raises(ValueError, match="unique"):
        audit.qualify([row, row], [baseline])


def test_real_quiet_CLI_schema_retains_a_structured_rejection(tmp_path, monkeypatch):
    import numpy as np

    from tests.test_coherent_captured_frame import profile_case

    profile, *_ = profile_case()
    file = tmp_path / "quiet.npz"
    np.savez(file, iq=np.zeros(256, complex), quiet=np.full(1024, .002, complex))
    cfg = vars(profile.config).copy()
    cfg.pop("_mcs")
    inputs, out = tmp_path / "inputs.json", tmp_path / "run"
    inputs.write_text(json.dumps([dict(id=0, file=file.name,
        SHA256=audit.old.reference.shared.sha(file), config=cfg,
        head_backend=profile.head_backend, expected=[], negative=True,
        truth_diagnostic_only=[])]))
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out), "--inputs", str(inputs)])
    assert audit.main() == 1
    result = json.loads((out / "decision.json").read_text())
    assert "incomplete_stream_set" in result["qualification"]["reasons"]
    assert result["input_manifest_SHA256"] == audit.old.reference.shared.sha(inputs)
    assert result["qualification"]["M37_resource_exit_deltas"]["frame_resource_drops"] == 0


@pytest.mark.parametrize("role,passed", [("development", False), ("fixed-regression", True)])
def test_failed_development_or_seen_regression_cannot_be_a_new_held_freeze(
    tmp_path, monkeypatch, role, passed
):
    freeze, out = tmp_path / "freeze.json", tmp_path / "uncreated"
    freeze.write_text(json.dumps(dict(source={}, policy=audit.POLICY,
        qualification=dict(passed=passed), role=role)))
    monkeypatch.setattr(audit, "sources", lambda: {})
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out), "--split", "validation",
        "--freeze", str(freeze), "--regression-freeze", str(freeze)])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2 and not out.exists()
