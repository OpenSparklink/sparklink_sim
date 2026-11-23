"""Adoption cannot hide original-success loss,cost or a changed frozen reference."""

import json

import pytest
from scripts import model_head_prefetch as audit


def _row():
    return dict(id=0, input_SHA256="literal", comparison=dict(cost=dict(
        bounded_head_prefetch=dict(additional_register_reservation_bits=84,
            additional_DSP=0, additional_BRAM36=0, additional_source_read_ports=0)),
        resource_exit_deltas=dict(queue_drops=0, history_drops=0, frame_resource_drops=0)),
        modes=dict(original=dict(complete=[0]), finite=dict(complete=[0], fine_cycles=50,
                                                          queue_drops=0, history_drops=0)))


def test_prefetch_gain_does_not_cover_a_previous_numeric_or_success_failure(monkeypatch):
    row, baseline = _row(), _row()
    baseline["modes"]["finite"]["fine_cycles"] = 60
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(
        passed=False, reasons=["original_success_lost", "structural_or_numerical_failure"]))
    result = audit.qualify([row], [baseline])
    assert not result["passed"] and "original_success_lost" in result["reasons"]
    assert "structural_or_numerical_failure" in result["reasons"]
    assert result["fine_request_elapsed_cycles"] < result["M36_fine_request_elapsed_cycles"]


def test_explicit_prefetch_cost_loss_resource_and_no_benefit_gates(monkeypatch):
    monkeypatch.setattr(audit.old, "qualify", lambda rows: dict(passed=True, reasons=[]))
    row, baseline = _row(), _row()
    assert "no_recovery_or_schedule_benefit" in audit.qualify([row], [baseline])["reasons"]
    baseline["modes"]["finite"]["complete"] = [0, 1]
    row["comparison"]["resource_exit_deltas"]["queue_drops"] = 1
    row["comparison"]["cost"]["bounded_head_prefetch"]["additional_BRAM36"] = 1
    reasons = audit.qualify([row], [baseline])["reasons"]
    assert {"M36_success_lost", "M36_resource_exits_increased", "prefetch_cost_gate_failed"} <= set(
        reasons)
    row["input_SHA256"] = "changed"
    with pytest.raises(ValueError, match="SHA"):
        audit.qualify([row], [baseline])
    with pytest.raises(ValueError, match="every"):
        audit.qualify([], [baseline])


def test_real_CLI_empty_stream_schema_retains_structured_rejection(tmp_path, monkeypatch):
    # Actual RRC/capture/consumer factories and output schemas;no mocked summary
    # with counters the real generator never produced.
    import numpy as np

    from tests.test_coherent_captured_frame import profile_case

    profile, *_ = profile_case()
    file = tmp_path / "quiet.npz"
    np.savez(file, iq=np.zeros(256, complex), quiet=np.full(1024, .002, complex))
    cfg = vars(profile.config).copy()
    cfg.pop("_mcs")
    row = dict(id=0, file=file.name, SHA256=audit.old.reference.shared.sha(file),
        config=cfg, head_backend=profile.head_backend, expected=[], negative=True,
        truth_diagnostic_only=[])
    inputs, out = tmp_path / "inputs.json", tmp_path / "run"
    inputs.write_text(json.dumps([row]))
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out), "--inputs", str(inputs)])
    assert audit.main() == 1
    result = json.loads((out / "decision.json").read_text())
    assert not result["qualification"]["passed"]
    assert result["input_manifest_SHA256"] == audit.old.reference.shared.sha(inputs)
    assert "incomplete_stream_set" in result["qualification"]["reasons"]
    assert result["qualification"]["M36_resource_exit_deltas"]["frame_resource_drops"] == 0


@pytest.mark.parametrize("role", ["development", "fixed-regression"])
def test_rejected_or_seen_freeze_never_generates_new_held_input(tmp_path, monkeypatch, role):
    freeze, out = tmp_path / "freeze.json", tmp_path / "uncreated"
    freeze.write_text(json.dumps(dict(source={}, policy=audit.POLICY,
        qualification=dict(passed=role != "development"), role=role)))
    monkeypatch.setattr(audit.old.reference.shared, "sources", lambda: {})
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out), "--split", "validation",
                                    "--freeze", str(freeze)])
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2 and not out.exists()
