"""Validation cannot run with missing or changed frozen source/policy."""

import json
import sys
from pathlib import Path

import pytest
from scripts import model_rrc_bridge as model


@pytest.mark.parametrize("freeze", [None, "changed-policy", "changed-source"])
def test_validation_refuses_before_evaluating_or_creating_outputs(tmp_path, monkeypatch, freeze):
    output = tmp_path / "validation"
    args = ["model_rrc_bridge", "--output", str(output), "--split", "validation"]
    if freeze is not None:
        value = dict(policy=dict(model.POLICY), sources={})
        if freeze == "changed-policy":
            value["policy"]["register_bits_upper"] += 1
        path = tmp_path / "freeze.json"
        path.write_text(json.dumps(value))
        args += ["--freeze", str(path)]
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match=r"frozen|Source or policy"):
        model.main()
    assert not output.exists()


def test_independent_reference_and_lock_are_frozen_before_validation(tmp_path, monkeypatch):
    bindings = model.source_bindings()
    assert bindings["tests/test_rrc_bridge.py"] == model.sha(Path("tests/test_rrc_bridge.py"))
    assert bindings["uv.lock"] == model.sha(Path("uv.lock"))
    freeze, output = tmp_path / "freeze.json", tmp_path / "validation"
    freeze.write_text(json.dumps(dict(policy=model.POLICY, sources=bindings)))
    original_sha = model.sha
    monkeypatch.setattr(
        model, "sha", lambda p: "0" * 64 if p.name == "test_rrc_bridge.py" else original_sha(p)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_rrc_bridge",
            "--split",
            "validation",
            "--freeze",
            str(freeze),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Source or policy"):
        model.main()
    assert not output.exists()
