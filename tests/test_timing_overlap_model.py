"""A stale source or policy cannot silently consume an already viewed validation."""

import importlib
import json
from pathlib import Path

import pytest


def test_model_rejects_changed_source_before_creating_evidence(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("scripts.model_timing_overlap")
    monkeypatch.setattr(module, "source_identity", lambda: {"changed": "source"})
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(dict(policy=json.loads(module.POLICY.read_text()), source={"old": "source"}))
    )
    target = tmp_path / "result"
    with pytest.raises(AssertionError):
        module.run(target, freeze=freeze)
    assert not target.exists()
    with pytest.raises(ValueError):
        module.run(tmp_path)
