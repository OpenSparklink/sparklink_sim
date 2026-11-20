"""Source/evidence and empty-result failures cannot masquerade as expected exit1."""

import importlib
import json

import pytest


def model(monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    return importlib.import_module("model_retained_stencil")


def test_changed_frozen_source_rejects_before_validation_output(tmp_path, monkeypatch):
    m = model(monkeypatch)
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps(dict(source={"wrong": "source"},
                                    policy=json.loads(m.POLICY.read_text()))))
    output = tmp_path / "validation"
    with pytest.raises(AssertionError):
        m.run(output, freeze=path)
    assert not output.exists()


def test_existing_evidence_remains_intact(tmp_path, monkeypatch):
    m = model(monkeypatch)
    sentinel = tmp_path / "keep"
    sentinel.write_bytes(b"original failure evidence")
    with pytest.raises(ValueError, match="fresh evidence"):
        m.run(tmp_path)
    assert sentinel.read_bytes() == b"original failure evidence"


def test_empty_nodes_cannot_generate_success_report(tmp_path, monkeypatch):
    m = model(monkeypatch)
    monkeypatch.setattr(m, "nodes", lambda *args: dict(cases=0, exact=True))

    def forbidden(*args):
        pytest.fail("Invalid node evidence must stop before frame evaluation")

    monkeypatch.setattr(m, "fixed", forbidden)
    output = tmp_path / "invalid"
    with pytest.raises(AssertionError):
        m.run(output)
    assert not (output / "model.json").exists()
