"""The independent validation generator is unreachable before exact policy/source freeze."""

import copy
import json
import sys

import pytest
from scripts import model_banked_scaled as model


@pytest.mark.parametrize("changed", [None, "policy", "sources"])
def test_validation_refuses_missing_or_changed_freeze_before_generating_input(
    tmp_path, monkeypatch, changed
):
    out = tmp_path / "unseen-input"
    args = ["model_banked_scaled", "--output", str(out), "--split", "validation"]
    if changed is not None:
        frozen = dict(policy=copy.deepcopy(model.POLICY), sources=model.sources())
        frozen[changed][next(iter(frozen[changed]))] = "changed"
        path = tmp_path / "freeze.json"
        path.write_text(json.dumps(frozen))
        args += ["--freeze", str(path)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(model, "inputs", lambda *_: pytest.fail("Generated before freeze"))
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not out.exists()


def test_valid_frozen_contract_reaches_generator_after_writing_exact_binding(tmp_path, monkeypatch):
    class GeneratorReached(Exception):
        pass

    out = tmp_path / "unseen-input"
    path = tmp_path / "freeze.json"
    frozen = dict(policy=model.POLICY, sources=model.sources())
    path.write_text(json.dumps(frozen))

    def stop_before_generation(directory, split):
        assert directory == out and split == "validation"
        assert json.loads((directory / "freeze.json").read_text()) == frozen
        raise GeneratorReached

    monkeypatch.setattr(model, "inputs", stop_before_generation)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_banked_scaled",
            "--output",
            str(out),
            "--split",
            "validation",
            "--freeze",
            str(path),
        ],
    )
    with pytest.raises(GeneratorReached):
        model.main()
    assert list(out.iterdir()) == [out / "freeze.json"]
