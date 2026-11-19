"""Source/parameters/budget freeze precedes opening independent new inputs."""

import importlib

import pytest


@pytest.mark.parametrize("missing", [True, False])
def test_holdout_guard_precedes_inputs(tmp_path, monkeypatch, missing):
    monkeypatch.syspath_prepend("scripts")
    module = importlib.import_module("model_hard_header")
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n")
    output = tmp_path / "validation"
    with pytest.raises(ValueError, match="differs before opening"):
        module.run(output, "validation", None if missing else freeze)
    assert not output.exists()
