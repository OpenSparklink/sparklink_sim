"""The independent holdout must not open before its frozen source/policy agrees."""

import importlib

import pytest


@pytest.mark.parametrize("missing", [True, False])
def test_holdout_guard_runs_before_creating_any_input(tmp_path, monkeypatch, missing):
    monkeypatch.syspath_prepend("scripts")
    module = importlib.import_module("model_header_words")
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}\n")
    output = tmp_path / "validation"
    with pytest.raises(ValueError, match=r"requires|differs"):
        module.run(output, "validation", None if missing else freeze)
    assert not output.exists()
