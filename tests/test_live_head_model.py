"""Reproduction metadata guards and complete frozen input enumeration."""

import importlib

import numpy as np
import pytest


def test_fixed_case_names_are_unique_and_include_all_six_original_vectors(monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_live_head")
    rows = list(model.cases())
    assert len(rows) == 9 and len({r[0] for r in rows}) == 9
    assert {r[0] for r in rows if r[0].startswith("standard-")} == {
        f"standard-ft{ft}-crc{width}" for ft in (2, 3, 4) for width in (24, 32)
    }
    for _name, _profile, _values, expected, selection, origin in rows:
        assert isinstance(expected, bytes) and selection["start_tick"] > 0
        assert origin["kind"].endswith("regression")


def test_reproduction_refuses_existing_evidence_without_overwriting(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_live_head")
    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"keep all original evidence")
    with pytest.raises(ValueError, match="fresh evidence"):
        model.run(tmp_path)
    assert sentinel.read_bytes() == b"keep all original evidence"


def test_reported_missing_fixture_candidate_is_explicit_and_not_truth_repaired(monkeypatch):
    monkeypatch.syspath_prepend("scripts")
    model = importlib.import_module("model_live_head")
    # The search API may return a zero-score maximum for silence. Its explicit
    # None path is checked here;coherent confirmation stays with the real receiver.
    monkeypatch.setattr(model, "search_psk_sync", lambda *_args: None)
    with pytest.raises(ValueError, match="upstream sync candidate"):
        model.selected(model.ordinary(), np.zeros(4096, complex))
