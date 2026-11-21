"""Adoption needs distinct inputs,positive measured work and frozen evidence."""

import importlib
from copy import deepcopy
from pathlib import Path

import pytest


@pytest.fixture
def model(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("scripts.model_spline_gauss")


def records():
    rows = []
    for j in range(6):
        source = f"{j:064x}"
        row = dict(
            input_SHA256=source, source_window_end=1024, frame_type=2 + j % 3,
            scored=True, numerical_rejected=False, numerical_evaluation_unknown=False,
            baseline_cycles=1000, candidate_cycles=950,
            every_sample_score_rank_and_selected_word_exact=True,
            incremental_register_bits=1863, total_word_context_bits=22799,
        )
        rows.append(dict(id=j, input_SHA256=source, records=[row]))
    return rows


def test_eligible_exact_candidate_and_three_percent_boundary(model):
    rows = records()
    assert model.qualify(rows)
    for row in rows:
        row["records"][0]["candidate_cycles"] = 970
    assert model.qualify(rows)
    rows[0]["records"][0]["candidate_cycles"] = 971
    assert not model.qualify(rows)


def test_duplicate_source_windows_cannot_be_counted_as_independent_work(model):
    rows = records()
    rows += deepcopy(rows)
    assert not model.qualify(rows)


def test_negative_measured_cycles_cannot_manufacture_gain(model):
    rows = records()
    rows[0]["records"][0]["candidate_cycles"] = -1
    assert not model.qualify(rows)


@pytest.mark.parametrize("field,value", [
    ("every_sample_score_rank_and_selected_word_exact", False),
    ("incremental_register_bits", 2049), ("total_word_context_bits", 24577),
])
def test_numeric_and_resource_gates_do_not_follow_cycle_gain(model, field, value):
    rows = records()
    rows[0]["records"][0][field] = value
    assert not model.qualify(rows)


def test_unknown_or_missing_frame_work_does_not_count_as_numeric_pass(model):
    rows = records()
    for row in rows:
        if row["records"][0]["frame_type"] == 4:
            row["records"][0].update(scored=False, numerical_evaluation_unknown=True)
    assert not model.qualify(rows)
    assert not model.qualify([])


def test_held_requires_frozen_eligible_source_before_creating_any_input(
    model, tmp_path, monkeypatch
):
    target = tmp_path / "held"
    monkeypatch.setattr("sys.argv", ["model", "--split", "validation", "--output", str(target)])
    with pytest.raises(ValueError, match="frozen eligible"):
        model.main()
    assert not target.exists()
