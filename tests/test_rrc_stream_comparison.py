"""Comparison cannot alter a reference payload, hide loss or grant numeric acceptance."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module("compare_rrc_stream")
    iq = tmp_path / "input.npz"
    np.savez_compressed(iq, iq=np.zeros(16))
    meta = dict(
        id="fixed",
        path=str(iq),
        sha256=hashlib.sha256(iq.read_bytes()).hexdigest(),
        packets=[dict(sync_start=40, expected_hex="00")],
    )
    frame = dict(
        status="valid",
        header_crc=True,
        data_crc=True,
        length_ok=True,
        correct=True,
        payload_hex="00",
        sync_start=40,
    )
    row = dict(
        input=meta,
        frames=[frame],
        delivered_payloads=["00"],
        captured=1,
        correct=1,
        false_candidates=0,
        false_deliveries=0,
        fir_array_bytes=1696,
        receiver_events=[dict(sync_start=40, sync_score=0.9, cfo_hz=0)],
        actual_capture_config=dict(energy_multiplier=2, integer_screen=0.6),
        actual_timing_config=dict(bandwidth=0.003, max_rate_ppm=500),
        carrier_method="coherent",
    )
    dirs = [tmp_path / "old", tmp_path / "new"]
    for directory in dirs:
        (directory / "evidence").mkdir(parents=True)
        (directory / "results.json").write_text(json.dumps([row]))
        np.savez_compressed(
            directory / "evidence/fixed.npz", **{"0-symbol_centers_raw": np.array([40.0, 44.0])}
        )
    budget = tmp_path / "budget.json"
    budget.write_text(
        json.dumps(
            dict(
                budgets=dict(
                    sync_start_difference_max=0.125,
                    paired_centers_difference_max=0.25,
                    score_difference_max=0.005,
                    host_fir_numpy_arrays_max_bytes=1928,
                    known_rrc_numeric_gate_passed=False,
                )
            )
        )
    )
    return module, *dirs, budget, tmp_path / "report.json", row


def test_functional_match_cannot_overrule_known_numerical_failure(comparison):
    module, old, new, budget, output, _ = comparison
    assert module.compare(old, new, budget, output) == 1
    report = json.loads(output.read_text())
    assert report["integration_functional_budget_passed"]
    assert not report["rrc_numeric_gate_passed"] and not report["receiver_promotion"]


def test_bad_data_and_trajectory_regression_are_retained(comparison):
    module, old, new, budget, output, row = comparison
    candidate = copy.deepcopy(row)
    candidate["frames"][0].update(
        status="bad_data_crc", data_crc=False, correct=False, payload_hex=None
    )
    candidate["delivered_payloads"] = []
    (new / "results.json").write_text(json.dumps([candidate]))
    np.savez_compressed(
        new / "evidence/fixed.npz", **{"0-symbol_centers_raw": np.array([40.0, 44.5])}
    )
    assert module.compare(old, new, budget, output) == 1
    report = json.loads(output.read_text())
    assert len(report["original_success_losses"]) == 1
    assert not report["gates"]["trajectory"] and not report["gates"]["original_successes"]


@pytest.mark.parametrize("change", ["reference", "crc", "config"])
def test_changed_expected_prefix_false_crc_or_receive_parameter_cannot_pass(comparison, change):
    module, old, new, budget, output, row = comparison
    candidate = copy.deepcopy(row)
    if change == "reference":
        candidate["input"]["packets"][0]["expected_hex"] = "ff00"
    elif change == "crc":
        candidate["frames"][0]["header_crc"] = False
    else:
        candidate["actual_capture_config"]["integer_screen"] = 0.5
    (new / "results.json").write_text(json.dumps([candidate]))
    with pytest.raises(AssertionError):
        module.compare(old, new, budget, output)
