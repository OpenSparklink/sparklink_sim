"""Standalone model evidence cannot award header CRC or whole-RX acceptance."""

import json
import subprocess
import sys

import pytest


def test_reproducible_model_has_actual_words_ports_and_unchecked_CRC(tmp_path):
    output = tmp_path / "new"
    result = subprocess.run([sys.executable, "scripts/model_control_polar.py", "--output",
                             str(output), "--random-cases", "0"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    model = json.loads((output / "model.json").read_text())
    assert not model["full_RX_gates_passed"] and not model["RF_holdout_evaluation"]
    assert model["reception_gain"] is None and model["known_standard_matrix_vectors"] == 6
    assert len(model["source"]) == 3
    assert len(model["rows"]) == 4
    for row in model["rows"]:
        assert row["vectors"] == 13 and row["CRC"] == "not_checked"
        assert row["decoded_bit_differences"] == 0
        assert row["cycles_min"] == row["cycles_max"] > 0
        trace = json.loads(next(output.glob(f"N{row['N']}-K{row['K']}-{row['layout']}-*json"))
                           .read_text())
        assert any(e["event"] == "read" for e in trace)
        assert max(row["physical_bank_peak_ports"].values()) <= 2


@pytest.mark.parametrize("count", ["-1", "513"])
def test_model_rejects_unbounded_case_count_before_output(tmp_path, count):
    output = tmp_path / "new"
    result = subprocess.run([sys.executable, "scripts/model_control_polar.py", "--output",
                             str(output), "--random-cases", count], capture_output=True)
    assert result.returncode != 0 and not output.exists()
