"""Adoption integrity and source/freeze guards; no successful-looking fake rows."""

import copy
import sys

import pytest
from scripts import model_fine_symbols as model


def rows():
    return [dict(id=k, input_SHA256=f"{k + 1:064x}", frame_type=(2, 3, 4)[k % 3],
                 all_semantic_values_exact=True, scored=True, numerical_pass=True,
                 numerical_rejection=False, baseline_cycles=100, candidate_cycles=75,
                 new_DSP=16, new_register_bits=11651, extra_RAM_and_sample_ports=0)
            for k in range(6)]


@pytest.mark.parametrize("edit", [
    dict(candidate_cycles=-1), dict(new_register_bits=-1), dict(new_DSP=17),
    dict(new_register_bits=12289), dict(numerical_pass=None), dict(scored=False),
    dict(numerical_rejection=True), dict(all_semantic_values_exact=False),
])
def test_reject_resource_unknown_numeric_nonpositive_and_semantic_failures(edit):
    data = rows()
    data[1].update(edit)
    assert not model.qualify(data)


def test_duplicate_inputs_and_exact_performance_boundary():
    data = rows()
    assert model.qualify(data)
    data[0]["candidate_cycles"] += 1
    assert not model.qualify(data)
    data = rows()
    data[1]["input_SHA256"] = data[0]["input_SHA256"]
    assert not model.qualify(data)
    data = rows()
    data[1] = copy.deepcopy(data[0])
    assert not model.qualify(data)
    assert not model.qualify(rows()[:-1])


def test_validation_requires_freeze_before_creating_directory(tmp_path, monkeypatch):
    out = tmp_path / "uncreated"
    monkeypatch.setattr(sys, "argv", ["model_fine_symbols", "--split", "validation",
                                     "--output", str(out)])
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not out.exists()


def test_actual_fixture_configuration_roundtrips_without_cached_mcs_fraction(tmp_path):
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    data = model.fixtures(tmp_path, "development")
    assert len(data) == 6 and len({r["SHA256"] for r in data}) == 6
    for row in data:
        assert "_mcs" not in row["config"]
        cfg = TxConfig(**row["config"])
        assert cfg.frame_type == row["frame_type"] and cfg.pid == row["sync_PID"]
