"""RRC experiment guards and independent zero-energy/error counterexamples."""

import json

import numpy as np
import pytest
from scripts import model_rrc_full14 as model


def test_all_predeclared_configs_are_validated_before_new_iq():
    policy = json.loads(model.POLICY.read_text())
    for split, count in [("development", 24), ("validation", 30)]:
        rows = model.specs(policy, split)
        assert len(rows) == count and len({r["id"] for r in rows}) == count
        assert {r["config"]["frame_type"] for r in rows} == {2, 3, 4}
        assert {r["config"]["mcs_index"] for r in rows} == {8, 12}
        assert {r["config"]["pilot_interval"] for r in rows} == {0, 4, 8, 16}
        assert all("_mcs" not in r["config"] for r in rows)


def test_validation_refuses_without_qualified_byte_identical_source_before_creating_output(tmp_path):
    out, dev = tmp_path/"held", tmp_path/"dev"
    with pytest.raises(ValueError, match="qualified development"):
        model.run(out, "validation")
    dev.mkdir()
    (dev/"qualification.json").write_text(json.dumps(dict(qualified=False, sources={})))
    with pytest.raises(ValueError, match="Unqualified or changed"):
        model.run(out, "validation", dev)
    assert not out.exists()
    (dev/"qualification.json").write_text(json.dumps(dict(qualified=True, sources={})))
    with pytest.raises(ValueError, match="Unqualified or changed"):
        model.run(out, "validation", dev)
    assert not out.exists()


def test_zero_energy_and_bad_range_events_are_not_successful_numeric_zeros():
    policy = json.loads(model.POLICY.read_text())
    x, counts = np.zeros(128, complex), dict.fromkeys(model.RANGE, 0)
    good = model.metrics(x, x, counts, policy["numerical_budget"])
    assert good["passed"] and good["values"]["local64_relative_rms"] == 0
    bad = model.metrics(x+1, x, counts, policy["numerical_budget"])
    assert not bad["passed"] and bad["values"]["global_normalized_rms"] is None
    bad = model.metrics(x, x, counts | dict(intermediate_overflows=1), policy["numerical_budget"])
    assert not bad["passed"] and bad["values"]["range_events"] == 1


def test_error_in_quiet_window_cannot_hide_behind_global_reference_peak():
    x = np.r_[np.ones(64), np.ones(64)*1e-6].astype(complex)
    y = x.copy()
    y[64:] += 1e-7
    policy = json.loads(model.POLICY.read_text())
    result = model.metrics(y, x, dict.fromkeys(model.RANGE, 0), policy["numerical_budget"])
    assert result["values"]["global_normalized_rms"] < 1e-6
    assert result["values"]["local64_relative_rms"] > .02 and not result["passed"]
    assert result["worst_local64_start"] == 64


def test_driver_eof_words_equal_actual_public_finish_without_double_padding():
    x = np.arange(137)/8192+1j*np.sin(np.arange(137))
    a, b = model.Full14MantissaRRC(), model.Full14MantissaRRC()
    y, _, counts = model.filtered(a, x, [3, 17, 1, 53])
    expected = np.r_[b.feed(x), b.finish()]
    np.testing.assert_array_equal(y, expected)
    assert counts == b.stats and counts["zero_flush_samples"] == 80
