"""Truth diagnostics cannot select a production answer or relabel failed CRCs."""

import importlib
from pathlib import Path

import pytest

from tests.test_strict_receive import transmitted
from tests.test_sync_refine import profile


@pytest.fixture
def diagnostic(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("diagnose_coherent_failures")


def test_expected_content_changes_only_post_decode_assessment(diagnostic):
    p = profile()
    wire, expected = transmitted(p)
    target = dict(expected_hex=expected.hex(), length=len(expected), mcs=p.config.mcs_index)
    a = diagnostic.assess(wire[40:], p, target, len(wire)-40)
    target["expected_hex"] = bytes([expected[0] ^ 1, *expected[1:]]).hex()
    b = diagnostic.assess(wire[40:], p, target, len(wire)-40)
    assert a["correct"] and a["status"] == "valid"
    assert not b["correct"] and b["status"] == "valid"
    assert a["payload_hex"] == b["payload_hex"]
    assert a["data_crc"] is b["data_crc"] is True
    assert (a["bit_errors"], b["bit_errors"]) == (0, 1)


def test_partial_header_control_is_not_a_full_packet_failure_oracle(diagnostic):
    p = profile()
    wire, expected = transmitted(p)
    a = diagnostic.assess(wire[40:50], p,
        dict(expected_hex=expected.hex(), length=len(expected), mcs=p.config.mcs_index), len(wire)-40)
    assert not a["complete_span"] and a["bit_errors"] is None
    assert diagnostic.classify(dict(truth_timing_truth_CFO=a), None) == "insufficient_evidence"


@pytest.mark.parametrize("category,arms,actual", [
    ("diagnostic_recovered_actual_failed", dict(truth_timing_truth_CFO=dict(correct=True)), None),
    ("errors_reduced_still_failed", dict(actual_symbols_kappa2=dict(correct=False, bit_errors=1)),
     dict(bit_errors=20)),
    ("tested_controls_still_failed",
     dict(truth_timing_truth_CFO=dict(correct=False, complete_span=True)), None),
    ("insufficient_evidence", {}, None),
])
def test_failure_classes_never_mean_formal_or_theoretical_recovery(diagnostic, category, arms, actual):
    assert diagnostic.classify(arms, actual) == category


def test_diagnostic_keeps_real_bad_crc_failed(diagnostic):
    p = profile()
    wire, expected = transmitted(p, bad_data=True)
    a = diagnostic.assess(wire[40:], p,
        dict(expected_hex=expected.hex(), length=len(expected), mcs=p.config.mcs_index), len(wire)-40)
    assert a["header_crc"] and a["data_crc"] is False
    assert a["status"] == "bad_data_crc" and not a["correct"] and a["bit_errors"] > 0
