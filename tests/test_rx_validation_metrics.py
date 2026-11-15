"""Counterexamples for audit labels: uncertainty, loss/slips and one-bit CRC failure."""

import importlib
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def boundary(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_rx_boundary")


def test_binomial_zero_and_full_success_are_not_certainty(boundary):
    np.testing.assert_allclose(boundary.wilson(0, 24), [0, 0.13797620467498017], atol=1e-15, rtol=0)
    np.testing.assert_allclose(boundary.wilson(24, 24), [0.8620237953250198, 1], atol=1e-15, rtol=0)
    middle = boundary.wilson(12, 24)
    assert middle[0] < 0.5 < middle[1]
    assert middle[0] + middle[1] == pytest.approx(1)


def test_truth_error_metrics_count_transitions_and_sustained_loss(boundary):
    clean = boundary.trajectory(np.zeros(100))
    assert clean["symbol_cycle_change_count"] == 0
    assert clean["settling_0p05_sample_suffix_symbol"] == 0
    assert not clean["sustained_32_symbol_unlock"]
    ramp = np.r_[np.zeros(10), np.full(40, 4.0)]
    changed = boundary.trajectory(ramp)
    assert changed["symbol_cycle_change_count"] == 1  # Not forty samples called forty slips.
    assert changed["sustained_32_symbol_unlock"]
    assert changed["settling_0p05_sample_suffix_symbol"] is None
    short = boundary.trajectory(np.r_[np.zeros(50), np.full(31, 4.0), np.zeros(20)])
    assert not short["sustained_32_symbol_unlock"]
    assert short["symbol_cycle_change_count"] == 2
    assert short["settling_0p05_sample_suffix_symbol"] == 81


def test_oracle_diagnostic_preserves_header_gate_and_one_bit_crc_failure(boundary):
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile

    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=boundary.audit.WORD, data_crc_seed=0x555555, pilot_interval=0
    )
    wire, _ = boundary.audit.packet("8PSK", 255)
    y = wire[40:].copy()
    good = boundary.diagnostic_no_carrier_refinement(y, profile, 255)
    assert good["status"] == "valid" and good["bit_errors"] == 0
    y[65] *= np.exp(1j * np.pi / 4)
    bad = boundary.diagnostic_no_carrier_refinement(y, profile, 255)
    assert bad["header_crc"] and not bad["data_crc"]
    assert bad["status"] == "bad_data_crc" and bad["bit_errors"] == 1
    assert (
        boundary.diagnostic_no_carrier_refinement(y[:40], profile, 255)["status"]
        == "truncated_header"
    )
