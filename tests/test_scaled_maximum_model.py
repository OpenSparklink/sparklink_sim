"""Frozen-source guards and independent standard fields;no RF outcome chosen by CRC."""

import json
import sys

import numpy as np
import pytest
from scripts import model_scaled_maximum as model

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from tests.maximum_reference import channel, packet
from tests.test_acquisition_maximum import events


@pytest.mark.parametrize("changed", [None, "policy", "sources"])
def test_validation_guard_precedes_generation(tmp_path, monkeypatch, changed):
    output = tmp_path / "validation"
    args = ["model_scaled_maximum", "--output", str(output), "--split", "validation"]
    if changed is not None:
        frozen = dict(policy=dict(model.POLICY), sources=model.sources())
        frozen[changed][next(iter(frozen[changed]))] = "changed"
        file = tmp_path / "freeze.json"
        file.write_text(json.dumps(frozen))
        args += ["--freeze", str(file)]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(model, "numeric", lambda *_: pytest.fail("Input generated before freeze"))
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not output.exists()


@pytest.mark.parametrize("frame,mcs,pilot", [(2, 8, 0), (3, 12, 8), (4, 10, 4)])
@pytest.mark.parametrize("family", [0, 1, 2, 3])
def test_independent_mapper_preserves_fields_crc_and_chosen_payload(frame, mcs, pilot, family):
    cfg = model.configuration(frame, mcs, pilot, family)
    profile = PSKReceiveProfile.ordinary(cfg)
    expected = model.payload(23 + family * 4, family)
    wire = packet(profile, expected)
    row = decode_psk_frame(wire[40:], profile)
    assert row["header_crc"] and row["data_crc"] and row["length_ok"]
    assert row["mcs"] == mcs and row["payload"] == expected


def test_calendar_retains_initialization_loss_and_finite_backpressure():
    trace = [(tag, *old, *new, maximum) for tag, old, new, maximum in events(125, 600)]
    for store in ["register2r2w", "serial1r1w"]:
        open_row = model.calendar(trace, 200, store, False)
        assert open_row["component_continuous_pass"] and len(open_row["emitted"]) == 512
        blocked = model.calendar(trace, 200, store, True)
        assert blocked["failure"] and blocked["resource_rejected_tags"]
        assert blocked["difference_history_RRC_score_cycles"] is None
        startup = model.calendar(trace, 200, store, False, from_reset=True)
        assert startup["component_continuous_pass"] == (store == "register2r2w")
        assert open_row["initialization_write_counter_observation"] == (
            644 if store == "serial1r1w" else 0
        )
    assert not model.calendar(trace, 100, "serial1r1w", False)["component_continuous_pass"]


def test_source_and_representation_costs_are_not_full_receiver_credit():
    sources = model.sources()
    assert "tests/maximum_reference.py" in sources and "scripts/rx_timing_audit.py" in sources
    for terms in [31, 61, 125]:
        for store in ["register2r2w", "serial1r1w"]:
            cost = model.ExponentMaximumService(terms, counter_store=store).estimate()
            assert cost["research_cap_pass"] and cost["LUT"] is cost["Fmax"] is None
            assert cost["borrowed_history_reads_per_sample"] == 2
            assert cost["full_scaled_bank_cycles"] is cost["full_RX_cycles"] is None


def test_noise_injection_has_matching_quiet_calibration_and_independent_signal_power():
    profile = PSKReceiveProfile.ordinary(model.configuration(2, 8, 0, 1))
    wire = packet(profile, model.payload(31, 2))
    arguments = dict(offset=0.137, ppm=75, cfo=31000, prefix=19, amplitude=0.13, seed=381999)
    clean, _, t, step, _ = channel(wire, family="clean", **arguments)
    noisy, quiet, other_t, other_step, metadata = channel(
        wire, family="noise", snr_db=10, **arguments
    )
    sigma = metadata["noise_std_per_component"]
    assert t == other_t and step == other_step
    assert metadata["quiet_std_per_component"] == sigma
    assert 0.85 < float(np.mean(abs(quiet) ** 2)) / (2 * sigma**2) < 1.15
    assert 0.90 < float(np.mean(abs(noisy - clean) ** 2)) / (2 * sigma**2) < 1.10
