"""Experiment guards keep original references, cost, and held isolation strict."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scripts import model_queued_tail as audit

from nearlink_sdr.phy.control_info import ControlInfoB1
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from tests.test_model_legal_tail_ceiling import row


def candidate():
    value = row()
    value["modes"]["original"]["fine_cycles"] = value["modes"]["finite"]["fine_cycles"]
    gate = CaptureTransport(CaptureTransportAssumptions(paths=1)).estimate(
        measured_worker=True, released_requests=True,
        retire_before_fine=True, retire_queued=True)["queued_retirement_gate"]
    value["comparison"]["queued_gate"] = dict(cost=gate, registered=True,
                                              prior_consumed_frontier=True)
    return value


def test_original_failure_extra_ports_and_fine_work_remain_failures(monkeypatch):
    value, before = candidate(), row()
    monkeypatch.setattr(audit.base, "qualify", lambda rows, baseline:
                        dict(passed=False, reasons=["structural_or_numerical_failure"]))
    assert "structural_or_numerical_failure" in audit.qualify([value], [before])["reasons"]
    monkeypatch.setattr(audit.base, "qualify", lambda rows, baseline:
                        dict(passed=True, reasons=[]))
    value["modes"]["original"]["fine_cycles"] = value["modes"]["finite"]["fine_cycles"]
    assert audit.qualify([value], [before])["passed"]
    changed = deepcopy(value)
    changed["modes"]["finite"]["fine_cycles"] += 1
    changed["comparison"]["queued_gate"]["cost"]["extra_IQ_RAM_DSP_or_sample_ports"] = 1
    result = audit.qualify([changed], [before])
    required = {"original_reference_fine_work_cap_failed", "queued_gate_cost_or_causality_failed"}
    assert required <= set(result["reasons"])
    assert not result["passed"]


def test_bad_experiment_choices_fail_before_input_load():
    with pytest.raises(ValueError, match="explicit"):
        audit.old.compare(Path("unused"), Path("unused"), {}, retire_queued=1)
    for enabled, queued in [(False, True), (True, 1)]:
        with pytest.raises(ValueError, match="pre-fine"):
            audit.old.reference.run_arm(None, None, None, enabled, retire_queued=queued)


@pytest.mark.parametrize("split", ["development", "validation"])
def test_prospective_configuration_lengths_fit_actual_CRC_without_generating_held_IQ(split):
    for ft, mcs, pilots, lengths, _, _ in audit.POLICY[split + "_specs"]:
        for n in lengths:
            cfg = audit.old.reference.shared.shared.configuration(ft, mcs, pilots, n)
            wire = n + cfg.crc_len // 8 if ft in (3, 4) else n
            assert wire <= 2047
            if ft in (3, 4):
                bits = ControlInfoB1(0, 0, 0, mcs, wire, 0, 0).pack(cfg.head_llid)
                assert ControlInfoB1.unpack(bits, cfg.head_llid).data_length == wire


@pytest.mark.parametrize("defect", ["rejected", "duplicate", "source"])
def test_four_distinct_freezes_and_new_development_are_required(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(audit, "sources", lambda: {})
    files = []
    for index in range(5):
        v = dict(source={}, policy=audit.POLICY, role="fixed-regression" if index else
                 "development", qualification=dict(passed=True), input_manifest_SHA256="fresh")
        if index:
            v["input_manifest_SHA256"] = audit.POLICY["fixed_manifest_SHA256"][index - 1]
        if defect == "rejected" and index == 0:
            v["qualification"]["passed"] = False
        if defect == "duplicate" and index == 4:
            v["input_manifest_SHA256"] = audit.POLICY["fixed_manifest_SHA256"][0]
        if defect == "source" and index == 4:
            v["source"] = {"wrong": "SHA"}
        file = tmp_path / f"freeze{index}.json"
        file.write_text(json.dumps(v))
        files.append(file)
    out = tmp_path / "uncreated"
    args = ["audit", "--output", str(out), "--split", "validation", "--freeze", str(files[0])]
    for file in files[1:]:
        args += ["--regression-freeze", str(file)]
    monkeypatch.setattr("sys.argv", args)
    with pytest.raises(SystemExit) as result:
        audit.main()
    assert result.value.code == 2 and not out.exists()


def test_actual_quiet_CLI_saves_all_cost_and_rejection_fields(tmp_path, monkeypatch):
    from tests.test_coherent_captured_frame import profile_case

    profile, *_ = profile_case()
    file = tmp_path / "quiet.npz"
    np.savez(file, iq=np.zeros(256, complex), quiet=np.full(1024, .002, complex))
    config = vars(profile.config).copy()
    config.pop("_mcs")
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps([dict(id=0, file=file.name,
        SHA256=audit.old.reference.shared.sha(file), config=config,
        head_backend=profile.head_backend, expected=[], negative=True, truth_diagnostic_only=[])]))
    out = tmp_path / "result"
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out), "--inputs", str(manifest)])
    assert audit.main() == 1
    decision = json.loads((out / "decision.json").read_text())
    assert not decision["qualification"]["passed"]
    assert decision["input_manifest_SHA256"] == audit.old.reference.shared.sha(manifest)
    result = json.loads((out / "stream0-finite.json").read_text())["result"]
    original = json.loads((out / "stream0-original.json").read_text())["result"]
    assert "queued_retirement_gate_cost" not in original
    assert result["queued_retirement_gate_cost"]["register_bits"] == 66
