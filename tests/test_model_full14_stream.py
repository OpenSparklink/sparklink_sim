"""Independent experiment qualification counterexamples;no new RF set here."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scripts import model_full14_stream as model


def test_all_new_configs_validate_before_any_iq_is_generated():
    policy = json.loads(model.POLICY.read_text())
    for split, count in [("development", 29), ("validation", 35)]:
        rows = model.specifications(policy, split)
        assert len(rows) == count and len({r["id"] for r in rows}) == count
        assert all(model.profile(r).length_contract == "standard" for r in rows)
        assert {r["config"]["frame_type"] for r in rows} == {2, 3, 4}
        assert {r["config"]["mcs_index"] for r in rows} == {8, 12}
        assert {r["config"]["pilot_interval"] for r in rows} == {0, 4, 8, 16}


def test_new_validation_rejects_unqualified_or_changed_sources_before_creating_input(tmp_path):
    out, dev = tmp_path/"new-held", tmp_path/"dev"
    with pytest.raises(ValueError, match="qualified development"):
        model.run(out, "validation")
    dev.mkdir()
    for qualified in [False, True]:
        (dev/"qualification.json").write_text(json.dumps(dict(qualified=qualified, sources={})))
        with pytest.raises(ValueError, match="Unqualified or changed"):
            model.run(out, "validation", dev)
    assert not out.exists()
    (dev/"qualification.json").write_text(json.dumps(dict(qualified=True, sources=model.sources())))
    with pytest.raises(ValueError, match="qualified fixed regression"):
        model.run(out, "validation", dev)
    assert not out.exists()


@pytest.mark.parametrize("bad_crc", [False, True])
def test_crc_content_mcs_and_length_are_independent_not_a_fixed_success_flag(monkeypatch, bad_crc):
    p = model.profile(model.specifications(json.loads(model.POLICY.read_text()), "development")[0])
    wanted, received = b"\x80", b"\x81"
    bits = np.unpackbits(np.frombuffer(wanted if bad_crc else received, np.uint8))
    monkeypatch.setattr(model, "diagnostic_bits", lambda e, profile: bits)
    event = dict(sync_start_raw=37., symbols=np.array([1+0j]), symbol_centers_raw=np.array([37.]),
        cfo_hz=0., frame=dict(status="bad_data_crc" if bad_crc else "valid", header_crc=True,
        data_crc=not bad_crc, length_ok=True, mcs=8, header=dict(data_length=1),
        payload=None if bad_crc else received))
    meta = dict(expected=[dict(sync_start=37., step=4., expected_hex=wanted.hex(), mcs=8,
                              wire_length=1, ppm=0, cfo_Hz=0)])
    a = dict(events=[event], packets=[] if bad_crc else [SimpleNamespace(mac_payload=received)],
             before_EOF=0 if bad_crc else 1)
    result = model.evaluate(a, p, meta)
    assert not result["correct"] and result["captured"] == [0]
    assert result["header_crc"] == result["expected_mcs_ok"] == result["length_ok"] == 1
    assert result["data_crc"] == (0 if bad_crc else 1)
    assert result["bit_errors"] == (0 if bad_crc else 1)
    assert result["content_exact"] == (1 if bad_crc else 0)
    assert result["false_MAC"] == (0 if bad_crc else 1)


def test_adoption_keeps_original_success_and_relative_trajectory_budget(tmp_path):
    policy = json.loads(model.POLICY.read_text())
    for name, centers in [("old", [37., 41., 45.]), ("new", [37., 41., 45.250001])]:
        np.savez_compressed(tmp_path/(name+".npz"), **{"0-symbol_centers_raw": np.array(centers)})

    def result(name):
        return dict(correct=[0], frames=[dict(target=0, sync_start=37.)], false_candidates=0,
                    false_MAC=0, duplicate_MAC=0, trace_files=[name+".npz"],
                    history_limit=384, frame_symbol_capacity=8192, frame_array_host_bytes=327680,
                    existing_search_paths=2,
                    RRC_numeric=dict(passed=True))

    row = dict(id="counterexample", positive=True, arms={
        "reference": result("old"), "word-full12-c18": result("old"),
        "word-full14-c18": result("new")})
    assert not model.acceptance([row], tmp_path, policy)
    failures = json.loads((tmp_path/"adoption-failures.json").read_text())
    assert len(failures) == 2 and all(r["reason"] == "relative_trajectory_budget" for r in failures)
    row["arms"]["word-full14-c18"] = result("old") | dict(correct=[])
    assert not model.acceptance([row], tmp_path, policy)
    failures = json.loads((tmp_path/"adoption-failures.json").read_text())
    assert len(failures) == 2 and all(r["reason"] == "original_success_lost" for r in failures)
