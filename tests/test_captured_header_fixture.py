"""Fixed CI bytes preserve original RF regression inputs,including container identity."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scripts import model_captured_header as model

ROOT = Path(__file__).parent / "data/captured-header-calendar"
MANIFEST_SHA = "cdda54e1ac97aeb7a680f3c44bf5e724c892bd9d3b29d15ad337f90258aa12cf"
PROVENANCE_SHA = "ea307fb6e371d40eb71d660ac16508ae2ead8430fe87bdd8b0132f736813e300"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def test_selected_fixed_rows_have_explicit_original_and_subset_identity():
    assert sha((ROOT / "inputs.json").read_bytes()) == MANIFEST_SHA
    assert sha((ROOT / "provenance.json").read_bytes()) == PROVENANCE_SHA
    ledger = json.loads((ROOT / "provenance.json").read_text())
    rows = json.loads((ROOT / "inputs.json").read_text())
    assert [r["id"] for r in rows] == ledger["selection"]["ids"] == [3, 6]
    assert ledger["original"]["manifest_SHA256"] == (
        "0f9c1c0a2a88a48c768563380306bcb3f109fe2856e5acfa114a9ea609cf5c73"
    )
    assert ledger["original"]["frozen_index_SHA256"] == (
        "f5a29c9851c92687e3a0dec8e016aa60e2d54606d8d98401261eaf150e6df129"
    )
    assert ledger["subset_manifest_SHA256"] == MANIFEST_SHA
    assert not ledger["new_RF_holdout"] and not ledger["receiver_and_parameters_changed"]
    for row, frozen in zip(rows, ledger["files"], strict=True):
        assert (
            sha(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
            == (frozen["canonical_original_row_SHA256"])
        )


@pytest.mark.parametrize("ident", [3, 6])
def test_IQ_containers_and_all_NPY_bytes_match_original_frozen_input(ident):
    ledger = json.loads((ROOT / "provenance.json").read_text())
    row = next(r for r in json.loads((ROOT / "inputs.json").read_text()) if r["id"] == ident)
    frozen = next(r for r in ledger["files"] if r["id"] == ident)
    raw = (ROOT / row["file"]).read_bytes()
    assert sha(raw) == row["SHA256"] == frozen["SHA256"]
    assert len(raw) == frozen["bytes"]
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        assert {name: sha(z.read(name)) for name in z.namelist()} == frozen["NPY_members_SHA256"]
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        assert set(data.files) == {"iq", "quiet"}
        assert data["iq"].ndim == data["quiet"].ndim == 1
        assert np.isfinite(data["iq"]).all() and np.isfinite(data["quiet"]).all()


@pytest.mark.parametrize("change", ["bit", "recompression"])
def test_changed_or_recompressed_IQ_is_rejected_before_any_receive(change, tmp_path, monkeypatch):
    row = json.loads((ROOT / "inputs.json").read_text())[0]
    raw = (ROOT / row["file"]).read_bytes()
    path = tmp_path / row["file"]
    if change == "bit":
        modified = bytearray(raw)
        modified[len(modified) // 2] ^= 1
        path.write_bytes(modified)
    else:
        with (
            zipfile.ZipFile(io.BytesIO(raw)) as a,
            zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as b,
        ):
            for name in a.namelist():
                b.writestr(name, a.read(name))
        with zipfile.ZipFile(io.BytesIO(raw)) as a, zipfile.ZipFile(path) as b:
            assert a.namelist() == b.namelist()
            assert all(a.read(name) == b.read(name) for name in a.namelist())
    assert sha(path.read_bytes()) != row["SHA256"]
    monkeypatch.setattr(
        model.reference, "run_arm", lambda *a, **k: pytest.fail("Receive before SHA")
    )
    with pytest.raises(ValueError, match="Frozen IQ path/hash"):
        model.compare(tmp_path / "unused-output", tmp_path, row)
    assert not (tmp_path / "unused-output").exists()


def test_changed_valid_JSON_manifest_is_not_rebound_to_a_new_reference(tmp_path, monkeypatch):
    (tmp_path / "inputs.json").write_bytes((ROOT / "inputs.json").read_bytes() + b" ")
    monkeypatch.setattr(
        "sys.argv",
        [
            "model_captured_header",
            "--input-dir",
            str(tmp_path),
            "--manifest-sha256",
            MANIFEST_SHA,
            "--output",
            str(tmp_path / "unused-output"),
        ],
    )
    with pytest.raises(SystemExit) as failed:
        model.main()
    assert failed.value.code == 2 and not (tmp_path / "unused-output").exists()


def test_CI_uses_frozen_subset_and_keeps_real_resource_failure_assertions():
    workflow = (ROOT.parents[2] / ".github/workflows/ci.yml").read_text()
    step = workflow.split("name: Preserve actual word-head wait and dense source queue failure", 1)[
        1
    ]
    step = step.split("\n      - name:", 1)[0]
    assert "--input-dir tests/data/captured-header-calendar" in step
    assert "--manifest-sha256 " + MANIFEST_SHA in step
    assert "--input-dir receiver-diagnostics/retired-capture-development" not in step
    assert 'test "$status" -eq 1' in step and 'a["original_payload_losses"] == [1]' in step
    assert 'a["resource_drop_deltas"]["queue_drops"] == 2' in step
