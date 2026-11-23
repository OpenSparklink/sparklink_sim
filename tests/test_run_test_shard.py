"""Shared selection fails closed and preserves the real pytest exit status."""

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from scripts import run_test_shard as runner


def fixture_partition(tmp_path):
    folder = tmp_path / "tests"
    folder.mkdir()
    for name in ("a", "b", "new"):
        (folder / f"test_{name}.py").write_text("def test_case(): pass\n")
    manifest = tmp_path / "partition.json"
    data = dict(version=1, core_is_exact_complement=True,
                groups=dict(group1=["tests/test_a.py"], group2=["tests/test_b.py"]))
    manifest.write_text(json.dumps(data))
    return manifest, data


def test_new_unlisted_module_stays_in_the_core_complement(tmp_path):
    manifest, _ = fixture_partition(tmp_path)
    paths, _ = runner.shard_paths("core", root=tmp_path, manifest=manifest)
    ignored = {p.removeprefix("--ignore=") for p in paths[1:]}
    all_files = {str(p.relative_to(tmp_path)) for p in (tmp_path / "tests").glob("*.py")}
    assert all_files - ignored == {"tests/test_new.py"}
    assert runner.shard_paths("group1", root=tmp_path, manifest=manifest)[0] == [
        "tests/test_a.py"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "outside", "empty", "version"])
def test_bad_manifest_cannot_expand_to_a_default_pytest_run(tmp_path, bad):
    manifest, data = fixture_partition(tmp_path)
    if bad == "missing":
        data["groups"]["group1"] = ["tests/test_missing.py"]
    elif bad == "duplicate":
        data["groups"]["group2"] = data["groups"]["group1"]
    elif bad == "outside":
        data["groups"]["group1"] = ["../tests/test_a.py"]
    elif bad == "empty":
        data["groups"]["group1"] = []
    else:
        data["version"] = 0
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner.shard_paths("group1", root=tmp_path, manifest=manifest)


def test_unknown_group_and_symlink_escape_fail_before_pytest(tmp_path):
    manifest, data = fixture_partition(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        runner.shard_paths("missing", root=tmp_path, manifest=manifest)
    outside = tmp_path / "external.py"
    outside.write_text("def test_external(): pass\n")
    (tmp_path / "tests/test_escape.py").symlink_to(outside)
    data["groups"]["group1"] = ["tests/test_escape.py"]
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="existing"):
        runner.shard_paths("group1", root=tmp_path, manifest=manifest)


@pytest.mark.parametrize("coverage", [False, True])
def test_runner_propagates_failure_and_records_the_exact_selection(tmp_path, monkeypatch, coverage):
    manifest, _ = fixture_partition(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    calls = []

    def failed(command, *, cwd):
        calls.append((command, cwd))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(runner.subprocess, "run", failed)
    selected = tmp_path / "selected.json"
    monkeypatch.setattr("sys.argv", ["runner", "--shard", "group1", "--junitxml", "one.xml",
        "--selection-output", str(selected), *(["--coverage"] if coverage else [])])
    assert runner.main() == 7
    receipt = json.loads(selected.read_text())
    assert receipt["pytest_command"] == calls[0][0] and calls[0][1] == tmp_path
    assert ("--cov=nearlink_sdr" in calls[0][0]) is coverage
    assert ("-v" in calls[0][0]) is not coverage
    assert "tests/test_a.py" in calls[0][0] and "tests/test_b.py" not in calls[0][0]
    assert "--junitxml=one.xml" in calls[0][0]
    assert receipt["manifest_SHA256"] == runner.shard_paths("group1")[1]
    assert not receipt["whole_receiver_acceptance"]


def test_truncated_manifest_is_an_error_not_a_success(tmp_path):
    file = tmp_path / "bad.json"
    file.write_text('{"version":')
    with pytest.raises(json.JSONDecodeError):
        runner.shard_paths("core", root=Path(tmp_path), manifest=file)


def test_actual_workflow_matrices_and_coverage_reports_match_the_shared_manifest():
    data = json.loads(runner.MANIFEST.read_text())
    names = ["core", *data["groups"]]
    workflow = (runner.ROOT / ".github/workflows/ci.yml").read_text()
    matrices = re.findall(r"shard: \[([^\]]+)\]", workflow)
    assert len(matrices) == 2
    assert all([v.strip() for v in matrix.split(",")] == names for matrix in matrices)
    line = next(v for v in workflow.splitlines() if "coverage combine --keep " in v)
    assert re.findall(r"coverage-([a-z0-9]+)\.data", line) == names
    lines = [v for v in workflow.splitlines() if "python scripts/check_test_shards.py " in v]
    assert len(lines) == 2
    assert re.findall(r"report-([a-z0-9]+)\.xml", lines[0]) == names
    assert re.findall(r"coverage-tests-([a-z0-9]+)\.xml", lines[1]) == names
    for name in names:
        assert runner.shard_paths(name)[0]


@pytest.mark.parametrize("coverage", [False, True])
def test_receipt_aggregation_requires_actual_backend_selection_and_budget(
    tmp_path, monkeypatch, coverage,
):
    manifest, data = fixture_partition(tmp_path)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    base = dict(commit="source", polar_rust=False, crc_rust=False,
                stream_timing_rust=False, integer_sync_rust=False)
    names = ["core", *data["groups"]]
    prefix = "coverage-" if coverage else ""
    for name in names:
        (tmp_path / f"{prefix}backend-{name}.json").write_text(json.dumps(base))
        paths, sha = runner.shard_paths(name)
        report = f"coverage-tests-{name}.xml" if coverage else f"report-{name}.xml"
        command = ["python", "-m", "pytest", *paths, "--junitxml=" + report]
        if coverage:
            command.append("--cov=nearlink_sdr")
        selected = dict(shard=name, manifest_SHA256=sha, pytest_command=command)
        (tmp_path / f"{prefix}selection-{name}.json").write_text(json.dumps(selected))
        (tmp_path / f"unit-budget-{name}.json").write_text('{"passed":true}')
    assert runner.verify_receipts(base, directory=tmp_path, coverage=coverage) == names
    backend = tmp_path / f"{prefix}backend-group1.json"
    backend.write_text(json.dumps(base | dict(commit="stale")))
    with pytest.raises(ValueError, match="backend"):
        runner.verify_receipts(base, directory=tmp_path, coverage=coverage)
    backend.write_text(json.dumps(base))
    selected_path = tmp_path / f"{prefix}selection-group1.json"
    original = selected_path.read_text()
    selected_path.write_text(json.dumps(json.loads(original) | dict(manifest_SHA256="other")))
    with pytest.raises(ValueError, match="selection"):
        runner.verify_receipts(base, directory=tmp_path, coverage=coverage)
    selected_path.write_text(original)
    if not coverage:
        (tmp_path / "unit-budget-group1.json").write_text('{"passed":false}')
        with pytest.raises(ValueError, match="budget"):
            runner.verify_receipts(base, directory=tmp_path)
        (tmp_path / "unit-budget-group1.json").write_text('{"passed":true}')
    selected_path.unlink()
    with pytest.raises(FileNotFoundError):
        runner.verify_receipts(base, directory=tmp_path, coverage=coverage)
