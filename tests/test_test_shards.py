"""All independent shards must be present, disjoint, successful and below120seconds."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/check_test_shards.py"


def report(path, identities, *, seconds=1, failures=0, errors=0, declared=None):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests=str(len(identities) if declared is None
                                                     else declared), time=str(seconds),
                          failures=str(failures), errors=str(errors), skipped="0")
    for name in identities:
        ET.SubElement(suite, "testcase", classname="independent_suite", name=name)
    ET.ElementTree(root).write(path)
    return path


def run(directory, reports):
    output = directory / "budget.json"
    result = subprocess.run([sys.executable, str(SCRIPT), *map(str, reports),
                             "--output", str(output)], capture_output=True, check=False)
    return result, json.loads(output.read_text()) if output.exists() else None


@pytest.mark.parametrize("shards", [2, 3])
def test_two_and_three_complete_reports_preserve_each_budget_and_union(tmp_path, shards):
    paths = [report(tmp_path / f"{n}.xml", [f"case{n}a", f"case{n}b"], seconds=119.999)
             for n in range(shards)]
    result, data = run(tmp_path, paths)
    assert result.returncode == 0 and data["passed"]
    assert data["unique_cases"] == shards * 2 and len(data["reports"]) == shards
    assert data["walltime_budget_seconds"] == 120 and data["walltimes_not_summed"]
    assert data["duplicate_cases"] == [] and not data["receiver_acceptance"]


def test_duplicate_between_first_and_third_report_fails_even_when_second_disjoint(tmp_path):
    paths = [report(tmp_path / f"{n}.xml", [name])
             for n, name in enumerate(["repeated", "other", "repeated"])]
    result, data = run(tmp_path, paths)
    assert result.returncode == 1 and not data["passed"]
    assert data["duplicate_cases"] == [["independent_suite", "repeated"]]


@pytest.mark.parametrize("bad", [dict(seconds=120), dict(seconds=-1), dict(failures=1),
                                 dict(errors=1), dict(declared=2)])
def test_any_invalid_third_report_remains_failure(tmp_path, bad):
    paths = [report(tmp_path / "a.xml", ["a"]), report(tmp_path / "b.xml", ["b"]),
             report(tmp_path / "c.xml", ["c"], **bad)]
    result, data = run(tmp_path, paths)
    assert result.returncode == 1 and not data["passed"]
    assert data["unique_cases"] == 3


@pytest.mark.parametrize("bad", ["missing", "empty", "malformed", "one_only"])
def test_missing_or_invalid_reports_cannot_be_counted_as_pass(tmp_path, bad):
    a = report(tmp_path / "a.xml", ["a"])
    b = tmp_path / "b.xml"
    if bad == "empty":
        report(b, [])
    elif bad == "malformed":
        b.write_text("<testsuites/>")
    result, data = run(tmp_path, [a] if bad == "one_only" else [a, b])
    assert result.returncode != 0 and (data is None or not data["passed"])
