"""One shared whole-file pytest partition for unit and coverage jobs.

Missing/duplicate/outside paths fail before pytest. Core remains the exact
complement,so new tests cannot silently disappear from the complete suite.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("ci_test_shards.json")


def shard_paths(shard, *, root=None, manifest=None):
    root = ROOT if root is None else root
    manifest = MANIFEST if manifest is None else manifest
    snapshot = manifest.read_bytes()
    data = json.loads(snapshot)
    groups = data["groups"]
    if (data["version"] != 1 or data["core_is_exact_complement"] is not True
            or not isinstance(groups, dict) or not groups):
        raise ValueError("Need version1 and a nonempty exact-complement partition")
    seen = set()
    for name, files in groups.items():
        if not re.fullmatch(r"group[1-9][0-9]*", name) or not isinstance(files, list) or not files:
            raise ValueError("Need named nonempty whole-file groups")
        for file in files:
            path = Path(file)
            if (path.is_absolute() or ".." in path.parts or path.parts[0] != "tests"
                    or path.suffix != ".py" or not (root / path).is_file()
                    or not (root / path).resolve().is_relative_to((root / "tests").resolve())):
                raise ValueError("Need existing project test files: " + str(file))
            if file in seen:
                raise ValueError("Duplicate test file: " + file)
            seen.add(file)
    if shard != "core" and shard not in groups:
        raise ValueError("Unknown test shard: " + shard)
    paths = (["tests/", *["--ignore=" + f for f in sorted(seen)]] if shard == "core"
             else groups[shard])
    return paths, hashlib.sha256(snapshot).hexdigest()


def verify_receipts(base, *, directory=None, coverage=False):
    """Reject missing,stale or wrong-backend shard artifacts before aggregation."""
    directory = Path.cwd() if directory is None else directory
    names = ["core", *json.loads(MANIFEST.read_text())["groups"]]
    for shard in names:
        prefix = "coverage-" if coverage else ""
        actual = json.loads((directory / f"{prefix}backend-{shard}.json").read_text())
        for key in ("commit", "polar_rust", "crc_rust", "stream_timing_rust", "integer_sync_rust"):
            if actual[key] != base[key]:
                raise ValueError(f"Wrong shard backend or commit: {shard}/{key}")
        selected = json.loads((directory / f"{prefix}selection-{shard}.json").read_text())
        paths, sha = shard_paths(shard)
        command = selected["pytest_command"]
        report = f"coverage-tests-{shard}.xml" if coverage else f"report-{shard}.xml"
        if (selected["shard"] != shard or selected["manifest_SHA256"] != sha
                or command[1:3] != ["-m", "pytest"] or command[3:3 + len(paths)] != paths
                or ("--cov=nearlink_sdr" in command) is not coverage
                or "--junitxml=" + report not in command):
            raise ValueError("Wrong shard selection or manifest: " + shard)
        if not coverage:
            budget = json.loads((directory / f"unit-budget-{shard}.json").read_text())
            if budget["passed"] is not True:
                raise ValueError("Actual unit shard budget failed: " + shard)
    return names


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--shard", required=True)
    parser.add_argument("--junitxml", required=True, type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--coverage", action="store_true")
    args = parser.parse_args()
    paths, sha = shard_paths(args.shard)
    command = [sys.executable, "-m", "pytest", *paths]
    if not args.coverage:
        command += ["-v", "--tb=short"]
    command += ["--junitxml=" + str(args.junitxml), "--durations=20"]
    if args.coverage:
        command += ["--cov=nearlink_sdr", "--cov-report=", "--cov-fail-under=0"]
    selection = dict(shard=args.shard, manifest_SHA256=sha, pytest_command=command,
                     core_is_exact_complement=True, whole_receiver_acceptance=False)
    if args.selection_output is not None:
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps(selection), flush=True)
    # Never convert a pytest exception,missing dependency or failing test to success.
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
