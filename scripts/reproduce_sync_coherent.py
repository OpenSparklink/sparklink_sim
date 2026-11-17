"""Reproduce frozen coherent-sync gates; return1 for the retained resource rejection."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from audit_capture_hardware import sha, write
from reproduce_carrier_weights import identity

_ROOT = Path(__file__).resolve().parents[1]
_BUDGET_SHA = "35ce5fb3c9999a83b5948af0ef268a281a263c0b7e027546d1826d478b52630f"
_SELECTION_SHA = "21e1b5673ba3bffae1ba5bbb1bef30f6fa8cf0f55e72dff67fafcbbb3cf8e052"
_COMPARER_SHA = "d7d8c44cc5b16496bd1c1196b9ed77c47d495538ad4e86a4cc19f19c82e46c27"


def frozen_snapshot(evidence):
    """Refuse replacement parameters/code before reading IQ or creating results."""
    assert sha(evidence / "predevelopment.json") == _BUDGET_SHA
    assert sha(evidence / "frozen-selection.json") == _SELECTION_SHA
    selected = json.loads((evidence / "frozen-selection.json").read_text())
    assert sha(evidence / "nco-format-predevelopment.json") == selected["nco_format_sha256"]
    for group in ["sources", "supporting_scripts"]:
        for name, digest in selected[group].items():
            assert sha(_ROOT / name) == digest, name
    return selected


def run(args):
    selected = frozen_snapshot(args.evidence)
    compare = args.evidence.parent / "bounded-sync-candidates/compare-regression.py"
    if args.full:
        assert sha(compare) == _COMPARER_SHA, "Frozen old-regression comparer differs"
    args.output.mkdir(parents=True, exist_ok=False)
    alternate = str(Path(args.alternate_python).absolute())
    env = dict(os.environ, PYTHONPATH=str(_ROOT / "src"))
    commands = []

    def execute(script, *options, python=sys.executable, expected=0):
        cmd = [python, str(_ROOT / "scripts" / script), *map(str, options)]
        log = args.output / f"command-{len(commands):02}.log"
        with log.open("w") as stream:
            completed = subprocess.run(cmd, cwd=_ROOT, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT)
        commands.append(dict(command=cmd, returncode=completed.returncode, expected=expected,
                             log=log.name, log_sha256=sha(log)))
        write(args.output / "commands.json", commands)
        assert completed.returncode == expected, (script, completed.returncode, log)

    execute("check_receiver_backend.py", "--expected-rust", "yes", "--output",
            args.output / "backend-rust.json")
    execute("check_receiver_backend.py", "--expected-rust", "no", "--output",
            args.output / "backend-python.json", python=alternate)
    common = ["--inputs", args.evidence / "held-inputs", "--budget",
              args.evidence / "predevelopment.json", "--selection",
              args.evidence / "frozen-selection.json", "--candidates", 1, "--normalized-coarse"]
    confirm = ["--coherent-threshold", selected["coherent_threshold"]]
    for arm, options, python in [
        ("reference", [], sys.executable), ("coherent", confirm, sys.executable),
        ("fixed", [*confirm, "--fixed-nco"], sys.executable),
        ("chunks-fixed", [*confirm, "--chunks", "fixed"], sys.executable),
        ("chunks-random", [*confirm, "--chunks", "random"], sys.executable),
        ("python", confirm, alternate), ("python-fixed", [*confirm, "--fixed-nco"], alternate),
    ]:
        execute("audit_sync_candidates.py", *common, *options, "--output", args.output / arm,
                python=python)
    execute("compare_capture_hardware.py", "--before", args.output / "reference", "--after",
            args.output / "coherent", "--output", args.output / "functional.json")
    execute("compare_capture_hardware.py", "--before", args.output / "coherent", "--after",
            args.output / "fixed", "--output", args.output / "fixed-functional.json")
    exact = {name: identity(args.output / "coherent", args.output / name)
             for name in ["chunks-fixed", "chunks-random", "python"]}
    exact["python-fixed"] = identity(args.output / "fixed", args.output / "python-fixed")
    execute("audit_sync_coherent_numeric.py", "--before", args.output / "coherent", "--after",
            args.output / "fixed", "--budget", args.evidence / "predevelopment.json",
            "--output", args.output / "numeric")
    execute("model_sync_coherent_work.py", "--output", args.output / "resource", expected=1)
    resource = json.loads((args.output / "resource/model.json").read_text())
    assert resource["whole_system_accepted"] is False and len(resource["scenarios"]) == 3
    assert all(r["known_capture_partial_total_bits"] > r["memory_budget_bits"]
               for r in resource["scenarios"])
    if args.full:
        inputs = ["--evidence-root", args.evidence.parent, "--dataset", args.dataset,
                  "--quiet-dataset", args.quiet_dataset]
        execute("replay_acquisition_regression.py", *inputs, "--selection",
            args.evidence.parent / "rx-interference-capture/rrc-stream-correlation-selection.json",
            "--backend", "scaled", "--tone-radius", .96875, "--sync-method", "bounded-grid",
            "--sync-candidate-count", 1, "--normalized-coarse", "--sync-coherent-threshold", .45,
            "--output", args.output / "old-regression")
        execute(compare, "--before",
            args.evidence.parent / "bounded-sync-candidates/old-regression-q12-fixed",
            "--after", args.output / "old-regression", "--output", args.output / "old-paired")
        for arm, mode in [("real-mac", []), ("real-mac-fixed", ["--fixed-nco"])]:
            execute("replay_carrier_mac_real.py", *inputs, "--frozen-reference",
                args.evidence.parent / "rx-interference-capture/rrc-stream-original-reference",
                "--experiment", "coherent-sync", "--coherent-threshold", .45, *mode,
                "--output", args.output / arm)
    result = dict(experiment_commands_completed=True, exact_identities=exact,
                  functional_gates_passed=True, nco_node_numeric_gate_passed=True,
                  resource_gate_passed=False, whole_system_accepted=False,
                  default_promote=False, expected_returncode=1,
                  scope="fixed frozen IQ;resource failure retained,not an execution error")
    write(args.output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alternate-python", required=True)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--dataset", type=Path, default=Path(
        "/home/sanchuanhehe/Documents/opensparklink/sparklink-hw/datasets/sle-rf/ws63/20261115-rx2"))
    parser.add_argument("--quiet-dataset", type=Path, default=Path(
        "/home/sanchuanhehe/Documents/opensparklink/sparklink-hw/build/evidence/20261116-ws63-release-r2/ws63-rx2-20261115-r2"))
    raise SystemExit(run(parser.parse_args()))
