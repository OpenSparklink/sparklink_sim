"""Reproduce the frozen carrier experiment, with explicitly separate acceptance gates."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write

BUDGET_SHA = "7065fc14cc4a63ff6af35752e9d911cbb079ab72c739c0c4199a23eb7783a4a0"
SELECTION_SHA = "c17752cf940686e341eac65c1654c5192f1084ea26528feb0184bc7c657a12b1"


def frozen_sources(root):
    """Fail before receiving any IQ if frozen code or parameter identities differ."""
    assert sha(root / "predevelopment.json") == BUDGET_SHA
    assert sha(root / "frozen-selection.json") == SELECTION_SHA
    selection = json.loads((root / "frozen-selection.json").read_text())
    for key in ["sources", "supporting_scripts"]:
        for name, digest in selection[key].items():
            assert sha(Path(name)) == digest, f"Frozen source differs: {name}"


def identity(before, after):
    """Compare actual results and every upstream trace, excluding backend labels."""
    left = json.loads((before / "results.json").read_text())
    right = json.loads((after / "results.json").read_text())
    arrays = 0
    for a, b in zip(left, right, strict=True):
        for row in [a, b]:
            row.pop("timing_backend")
            row.pop("integer_backend")
        assert a == b, a["input"]["id"]
        filename = a["input"]["id"] + ".npz"
        with np.load(before / "evidence" / filename) as x, np.load(
                after / "evidence" / filename) as y:
            assert x.files == y.files
            for key in x.files:
                assert np.array_equal(x[key], y[key]), (filename, key)
                arrays += 1
    return dict(inputs=len(left), trace_arrays=arrays, exact=True)


def run(args):
    frozen_sources(args.evidence)
    args.output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ, PYTHONPATH=str(Path("src").resolve()))
    commands = []

    def execute(script, *options, python=sys.executable, expected=0):
        cmd = [python, "scripts/" + script, *map(str, options)]
        log = args.output / f"command-{len(commands):02}.log"
        with log.open("w") as stream:
            result = subprocess.run(cmd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        commands.append(dict(command=cmd, returncode=result.returncode,
                             expected=expected, log=log.name, log_sha256=sha(log)))
        write(args.output / "commands.json", commands)
        assert result.returncode == expected, (script, result.returncode, log)

    common = ["--inputs", args.evidence / "held-inputs", "--budget",
              args.evidence / "predevelopment.json", "--selection",
              args.evidence / "frozen-selection.json"]
    for arm, options in [("reference", []), ("bounded", ["--kappa", 2]),
                         ("fixed", ["--kappa", 2, "--fixed"])]:
        execute("audit_carrier_weights.py", *common, *options, "--output", args.output / arm)
    for arm in ["bounded", "fixed"]:
        execute("compare_carrier_weights.py", "--before", args.output / "reference", "--after",
                args.output / arm, "--output", args.output / (arm + "-comparison.json"))
    execute("audit_carrier_weight_numeric.py", "--before", args.output / "bounded", "--after",
            args.output / "fixed", "--budget", args.evidence / "predevelopment.json",
            "--output", args.output / "numeric")
    identical = {}
    for chunks in ["fixed", "random"]:
        directory = args.output / ("chunks-" + chunks)
        execute("audit_carrier_weights.py", *common, "--kappa", 2, "--chunks", chunks,
                "--output", directory)
        identical[chunks] = identity(args.output / "bounded", directory)
    if args.alternate_python:
        execute("audit_carrier_weights.py", *common, "--kappa", 2, "--output",
                args.output / "alternate", python=args.alternate_python)
        identical["alternate"] = identity(args.output / "bounded", args.output / "alternate")
    execute("model_carrier_workload.py", "--output", args.output / "workload", expected=1)
    report = dict(
        experiment_gates_passed=True, identical=identical,
        summary=json.loads((args.output / "bounded/summary.json").read_text()),
        numerical=json.loads((args.output / "numeric/numeric.json").read_text())["gates"],
        default_promotion=False, whole_system_accepted=False,
        prior_failures="RRC 2.348%>2%, local4 start error and old input range events unchanged",
        scope="fixed carrier experiment only; old-original/real-MAC evidence separately indexed",
    )
    write(args.output / "summary.json", report)
    print(json.dumps(report, indent=2))
    return 0  # Only the stated experiment gates, never full RX/hardware acceptance.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alternate-python")
    raise SystemExit(run(parser.parse_args()))
