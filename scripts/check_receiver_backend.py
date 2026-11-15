"""Fail if a CI label differs from the loaded DSP backend; save actual dependency metadata."""

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path

from nearlink_sdr.common import crc, polar


def run(args):
    expected = args.expected_rust == "yes"
    versions = {}
    for package in (
        "numpy",
        "scipy",
        "pytest",
        "cryptography",
        "maturin",
        "nearlink_sdr_accel",
        "gmssl",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = dict(
        python=sys.version,
        executable=sys.executable,
        platform=platform.platform(),
        polar_rust=polar._HAS_RUST_ACCEL,
        crc_rust=crc._HAS_RUST_CRC,
        gmssl_available=importlib.util.find_spec("gmssl") is not None,
        dependencies=versions,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    assert result["polar_rust"] == expected and result["crc_rust"] == expected, result
    if args.require_sm3:
        assert result["gmssl_available"], "SM3 job must install gmssl; a skip is not verification"
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-rust", choices=["yes", "no"], required=True)
    parser.add_argument("--require-sm3", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
