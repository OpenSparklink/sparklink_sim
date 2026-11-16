"""Replay immutable low-amplitude and coded IQ evidence; count every acquisition stage."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    original = args.evidence_root / "rx-chain-hardening"
    old_source = subprocess.check_output(
        ["git", "show", "c3e2cbb:src/nearlink_sdr/phy/stream_rx.py"]
    )
    path = out / "baseline-stream.py"
    path.write_bytes(old_source)
    spec = importlib.util.spec_from_file_location("capture_baseline", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cases = []
    for row in json.loads((original / "low-snr-boundary-final/results.json").read_text()):
        if row["family"] != "noise-cfo-sro-amplitude":
            continue
        path = original / "low-snr-boundary-final" / row["input"]["path"]
        assert sha(path) == row["input"]["sha256"]
        profile = PSKReceiveProfile.ws63_ft2_a3(
            sync_word=0x6A28C4BD9D8EB6C5, data_crc_seed=0x555555, pilot_interval=0
        )
        cases.append(
            (
                row["id"],
                path,
                profile,
                np.packbits(prbs9(2040), bitorder="little").tobytes(),
                "amplitude",
            )
        )
    for row in json.loads((original / "soft-capture-python/frozen-plan.json").read_text())[
        "inputs"
    ]:
        path = Path(row["file"])
        assert sha(path) == row["sha256"]
        profile = PSKReceiveProfile.ordinary(
            TxConfig(
                mcs_index=row["mcs"], symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456
            )
        )
        with np.load(path) as data:
            expected = data["expected"].tobytes()
        cases.append((row["id"], path, profile, expected, "coded"))
    plan = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        source_sha256={str(p): sha(p) for p in Path("src").rglob("*.py")},
        driver_sha256=sha(Path(__file__)),
        inputs=[dict(id=c[0], file=str(c[1]), sha256=sha(c[1])) for c in cases],
    )
    write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for ident, path, profile, expected, group in cases:
        with np.load(path) as data:
            x, q = data["iq"], data["quiet"]
        begin = time.perf_counter()
        rx = PSKStreamReceiver(profile, q)
        events = rx.feed(x) + rx.finish()
        seconds = time.perf_counter() - begin
        old = module.PSKStreamReceiver(profile, q)
        previous = old.feed(x) + old.finish()
        assert len(events) == len(previous), ident
        arrays = {}
        for i, (a, b) in enumerate(zip(events, previous, strict=True)):
            assert (
                a["frame"] == b["frame"]
                and a["emitted_after_samples"] == b["emitted_after_samples"]
            )
            for key in (
                "symbols",
                "symbol_centers_raw",
                "timing_rate_ppm",
                "timing_detector_error",
            ):
                np.testing.assert_array_equal(a[key], b[key])
                arrays[f"{i}-{key}"] = a[key]
        np.savez_compressed(out / "evidence" / (ident + ".npz"), **arrays)
        reason = (
            "captured"
            if events
            else "energy_gate"
            if not rx.sync_searches
            else "no_ready_qualified_correlation"
        )
        rows.append(
            dict(
                id=ident,
                group=group,
                input=dict(file=str(path), sha256=sha(path)),
                samples=len(x),
                seconds=seconds,
                reason=reason,
                stats=rx.capture_stats,
                search_calls=rx.sync_searches,
                search_calls_per_Msample=rx.sync_searches * 1e6 / len(x),
                candidates_per_Msample=len(events) * 1e6 / len(x),
                correct=sum(e["frame"]["payload"] == expected for e in events),
                frames=[{k: v for k, v in e["frame"].items() if k != "payload"} for e in events],
                state_array_bytes=sum(
                    v.nbytes for v in vars(rx).values() if isinstance(v, np.ndarray)
                ),
                baseline_decisions_and_trajectories_exact=True,
            )
        )
    write(out / "results.json", rows)
    from collections import Counter

    write(
        out / "summary.json",
        {
            g: dict(
                inputs=sum(r["group"] == g for r in rows),
                reasons=dict(Counter(r["reason"] for r in rows if r["group"] == g)),
                correct=sum(r["correct"] for r in rows if r["group"] == g),
            )
            for g in ("amplitude", "coded")
        },
    )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
