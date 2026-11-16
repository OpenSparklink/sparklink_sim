"""Paired frozen negative-IQ costs; failures and slow interference are never omitted."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
from benchmark_capture_front import consume, equivalent

from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args):
    if args.repeats < 1 or args.chunk_size < 0:
        raise ValueError("Need positive repetitions and nonnegative chunk size")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(
        ["git", "show", args.baseline_ref + ":src/nearlink_sdr/phy/stream_rx.py"]
    )
    (out / "baseline-stream_rx.py").write_bytes(source)
    namespace = {}
    exec(compile(source, str(out / "baseline-stream_rx.py"), "exec"), namespace)
    classes = dict(baseline=namespace["PSKStreamReceiver"], candidate=PSKStreamReceiver)
    inputs = json.loads((args.inputs / "plan.json").read_text())["inputs"]
    plan = dict(
        inputs=inputs,
        baseline_ref=args.baseline_ref,
        driver_sha256=sha(Path(__file__)),
        source_sha256={str(p): sha(p) for p in Path("src").rglob("*.py")},
        rust_sources={
            str(p): sha(p)
            for p in [
                *Path("rust/src").rglob("*.rs"),
                Path("rust/Cargo.lock"),
                Path("rust/Cargo.toml"),
            ]
        },
        python=sys.version,
        platform=platform.platform(),
        dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        timing_scope="feed plus finish; independent quiet calibration initialization excluded",
        repetitions=args.repeats,
        chunk_size=args.chunk_size,
        budget="no false delivery/candidate; bounded cache; main quiet probe coverage <= input+4096",
    )
    (out / "frozen-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    with tarfile.open(out / "candidate-source.tar", "w") as archive:
        for p in [*plan["source_sha256"], *plan["rust_sources"], "pyproject.toml", "uv.lock"]:
            archive.add(p, arcname=p)
    rows = []
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    for meta in inputs:
        path = Path(meta["path"])
        assert sha(path) == meta["sha256"]
        with np.load(path) as d:
            x, q = d["iq"], d["quiet"]
        reference, measurements, states = None, [], {}
        for trial in range(args.repeats + 1):
            for name in list(classes) if trial % 2 == 0 else list(reversed(classes)):
                rx = classes[name](profile, q)
                begin, cpu_begin = time.perf_counter(), time.process_time()
                events, _ = consume(rx, x, args.chunk_size)
                elapsed, cpu_elapsed = time.perf_counter() - begin, time.process_time() - cpu_begin
                if reference is None:
                    reference = events
                try:
                    equivalent(reference, events)
                    assert not events, "Independent negative unexpectedly acquired"
                except AssertionError:
                    (out / (meta["id"] + "-failure.json")).write_text(
                        json.dumps(
                            dict(
                                input=meta,
                                arm=name,
                                trial=trial,
                                frames=[
                                    {k: v for k, v in e["frame"].items() if k != "payload"}
                                    for e in events
                                ],
                            ),
                            indent=2,
                        )
                        + "\n"
                    )
                    np.savez_compressed(
                        out / (meta["id"] + "-failure.npz"),
                        **{
                            f"{i}-{key}": event[key]
                            for i, event in enumerate(events)
                            for key in (
                                "symbols",
                                "symbol_centers_raw",
                                "timing_detector_error",
                                "timing_rate_ppm",
                            )
                        },
                    )
                    raise
                if hasattr(rx, "quiet_probe_samples"):
                    assert rx.quiet_probe_samples <= len(x) + rx.quiet_batch_samples
                states[name] = {
                    k: getattr(rx, k).copy()
                    for k in ("_filtered", "_energy", "_fir_state", "_energy_state")
                }
                if trial:
                    measurements.append(
                        dict(
                            arm=name,
                            trial=trial,
                            seconds=elapsed,
                            cpu_seconds=cpu_elapsed,
                            throughput_Msamples=len(x) / elapsed / 1e6,
                            candidates=len(events),
                            valid_deliveries=sum(e["frame"]["status"] == "valid" for e in events),
                            searches=rx.sync_searches,
                            timing_backend=rx.timing_backend,
                            integer_screen_backend=rx.integer_screen_backend,
                            stats=rx.capture_stats,
                            quiet_probe_samples=getattr(rx, "quiet_probe_samples", None),
                            quiet_probe_state_samples=getattr(
                                rx, "quiet_probe_state_samples", None
                            ),
                            peak_history_samples=rx.peak_history_samples,
                            state_array_bytes=sum(
                                v.nbytes for v in vars(rx).values() if isinstance(v, np.ndarray)
                            ),
                        )
                    )
        np.savez_compressed(
            out / (meta["id"] + "-states.npz"),
            **{a + "-" + k: v for a, t in states.items() for k, v in t.items()},
        )
        for key in states["baseline"]:
            np.testing.assert_array_equal(
                states["baseline"][key], states["candidate"][key], err_msg=meta["id"] + ":" + key
            )
        medians = {
            a: float(np.median([r["seconds"] for r in measurements if r["arm"] == a]))
            for a in classes
        }
        row = dict(
            meta,
            measurements=measurements,
            median_seconds=medians,
            ratio=medians["candidate"] / medians["baseline"],
            final_states_exact=True,
        )
        rows.append(row)
        (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert all(sha(Path(p)) == h for p, h in plan["source_sha256"].items())
    assert all(sha(Path(p)) == h for p, h in plan["rust_sources"].items())
    assert sha(Path(__file__)) == plan["driver_sha256"]
    summary = {}
    for kind in sorted({r["kind"] for r in rows}):
        group = [r for r in rows if r["kind"] == kind]
        samples = sum(r["samples"] for r in group)
        summary[kind] = dict(
            samples=samples,
            inputs=len(group),
            median_seconds={a: sum(r["median_seconds"][a] for r in group) for a in classes},
            throughput_Msamples={
                a: samples / sum(r["median_seconds"][a] for r in group) / 1e6 for a in classes
            },
            max_stream_cost_ratio=max(r["ratio"] for r in group),
            candidates=0,
            false_deliveries=0,
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "manifest.json").write_text(
        json.dumps({p.name: sha(p) for p in out.iterdir() if p.is_file()}, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-ref", required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--chunk-size", type=int, default=4096)
    run(p.parse_args())
