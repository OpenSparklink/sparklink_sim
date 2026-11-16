"""Paired receiver-module benchmark on immutable development IQ; no profiler in timings."""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
import tracemalloc
import types
from pathlib import Path

import numpy as np
from profile_rx_stream import load, number

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL
from nearlink_sdr.phy import psk_fields
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile

FIELDS = {
    "symbols": 1e-12,
    "symbol_centers_raw": 1e-10,
    "timing_rate_ppm": 1e-8,
    "timing_detector_error": 1e-12,
}


def equivalent(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b, strict=True):
        assert x["frame"] == y["frame"]
        assert x["emitted_after_samples"] == y["emitted_after_samples"]
        for key, tolerance in FIELDS.items():
            np.testing.assert_allclose(x[key], y[key], atol=tolerance, rtol=0)


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((args.original / "frozen-plan.json").read_text())
    source = subprocess.check_output(
        ["git", "show", args.baseline_ref + ":src/nearlink_sdr/phy/stream_rx.py"]
    )
    (out / "baseline-stream_rx.py").write_bytes(source)
    namespace = {}
    exec(compile(source, str(out / "baseline-stream_rx.py"), "exec"), namespace)
    if args.baseline_sync:
        sync_source = subprocess.check_output(
            ["git", "show", args.baseline_ref + ":src/nearlink_sdr/phy/psk_sync.py"]
        )
        path = out / "baseline-psk_sync.py"
        path.write_bytes(sync_source)
        module = types.ModuleType("capture_benchmark_baseline_sync")
        sys.modules[module.__name__] = module
        exec(compile(sync_source, str(path), "exec"), module.__dict__)
        namespace["search_psk_sync"] = module.search_psk_sync
    arms = {"baseline": namespace["PSKStreamReceiver"], "candidate": PSKStreamReceiver}
    current_demapper = psk_fields.PSKDemodulator
    demappers = dict.fromkeys(arms, current_demapper)
    if args.baseline_demapper:
        psk_source = subprocess.check_output(
            ["git", "show", args.baseline_ref + ":src/nearlink_sdr/phy/psk.py"]
        )
        path = out / "baseline-psk.py"
        path.write_bytes(psk_source)
        module = types.ModuleType("capture_benchmark_baseline_psk")
        sys.modules[module.__name__] = module
        exec(compile(psk_source, str(path), "exec"), module.__dict__)
        demappers["baseline"] = module.PSKDemodulator
    plan = dict(
        baseline_receiver_ref=args.baseline_ref,
        baseline_receiver_sha256=hashlib.sha256(source).hexdigest(),
        shared_dependencies="All modules except stream_rx and optional psk_sync use current source",
        baseline_sync=args.baseline_sync,
        baseline_demapper=args.baseline_demapper,
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        sources={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("src").rglob("*.py")
        },
        rust_sources={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                *Path("rust/src").rglob("*.rs"),
                Path("rust/Cargo.toml"),
                Path("rust/Cargo.lock"),
            ]
        },
        python=platform.python_version(),
        rust_polar_crc=_HAS_RUST_ACCEL,
        dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        tolerance=FIELDS,
        repeats=args.repeats,
        input_rate_hz=16000000,
        parameters=dict(
            energy_multiplier=2,
            integer_screen=0.6,
            quantum=32,
            subdivisions=8,
            bandwidth=0.003,
            max_rate_ppm=500,
        ),
        inputs=[],
        calibration=[],
    )
    results = []
    for ident in args.sample_id:
        assert ident.endswith("rx-00"), "Development captures only"
        meta = next(v for v in frozen["real_inputs"] if v["sample_id"] == ident)
        quiet = [v for v in frozen["quiet_inputs"] if v["sample_id"] in meta["quiet_control_ids"]]
        plan["inputs"].append(meta)
        plan["calibration"].extend(quiet)
        x, q = load(args.dataset, meta), np.concatenate([load(args.dataset, v) for v in quiet])
        profile = PSKReceiveProfile.ws63_ft2_a3(
            sync_word=number(frozen["sync_word"]),
            data_crc_seed=number(frozen["crc_seed"]),
            pilot_interval=0,
        )
        rows, reference, traces, costs = [], None, {}, {}
        for repeat in range(args.repeats + 1):
            for name in list(arms) if repeat % 2 == 0 else list(reversed(arms)):
                psk_fields.PSKDemodulator = demappers[name]
                begin = time.perf_counter()
                rx = arms[name](profile, q)
                init = time.perf_counter() - begin
                begin = time.perf_counter()
                cpu_begin = time.process_time()
                events = rx.feed(x) + rx.finish()
                elapsed = time.perf_counter() - begin
                cpu_elapsed = time.process_time() - cpu_begin
                if reference is None:
                    reference = events
                equivalent(reference, events)
                for i, event in enumerate(events):
                    for key in FIELDS:
                        traces[f"{name}-{i}-{key}"] = event[key]
                costs[name] = dict(
                    timing_backend=getattr(rx, "timing_backend", "python"),
                    integer_screen_backend=getattr(rx, "integer_screen_backend", "python"),
                    stats=rx.capture_stats,
                    state_array_bytes=sum(
                        v.nbytes for v in vars(rx).values() if isinstance(v, np.ndarray)
                    ),
                    peak_history_samples=rx.peak_history_samples,
                    max_frame_symbols=rx.max_frame_symbols,
                )
                if repeat:
                    latency = [
                        e["emitted_after_samples"] - (e["symbol_centers_raw"][-1] + 1)
                        for e in events
                    ]
                    rows.append(
                        dict(
                            arm=name,
                            repeat=repeat,
                            seconds=elapsed,
                            cpu_seconds=cpu_elapsed,
                            initialization_seconds=init,
                            samples_per_second=len(x) / elapsed,
                            valid_packets=sum(e["frame"]["status"] == "valid" for e in events),
                            availability_delay_samples=np.percentile(
                                latency, [0, 50, 95, 100]
                            ).tolist(),
                        )
                    )
        memory, stream_memory = {}, {}
        for name, cls in arms.items():
            psk_fields.PSKDemodulator = demappers[name]
            tracemalloc.start()
            rx = cls(profile, q)
            rx.feed(x)
            rx.finish()
            memory[name] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            rx = cls(profile, q)
            tracemalloc.start()
            rx.feed(x)
            rx.finish()
            stream_memory[name] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        np.savez_compressed(out / (ident.replace("/", "-") + "-traces.npz"), **traces)
        summary = {
            a: dict(
                median_Msamples_per_second=float(
                    np.median([r["samples_per_second"] / 1e6 for r in rows if r["arm"] == a])
                ),
                wall_seconds_percentiles=np.percentile(
                    [r["seconds"] for r in rows if r["arm"] == a], [0, 50, 95, 100]
                ).tolist(),
            )
            for a in arms
        }
        result = dict(
            id=ident,
            sha256=meta["sha256"],
            samples=len(x),
            summary=summary,
            measurements=rows,
            costs=costs,
            python_numpy_peak_bytes=memory,
            python_numpy_stream_peak_bytes=stream_memory,
            frames_and_availability_equal=True,
            trajectories_within_tolerance=True,
        )
        results.append(result)
        print(json.dumps(dict(id=ident, summary=summary, costs=costs, memory=memory)), flush=True)
    plan["process_peak_rss_KiB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
        for table in ("sources", "rust_sources")
        for p, h in plan[table].items()
    )
    plan["memory_scope"] = (
        "RSS includes both arms/input; tracemalloc excludes native Rust allocations"
    )
    for name, value in [("frozen-plan.json", plan), ("results.json", results)]:
        (out / name).write_text(json.dumps(value, indent=2) + "\n")
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    psk_fields.PSKDemodulator = current_demapper
    (out / "manifest.json").write_text(
        json.dumps(
            {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.iterdir()
                if p.is_file()
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--baseline-ref", required=True)
    p.add_argument("--baseline-sync", action="store_true")
    p.add_argument("--baseline-demapper", action="store_true")
    p.add_argument("--sample-id", action="append", required=True)
    p.add_argument("--repeats", type=int, default=5)
    run(p.parse_args())
