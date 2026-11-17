"""Frozen finite-word RRC -> strict MAC experiment; not a receiver promotion gate."""

import argparse
import hashlib
import importlib.metadata
import json
import resource
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audit_capture_hardware import KINDS, profile, score, sha, vector, write

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def generate(args):
    budget = json.loads(args.budget.read_text())
    held = args.generate == "held-out"
    if held:
        if args.selection is None:
            raise ValueError("Freeze integration parameters/source before generating held-out")
        selection = json.loads(args.selection.read_text())
        assert selection["held_out_seen"] is False
        assert selection["budget_sha256"] == sha(args.budget)
        assert all(sha(Path(f)) == h for f, h in selection["sources"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    seed = budget["held_seed" if held else "development_seed"]
    repeats = budget["held_repeats" if held else "development_repeats"]
    cases = []
    for kind in KINDS:
        for positive in (False,) if kind == "zero" else (False, True):
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + repeat)
                options = dict(tone_frequency=float(rng.uniform(-0.2, 0.2)), tone_phase=1.03)
                x, q, meta = vector(
                    kind, repeat, seed, positive, **(options if kind == "tone" else {})
                )
                ident = f"{kind}-{int(positive)}-{repeat}"
                path = args.output / (ident + ".npz")
                np.savez_compressed(path, iq=x, quiet=q)
                cases.append(
                    dict(
                        id=ident,
                        path=str(path.resolve()),
                        sha256=sha(path),
                        samples=len(x),
                        quiet_samples=len(q),
                        iq_array_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                        quiet_array_sha256=hashlib.sha256(q.tobytes()).hexdigest(),
                        **meta,
                    )
                )
    write(
        args.output / "frozen-inputs.json",
        dict(
            seed=seed,
            split=args.generate,
            cases=cases,
            budget_sha256=sha(args.budget),
            selection_sha256=None if args.selection is None else sha(args.selection),
            generator_sha256=sha(Path(__file__)),
            reference="independent pulse/angle/resampling; existing bit encoder; not sole conformance",
        ),
    )
    (args.output / "generator.executed.py").write_bytes(Path(__file__).read_bytes())


def run(args):
    inputs = json.loads((args.inputs / "frozen-inputs.json").read_text())
    assert inputs["budget_sha256"] == sha(args.budget)
    if inputs["split"] == "held-out":
        if args.selection is None or sha(args.selection) != inputs["selection_sha256"]:
            raise ValueError("Need exact frozen selection for held-out")
        selected = json.loads(args.selection.read_text())
        assert all(sha(Path(f)) == h for f, h in selected["sources"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence").mkdir()
    options = dict(
        quiet_batching=False,
        defer_quiet_fir=False,
        correlation_backend="scaled",
        correlation_mantissa_bits=10,
        tone_notch=ToneNotchConfig(radius=0.96875),
        timing_subdivisions=8,
        sync_search_method="reference",
    )
    # Old source can execute the baseline before the new optional API exists.
    if args.rrc_backend != "reference":
        options["rrc_backend"] = args.rrc_backend
    sources = {str(f): sha(f) for f in Path("src").rglob("*.py")}
    write(
        args.output / "plan.json",
        dict(
            input_plan=str((args.inputs / "frozen-inputs.json").resolve()),
            input_plan_sha256=sha(args.inputs / "frozen-inputs.json"),
            budget_sha256=sha(args.budget),
            backend=args.rrc_backend,
            chunks=args.chunks,
            chunk_seed=510120,
            options={**options, "tone_notch": asdict(options["tone_notch"])},
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            sources=sources,
            driver_sha256=sha(Path(__file__)),
            dependencies={n: importlib.metadata.version(n) for n in ["numpy", "scipy"]},
            association_tolerance_samples=4,
            receiver_promotion=False,
        ),
    )
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta in inputs["cases"]:
        path = Path(meta["path"])
        assert sha(path) == meta["sha256"]
        with np.load(path) as data:
            x, q = data["iq"], data["quiet"]
        assert hashlib.sha256(x.tobytes()).hexdigest() == meta["iq_array_sha256"]
        assert hashlib.sha256(q.tobytes()).hexdigest() == meta["quiet_array_sha256"]
        start = time.perf_counter()
        rx = MacStreamReceiver(profile(meta["mcs"]), q, **options)
        events, packets, offset = [], [], 0
        rng = np.random.default_rng(510120)
        while offset < len(x):
            n = (
                len(x)
                if args.chunks == "whole"
                else (257 if args.chunks == "fixed" else int(rng.integers(1, 1001)))
            )
            result = rx.feed(x[offset : offset + n])
            events += result.diagnostics["events"]
            packets += result.packets
            offset += n
        before_eof = len(packets)
        result = rx.finish()
        events += result.diagnostics["events"]
        packets += result.packets
        elapsed = time.perf_counter() - start
        phy = rx.phy
        result = score(events, meta)  # Truth/content only after all receive decisions.
        traces = {
            f"{i}-{k}": e[k]
            for i, e in enumerate(events)
            for k in ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]
        }
        np.savez_compressed(args.output / "evidence" / (meta["id"] + ".npz"), **traces)
        word = getattr(phy, "_rrc_word_model", None)
        arrays = getattr(phy, "fir_array_bytes", phy.h.nbytes + phy._fir_state.nbytes)
        rows.append(
            dict(
                input=meta,
                **result,
                delivered_payloads=[p.mac_payload.hex() for p in packets],
                delivered_before_eof=before_eof,
                receiver_events=[
                    dict(
                        sync_start=e["sync_start_raw"],
                        sync_score=e["sync_score"],
                        cfo_hz=e["cfo_hz"],
                        emitted_after_samples=e["emitted_after_samples"],
                    )
                    for e in events
                ],
                actual_capture_config=asdict(phy.capture_config),
                actual_timing_config=asdict(phy.timing_config),
                carrier_method=phy.carrier_method,
                stats=phy.capture_stats,
                timing_backend=phy.timing_backend,
                integer_backend=phy.integer_screen_backend,
                peak_history_samples=phy.peak_history_samples,
                history_limit=phy.history_limit,
                fir_array_bytes=arrays,
                rrc_stats=None if word is None else dict(word.stats),
                input_samples=phy.input_samples,
                processed_samples=phy._processed,
                future_samples=phy.future_samples,
                per_million={k: v * 1e6 / len(x) for k, v in phy.capture_stats.items()},
                elapsed_seconds=elapsed,
                wallclock_samples_per_second=len(x) / elapsed,
                rss_peak_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
        )
        write(args.output / "results.json", rows)
    summary = {
        k: sum(r[k] for r in rows)
        for k in [
            "known_packets",
            "captured",
            "correct",
            "missed",
            "false_candidates",
            "false_deliveries",
        ]
    }
    summary.update(
        inputs=len(rows),
        samples=sum(r["input"]["samples"] for r in rows),
        delivered=sum(len(r["delivered_payloads"]) for r in rows),
        max_fir_array_bytes=max(r["fir_array_bytes"] for r in rows),
        receiver_promotion=False,
        numeric_gate_already_failed=True,
        whole_rx_16msps=None,
    )
    write(args.output / "summary.json", summary)
    write(
        args.output / "failures.json",
        [
            r
            for r in rows
            if r["correct"] < r["known_packets"] or r["false_candidates"] or r["false_deliveries"]
        ],
    )
    assert sources == {str(f): sha(f) for f in Path("src").rglob("*.py")}
    write(
        args.output / "manifest.json",
        {str(f.relative_to(args.output)): sha(f) for f in args.output.rglob("*") if f.is_file()},
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", choices=["development", "held-out"])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rrc-backend", choices=["reference", "word-full12-c18"], default="reference"
    )
    parser.add_argument("--chunks", choices=["whole", "fixed", "random"], default="whole")
    args = parser.parse_args()
    if args.generate:
        generate(args)
    elif args.inputs:
        run(args)
    else:
        parser.error("Need --generate or --inputs")
