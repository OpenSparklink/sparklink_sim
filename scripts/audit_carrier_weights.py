"""Single-factor bounded carrier weights: frozen IQ to actual strict MAC.

Expected bits and time boundaries enter only the post-receive score. No retry,
phase choice or gain estimate reads them. New held inputs require a selection.
"""

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audit_capture_hardware import KINDS, profile, score, sha, vector, write

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def generate(args):
    budget = json.loads(args.budget.read_text())
    held = args.generate == "held"
    if held:
        if args.selection is None:
            raise ValueError("Freeze selection and source before generating held inputs")
        selection = json.loads(args.selection.read_text())
        assert selection["held_seen"] is False
        assert selection["budget_sha256"] == sha(args.budget)
        assert all(sha(Path(f)) == h for f, h in selection["sources"].items())
    seed = budget["held_seed" if held else "development_seed"]
    repeats = budget["held_repeats" if held else "development_repeats"]
    args.output.mkdir(parents=True, exist_ok=False)
    cases = []
    for kind in KINDS:
        for positive in (False,) if kind == "zero" else (False, True):
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + repeat)
                options = dict(tone_frequency=float(rng.uniform(-.2, .2)), tone_phase=1.03)
                x, q, meta = vector(kind, repeat, seed, positive,
                                    **(options if kind == "tone" else {}))
                ident = f"{kind}-{int(positive)}-{repeat}"
                path = args.output / (ident + ".npz")
                np.savez_compressed(path, iq=x, quiet=q)
                cases.append(dict(id=ident, path=str(path.resolve()), sha256=sha(path),
                                  samples=len(x), iq_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                                  burst_ranges=([[s, min(s + 31 + repeat, len(x))]
                                                 for s in range(97 + repeat, len(x), 997)]
                                                if kind == "burst" else []), **meta))
    write(args.output / "inputs.json", dict(
        split=args.generate, seed=seed, cases=cases, budget_sha256=sha(args.budget),
        selection_sha256=None if args.selection is None else sha(args.selection),
        generator_sha256=sha(Path(__file__)),
        independence="independent analytic pulse/angle and channel resampling; shared bit encoder, not sole conformance",
    ))
    (args.output / "generator.executed.py").write_bytes(Path(__file__).read_bytes())


def run(args):
    inputs = json.loads((args.inputs / "inputs.json").read_text())
    assert inputs["budget_sha256"] == sha(args.budget)
    if inputs["split"] == "held":
        if args.selection is None or sha(args.selection) != inputs["selection_sha256"]:
            raise ValueError("Need exact frozen held selection")
        selected = json.loads(args.selection.read_text())
        assert args.kappa in [None, selected["kappa"]]
        assert all(sha(Path(f)) == h for f, h in selected["sources"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence").mkdir()
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(args.output / "plan.json", dict(
        budget_sha256=sha(args.budget), input_plan_sha256=sha(args.inputs / "inputs.json"),
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        sources=sources, driver_sha256=sha(Path(__file__)), kappa=args.kappa,
        fixed=args.fixed, chunks=args.chunks, chunk_seed=530120,
        rrc_backend="reference", promotion=False,
    ))
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta in inputs["cases"]:
        path = Path(meta["path"])
        assert sha(path) == meta["sha256"]
        with np.load(path) as data:
            x, q = data["iq"], data["quiet"]
        assert hashlib.sha256(x.tobytes()).hexdigest() == meta["iq_sha256"]
        options = dict(quiet_batching=False, defer_quiet_fir=False,
                       correlation_backend="scaled", correlation_mantissa_bits=10,
                       tone_notch=ToneNotchConfig(radius=31/32))
        if args.kappa is not None:
            options.update(carrier_weight_kappa=args.kappa, carrier_weight_fixed=args.fixed)
        elif args.fixed:
            raise ValueError("Fixed weights require explicit kappa")
        rx = MacStreamReceiver(profile(meta["mcs"]), q, **options)
        events, packets, offset = [], [], 0
        rng = np.random.default_rng(530120)
        while offset < len(x):
            n = len(x) if args.chunks == "whole" else (
                257 if args.chunks == "fixed" else int(rng.integers(1, 1001)))
            result = rx.feed(x[offset:offset + n])
            events += result.diagnostics["events"]
            packets += result.packets
            offset += n
        before_eof = len(packets)
        result = rx.finish()
        events += result.diagnostics["events"]
        packets += result.packets
        row = score(events, meta)
        np.savez_compressed(args.output / "evidence" / (meta["id"] + ".npz"), **{
            f"{i}-{k}": e[k] for i, e in enumerate(events)
            for k in ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]
        })
        rows.append(dict(
            input=meta, **row, delivered_payloads=[p.mac_payload.hex() for p in packets],
            delivered_before_eof=before_eof, stats=rx.phy.capture_stats,
            capture_config=asdict(rx.phy.capture_config),
            timing_config=asdict(rx.phy.timing_config),
            receiver_events=[dict(sync_start=e["sync_start_raw"], score=e["sync_score"],
                                  cfo_hz=e["cfo_hz"], emitted=e["emitted_after_samples"])
                             for e in events],
            timing_backend=rx.phy.timing_backend, integer_backend=rx.phy.integer_screen_backend,
            history_limit=rx.phy.history_limit, peak_history_samples=rx.phy.peak_history_samples,
        ))
        write(args.output / "results.json", rows)
    summary = {k: sum(r[k] for r in rows) for k in [
        "known_packets", "captured", "correct", "missed", "false_candidates", "false_deliveries"]}
    summary.update(inputs=len(rows), samples=sum(r["input"]["samples"] for r in rows),
                   promotion=False, whole_system_accepted=False)
    write(args.output / "summary.json", summary)
    write(args.output / "failures.json", [r for r in rows
          if r["correct"] < r["known_packets"] or r["false_candidates"] or r["false_deliveries"]])
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(args.output / "manifest.json", {str(p.relative_to(args.output)): sha(p)
                                          for p in args.output.rglob("*") if p.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--generate", choices=["development", "held"])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--kappa", type=int, choices=[2, 4, 8])
    parser.add_argument("--fixed", action="store_true")
    parser.add_argument("--chunks", choices=["whole", "fixed", "random"], default="whole")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.generate:
        generate(args)
    elif args.inputs:
        run(args)
    else:
        parser.error("Need --generate or --inputs")
