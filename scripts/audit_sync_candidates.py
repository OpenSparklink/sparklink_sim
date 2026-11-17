"""Audit bounded search geometry on frozen IQ; full-grid oracle is diagnostic only."""

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import profile, score, sha, write

from nearlink_sdr.phy import sync_candidates
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def run(args):
    plan = json.loads((args.inputs / "inputs.json").read_text())
    assert plan["budget_sha256"] == sha(args.budget)
    if plan["split"] == "held":
        assert args.selection and sha(args.selection) == plan["selection_sha256"]
        frozen = json.loads(args.selection.read_text())
        assert args.candidates == frozen["candidate_count"]
        assert args.normalized_coarse == frozen["normalized_coarse"]
        assert all(sha(Path(f)) == h for f, h in frozen["sources"].items())
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence").mkdir()
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(args.output / "plan.json", dict(
        sources=sources, driver_sha256=sha(Path(__file__)), input_plan_sha256=sha(args.inputs / "inputs.json"),
        budget_sha256=sha(args.budget), source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(), candidates=args.candidates,
        normalized_coarse=args.normalized_coarse, chunks=args.chunks, chunk_seed=550120,
        diagnostic_full_search="post-selection only; never modifies algorithm return or parameters",
        rrc_backend="reference", carrier_weights="default amplitude squared",
    ))
    rows, all_windows = [], []
    real_search = sync_candidates.search_candidate_grid
    for meta in plan["cases"]:
        assert sha(Path(meta["path"])) == meta["sha256"]
        with np.load(meta["path"]) as data:
            x, q = data["iq"], data["quiet"]
        windows = []

        def diagnostic(*a, **kw):
            # The receiver's answer is computed first and returned unchanged.
            chosen = real_search(*a, **kw)
            observed = dict(kw["diagnostics"])
            full = search_psk_sync(*a, min_start=kw["min_start"], lookahead=0)
            full_start = None if full is None else full["phase"]+full["offset"]*a[1]
            ready = (full is not None and full_start+(len(a[2])-1)*a[1]
                     + kw["lookahead"] <= len(a[0])-1)
            record = dict(index=len(windows), input_id=meta["id"], input_sha256=meta["sha256"],
                          local=observed, full_start=full_start,
                          full_score=None if full is None else full["score"], full_ready=ready,
                          full_peak_in_local_set=full_start is not None and any(
                              abs(full_start-p) <= .5 for p in observed.get("coarse_starts", [])))
            if full and full["score"] >= .85 and (
                    not record["full_peak_in_local_set"] or
                    abs(observed.get("best_start", -1000)-full_start) > .125):
                name = f"{meta['id']}-window{len(windows)}.npz"
                np.savez_compressed(args.output / "evidence" / name, filtered=a[0], reference=a[2])
                record.update(window_file=name, window_sha256=sha(args.output / "evidence" / name))
            windows.append(record)
            return chosen

        options = dict(quiet_batching=False, defer_quiet_fir=False,
                       correlation_backend="scaled", correlation_mantissa_bits=10,
                       tone_notch=ToneNotchConfig(radius=31/32),
                       sync_search_method="bounded-grid", sync_candidate_count=args.candidates,
                       normalized_coarse=args.normalized_coarse)
        rx = MacStreamReceiver(profile(meta["mcs"]), q, **options)
        events, packets, start = [], [], 0
        rng = np.random.default_rng(550120)
        with patch.object(sync_candidates, "search_candidate_grid", diagnostic):
            while start < len(x):
                count = len(x) if args.chunks == "whole" else (
                    257 if args.chunks == "fixed" else int(rng.integers(1, 1001)))
                result = rx.feed(x[start:start+count])
                events += result.diagnostics["events"]
                packets += result.packets
                start += count
            result = rx.finish()
            events += result.diagnostics["events"]
            packets += result.packets
        np.savez_compressed(args.output / "evidence" / (meta["id"] + ".npz"), **{
            f"{i}-{key}": e[key] for i, e in enumerate(events) for key in
            ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]})
        row = dict(input=meta, **score(events, meta), stats=rx.phy.capture_stats,
                   delivered_payloads=[p.mac_payload.hex() for p in packets],
                   timing_backend=rx.phy.timing_backend, integer_backend=rx.phy.integer_screen_backend,
                   bank_stats=rx.phy.correlation_bank.stats,
                   tone_bank_stats=rx.phy.tone_correlation_bank.stats,
                   windows=windows, history_limit=rx.phy.history_limit,
                   peak_history_samples=rx.phy.peak_history_samples)
        rows.append(row)
        all_windows += windows
        write(args.output / "results.json", rows)
    summary = {k: sum(r[k] for r in rows) for k in ["known_packets", "captured", "correct", "missed",
               "false_candidates", "false_deliveries"]}
    summary.update(inputs=len(rows), samples=sum(r["input"]["samples"] for r in rows),
                   requests=len(all_windows), fine_positions=sum(w["local"]["scored_positions"]
                   for w in all_windows), full_qualifying_windows=sum(w["full_score"] is not None
                   and w["full_score"] >= .85 for w in all_windows),
                   local_set_excluded_full_peak=sum(w["full_score"] is not None
                   and w["full_score"] >= .85 and not w["full_peak_in_local_set"] for w in all_windows),
                   whole_system_accepted=False, default_promotion=False)
    write(args.output / "summary.json", summary)
    write(args.output / "windows.json", all_windows)
    write(args.output / "failures.json", [r for r in rows if r["correct"] < r["known_packets"]
                                          or r["false_candidates"] or r["false_deliveries"]])
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {str(p.relative_to(args.output)): sha(p)
                                          for p in args.output.rglob("*") if p.is_file()})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ["inputs", "budget", "output"]:
        parser.add_argument("--" + arg, type=Path, required=True)
    parser.add_argument("--candidates", type=int, choices=[1, 2], required=True)
    parser.add_argument("--normalized-coarse", action="store_true")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--chunks", choices=["whole", "fixed", "random"], default="whole")
    run(parser.parse_args())
