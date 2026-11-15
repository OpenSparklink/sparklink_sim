"""Frozen offline/whole-stream/chunked-stream comparison with retained failures and traces."""

import argparse
import json
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rx_mac_frozen_replay as frozen
import rx_timing_audit as audit
from profile_rx_stream import number

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL
from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile


def load_input(p, replay, original, dataset, quiet_dataset, old_quiet, new_quiet):
    if p["source"] == "simulation":
        path = (replay if p["id"].startswith("new-noise-") else original) / p["input_iq"]
        expected = p.get("input_identity") or audit.identity(path)
        assert audit.identity(path) == expected
        with np.load(path) as data:
            return data["iq"], data["quiet"]
    if p["source"] in ("real", "real-derived"):
        r = p["input"]
        q = np.concatenate(
            [frozen.checked_raw(dataset, old_quiet[s]) for s in r["quiet_control_ids"]]
        )
        if p["source"] == "real":
            return frozen.checked_raw(dataset, r), q
        path = original / (p["id"] + ".npz")
        assert audit.identity(path) == p["derivative"]
        with np.load(path) as data:
            return data["iq"], q
    r = p["input"]
    other = next(new_quiet[s] for s in r["quiet_control_ids"] if s != r["sample_id"])
    return frozen.checked_raw(quiet_dataset, r), frozen.checked_raw(quiet_dataset, other)


def receive(x, q, profile, pattern, seed):
    t = time.perf_counter()
    rx = PSKStreamReceiver(profile, q)
    initialization = time.perf_counter() - t
    rng = np.random.default_rng(seed)
    events = []
    t = time.perf_counter()
    pos = 0
    while pos < len(x):
        n = (
            len(x)
            if pattern == "whole"
            else (4096 if pattern == "fixed4096" else int(rng.integers(1, 8193)))
        )
        events.extend(rx.iter_feed(x[pos : pos + n]))
        pos += n
    before_eof = len(events)
    events.extend(rx.finish())
    return events, dict(
        seconds=time.perf_counter() - t,
        init_seconds=initialization,
        events_before_eof=before_eof,
        peak_history_samples=rx.peak_history_samples,
        max_frame_symbols=rx.max_frame_symbols,
        quiet_fast_path_samples=rx.quiet_fast_path_samples,
        state_array_bytes=sum(v.nbytes for v in vars(rx).values() if isinstance(v, np.ndarray)),
    )


def compare(a, b):
    if len(a) != len(b):
        return dict(equal=False, reason="candidate count", baseline=len(a), observed=len(b))
    maxima = dict(
        symbols=0.0, symbol_centers_raw=0.0, timing_rate_ppm=0.0, timing_detector_error=0.0
    )
    tolerances = dict(
        symbols=1e-12, symbol_centers_raw=1e-10, timing_rate_ppm=1e-8, timing_detector_error=1e-12
    )
    for expected, actual in zip(a, b, strict=True):
        if (
            expected["frame"] != actual["frame"]
            or expected["emitted_after_samples"] != actual["emitted_after_samples"]
        ):
            return dict(equal=False, reason="frame decision or availability", maxima=maxima)
        for field in maxima:
            if expected[field].shape != actual[field].shape:
                return dict(equal=False, reason="trajectory length", field=field)
            maxima[field] = max(
                maxima[field], float(np.max(abs(expected[field] - actual[field]), initial=0))
            )
    return dict(equal=all(maxima[k] <= tolerances[k] for k in maxima), maxima=maxima)


def score(p, events):
    expected = np.packbits(prbs9(p["length"] * 8), bitorder="little").tobytes()
    frames = []
    for e in events:
        f = e["frame"]
        frame = {k: v for k, v in f.items() if k != "payload"}
        content = f["payload"] == expected
        frame.update(
            sync_start_raw=e["sync_start_raw"],
            sync_score=e["sync_score"],
            content_ok=content,
            whole_packet_ok=bool(
                f["status"] == "valid"
                and content
                and f["header"]["data_length"] == p["length"]
                and f["mcs"] == (12 if p["kind"] == "8PSK" else 8)
            ),
            emitted_after_samples=e["emitted_after_samples"],
            used_end_padding=e["used_end_padding"],
        )
        if f.get("header_crc") and p["source"] != "quiet":
            diagnostic = audit.score_symbols(e["symbols"], f, p["kind"], p["length"])
            frame["diagnostic_ber"] = diagnostic.get("ber")
            frame["diagnostic_bit_errors"] = diagnostic.get("bit_errors")
        truth = p.get("truth")
        if truth:
            error = e["symbol_centers_raw"] - (
                truth["sync_start"] + np.arange(len(e["symbols"])) * truth["step"]
            )
            n = min(len(error), f.get("frame_end_symbol", len(error)))
            frame["timing_error_samples_head_mid_tail"] = [
                float(error[i]) for i in (0, n // 2, n - 1)
            ]
            frame["timing_error_rms_samples"] = float(np.sqrt(np.mean(error[:n] ** 2)))
            frame["timing_error_max_samples"] = float(np.max(abs(error[:n])))
            frame["half_symbol_boundary_crossings"] = int(np.sum(abs(error[:n]) >= 2))
            # Earliest suffix remaining within the frozen .05-sample diagnostic band.
            over = np.flatnonzero(abs(error[:n]) > 0.05)
            settling = int(over[-1] + 1) if len(over) else 0
            frame["settling_symbol_0p05_samples"] = settling if settling < n else None
        else:
            frame["timing_truth"] = None
            frame["original_sro"] = "unknown"
        frame["timing_loop"] = e["timing_loop"]
        frames.append(frame)
    return frames


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    frozen.verify_manifest(args.replay)
    frozen.verify_manifest(args.original)
    rows = [json.loads(line) for line in (args.replay / "results.jsonl").read_text().splitlines()]
    original_plan = json.loads((args.original / "frozen-plan.json").read_text())
    old_quiet = {r["sample_id"]: r for r in original_plan["quiet_inputs"]}
    new_quiet = {r["input"]["sample_id"]: r["input"] for r in rows if r["source"] == "quiet"}
    root = Path(__file__).resolve().parents[1]
    plan = dict(
        commit=audit.git("rev-parse", "HEAD"),
        rust_accel=_HAS_RUST_ACCEL,
        driver=audit.identity(Path(__file__)),
        source_sha256={
            str(p.relative_to(root)): audit.identity(p) for p in (root / "src").rglob("*.py")
        },
        baseline_manifest=audit.identity(args.replay / "manifest.json"),
        original_manifest=audit.identity(args.original / "manifest.json"),
        patterns=["whole", "fixed4096"],
        random_scope="all real/derived and new-noise; selected simulator lengths",
        random_seed=69016,
        chunk_tolerance=dict(symbol_abs=1e-12, center_samples_abs=1e-10, rate_ppm_abs=1e-8),
        offline_center_tolerance_samples=0.25,
        packet_association_samples=4,
        parameters=dict(
            subdivisions=8,
            energy_window_symbols=8,
            bandwidth=0.003,
            max_rate_ppm=500,
            quiet_batching=True,
        ),
        inputs=[
            {k: v for k, v in p.items() if k not in ("arms", "regressions", "carrier_regressions")}
            for p in rows
        ],
        performance_scope="correctness replay; times are not isolated formal benchmarks",
    )
    audit.write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=number(original_plan["sync_word"]),
        data_crc_seed=number(original_plan["crc_seed"]),
        pilot_interval=0,
    )
    results = []
    failures = []
    regressions = []
    invariance = []
    recoveries = []
    stream = (out / "results.jsonl").open("w")
    for i, p in enumerate(rows):
        x, q = load_input(
            p, args.replay, args.original, args.dataset, args.quiet_dataset, old_quiet, new_quiet
        )
        baseline, metrics = receive(x, q, profile, "whole", 69016 + i)
        comparisons = {}
        arrays = {}
        for j, e in enumerate(baseline):
            for key in (
                "symbols",
                "symbol_centers_raw",
                "timing_rate_ppm",
                "timing_detector_error",
            ):
                arrays[f"whole-{j}-{key}"] = e[key]
        patterns = ["fixed4096"]
        if (
            p["source"] in ("real", "real-derived")
            or p["id"].startswith("new-noise-")
            or (
                p["source"] == "simulation"
                and p.get("family") == "length"
                and p.get("split") == "held-out"
            )
        ):
            patterns.append("random")
        for pattern in patterns:
            events, cost = receive(x, q, profile, pattern, 69016 + i)
            comparisons[pattern] = dict(**compare(baseline, events), cost=cost)
            for j, e in enumerate(events):
                for key in (
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                ):
                    arrays[f"{pattern}-{j}-{key}"] = e[key]
        frames = score(p, baseline)
        old = [
            r["sync_start_raw"] for r in p["arms"]["coherent"]["rows"] if r.get("whole_packet_ok")
        ]
        new = [r["sync_start_raw"] for r in frames if r["whole_packet_ok"]]
        missed = [v for v in old if not any(abs(v - w) <= 4 for w in new)]
        added = [w for w in new if not any(abs(v - w) <= 4 for v in old)]
        np.savez_compressed(out / "evidence" / (p["id"] + ".npz"), **arrays)
        row = dict(
            id=p["id"],
            source=p["source"],
            kind=p["kind"],
            frames=frames,
            cost=metrics,
            known_tx_packets=1 if p["source"] == "simulation" else None,
            successful_sync_positions=new,
            previous_success_positions=old,
            missed_previous_successes=missed,
            additional_successes=added,
            comparisons=comparisons,
            diagnostic_reference=str(args.replay / p["arms"]["coherent"]["evidence"]),
            evidence="evidence/" + p["id"] + ".npz",
            input=p.get("input_identity", p.get("input", p.get("input_iq"))),
        )
        results.append(row)
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        if missed:
            regressions.append(row)
        if added:
            recoveries.append(row)
        if any(not c["equal"] for c in comparisons.values()):
            invariance.append(row)
        if (p["source"] != "quiet" and not new) or any(
            f["status"] != "valid" or not f["whole_packet_ok"] for f in frames
        ):
            failures.append(row)
        if (i + 1) % 20 == 0:
            print(
                i + 1,
                "inputs; regressions",
                len(regressions),
                "chunk mismatches",
                len(invariance),
                flush=True,
            )
    stream.close()
    assert all(audit.identity(root / p) == v for p, v in plan["source_sha256"].items())
    summary = []
    for group in ["simulation", "real", "real-derived", "quiet"]:
        selected = [r for r in results if r["source"] == group]
        summary.append(
            dict(
                source=group,
                inputs=len(selected),
                correct_packets=sum(len(r["successful_sync_positions"]) for r in selected),
                statuses=dict(Counter(f["status"] for r in selected for f in r["frames"])),
                whole_stream_seconds=sum(r["cost"]["seconds"] for r in selected),
                max_state_array_bytes=max(
                    (r["cost"]["state_array_bytes"] for r in selected), default=0
                ),
            )
        )
    for filename, value in [
        (
            "summary.json",
            dict(
                groups=summary,
                regressions=len(regressions),
                recoveries=len(recoveries),
                chunk_mismatches=len(invariance),
                sources_unchanged=True,
                peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ),
        ),
        ("failures.json", failures),
        ("regressions.json", regressions),
        ("recoveries.json", recoveries),
        ("chunk-mismatches.json", invariance),
    ]:
        audit.write(out / filename, value)
    audit.write(
        out / "manifest.json",
        {str(p.relative_to(out)): audit.identity(p) for p in out.rglob("*") if p.is_file()},
    )
    return bool(regressions or invariance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("replay", "original", "dataset", "quiet-dataset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
