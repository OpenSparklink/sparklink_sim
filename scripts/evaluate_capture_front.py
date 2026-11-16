"""Frozen gate/screen evaluation; references are scoring-only and never enter acquisition."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_strict_receive import iq_record, transmitted  # noqa: E402

from nearlink_sdr.phy.psk_sync import PSKCaptureConfig  # noqa: E402
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver  # noqa: E402
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile  # noqa: E402
from nearlink_sdr.phy.tx_pipeline import TxConfig  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def profile(mcs):
    return PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=mcs, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )


def generate(out, split):
    out.mkdir(parents=True, exist_ok=False)
    seed = 114016 if split == "development" else 124016
    cases = []
    for mcs in (6, 10, 8, 12):
        p = profile(mcs)
        for amplitude in (0.5, 1.0):
            for snr in (4, 6, 8):
                for repeat in range(2 if split == "development" else 4):
                    rng = np.random.default_rng(
                        np.random.SeedSequence([seed, mcs, int(amplitude * 10), snr, repeat])
                    )
                    wire, expected = transmitted(p, length=255)
                    clean, _, truth, step = iq_record(
                        wire, ppm=100, cfo=30000, offset=float(rng.uniform(0, 4))
                    )
                    sigma = 50 * 10 ** (-snr / 20)
                    x = amplitude * clean + sigma / np.sqrt(2) * (
                        rng.normal(size=len(clean)) + 1j * rng.normal(size=len(clean))
                    )
                    q = sigma / np.sqrt(2) * (rng.normal(size=8192) + 1j * rng.normal(size=8192))
                    ident = f"m{mcs}-a{amplitude}-n{snr}-r{repeat}"
                    path = out / (ident + ".npz")
                    np.savez_compressed(
                        path, iq=x, quiet=q, expected=np.frombuffer(expected, dtype=np.uint8)
                    )
                    cases.append(
                        dict(
                            id=ident,
                            mcs=mcs,
                            amplitude=amplitude,
                            snr=snr,
                            repeat=repeat,
                            truth=truth,
                            step=step,
                            path=str(path),
                            sha256=sha(path),
                        )
                    )
    write(out / "plan.json", dict(split=split, seed=seed, inputs=cases))
    return cases


def receive(x, q, p, config, pattern):
    rx = PSKStreamReceiver(p, q, capture_config=config)
    events = []
    rng = np.random.default_rng(125016)
    pos = 0
    begin = time.perf_counter()
    while pos < len(x):
        n = (
            len(x)
            if pattern == "whole"
            else (4096 if pattern == "fixed" else int(rng.integers(1, 513)))
        )
        events.extend(rx.feed(x[pos : pos + n]))
        pos += n
    events.extend(rx.finish())
    return events, dict(
        seconds=time.perf_counter() - begin,
        stats=rx.capture_stats,
        searches=rx.sync_searches,
        searches_per_Msample=rx.sync_searches * 1e6 / len(x),
        full_searches_per_Msample=rx.capture_stats["full_searches"] * 1e6 / len(x),
        state_array_bytes=sum(v.nbytes for v in vars(rx).values() if isinstance(v, np.ndarray)),
        history_limit=rx.history_limit,
        peak_history_samples=rx.peak_history_samples,
        threshold=rx.threshold,
    )


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    if args.inputs:
        index = args.inputs / "plan.json"
        if not index.exists():
            index = args.inputs / "frozen-plan.json"
        cases = json.loads(index.read_text())["inputs"]
    else:
        cases = generate(out / "inputs", args.split)
    configs = {"legacy": PSKCaptureConfig.legacy(), "selected": PSKCaptureConfig()}
    if args.split == "development":
        configs.update(gate2=PSKCaptureConfig(2, None), gate125=PSKCaptureConfig(1.25, None))
    plan = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")},
        driver_sha256=sha(Path(__file__)),
        split=args.split,
        inputs=cases,
        configurations={k: vars(v) for k, v in configs.items()},
        waveform="ordinary A2, 255-byte fixed random seed65016, pilot8, explicit seed/whitening",
        false_candidate_association_samples=4,
        chunk_tolerance=dict(symbols=1e-12, centers=1e-10, rate=1e-8),
    )
    write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta in cases:
        path = Path(meta["path"])
        assert sha(path) == meta["sha256"]
        with np.load(path) as d:
            x, q, expected = d["iq"], d["quiet"], d["expected"].tobytes()
        row = dict(meta, arms={})
        arrays = {}
        for arm, config in configs.items():
            events, cost = receive(x, q, profile(meta["mcs"]), config, "whole")
            frames = []
            for j, e in enumerate(events):
                f = e["frame"]
                frames.append(
                    dict(
                        status=f["status"],
                        header_crc=f["header_crc"],
                        data_crc=f["data_crc"],
                        mcs=f.get("mcs"),
                        sync=e["sync_start_raw"],
                        matched=abs(e["sync_start_raw"] - meta["truth"]) < 4,
                        correct=f["status"] == "valid"
                        and f["payload"] == expected
                        and f["mcs"] == meta["mcs"]
                        and f["header"]["data_length"] == 255,
                    )
                )
                for key in (
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                ):
                    arrays[f"{arm}-{j}-{key}"] = e[key]
            row["arms"][arm] = dict(
                cost,
                frames=frames,
                candidates=len(events),
                correct=sum(f["correct"] for f in frames),
                false_candidates=sum(not f["matched"] for f in frames),
            )
            if arm == "selected" and args.check_chunks:
                for pattern in ("fixed", "random"):
                    other, _ = receive(x, q, profile(meta["mcs"]), config, pattern)
                    assert len(events) == len(other), meta["id"]
                    for a, b in zip(events, other, strict=True):
                        assert (
                            a["frame"] == b["frame"]
                            and a["emitted_after_samples"] == b["emitted_after_samples"]
                        )
                        for key, tol in [
                            ("symbols", 1e-12),
                            ("symbol_centers_raw", 1e-10),
                            ("timing_rate_ppm", 1e-8),
                            ("timing_detector_error", 1e-12),
                        ]:
                            np.testing.assert_allclose(a[key], b[key], atol=tol, rtol=0)
                row["selected_chunk_equivalence"] = True
        np.savez_compressed(out / "evidence" / (meta["id"] + ".npz"), **arrays)
        rows.append(row)
    summary = {
        a: dict(
            inputs=len(rows),
            candidates=sum(r["arms"][a]["candidates"] for r in rows),
            correct=sum(r["arms"][a]["correct"] for r in rows),
            false_candidates=sum(r["arms"][a]["false_candidates"] for r in rows),
            searches=sum(r["arms"][a]["searches"] for r in rows),
            full_searches=sum(r["arms"][a]["stats"]["full_searches"] for r in rows),
        )
        for a in configs
    }
    regressions = [
        r for r in rows if r["arms"]["legacy"]["correct"] > r["arms"]["selected"]["correct"]
    ]
    write(out / "results.json", rows)
    write(out / "summary.json", summary)
    write(out / "regressions.json", regressions)
    write(out / "failures.json", [r for r in rows if r["arms"]["selected"]["correct"] != 1])
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    assert not regressions, "Original success regression retained in regressions.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--split", choices=["development", "held-out"], required=True)
    parser.add_argument("--check-chunks", action="store_true")
    run(parser.parse_args())
