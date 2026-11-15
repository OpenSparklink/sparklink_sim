"""Verify receiver evidence cross-backend, retain failure links and render final cost figures."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def json_rows(p):
    return [json.loads(line) for line in p.read_text().splitlines()]


def arrays_equal(a, b):
    with np.load(a) as x, np.load(b) as y:
        assert x.files == y.files, (a, b)
        for key in x.files:
            np.testing.assert_array_equal(x[key], y[key], err_msg=f"{a}: {key}")


def run(args):
    root = args.evidence_root / "rx-chain-hardening"
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    stream_a = json_rows(root / "final-stream-python/results.jsonl")
    stream_b = json_rows(root / "final-stream-rust/results.jsonl")
    before = json_rows(root / "stream-fixed-final/results.jsonl")
    for a, b, c in zip(stream_a, stream_b, before, strict=True):
        assert a["id"] == b["id"] == c["id"]
        assert a["frames"] == b["frames"] == c["frames"], a["id"]
        arrays_equal(
            root / "final-stream-python" / a["evidence"], root / "final-stream-rust" / b["evidence"]
        )
        arrays_equal(
            root / "final-stream-python" / a["evidence"],
            root / "stream-fixed-final" / c["evidence"],
        )
    for a, b in zip(
        read(root / "soft-awgn-python-serial/results.json"),
        read(root / "soft-awgn-rust-serial/results.json"),
        strict=True,
    ):
        assert a["id"] == b["id"]
        for method in a["arms"]:
            assert {k: v for k, v in a["arms"][method].items() if k != "seconds"} == {
                k: v for k, v in b["arms"][method].items() if k != "seconds"
            }
        arrays_equal(
            root / "soft-awgn-python-serial" / a["evidence"],
            root / "soft-awgn-rust-serial" / b["evidence"],
        )
    for a, b in zip(
        read(root / "soft-capture-python/results.json"),
        read(root / "soft-capture-rust/results.json"),
        strict=True,
    ):
        assert a["id"] == b["id"]
        for method in a["arms"]:
            assert a["arms"][method]["frames"] == b["arms"][method]["frames"]
        arrays_equal(
            root / "soft-capture-python/evidence" / (a["id"] + "-results.npz"),
            root / "soft-capture-rust/evidence" / (b["id"] + "-results.npz"),
        )
    for a, b in zip(
        read(root / "low-snr-boundary-final/results.json"),
        read(root / "low-snr-boundary-replay/results.json"),
        strict=True,
    ):
        assert a["id"] == b["id"] and a["input"]["sha256"] == b["input"]["sha256"]
        assert a["frames"] == b["frames"] and a["quiet_candidates"] == b["quiet_candidates"]
    write(
        out / "cross-checks.json",
        dict(
            stream_inputs=364,
            stream_decisions_exact=True,
            stream_trajectory_arrays_exact=True,
            uncoded_before_after_soft_exact=True,
            coded_awgn_inputs=240,
            coded_awgn_decisions_and_arrays_exact=True,
            coded_raw_iq_inputs=96,
            coded_raw_iq_decisions_and_arrays_exact=True,
            independent_noise_replay_inputs=220,
            independent_noise_decisions_and_inputs_exact=True,
        ),
    )
    inventory = []
    for folder in [
        "parser-replay",
        "mac-replay-final",
        "stream-frozen",
        "stream-fixed-final",
        "final-stream-python",
        "final-stream-rust",
        "soft-awgn-python-serial",
        "soft-awgn-rust-serial",
        "soft-capture-python",
        "soft-capture-rust",
        "low-snr-boundary-final",
        "low-snr-boundary-replay",
    ]:
        p = root / folder
        index = dict(run=folder, path=str(p), references={})
        for name in [
            "frozen-plan.json",
            "manifest.json",
            "summary.json",
            "failures.json",
            "regressions.json",
            "recoveries.json",
        ]:
            f = p / name
            if f.exists():
                index["references"][name] = dict(path=str(f), sha256=sha(f))
        if (p / "failures.json").exists():
            index["failure_entries"] = len(read(p / "failures.json"))
        inventory.append(index)
    write(out / "evidence-index.json", inventory)
    perf = []
    for backend in ["python", "rust"]:
        p = root / ("final-benchmark-" + backend)
        for r in read(p / "results.json"):
            measurements = [m for m in r["measurements"] if m["enabled"]]
            perf.append(
                dict(
                    backend=backend,
                    sample_id=r["id"],
                    input_sha256=r["input_sha256"],
                    samples=r["samples"],
                    baseline_seconds=r["baseline_median_seconds"],
                    optimized_seconds=r["optimized_median_seconds"],
                    speedup=r["median_speedup"],
                    Msample_s=r["optimized_Msamples_per_second"],
                    state_array_bytes=measurements[0]["state_array_bytes"],
                    delay_samples=measurements[0]["availability_delay_samples_percentiles"],
                    cumulative_process_peak_rss_KiB=read(p / "memory.json")[
                        "cumulative_process_peak_rss_KiB"
                    ],
                )
            )
    report = dict(
        algorithm_commit=read(root / "final-stream-python/frozen-plan.json")["commit"],
        baseline_commit="e2f7b54",
        current_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        source_sha256=read(root / "final-stream-python/frozen-plan.json")["source_sha256"],
        stream=read(root / "final-stream-python/summary.json"),
        coded_awgn_held_out=[
            r
            for r in read(root / "soft-awgn-python-serial/summary.json")
            if r["split"] == "held-out"
        ],
        coded_raw_iq_held_out=[
            r for r in read(root / "soft-capture-python/summary.json") if r["split"] == "held-out"
        ],
        independent_noise=read(root / "low-snr-boundary-final/summary.json"),
        performance=perf,
        remaining_frozen_failures=[
            dict(
                id=r["id"],
                source=r["source"],
                input=r["input"],
                statuses=[(f["status"], f.get("diagnostic_bit_errors")) for f in r["frames"]],
                evidence=str(root / "final-stream-python" / r["evidence"]),
            )
            for r in read(root / "final-stream-python/failures.json")
        ],
        limitations=[
            "16 MS/s target unmet",
            "new strict stream entry supports PSK FT2/3/4, not GFSK FT1",
            "header remains hard decision",
            "WS63 encoded prefix remains unknown",
            "real IQ has no absolute timing/SRO truth",
            "decoded conditional BER excludes erasures",
            "no device, firmware, production RTL, release or FPGA real-time validation",
        ],
    )
    write(out / "results.json", report)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    xx = np.arange(len(perf))
    axes[0].bar(xx - 0.17, [p["baseline_seconds"] for p in perf], 0.34, label="no quiet batching")
    axes[0].bar(xx + 0.17, [p["optimized_seconds"] for p in perf], 0.34, label="quiet batching")
    axes[0].axhline(1048576 / 16e6, color="red", linestyle="--", label="16 MS/s input budget")
    axes[0].set(ylabel="Seconds / 1,048,576 samples\n(median, 5 runs)")
    for i, p in enumerate(perf):
        d = p["delay_samples"]
        axes[1].plot([i, i], [d["min"] / 16, d["max"] / 16], color="grey")
        axes[1].plot(i, d["p50"] / 16, "o", color="blue")
        axes[1].plot(i, d["p95"] / 16, "x", color="orange")
    axes[1].plot([], [], "o", color="blue", label="p50")
    axes[1].plot([], [], "x", color="orange", label="p95")
    axes[1].set(ylabel="Selected-center availability delay (µs)")
    for ax in axes:
        ax.set_xticks(
            xx,
            [p["backend"] + "\n" + ("QPSK" if "057" in p["sample_id"] else "8PSK") for p in perf],
        )
        ax.legend()
        ax.grid(axis="y")
    fig.suptitle(
        "Frozen development IQ; identical decisions/traces; delay is not true air-interface latency"
    )
    fig.tight_layout()
    fig.savefig(out / "performance.png", dpi=160)
    fig.savefig(out / "performance.svg")
    plt.close(fig)
    write(out / "manifest.json", {p.name: sha(p) for p in out.iterdir() if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
