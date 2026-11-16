"""Explicit independent noise/interference exposures, acquisition work and false deliveries."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import lfilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_strict_receive import iq_record, transmitted  # noqa: E402

from nearlink_sdr.phy.psk_sync import PSKCaptureConfig  # noqa: E402
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver  # noqa: E402
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile  # noqa: E402
from nearlink_sdr.phy.tx_pipeline import TxConfig  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    expected = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    cases = []
    for kind, count, n in [
        ("white", 8, 262144),
        ("colored", 4, 262144),
        ("impulse", 4, 65536),
        ("tone", 4, 8192),
        ("wrong-pid", 4, 8192),
    ]:
        for repeat in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([134016, len(cases)]))

            def noise(length):
                return 5 / np.sqrt(2) * (rng.normal(size=length) + 1j * rng.normal(size=length))

            x, q = noise(n), noise(8192)
            if kind == "colored":
                x = lfilter([1, 0.9, 0.4], [1], x)
                q = lfilter([1, 0.9, 0.4], [1], q)
            if kind == "impulse":
                x[rng.integers(0, n, 8)] += 1000 * (1 + 1j)
            if kind == "tone":
                x += 100 * np.exp(2j * np.pi * [0, 0.03125, 0.1375, 0.35][repeat] * np.arange(n))
            if kind == "wrong-pid":
                wrong = PSKReceiveProfile.ordinary(
                    TxConfig(mcs_index=8, symbol_rate_mhz=4, pid=0x123456 + repeat)
                )
                wire, _ = transmitted(wrong)
                x, _, _, _ = iq_record(wire, cfo=30000, ppm=100)
                x += noise(len(x))
            path = out / f"{kind}-{repeat}.npz"
            np.savez_compressed(path, iq=x, quiet=q)
            cases.append(
                dict(id=path.stem, kind=kind, path=str(path), sha256=sha(path), samples=len(x))
            )
    plan = dict(
        seed=134016,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")},
        driver_sha256=sha(Path(__file__)),
        inputs=cases,
        budget="no added sync candidates or false CRC-gated delivery vs legacy; fixed history/frame bounds",
    )
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for c in cases:
        with np.load(c["path"]) as d:
            x, q = d["iq"], d["quiet"]
        row = dict(c, arms={})
        for name, config in [
            ("legacy", PSKCaptureConfig.legacy()),
            ("selected", PSKCaptureConfig()),
        ]:
            rx = PSKStreamReceiver(expected, q, capture_config=config)
            t = time.perf_counter()
            events = rx.feed(x) + rx.finish()
            seconds = time.perf_counter() - t
            row["arms"][name] = dict(
                candidates=len(events),
                false_deliveries=sum(e["frame"]["status"] == "valid" for e in events),
                searches=rx.sync_searches,
                stats=rx.capture_stats,
                seconds=seconds,
                candidates_per_Msample=len(events) * 1e6 / len(x),
                searches_per_Msample=rx.sync_searches * 1e6 / len(x),
                full_searches_per_Msample=rx.capture_stats["full_searches"] * 1e6 / len(x),
                frames=[{k: v for k, v in e["frame"].items() if k != "payload"} for e in events],
            )
        rows.append(row)
    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    summary = []
    for kind in sorted({r["kind"] for r in rows}):
        selected = [r for r in rows if r["kind"] == kind]
        samples = sum(r["samples"] for r in selected)
        for arm in ("legacy", "selected"):
            summary.append(
                dict(
                    kind=kind,
                    arm=arm,
                    samples=samples,
                    seconds_of_input=samples / 16e6,
                    candidates=sum(r["arms"][arm]["candidates"] for r in selected),
                    false_deliveries=sum(r["arms"][arm]["false_deliveries"] for r in selected),
                    searches_per_Msample=sum(r["arms"][arm]["searches"] for r in selected)
                    * 1e6
                    / samples,
                    full_searches_per_Msample=sum(
                        r["arms"][arm]["stats"]["full_searches"] for r in selected
                    )
                    * 1e6
                    / samples,
                    observed_processing_seconds=sum(r["arms"][arm]["seconds"] for r in selected),
                )
            )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "manifest.json").write_text(
        json.dumps(
            {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}, indent=2
        )
        + "\n"
    )
    assert all(
        r["arms"]["selected"]["candidates"] <= r["arms"]["legacy"]["candidates"]
        and r["arms"]["selected"]["false_deliveries"] == 0
        for r in rows
    ), "negative budget exceeded; evidence retained"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
