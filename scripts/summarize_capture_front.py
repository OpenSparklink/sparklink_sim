"""Associate frozen old/new outcomes, traces, failures and exposure costs by input identity."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from collections import Counter
from pathlib import Path

import numpy as np


def read(p):
    return json.loads(p.read_text())


def lines(p):
    return [json.loads(v) for v in p.read_text().splitlines()]


def manifest_hashes(root, output):
    """Index manifests relative to the evidence root, including external output trees."""
    return {
        os.path.relpath(p.resolve(), root.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("manifest.json"))
    }


def run(root, capture_root=None):
    old, new = root / "rx-chain-hardening", capture_root or root / "rx-capture-front"
    regressions, parity = [], []
    previous = {r["id"]: r for r in lines(old / "final-stream-python/results.jsonl")}
    for r in lines(new / "original-frozen/results.jsonl"):
        p = previous[r["id"]]
        for t in p["successful_sync_positions"]:
            if not any(abs(t - u) <= 4 for u in r["successful_sync_positions"]):
                regressions.append(dict(group="original", id=r["id"], previous_sync=t))
    groups = {}
    for name, oldfolder, newfolder, coded in [
        ("uncoded", "low-snr-boundary-final", "uncoded-boundary", False),
        ("coded", "soft-capture-python", "coded-replay", True),
    ]:
        before = {r["id"]: r for r in read(old / oldfolder / "results.json")}
        after = read(new / newfolder / "results.json")
        arms = ("hard", "maxlog") if coded else ("uncoded",)
        for arm in arms:

            def correct(row):
                return (
                    any(f["correct"] for f in row["arms"][arm]["frames"])
                    if coded
                    else bool(row["correct"])
                )

            for r in after:
                p = before[r["id"]]
                assert (
                    p["sha256"] == r["sha256"]
                    if coded
                    else p["input"]["sha256"] == r["input"]["sha256"]
                )
                if correct(p) and not correct(r):
                    regressions.append(dict(group=name, arm=arm, id=r["id"]))
            groups[name + "-" + arm] = dict(
                inputs=len(after),
                before=sum(correct(r) for r in before.values()),
                after=sum(correct(r) for r in after),
                remaining_failure_ids=[r["id"] for r in after if not correct(r)],
            )
    for folder, other, filename, suffix in [
        ("held-out", "held-out-rust", "results.json", ".npz"),
        ("coded-replay", "coded-replay-rust", "results.json", "-results.npz"),
    ]:
        a, b = read(new / folder / filename), read(new / other / filename)
        assert [r["id"] for r in a] == [r["id"] for r in b]
        for x, y in zip(a, b, strict=True):
            for arm in x["arms"]:
                assert x["arms"][arm]["frames"] == y["arms"][arm]["frames"]
            with (
                np.load(new / folder / "evidence" / (x["id"] + suffix)) as u,
                np.load(new / other / "evidence" / (x["id"] + suffix)) as v,
            ):
                assert set(u.files) == set(v.files)
                for k in u.files:
                    np.testing.assert_array_equal(u[k], v[k])
        parity.append(dict(group=folder, inputs=len(a), frames_and_traces_exact=True))
    held = read(new / "held-out/results.json")
    funnel = {
        a: dict(
            Counter(
                {
                    k: sum(r["arms"][a]["stats"][k] for r in held)
                    for k in held[0]["arms"][a]["stats"]
                }
            )
        )
        for a in ("legacy", "selected")
    }
    samples = sum(len(np.load(r["path"])["iq"]) for r in held)
    rates = {
        a: dict(
            samples=samples,
            candidates_per_Msample=funnel[a]["sync_candidates"] * 1e6 / samples,
            searches_per_Msample=funnel[a]["energy_triggers"] * 1e6 / samples,
            full_searches_per_Msample=funnel[a]["full_searches"] * 1e6 / samples,
            false_deliveries=sum(
                not f["matched"] and f["status"] == "valid"
                for r in held
                for f in r["arms"][a]["frames"]
            ),
        )
        for a in funnel
    }
    report = dict(
        groups=groups,
        original_success_regressions=regressions,
        backend_parity=parity,
        held_out_funnel=funnel,
        held_out_rates=rates,
        environment=dict(
            python=platform.python_version(),
            dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        ),
    )
    (new / "acceptance-associations.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    assert not regressions
    manifests = manifest_hashes(root, new)
    (new / "stage2-evidence-manifests.json").write_text(json.dumps(manifests, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    stages = ["energy_triggers", "sync_candidates", "header_crc_ok", "data_crc_ok"]
    for i, a in enumerate(("legacy", "selected")):
        axes[0].plot(range(4), [funnel[a][k] for k in stages], "o-", label=a)
        axes[1].bar(
            np.arange(2) + i * 0.35,
            [rates[a][k] for k in ("searches_per_Msample", "full_searches_per_Msample")],
            0.35,
            label=a,
        )
    axes[0].set_xticks(range(4), ["Energy", "Sync", "Head CRC", "Data CRC"])
    axes[0].set_ylabel("Count (log scale)")
    axes[0].set_title("96 held-out inputs: windows / packets")
    axes[0].set_yscale("log")
    axes[1].set_xticks(np.arange(2) + 0.175, ["All entries", "Full fractional search"])
    axes[1].set_ylabel("Calls / million input samples")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(new / "capture-funnel.png", dpi=160)
    fig.savefig(new / "capture-funnel.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path)
    args = parser.parse_args()
    run(args.evidence_root, args.capture_root)
