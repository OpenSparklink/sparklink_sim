"""Compare frozen receiver outcomes and complete traces across two acquisition runs."""

import argparse
import json
from pathlib import Path

import numpy as np


def run(before, after, output):
    comparisons = []
    for folder, lines, suffix in [
        ("original-frozen", True, ".npz"),
        ("held-out", False, ".npz"),
        ("uncoded-boundary", False, ".npz"),
        ("coded-replay", False, "-results.npz"),
    ]:
        name = "results.jsonl" if lines else "results.json"

        def read(root):
            text = (root / folder / name).read_text()
            rows = [json.loads(line) for line in text.splitlines()] if lines else json.loads(text)
            return {r["id"]: r for r in rows}

        a, b = read(before), read(after)
        assert a.keys() == b.keys(), folder
        maxima = {}
        for ident, x in a.items():
            y = b[ident]
            for key in (
                "frames",
                "correct",
                "successful_sync_positions",
                "quiet_candidates",
                "quiet_deliveries",
            ):
                if key in x:
                    assert x[key] == y[key], (folder, ident, key)
            if "arms" in x:
                for arm in x["arms"]:
                    assert x["arms"][arm]["frames"] == y["arms"][arm]["frames"], (
                        folder,
                        ident,
                        arm,
                    )
            with (
                np.load(before / folder / "evidence" / (ident + suffix)) as u,
                np.load(after / folder / "evidence" / (ident + suffix)) as v,
            ):
                assert set(u.files) == set(v.files)
                for key in u.files:
                    tolerance = 1e-8 if "rate" in key else 1e-10 if "centers" in key else 1e-12
                    np.testing.assert_allclose(u[key], v[key], atol=tolerance, rtol=0)
                    delta = float(np.max(abs(u[key] - v[key]))) if u[key].size else 0.0
                    maxima[key.split("-")[-1]] = max(maxima.get(key.split("-")[-1], 0.0), delta)
        comparisons.append(
            dict(group=folder, inputs=len(a), outcomes_equal=True, max_trace_deltas=maxima)
        )
    output.write_text(json.dumps(comparisons, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("before", "after", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    args = p.parse_args()
    run(args.before, args.after, args.output)
