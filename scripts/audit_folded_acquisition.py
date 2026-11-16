"""Compare every integer point/hit with the hash-frozen pre-folding model."""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, sha, write
from scipy.signal import lfilter

from nearlink_sdr.phy.acquisition_scaled import ScaledDifferentialBank
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.tone_notch import CausalToneNotch


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    expected = json.loads(args.source_manifest.read_text())["source_hashes"][
        "src/nearlink_sdr/phy/acquisition_scaled.py"
    ]
    assert sha(args.reference_source) == expected
    old = types.ModuleType("unfolded_reference")
    sys.modules[old.__name__] = old
    exec(
        compile(args.reference_source.read_bytes(), str(args.reference_source), "exec"),
        old.__dict__,
    )
    plan = json.loads((args.inputs / "frozen-inputs.json").read_text())
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        out / "frozen-plan.json",
        dict(
            input_plan=str(args.inputs / "frozen-inputs.json"),
            input_plan_sha256=sha(args.inputs / "frozen-inputs.json"),
            reference_source_sha256=expected,
            sources=sources,
            driver_sha256=sha(Path(__file__)),
            scope="integer point/hit/state identity; no packet acceptance",
        ),
    )
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    failures, rows = [], []
    for case in plan["cases"]:
        path = Path(case["path"])
        assert sha(path) == case["sha256"]
        with np.load(path) as data:
            x = data["iq"]
            numeric = "reference" in data
            ref = data["reference"] if numeric else profile(case["mcs"]).sync_symbols()
        filtered = x if numeric else lfilter(rrc_filter(0.4, 4), [1.0], x)
        notch = CausalToneNotch()
        before = [old.ScaledDifferentialBank(ref, 4, 256) for _ in range(2)]
        after = [ScaledDifferentialBank(ref, 4, 256) for _ in range(2)]
        points = queries = 0
        for start in range(0, len(filtered), 32):
            block = filtered[start : start + 32]
            for index, values in enumerate([block, notch.advance(block)]):
                a, b = before[index], after[index]
                old_points, new_points = (
                    a.advance(values, inspect=True),
                    b.advance(values, inspect=True),
                )
                lower, upper = max(0, a.samples - 255), a.samples - 125
                old_gate, new_gate = a.gate(lower, upper), b.gate(lower, upper)
                points += len(old_points)
                queries += 1
                if (
                    old_points != new_points
                    or old_gate != new_gate
                    or not np.array_equal(a._hits, b._hits)
                    or a.stats != b.stats
                ):
                    failures.append(
                        dict(
                            input=case,
                            path=["raw", "tone"][index],
                            after_samples=a.samples,
                            points_equal=old_points == new_points,
                            gate_equal=old_gate == new_gate,
                            hits_equal=np.array_equal(a._hits, b._hits),
                            old_stats=a.stats.copy(),
                            new_stats=b.stats.copy(),
                        )
                    )
        rows.append(
            dict(
                input=case,
                integer_points=points,
                queries=queries,
                old_logical_bank_bits=before[0].logical_storage_bits,
                new_logical_bank_bits=after[0].logical_storage_bits,
                raw_stats=after[0].stats,
                tone_stats=after[1].stats,
            )
        )
    write(out / "results.json", rows)
    write(out / "failures.json", failures)
    summary = dict(
        inputs=len(rows),
        integer_points=sum(r["integer_points"] for r in rows),
        queries=sum(r["queries"] for r in rows),
        mismatches=len(failures),
        reference_sha256=expected,
    )
    write(out / "summary.json", summary)
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return bool(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "output", "reference-source", "source-manifest"]:
        parser.add_argument("--" + name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
