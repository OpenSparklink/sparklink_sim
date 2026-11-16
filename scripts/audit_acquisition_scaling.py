"""Independent numeric port stress for finite exponent-tagged correlation, not packet BER."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, sha, write

from nearlink_sdr.phy.acquisition_bank import normalized_greater
from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, ScaledDifferentialBank


def generate(output, split):
    output.mkdir(parents=True, exist_ok=False)
    seed = 390117 if split == "development" else 400117
    ref = profile(8).sync_symbols()
    rows = []
    for rep in range(24 if split == "development" else 32):
        rng = np.random.default_rng(seed + rep)
        x = np.tile(ref, 8).repeat(4)
        x *= np.exp(1j * np.arange(len(x)) * rng.uniform(-0.04, 0.04))
        x += rng.uniform(0.01, 0.6) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
        kind = rep % 4
        if kind == 0:
            gain = np.full(len(x), 10 ** rng.uniform(-8, 5))
        elif kind == 1:
            gain = 10 ** np.repeat(rng.uniform(-8, 5, (len(x) + 127) // 128), 128)[: len(x)]
        elif kind == 2:
            gain = 10 ** np.linspace(-8, 5, len(x))
        else:
            gain = np.where((np.arange(len(x)) // 128) % 2 == 0, 1e-8, 1e5)
        path = output / f"{rep}.npz"
        np.savez_compressed(path, iq=x * gain, reference=ref)
        rows.append(
            dict(id=str(rep), path=str(path.resolve()), sha256=sha(path), samples=len(x), kind=kind)
        )
    write(
        output / "frozen-inputs.json",
        dict(
            seed=seed,
            split=split,
            cases=rows,
            scope="numeric input port stress, repeated sync/noise and range transitions; no packet claim",
            generator_sha256=sha(Path(__file__)),
        ),
    )


def independent_scores(x, ref):
    """Direct full-window energy/dot reference, no receiver quantizer or recurrence."""
    k = len(ref) - 1
    indices = np.arange(len(x) - 4 * k)[:, None] + 4 * np.arange(len(ref))
    y = x[indices]
    d = y[:, 1:] * y[:, :-1].conj()
    dr = ref[1:] * ref[:-1].conj()
    energy = np.sum(abs(d) ** 2, axis=1) * sum(abs(dr) ** 2)
    return np.divide(abs(d @ dr.conj()), np.sqrt(energy), out=np.zeros(len(d)), where=energy > 0)


def evaluate(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    inputs = args.inputs / "frozen-inputs.json"
    plan = json.loads(inputs.read_text())
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        out / "frozen-plan.json",
        dict(
            input_plan=str(inputs),
            input_plan_sha256=sha(inputs),
            widths=args.widths,
            score_budget=0.005,
            rank_budget=0.005,
            reference_min=0.55,
            sources=sources,
            driver_sha256=sha(Path(__file__)),
        ),
    )
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for case in plan["cases"]:
        path = Path(case["path"])
        assert sha(path) == case["sha256"]
        with np.load(path) as data:
            x, ref = data["iq"], data["reference"]
        scores = independent_scores(x, ref)
        for bits in args.widths:
            bank = ScaledDifferentialBank(ref, 4, 256, scaled_format=ScaledBankFormat(bits))
            points = []
            for j in range(0, len(x), 32):
                points += bank.advance(x[j : j + 32], inspect=True)
            model = np.sqrt([p.score_squared for p in points])
            eligible = scores >= 0.55
            errors = abs(scores - model)
            rank_losses = []
            for s in range(0, len(points), 32):
                ps = points[s : s + 132]
                chosen = ps[0]
                for p in ps[1:]:
                    if normalized_greater(p, chosen):
                        chosen = p
                reference_best = max(scores[s : s + 132])
                if reference_best >= 0.55:
                    rank_losses.append(float(reference_best - scores[chosen.start]))
            maximum = float(np.max(errors[eligible], initial=0))
            loss = max(rank_losses, default=0.0)
            row = dict(
                input=case,
                mantissa_bits=bits,
                max_score_error=maximum,
                max_rank_loss=loss,
                eligible_points=int(sum(eligible)),
                gate_disagreements=int(sum((scores >= 0.6) != (model >= 0.6))),
                stats=bank.stats,
                widths=bank.widths,
                logical_state_bits=bank.logical_storage_bits,
                operations=bank.worst_operations_per_sample(),
                passed=maximum <= 0.005
                and loss <= 0.005
                and not any(
                    bank.stats[k]
                    for k in [
                        "input_saturations",
                        "intermediate_overflows",
                        "exponent_underflows",
                        "exponent_overflows",
                    ]
                ),
            )
            rows.append(row)
            if not row["passed"]:
                np.savez_compressed(
                    out / f"failed-{case['id']}-{bits}.npz",
                    reference_scores=scores,
                    model_scores=model,
                    eligible=eligible,
                )
    write(out / "results.json", rows)
    write(out / "failures.json", [r for r in rows if not r["passed"]])
    summary = {
        str(bits): dict(
            max_score_error=max(r["max_score_error"] for r in rows if r["mantissa_bits"] == bits),
            max_rank_loss=max(r["max_rank_loss"] for r in rows if r["mantissa_bits"] == bits),
            failed_inputs=sum(not r["passed"] for r in rows if r["mantissa_bits"] == bits),
        )
        for bits in args.widths
    }
    write(out / "summary.json", summary)
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return any(not r["passed"] for r in rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", choices=["development", "held-out"])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--widths", type=int, nargs="+", default=[10])
    args = parser.parse_args()
    if args.generate:
        generate(args.output, args.generate)
    elif args.inputs:
        raise SystemExit(evaluate(args))
    else:
        parser.error("Need --generate or --inputs")
