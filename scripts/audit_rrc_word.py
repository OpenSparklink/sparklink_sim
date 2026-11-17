"""Independent new RRC range inputs, finite-word errors and module cycle boundaries."""

import argparse
import ast
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import rx_timing_audit as independent
from audit_capture_hardware import sha, write
from scipy.signal import lfilter

from nearlink_sdr.phy.rrc_flow import RRCFlow, RRCFlowAssumptions
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC, PairProductRRC
from nearlink_sdr.phy.rrc_word import FixedRRC, FixedRRCFormat, SymmetricRRC


def check_selection_sources(selection, args):
    """Exact source SHA, with one explicit and AST-verified reporting correction.

    The sole exception is RRCFlow.resource_estimate. Word arithmetic, schedule,
    source inputs and all other receiver functions must remain identical. This
    never permits retuning an algorithm against an already viewed held-out set.
    """
    changed = [p for p, h in selection["sources"].items() if sha(Path(p)) != h]
    if not changed:
        return
    path = "src/nearlink_sdr/phy/rrc_flow.py"
    if changed != [path] or args.cost_lineage is None:
        raise ValueError("Frozen source mismatch; no matching cost-only lineage")
    lineage = json.loads(args.cost_lineage.read_text())
    assert lineage["selection_sha256"] == sha(args.selection)
    old_source = Path(lineage["old_source"])
    assert sha(old_source) == selection["sources"][path] == lineage["old_sha256"]
    assert sha(Path(path)) == lineage["new_sha256"]

    def without_report(source):
        tree = ast.parse(source.read_text())
        methods = [
            m
            for c in tree.body
            if isinstance(c, ast.ClassDef) and c.name == "RRCFlow"
            for m in c.body
            if isinstance(m, ast.FunctionDef) and m.name == "resource_estimate"
        ]
        assert len(methods) == 1
        methods[0].body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)

    assert without_report(old_source) == without_report(Path(path)), "Algorithm/schedule changed"


def generate(args):
    args.output.mkdir(parents=True, exist_ok=False)
    plan = json.loads(args.budget.read_text())
    held = args.generate == "held-out"
    if held and (args.selection is None or not args.selection.is_file()):
        raise ValueError("Generate held-out only after an explicit frozen format selection")
    if held:
        selection = json.loads(args.selection.read_text())
        assert selection["held_out_seen"] is False
        FixedRRCFormat(**selection["format"])
        check_selection_sources(selection, args)
    seed = plan["held_seed" if held else "development_seed"]
    cases = []
    gains = np.array([1e-8, 1e-4, 0.125, 1.0, 32.0, 1e5])
    for family in ["static", "blocks", "strong-weak", "impulses"]:
        for rep in range(8 if held else 6):
            rng = np.random.default_rng(seed + rep)
            k = rng.integers(0, 8, 512)
            symbols = np.exp(1j * (2 * np.pi * k / 8 - np.pi / 8 * (np.arange(512) % 2)))
            x, _, _ = independent.reference(symbols, float(rng.uniform(0, 4)), (-100, 100)[rep % 2])
            x /= np.sqrt(np.mean(abs(x) ** 2))
            n = np.arange(len(x))
            x *= np.exp(2j * np.pi * (-25000, 30000)[rep % 2] * n / 16e6)
            x += 0.6 * np.exp(2j * np.pi * float(rng.uniform(-0.2, 0.2)) * n)
            x += 0.01 * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
            if family == "static":
                x *= gains[rep % 6]
            elif family == "blocks":
                x *= np.repeat(rng.choice(gains, (len(x) + 127) // 128), 128)[: len(x)]
            elif family == "strong-weak":
                x *= np.where(n < len(x) // 2, gains[-1], gains[0])
                if rep % 2:
                    x = x[::-1].copy()
            else:
                x[:] = 0
                positions = rng.choice(len(x), 17, replace=False)
                x[positions] = gains[rep % 6] * rng.choice([-8192, 8191], 17) * (1 + 1j)
            path = args.output / f"{family}-{rep}.npz"
            np.savez_compressed(path, iq=x)
            cases.append(
                dict(
                    id=path.stem,
                    path=str(path.resolve()),
                    sha256=sha(path),
                    family=family,
                    samples=len(x),
                    repeat=rep,
                )
            )
    write(
        args.output / "inputs.json",
        dict(
            seed=seed,
            split=args.generate,
            cases=cases,
            budget_sha256=sha(args.budget),
            selection_sha256=None if args.selection is None else sha(args.selection),
            driver_sha256=sha(Path(__file__)),
            independent_pulse_sha256=sha(Path(independent.__file__)),
            scope="range/filter waveforms, not legal packet acquisition trials",
        ),
    )
    (args.output / "generator.executed.py").write_bytes(Path(__file__).read_bytes())


class WordTrace:
    """Caller-owned offline trace; not proposed filter storage."""

    def __init__(self, format):
        super().__init__(format)
        self.words = []

    def feed_words(self, *args):
        words = super().feed_words(*args)
        self.words.append(words)
        return words


class WordObserver(WordTrace, FixedRRC):
    """Original window-input reference arm."""


class PairWordObserver(WordTrace, PairProductRRC):
    """New pair-product arm, traced without changing receiver decisions."""


class FullWordObserver(WordTrace, FullMantissaRRC):
    """Full signed-input representation, otherwise identical pair/product arm."""


def relative_norm(error, reference):
    """Exact zero-reference rule; never hide a nonzero error with an epsilon."""
    denominator = float(np.linalg.norm(reference))
    numerator = float(np.linalg.norm(error))
    return numerator / denominator if denominator else (float("inf") if numerator else 0.0)


def check_model_schedule(model, cycle):
    if model != "global" and cycle:
        raise ValueError("Pair-product physical cycle model not yet available; no reference reuse")


def evaluate(args):
    check_model_schedule(args.model, args.cycle)
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(args.budget.read_text())
    plan = json.loads((args.inputs / "inputs.json").read_text())
    assert plan["budget_sha256"] == sha(args.budget)
    format = FixedRRCFormat(args.input_bits, args.coefficient_bits)
    if plan["split"] == "held-out":
        if args.selection is None or sha(args.selection) != plan["selection_sha256"]:
            raise ValueError("Held-out evaluation requires its frozen format selection")
        selection = json.loads(args.selection.read_text())
        arms = selection.get("arms", [dict(model="global", format=selection["format"])])
        assert dict(model=args.model, format=asdict(format)) in arms
        check_selection_sources(selection, args)
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        args.output / "plan.json",
        dict(
            input_sha256=sha(args.inputs / "inputs.json"),
            budget_sha256=sha(args.budget),
            input_plan=str((args.inputs / "inputs.json").resolve()),
            format=asdict(format),
            model=args.model,
            sources=sources,
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            driver_sha256=sha(Path(__file__)),
            cost_lineage_sha256=None if args.cost_lineage is None else sha(args.cost_lineage),
            cycle=args.cycle,
            dependency_versions=dict(numpy=np.__version__),
            scope="isolated RRC; packet and whole-RX acceptance not evaluated",
        ),
    )
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    h = independent.pulse(np.arange(-40, 41) / 4)
    h /= np.linalg.norm(h)
    rows = []
    for meta in plan["cases"]:
        assert sha(Path(meta["path"])) == meta["sha256"]
        with np.load(meta["path"]) as data:
            x = data["iq"]
        reference = lfilter(h, [1.0], np.r_[x, np.zeros(80)])
        floating = SymmetricRRC()
        fixed = {"pair": PairWordObserver, "full": FullWordObserver, "global": WordObserver}[
            args.model
        ](format)
        structural = np.r_[floating.feed(x), floating.finish()]
        result = np.r_[fixed.feed(x), fixed.finish()]
        difference = result - reference
        relative = relative_norm(difference, reference)
        ref_peak, error_peak = float(np.max(abs(reference))), float(np.max(abs(difference)))
        peak_error = error_peak / ref_peak if ref_peak else (float("inf") if error_peak else 0.0)
        local_errors = [
            relative_norm(difference[k : k + 64], reference[k : k + 64])
            for k in range(0, len(result), 64)
        ]
        range_events = sum(
            fixed.stats[n]
            for n in [
                "exponent_underflows",
                "exponent_overflows",
                "input_saturations",
                "output_underflows",
                "output_overflows",
                "output_saturations",
                "intermediate_overflows",
            ]
        )
        limits = budget["numerical_budget"]
        rms_limit = limits.get(
            "global_normalized_rms", limits.get("normalized_filter_error_rms_max")
        )
        peak_limit = limits.get(
            "global_relative_peak", limits.get("max_sample_error_over_reference_peak")
        )
        local_limit = limits.get("local64_relative_rms")
        accepted = (
            relative <= rms_limit
            and peak_error <= peak_limit
            and range_events == 0
            and (local_limit is None or max(local_errors) <= local_limit)
        )
        row = dict(
            input=meta,
            normalized_rms_error=relative,
            relative_peak_error=peak_error,
            max_64_sample_relative_rms_error=max(local_errors),
            float_structure_max_error=float(np.max(abs(structural - reference))),
            range_events=range_events,
            stats=fixed.stats,
            numerical_budget_passed=accepted,
        )
        r, i, e = [np.concatenate([w[j] for w in fixed.words]) for j in range(3)]
        np.savez_compressed(
            args.output / (meta["id"] + "-nodes.npz"),
            reference=reference,
            structure=structural,
            fixed=result,
            mantissa_real=r,
            mantissa_imag=i,
            exponents=e,
            coefficients=fixed.coefficients,
        )
        rows.append(row)
        write(args.output / "results.json", rows)
    estimate = RRCFlow(format).resource_estimate() if args.model == "global" else None
    cycles = None
    if args.cycle:
        cycles = {}
        for name, options, stalls in [
            ("base", {}, []),
            ("160MHz", dict(clock_hz=160_000_000), []),
            ("slow-feedback", dict(arithmetic_ii=2), []),
            ("one-coefficient-port", dict(coefficient_read_ports=1), []),
            ("blocked", {}, [(100, 4100)]),
        ]:
            model = RRCFlow(format, replace(RRCFlowAssumptions(), **options))
            cycles[name] = model.simulate(4096, downstream_stalls=stalls, trace=True)
            write(args.output / ("cycles-" + name + ".json"), cycles[name])
        cycles = {k: {n: v for n, v in r.items() if n != "trace"} for k, r in cycles.items()}
    passed = all(r["numerical_budget_passed"] for r in rows)
    cycle_passed = (
        None
        if cycles is None
        else (
            cycles["base"]["continuous_input_without_loss"]
            and estimate["capacity_necessary_condition"]
            and estimate["operators_within_module_budget"]
            and estimate["register_tap_storage_within_budget"]
        )
    )
    write(
        args.output / "summary.json",
        dict(
            cases=len(rows),
            format=asdict(format),
            numerical_budget_passed=passed,
            max_normalized_rms_error=max(r["normalized_rms_error"] for r in rows),
            max_relative_peak_error=max(r["relative_peak_error"] for r in rows),
            max_64_sample_relative_rms_error=max(
                r["max_64_sample_relative_rms_error"] for r in rows
            ),
            range_events=sum(r["range_events"] for r in rows),
            operation_estimate={
                "pair": PairProductRRC,
                "full": FullMantissaRRC,
                "global": FixedRRC,
            }[args.model](format).operation_estimate(),
            resource_estimate=estimate,
            cycles=cycles,
            isolated_schedule_under_assumptions_passed=cycle_passed,
            local_window_acceptance=None
            if local_limit is None
            else all(r["max_64_sample_relative_rms_error"] <= local_limit for r in rows),
            receiver_promotion=False,
            full_rx_acceptance=None,
        ),
    )
    write(args.output / "failures.json", [r for r in rows if not r["numerical_budget_passed"]])
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        args.output / "manifest.json",
        {str(p.relative_to(args.output)): sha(p) for p in args.output.rglob("*") if p.is_file()},
    )
    print(
        json.dumps(
            dict(
                format=asdict(format),
                numerical_budget_passed=passed,
                max_rms_error=max(r["normalized_rms_error"] for r in rows),
            ),
            indent=2,
        )
    )
    return int(not passed or cycle_passed is False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate", choices=["development", "held-out"])
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--cost-lineage", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-bits", type=int, default=12)
    parser.add_argument("--coefficient-bits", type=int, default=18)
    parser.add_argument("--model", choices=["global", "pair", "full"], default="global")
    parser.add_argument("--cycle", action="store_true")
    args = parser.parse_args()
    if args.generate:
        generate(args)
    elif args.inputs:
        raise SystemExit(evaluate(args))
    else:
        parser.error("Need --generate or --inputs")
