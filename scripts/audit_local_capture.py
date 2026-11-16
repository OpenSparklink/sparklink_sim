"""Replay the frozen local-refinement experiment; rejected budgets return nonzero."""

import argparse
import contextlib
import io
import json
import subprocess
import sys
from pathlib import Path

from audit_capture_hardware import sha, write
from compare_capture_hardware import run as compare


def operation_estimate():
    """Literal four-trial graph, real product equivalents; no schedule/synthesis.

    Cubic coefficient construction is counted separately from applying weights.
    There is no claim these are full-width25x18 primitives or that operations
    have been scheduled together with the continuously running frontend.
    """
    symbols, trials = 32, 4
    return dict(
        reference_symbols=symbols,
        trials_max=trials,
        symbol_interpolations=trials * symbols,
        interpolation_weight_products=8 * trials * symbols,
        differential_and_power_products=5 * trials * (symbols - 1),
        correlation_magnitude_squares=2 * trials,
        variable_real_product_equivalents=8 * trials * symbols + 5 * trials * (symbols - 1)
        + 2 * trials,
        cubic_coefficient_construction="4 cubics:8 variable products/position; shared fractional phase may reuse weights, not yet scheduled",
        cubic_coefficient_products_without_reuse=8 * trials * symbols,
        complex_interpolation_additions=6 * trials * symbols,
        complex_differential_additions=5 * trials * (symbols - 1),
        power_and_energy_additions=trials * (2 * (symbols - 1) - 1),
        signed_correlation_additions_upper=4 * trials * (symbols - 1),
        score_divisions=trials,
        square_roots=trials,
        angle_evaluations=trials,
        parabola_divisions_max=1,
        candidate_comparisons_max=3,
        coarse_rank_bits=12,
        coarse_rank_restore_steps_per_bank_sample=12,
        coarse_rank_scan_words_max=131,
        two_bank_ranking_extra_bits=2 * 3732,
        two_bank_logical_state_bits=2 * 16122,
        interpolator_format="floating experiment; no fixed word/overflow closure",
        scope="operation estimate only; no whole-receiver cycle, FIFO, port or synthesis acceptance",
    )


def evaluate(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    script = Path(__file__).with_name("audit_interference_capture.py")
    write(output / "plan.json", dict(
        inputs=str(inputs.resolve()), input_sha256=sha(inputs / "frozen-inputs.json"),
        sync_threshold=.85, integer_threshold=.6, lookahead=16,
        old_success_start_budget_samples=.125, center_budget_samples=.25,
        source_hashes={str(p): sha(p) for p in Path("src").rglob("*.py")},
        python=sys.executable, driver_sha256=sha(Path(__file__)),
    ))
    arms = {
        "reference": ["--backend", "structure"],
        "local-float": ["--backend", "structure", "--sync-method", "local-parabolic"],
        "local-q12": ["--backend", "scaled", "--sync-method", "local-parabolic",
                      "--normalized-coarse"],
    }
    for name, options in arms.items():
        subprocess.run([sys.executable, str(script), "--inputs", str(inputs),
                        "--output", str(output / name), "--radius", "0.96875", *options],
                       check=True)
    comparisons = {}
    for old, new in [("reference", "local-float"), ("local-float", "local-q12")]:
        path = output / f"{old}-vs-{new}.json"
        with contextlib.redirect_stdout(io.StringIO()):
            compare(output / old, output / new, path)
        comparisons[path.stem] = json.loads(path.read_text())
    passed = all(r["budget_passed"] for r in comparisons.values())
    write(output / "operation-estimate.json", operation_estimate())
    acceptance = dict(algorithm_budget_passed=passed, hardware_acceptance=None,
                      note="Unchanged packet totals do not waive timing-budget failures; finite fine words and closed-loop schedule remain unvalidated")
    write(output / "acceptance.json", acceptance)
    write(output / "manifest.json", {str(p.relative_to(output)): sha(p) for p in
                                    output.rglob("*") if p.is_file()})
    print(json.dumps(acceptance, indent=2))
    return int(not passed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(evaluate(args.inputs, args.output))
