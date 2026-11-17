"""Diagnostic-only independent convolution controls for a frozen RRC failure.

No control output is used by a receiver, format selector or packet success rate.
Returns1 when the existing input quantizer alone exceeds the frozen local budget.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_rrc_word import relative_norm
from scipy.signal import lfilter

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, quantize_scaled_words
from nearlink_sdr.phy.rrc_input import quantize_full_rrc_words


def diagnose(results, budget, output):
    rows = json.loads((results / "results.json").read_text())
    plan = json.loads((results / "plan.json").read_text())
    assert sha(budget) == plan["budget_sha256"]
    row = max(rows, key=lambda r: r["max_64_sample_relative_rms_error"])
    meta = row["input"]
    path = Path(meta["path"])
    assert sha(path) == meta["sha256"]
    nodes = results / (meta["id"] + "-nodes.npz")
    manifest = json.loads((results / "manifest.json").read_text())
    assert sha(nodes) == manifest[nodes.name]
    with np.load(path) as data:
        x = data["iq"]
    with np.load(nodes) as data:
        reference, fixed, coefficients = [data[n] for n in ("reference", "fixed", "coefficients")]
    # A different summation structure: full81-term causal SciPy convolution.
    from rx_timing_audit import pulse

    h = pulse(np.arange(-40, 41) / 4)
    h /= np.linalg.norm(h)
    if plan.get("model") == "full":
        words = [quantize_full_rrc_words(x[k : k + 32], stats={}) for k in range(0, len(x), 32)]
        r, i, e = [np.concatenate([w[k] for w in words]) for k in range(3)]
    else:
        r, i, e = quantize_scaled_words(
            x, ScaledBankFormat(plan["format"]["input_mantissa_bits"]), stats={}
        )
    qx = np.ldexp(r.astype(float), e.astype(int)) + 1j * np.ldexp(i.astype(float), e.astype(int))
    coefficient_fraction = plan["format"]["coefficient_bits"] - 1
    controls = dict(
        input_only=lfilter(h, [1], np.r_[qx, np.zeros(80)]),
        coefficient_only=lfilter(
            coefficients / 2**coefficient_fraction, [1], np.r_[x, np.zeros(80)]
        ),
        input_and_coefficient=lfilter(
            coefficients / 2**coefficient_fraction, [1], np.r_[qx, np.zeros(80)]
        ),
        finite=fixed,
    )
    starts = range(0, len(reference), 64)
    start = max(
        starts,
        key=lambda k: relative_norm(
            fixed[k : k + 64] - reference[k : k + 64], reference[k : k + 64]
        ),
    )
    sl = slice(start, start + 64)
    errors = {n: relative_norm(v[sl] - reference[sl], reference[sl]) for n, v in controls.items()}
    limit = json.loads(budget.read_text())["numerical_budget"]["local64_relative_rms"]
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "controls.npz", reference=reference, **controls)
    write(
        output / "diagnosis.json",
        dict(
            input=meta,
            results=str(results.resolve()),
            results_sha256=sha(results / "results.json"),
            budget_sha256=sha(budget),
            window_start=start,
            window_samples=64,
            relative_rms=errors,
            frozen_local_limit=limit,
            input_quantization_alone_failed=errors["input_only"] > limit,
            controls_sha256=sha(output / "controls.npz"),
            driver_sha256=sha(Path(__file__)),
            scope="diagnostic window only; no runtime truth/CRC/payload choice or acceptance change",
        ),
    )
    print(json.dumps(dict(window_start=start, relative_rms=errors), indent=2))
    return int(errors["input_only"] > limit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--budget", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(diagnose(args.results, args.budget, args.output))
