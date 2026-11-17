"""Isolated NCO/quality budgets; identical packet labels never waive node errors."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write

from nearlink_sdr.phy.sync_coherent import SyncConfirmationNCO, coherent_sync_confirmation


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    budget = json.loads(args.budget.read_text())["budgets"]
    left = json.loads((args.before / "results.json").read_text())
    right = json.loads((args.after / "results.json").read_text())
    records, lost, gate_changes, absent = [], [], [], []
    qmax = fmax = phase_max = 0.
    for old, new in zip(left, right, strict=True):
        assert old["input"] == new["input"]
        meta = old["input"]
        assert sha(Path(meta["path"])) == meta["sha256"]
        for frame in old["frames"]:
            if frame["correct"] and not any(f["correct"] and
                    abs(f["sync_start"]-frame["sync_start"]) <= .125 for f in new["frames"]):
                lost.append(dict(id=meta["id"], input_sha256=meta["sha256"], start=frame["sync_start"]))
        windows = {(w["query_after_samples"], w["start_raw"]): w for w in new["windows"]
                   if "coherent" in w}
        for w in old["windows"]:
            if "coherent" not in w:
                continue
            info = w["coherent"]
            assert info["nco"] is None
            path = args.before / "evidence" / w["sync_file"]
            assert sha(path) == w["sync_sha256"]
            with np.load(path) as data:
                y, ref = data["symbols"], data["reference"]
            a = coherent_sync_confirmation(y, ref, info["omega_rad_per_symbol"], info["threshold"])
            b = coherent_sync_confirmation(y, ref, info["omega_rad_per_symbol"],
                                           info["threshold"], fixed_nco=True)
            assert info == a
            twin = windows.get((w["query_after_samples"], w["start_raw"]))
            tag = dict(id=meta["id"], input_sha256=meta["sha256"],
                       after_samples=w["query_after_samples"], start_raw=w["start_raw"])
            if twin is None:
                absent.append(tag)
            else:
                assert b == twin["coherent"]
            qdelta = abs(a["quality"]-b["quality"])
            fdelta = abs((b["nco"]["quantized_cycles_per_symbol"]-a["cycles_per_symbol"]+.5) % 1-.5)
            nco = SyncConfirmationNCO(b["nco"]["step_word"])
            re, im = nco.advance(len(y))
            coefficients = (re+1j*im)/65536
            error = float(np.max(abs(np.angle(coefficients*np.exp(
                1j*a["omega_rad_per_symbol"]*np.arange(len(y)))))))
            phase_max = max(phase_max, error)
            qmax, fmax = max(qmax, qdelta), max(fmax, fdelta)
            if a["accepted"] != b["accepted"]:
                gate_changes.append(tag)
            node_file = f"node-{len(records)}.npz"
            np.savez_compressed(args.output / node_file, symbols=y, reference=ref,
                                 coefficient_real=re, coefficient_imag=im)
            records.append(dict(**tag, quality_difference=qdelta,
                                frequency_difference_cycles=fdelta, phase_error_radians=error,
                                accepted_pair=[a["accepted"], b["accepted"]],
                                node_file=node_file, node_sha256=sha(args.output / node_file)))
    gates = dict(
        evaluated_nodes=bool(records),
        quality=qmax <= budget["coherent_quality_fixed_difference"],
        frequency=fmax <= budget["CFO_fixed_cycles_per_symbol_difference"],
        original_successes=not lost,
    )
    report = dict(gates=gates, node_budget_passed=all(gates.values()), nodes=len(records),
                  max_quality_difference=qmax, max_frequency_difference_cycles=fmax,
                  max_coefficient_phase_error_radians=phase_max, gate_changes=gate_changes,
                  unpaired_windows=absent, original_success_losses=lost, records=records,
                  scope="NCO/coefficients only;CFO bridge/coherent sums/input normalization float",
                  budget_sha256=sha(args.budget), whole_system_accepted=False)
    write(args.output / "numeric.json", report)
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return int(not report["node_budget_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["before", "after", "budget", "output"]:
        parser.add_argument("--"+flag, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
