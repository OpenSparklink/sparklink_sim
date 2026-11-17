"""Fixed-vs-float weight-node budgets; equal packet results cannot waive a failure."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write

from nearlink_sdr.phy.carrier_weights import bounded_carrier_weights


def run(args):
    a = json.loads((args.before / "results.json").read_text())
    b = json.loads((args.after / "results.json").read_text())
    budget = json.loads(args.budget.read_text())["numerical_budget"]
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    maxima = dict(weight=0., frequency=0., phase=0.)
    events, lost = 0, []
    for old, new in zip(a, b, strict=True):
        assert old["input"] == new["input"]
        ident = old["input"]["id"]
        assert sha(Path(old["input"]["path"])) == old["input"]["sha256"]
        with np.load(args.before / "evidence" / (ident + ".npz")) as data:
            for index, (f, g) in enumerate(zip(old["frames"], new["frames"], strict=True)):
                if f["correct"] and not g["correct"]:
                    lost.append(dict(id=ident, event=index, input_sha256=old["input"]["sha256"]))
                if "weights" not in f.get("carrier", {}):
                    assert "weights" not in g.get("carrier", {})
                    continue
                c, d = f["carrier"], g["carrier"]
                assert not c["weights"]["fixed"] and d["weights"]["fixed"]
                kappa = c["weights"]["kappa"]
                assert kappa == d["weights"]["kappa"]
                samples = data[f"{index}-symbols"][
                    f["data_start_symbol"]:f["frame_end_symbol"]].copy()
                order = {"QPSK": 4, "8PSK": 8, "BPSK": 2}[f["modulation"]]
                odd = (np.arange(len(samples)) + f["data_start_symbol"]) % 2 == 1
                samples[odd] *= np.exp(1j * np.pi / order)
                power = abs(samples)**2
                wf, info = bounded_carrier_weights(power, kappa)
                wi, fixed = bounded_carrier_weights(power, kappa, fixed=True)
                assert c["weights"]["stats"] == info["stats"]
                assert d["weights"]["stats"] == fixed["stats"]
                diff = dict(
                    weight=float(np.max(abs(wf - wi), initial=0)
                                 / max(float(wf.max(initial=0)), 1e-30)),
                    frequency=abs(c["residual_cycles_per_symbol"]
                                  - d["residual_cycles_per_symbol"]),
                    phase=float(abs(np.angle(np.exp(1j * (c["residual_phase_radians"]
                                                          - d["residual_phase_radians"]))))),
                )
                for key, value in diff.items():
                    maxima[key] = max(maxima[key], value)
                count = fixed["stats"]["input_power_saturations"] + fixed["stats"]["input_power_erased"]
                events += count
                name = f"{ident}-{index}.npz"
                np.savez_compressed(args.output / name, power=power, float_weights=wf,
                                    fixed_weights=wi)
                records.append(dict(id=ident, event=index, input_sha256=old["input"]["sha256"],
                                    difference=diff, input_range_events=count,
                                    node_file=name, node_sha256=sha(args.output / name),
                                    status_pair=[f["status"], g["status"]],
                                    correct_pair=[f["correct"], g["correct"]]))
    gates = dict(
        evaluated_nodes=bool(records),
        weight=maxima["weight"] <= budget["weight_max_difference_over_max_reference"],
        frequency=maxima["frequency"] <= budget["carrier_frequency_cycles_per_symbol_difference"],
        phase=maxima["phase"] <= budget["carrier_phase_difference_radians"],
        original_successes=not lost, input_range=events == 0,
    )
    report = dict(gates=gates, numeric_pass=all(gates.values()), maxima=maxima,
                  input_range_events=events, original_success_losses=lost, records=records,
                  budget_sha256=sha(args.budget), whole_system_accepted=False,
                  scope="observed payload weight nodes only; Q16 zero-input deadband remains; upstream normalization/power conversion unverified")
    write(args.output / "numeric.json", report)
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))
    return int(not report["numeric_pass"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["before", "after", "budget", "output"]:
        parser.add_argument("--" + option, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
