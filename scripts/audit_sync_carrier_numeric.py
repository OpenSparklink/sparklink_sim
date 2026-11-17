"""NCO-only initialization errors on actual selected sync symbols, never CRC waivers."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, sha, write

from nearlink_sdr.phy.sync_carrier import initialize_sync_carrier


def run(args):
    budget = json.loads(args.budget.read_text())
    rows = json.loads((args.capture / "results.json").read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    for row in rows:
        assert sha(Path(row["input"]["path"])) == row["input"]["sha256"]
        ref = profile(row["input"]["mcs"]).sync_symbols()
        with np.load(args.capture / "evidence" / (row["input"]["id"]+".npz")) as saved:
            for index, frame in enumerate(row["frames"]):
                original = row["confirmations"][index]["carrier_initialization"]
                assert original["fixed_nco"] is False
                # Undo only the recorded actual initialization. Unknown gain is
                # common to all symbols and cancels in component normalization.
                omega, phase = original["omega_radians_per_symbol"], original["phase_radians"]
                y = saved[f"{index}-symbols"][:len(ref)]*np.exp(
                    1j*(phase+omega*np.arange(len(ref))))
                seed, steps = original["seed_radians_per_symbol"], original["iterations"]
                before = initialize_sync_carrier(y, ref, seed, steps)
                assert abs((before["omega_radians_per_symbol"]-omega+np.pi) % (2*np.pi)
                           - np.pi) < 1e-10
                after = initialize_sync_carrier(y, ref, seed, steps, fixed_nco=True)
                node = args.output / f"node-{len(records)}.npz"
                np.savez_compressed(node, raw_sync_over_gain=y, reference=ref)
                records.append(dict(
                    id=row["input"]["id"], input_sha256=row["input"]["sha256"],
                    start=frame["sync_start"], steps=steps,
                    frequency_cycles_difference=abs((after["omega_radians_per_symbol"]
                        - before["omega_radians_per_symbol"]+np.pi) % (2*np.pi)-np.pi)/(2*np.pi),
                    phase_difference=abs(float(np.angle(np.exp(1j*(after["phase_radians"]
                                                                  - before["phase_radians"]))))),
                    quality_difference=abs(after["quality"]-before["quality"]),
                    node_file=node.name, node_sha256=sha(node), word_result=after,
                ))
    maxima = {key: max((r[key] for r in records), default=0.) for key in
              ["frequency_cycles_difference", "phase_difference", "quality_difference"]}
    gates = dict(
        nonempty=bool(records),
        frequency=maxima["frequency_cycles_difference"] <= budget["frequency_cycles_difference_budget"],
        phase=maxima["phase_difference"] <= budget["phase_difference_radians_budget"],
        quality=maxima["quality_difference"] <= budget["quality_difference_budget"],
    )
    write(args.output / "numeric.json", dict(
        nodes=len(records), maxima=maxima, gates=gates, node_numeric_passed=all(gates.values()),
        records=records, budget_sha256=sha(args.budget),
        scope="actual sync/gain reconstruction;NCO coefficients only;moments/phase/division float",
        whole_system_accepted=False,
    ))
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps(dict(nodes=len(records), maxima=maxima, gates=gates)))
    return int(not all(gates.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["capture", "budget", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
