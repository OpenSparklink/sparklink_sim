"""Compare strict MAC outcomes; truth associates completed events only, never RX choices."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write


def associated(row):
    result = {}
    for i, f in enumerate(row["frames"]):
        if not row["input"]["packets"]:
            continue
        target = min(
            range(len(row["input"]["packets"])),
            key=lambda k: abs(row["input"]["packets"][k]["sync_start"] - f["sync_start"]),
        )
        if abs(row["input"]["packets"][target]["sync_start"] - f["sync_start"]) <= 4:
            result.setdefault(target, []).append(i)
    return result


def compare(before, after, budget_path, output):
    budget = json.loads(budget_path.read_text())["budgets"]
    a = {r["input"]["id"]: r for r in json.loads((before / "results.json").read_text())}
    b = {r["input"]["id"]: r for r in json.loads((after / "results.json").read_text())}
    assert a.keys() == b.keys(), "Input set changed"
    maxima = dict(sync_start=0.0, centers=0.0, sync_score=0.0, carrier_hz=0.0)
    losses, recoveries, regressions, changed, duplicates = [], [], [], [], []
    captured_loss = []
    for ident, old in a.items():
        new = b[ident]
        assert old["input"] == new["input"], "Input or expected payload metadata changed"
        assert sha(Path(old["input"]["path"])) == old["input"]["sha256"]
        for row in [old, new]:
            valid = [f for f in row["frames"] if f["status"] == "valid"]
            assert row["delivered_payloads"] == [f["payload_hex"] for f in valid]
            assert all(f["header_crc"] and f["data_crc"] and f["length_ok"] for f in valid)
        for name in ["actual_capture_config", "actual_timing_config", "carrier_method"]:
            assert old[name] == new[name], "Another receive parameter changed"
        x, y = associated(old), associated(new)
        old_good = {t for t, v in x.items() if any(old["frames"][i]["correct"] for i in v)}
        new_good = {t for t, v in y.items() if any(new["frames"][i]["correct"] for i in v)}
        tag = dict(id=ident, input_sha256=old["input"]["sha256"])
        losses += [dict(tag, target=t) for t in sorted(old_good - new_good)]
        recoveries += [dict(tag, target=t) for t in sorted(new_good - old_good)]
        captured_loss += [dict(tag, target=t) for t in sorted(x.keys() - y.keys())]
        for target, indices in y.items():
            if sum(new["frames"][i]["status"] == "valid" for i in indices) > 1:
                duplicates.append(dict(tag, target=target, events=indices))
        delta = dict(
            tag,
            captured=[old["captured"], new["captured"]],
            correct=[old["correct"], new["correct"]],
            false_candidates=[old["false_candidates"], new["false_candidates"]],
            false_deliveries=[old["false_deliveries"], new["false_deliveries"]],
        )
        if (
            new["false_candidates"] > old["false_candidates"]
            or new["false_deliveries"] > old["false_deliveries"]
        ):
            regressions.append(delta)
        if old["frames"] != new["frames"]:
            changed.append(delta)
        with (
            np.load(before / "evidence" / (ident + ".npz")) as u,
            np.load(after / "evidence" / (ident + ".npz")) as v,
        ):
            for target in x.keys() & y.keys():
                # Ambiguous duplicate candidates are reported, never resolved
                # by inspecting CRC/payload or choosing a more favorable phase.
                if len(x[target]) != 1 or len(y[target]) != 1:
                    continue
                i, j = x[target][0], y[target][0]
                p, q = old["receiver_events"][i], new["receiver_events"][j]
                for field in ["sync_start", "sync_score"]:
                    maxima[field] = max(maxima[field], abs(p[field] - q[field]))
                maxima["carrier_hz"] = max(maxima["carrier_hz"], abs(p["cfo_hz"] - q["cfo_hz"]))
                r, s = u[f"{i}-symbol_centers_raw"], v[f"{j}-symbol_centers_raw"]
                n = min(len(r), len(s))
                maxima["centers"] = max(
                    maxima["centers"], float(np.max(abs(r[:n] - s[:n]), initial=0))
                )
    gates = dict(
        original_successes=not losses,
        false_candidates_and_deliveries=not regressions,
        duplicate_deliveries=not duplicates,
        sync_start=maxima["sync_start"] <= budget["sync_start_difference_max"],
        trajectory=maxima["centers"] <= budget["paired_centers_difference_max"],
        score=maxima["sync_score"] <= budget["score_difference_max"],
        fir_numpy_cache=max(r["fir_array_bytes"] for r in b.values())
        <= budget["host_fir_numpy_arrays_max_bytes"],
    )
    functional = all(gates.values())
    report = dict(
        inputs=len(a),
        budget_sha256=sha(budget_path),
        gates=gates,
        integration_functional_budget_passed=functional,
        matched_maxima=maxima,
        original_success_losses=losses,
        recoveries=recoveries,
        lost_captures=captured_loss,
        false_regressions=regressions,
        duplicate_deliveries=duplicates,
        changed_inputs=changed,
        rrc_numeric_gate_passed=budget["known_rrc_numeric_gate_passed"],
        receiver_promotion=False,
        whole_rx_16msps=None,
        note="carrier difference is diagnostic only; existing RRC local2.348%failure remains; not full fixed RX",
    )
    write(output, report)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["changed_inputs", "original_success_losses", "recoveries"]
            },
            indent=2,
        )
    )
    return int(not functional or not budget["known_rrc_numeric_gate_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["before", "after", "budget", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(compare(args.before, args.after, args.budget, args.output))
