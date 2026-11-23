"""One legal upper-fence candidate;fresh development and three frozen regressions."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts import model_initialized_capture as old
from scripts.model_captured_frame import write

from nearlink_sdr.sim.legal_tail_ceiling import UpperTailCaptureReplay
from nearlink_sdr.sim.private_prefetch import ContinuousPrivatePrefetch

POLICY_FILE = Path(__file__).with_name("legal_tail_ceiling_policy.json")
POLICY = json.loads(POLICY_FILE.read_text())


def sources():
    return old.reference.shared.sources() | {
        "scripts/legal_tail_ceiling_policy.json": old.reference.shared.sha(POLICY_FILE),
        "scripts/private_prefetch_policy.json": old.reference.shared.sha(
            POLICY_FILE.with_name("private_prefetch_policy.json"))}


def payload(n, ident, held):
    data = []
    for k in range(n):
        word = ((k+1)*0x45D9 + ident*0x9E37 + held*0x1701) & 65535
        mask = 0xF0 if (k//17) % 2 else 0x0F
        data.append((((word >> 8) ^ (word & 255) ^ mask) + (k % 11)*23) & 255)
    return bytes(data)


def qualify(rows, baseline):
    result = old.qualify(rows)
    reasons = list(result["reasons"])
    lookup = {r["id"]: r for r in baseline}
    if (len(lookup) != len(baseline) or len({r["id"] for r in rows}) != len(rows)
            or {r["id"] for r in rows} != set(lookup)):
        raise ValueError("Need every unique baseline stream")
    recovered, lost, work, old_work, same_recall, deltas = [], [], 0, 0, True, {}
    for row in rows:
        previous = lookup[row["id"]]
        if row["input_SHA256"] != previous["input_SHA256"]:
            raise ValueError("Baseline IQ SHA changed")
        if row["modes"]["original"] != previous["modes"]["original"]:
            raise ValueError("Original receive baseline changed")
        a, b = previous["modes"]["finite"], row["modes"]["finite"]
        recovered += [[row["id"], p] for p in sorted(set(b["complete"]) - set(a["complete"]))]
        lost += [[row["id"], p] for p in sorted(set(a["complete"]) - set(b["complete"]))]
        work, old_work = work + b["fine_cycles"], old_work + a["fine_cycles"]
        same_recall &= (a["coherent"] == b["coherent"] and
            a["captured_packet_associations_diagnostic"] == b[
                "captured_packet_associations_diagnostic"])
        for key in ("queue_drops", "history_drops", "frame_resource_drops"):
            delta = (row["comparison"]["resource_exit_deltas"][key]
                     - previous["comparison"]["resource_exit_deltas"][key])
            deltas[key] = deltas.get(key, 0) + delta
        cost = row["comparison"]["cost"]["bounded_private_prefetch"]
        h = cost["horizon"]
        if (cost["additional_register_reservation_bits"] > POLICY["new_control_bit_cap"]
                or h["state_bits"] != POLICY["horizon_service"]["state_bits"]
                or h["latency_edges"] != POLICY["horizon_service"]["latency_edges"]
                or any(cost[k] for k in ("additional_DSP", "additional_BRAM36",
                                         "additional_source_read_ports"))):
            reasons.append("private_prefetch_cost_failed")
    if lost:
        reasons.append("static_success_lost")
    if any(v > 0 for v in deltas.values()):
        reasons.append("static_resource_exits_increased")
    if not recovered and not (same_recall and work < old_work):
        reasons.append("no_recovery_or_same_recall_schedule_benefit")
    return result | dict(passed=not reasons, reasons=sorted(set(reasons)),
        static_recovered=recovered, static_lost=lost, static_resource_exit_deltas=deltas,
        same_recall=same_recall, fine_elapsed_cycles=work, static_fine_elapsed_cycles=old_work,
        reduced_work_with_recall_loss_is_not_benefit=True, whole_RX16MSps=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--role", choices=("development", "fixed-regression"),
                        default="development")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--regression-freeze", type=Path, action="append", default=[])
    args = parser.parse_args()
    frozen = sources()
    if args.split == "validation":
        if (args.freeze is None or len(args.regression_freeze) != 3 or args.inputs is not None
                or args.role != "development"):
            parser.error("Need new development and fixed regression freezes before new held")
        for file, role in [(args.freeze, "development"),
                           *[(f, "fixed-regression") for f in args.regression_freeze]]:
            f = json.loads(file.read_text())
            if (f["source"] != frozen or f["policy"] != POLICY or f["role"] != role
                    or not f["qualification"]["passed"]):
                parser.error("Changed source/policy or rejected/seen freeze;no held generated")
        manifests = [json.loads(f.read_text())["input_manifest_SHA256"]
                     for f in args.regression_freeze]
        if sorted(manifests) != sorted(POLICY["fixed_manifest_SHA256"]):
            parser.error("Need all three distinct fixed input cohorts before held")
    elif args.freeze is not None or args.regression_freeze:
        parser.error("Development does not import a held decision")
    if args.baseline is not None and (args.inputs is None or args.role != "fixed-regression"):
        parser.error("External baseline needs a literal fixed-regression input manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "PRE-RUN-SOURCE-AND-POLICY.json", dict(source=frozen, policy=POLICY))
    if args.inputs is None:
        old.reference.make_inputs(args.output, args.split,
            specs=POLICY[args.split+"_specs"], payload_fn=payload,
            seed_bias=POLICY["seed_bias"][args.split])
        manifest = args.output / "inputs.json"
    else:
        manifest = args.inputs
    snapshot = manifest.read_bytes()
    rows, root = json.loads(snapshot), manifest.parent
    sha = hashlib.sha256(snapshot).hexdigest()
    if args.baseline is None:
        base_dir = args.output / "static-baseline"
        base_dir.mkdir()
        baseline = []
        for row in rows:
            with patch.object(old, "InitializedCapturedFrameReplay", ContinuousPrivatePrefetch):
                baseline.append(old.compare(base_dir, root, row))
            write(base_dir / "results.json", baseline)
            print(json.dumps(dict(baseline_id=row["id"],
                complete=baseline[-1]["modes"]["finite"]["complete"])), flush=True)
        baseline_file = base_dir / "results.json"
    else:
        baseline_file = args.baseline
        decision = json.loads(baseline_file.with_name("decision.json").read_text())
        if decision["input_manifest_SHA256"] != sha:
            raise ValueError("External baseline input manifest SHA changed")
        baseline = json.loads(baseline_file.read_text())
    baseline_sha = old.reference.shared.sha(baseline_file)
    write(args.output / "BASELINE-INPUT-BINDING.json", dict(
        input_manifest_SHA256=sha, baseline_results_SHA256=baseline_sha))
    results = []
    for row in rows:
        with patch.object(old, "InitializedCapturedFrameReplay", UpperTailCaptureReplay):
            value = old.compare(args.output, root, row)
        previous = json.loads((baseline_file.parent / f"stream{row['id']}-original.json")
                              .read_text())
        measured = json.loads((args.output / f"stream{row['id']}-original.json").read_text())
        if measured["result"] != previous["result"]:
            raise ValueError("Original transport changed;resource deltas are not comparable")
        results.append(value)
        write(args.output / "results.json", results)
        print(json.dumps(dict(id=row["id"], complete=value["modes"]["finite"]["complete"],
                              original_losses=value["comparison"]["old_success_losses"])),
              flush=True)
    if (sources() != frozen or old.reference.shared.sha(manifest) != sha
            or old.reference.shared.sha(baseline_file) != baseline_sha):
        raise ValueError("Source,policy,input or baseline changed during frozen run")
    qualification = qualify(results, baseline)
    write(args.output / "decision.json", dict(source=frozen, policy=POLICY, role=args.role,
        input_manifest_SHA256=sha, baseline_results_SHA256=baseline_sha,
        qualification=qualification))
    print(json.dumps(qualification))
    return 0 if qualification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
