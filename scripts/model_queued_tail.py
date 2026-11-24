"""Existing upper fence plus existing queued retirement;one new composition only."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts import model_initialized_capture as old
from scripts import model_legal_tail_ceiling as base
from scripts.model_captured_frame import write

from nearlink_sdr.sim.legal_tail_ceiling import UpperTailCaptureReplay

POLICY_FILE = Path(__file__).with_name("queued_tail_policy.json")
POLICY = json.loads(POLICY_FILE.read_text())
if (POLICY["same_age_budgets"] != base.POLICY["same_age_budgets"]
        or POLICY["minimum_complete_numeric_pairs"] != old.POLICY["minimum_numeric_pairs"]
        or POLICY["minimum_original_complete"] != old.POLICY["minimum_original_complete"]
        or POLICY["minimum_candidate_complete"] != old.POLICY["minimum_candidate_complete"]):
    raise ValueError("Inherited numerical gates differ from the prospective composition policy")


def sources():
    return base.sources() | {
        "scripts/queued_tail_policy.json": old.reference.shared.sha(POLICY_FILE)}


def payload(n, ident, held):
    return bytes((((k * 0x6D + ident * 31 + held * 0x47) ^ (k // 5)
                  ^ (0x3C if (k // 23) % 2 else 0xC3)) + (k % 7) * 19) & 255
                 for k in range(n))


def compare(out, root, row, *, enabled, overlap_fine=False):
    with patch.object(old, "InitializedCapturedFrameReplay", UpperTailCaptureReplay):
        options = dict(overlap_fine=True) if overlap_fine else {}
        value = old.compare(out, root, row, retire_queued=enabled, **options)
    if enabled:
        raw = json.loads((out / f"stream{row['id']}-finite.json").read_text())
        result = raw["result"]
        value["comparison"]["queued_gate"] = dict(
            cost=result["queued_retirement_gate_cost"],
            registered=result["queued_retirement_registered"],
            prior_consumed_frontier=result["queued_retirement_uses_prior_consumed_frontier"],
            retired=result["queued_retired"],
            checked=sum(v["event"] == "queued_check" for v in raw["checks"]),
            gate_tally_is_caller_diagnostic_not_free_hardware=True)
    return value


def qualify(rows, baseline):
    result = base.qualify(rows, baseline)
    reasons = list(result["reasons"])
    for row in rows:
        gate = row["comparison"]["queued_gate"]
        c = gate["cost"]
        if (c["register_bits"] != POLICY["extra_queue_gate_bits"]
                or c["combined_retirement_state_bits"] != POLICY["combined_pre_fine_queued_bits"]
                or c["extra_IQ_RAM_DSP_or_sample_ports"] != 0
                or c["additional_operand_mux2_bits"] != POLICY["extra_mux2_bits"]
                or not gate["registered"] or not gate["prior_consumed_frontier"]):
            reasons.append("queued_gate_cost_or_causality_failed")
    work = sum(r["modes"]["finite"]["fine_cycles"] for r in rows)
    original_work = sum(r["modes"]["original"]["fine_cycles"] for r in rows)
    if work > original_work:
        reasons.append("original_reference_fine_work_cap_failed")
    return result | dict(passed=not reasons, reasons=sorted(set(reasons)),
        original_reference_fine_cycles=original_work,
        additional_queue_control_bits=POLICY["extra_queue_gate_bits"],
        baseline="legal upper fence,queued retirement disabled",
        new_default=False, whole_RX16MSps=False)


def main():
    return run_study(__doc__, POLICY, sources, payload, compare, qualify)


def run_study(description, policy, source_map, payload_factory, evaluate_arm, qualification):
    """Evaluate a prospective transport factor on the same frozen input protocol."""
    parser = argparse.ArgumentParser(description)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--role", choices=("development", "fixed-regression"),
                        default="development")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--regression-freeze", type=Path, action="append", default=[])
    args = parser.parse_args()
    frozen = source_map()
    if args.split == "validation":
        if (args.freeze is None
                or len(args.regression_freeze) != len(policy["fixed_manifest_SHA256"])
                or args.inputs is not None or args.role != "development"):
            parser.error("Need new development and fixed regression freezes before new held")
        for file, role in [(args.freeze, "development"),
                           *[(f, "fixed-regression") for f in args.regression_freeze]]:
            f = json.loads(file.read_text())
            if (f["source"] != frozen or f["policy"] != policy or f["role"] != role
                    or not f["qualification"]["passed"]):
                parser.error("Changed source/policy or rejected/seen freeze;no held generated")
        manifests = [json.loads(f.read_text())["input_manifest_SHA256"]
                     for f in args.regression_freeze]
        if sorted(manifests) != sorted(policy["fixed_manifest_SHA256"]):
            parser.error("Need all distinct fixed input cohorts before held")
    elif args.freeze is not None or args.regression_freeze:
        parser.error("Development does not import a held decision")
    if args.baseline is not None and (args.inputs is None or args.role != "fixed-regression"):
        parser.error("External baseline needs a literal fixed-regression input manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "PRE-RUN-SOURCE-AND-POLICY.json", dict(source=frozen, policy=policy))
    if args.inputs is None:
        old.reference.make_inputs(args.output, args.split,
            specs=policy[args.split+"_specs"], payload_fn=payload_factory,
            seed_bias=policy["seed_bias"][args.split])
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
            baseline.append(evaluate_arm(base_dir, root, row, enabled=False))
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
        value = evaluate_arm(args.output, root, row, enabled=True)
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
    if (source_map() != frozen or old.reference.shared.sha(manifest) != sha
            or old.reference.shared.sha(baseline_file) != baseline_sha):
        raise ValueError("Source,policy,input or baseline changed during frozen run")
    decision = qualification(results, baseline)
    write(args.output / "decision.json", dict(source=frozen, policy=policy, role=args.role,
        input_manifest_SHA256=sha, baseline_results_SHA256=baseline_sha, qualification=decision))
    print(json.dumps(decision))
    return 0 if decision["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
