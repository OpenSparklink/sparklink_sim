"""One head-only prefetch candidate;unchanged capture/loops,explicit unknown payload cycles."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from scripts import model_initialized_capture as old
from scripts.model_captured_frame import write

from nearlink_sdr.sim.head_prefetched_capture import HeadPrefetchedCaptureReplay

POLICY = json.loads(Path(__file__).with_name("head_prefetch_policy.json").read_text())


def payload(n, ident, held):
    data = []
    for k in range(n):
        x = (k * 19 + k // 5 + ident * 43 + held * 59) & 255
        r = (k + ident) % 8
        x = ((x << r) | (x >> (8 - r))) & 255
        data.append(x ^ ((0xC3, 0x96, 0x5D, 0x2B)[(k // 13 + ident + held) % 4]))
    return bytes(data)


def qualify(rows, baseline):
    result = old.qualify(rows)
    reasons = list(result["reasons"])
    if {r["id"] for r in rows} != {r["id"] for r in baseline}:
        raise ValueError("Need every SHA-matched baseline stream")
    lookup = {r["id"]: r for r in baseline}
    recovered, lost, cycles, old_cycles, resource_delta = [], [], 0, 0, {}
    for row in rows:
        previous = lookup[row["id"]]
        if row["input_SHA256"] != previous["input_SHA256"]:
            raise ValueError("Baseline IQ SHA changed")
        if row["modes"]["original"] != previous["modes"]["original"]:
            raise ValueError("Original receive baseline changed")
        a, b = previous["modes"]["finite"], row["modes"]["finite"]
        recovered += [[row["id"], p] for p in sorted(set(b["complete"]) - set(a["complete"]))]
        lost += [[row["id"], p] for p in sorted(set(a["complete"]) - set(b["complete"]))]
        cycles += b["fine_cycles"]
        old_cycles += a["fine_cycles"]
        for key in ("queue_drops", "history_drops", "frame_resource_drops"):
            # These are measured finite-minus-original counters,including the
            # frame exit absent from the summary. Original transport is checked
            # exactly by main();never manufacture an absent counter as zero.
            delta = (row["comparison"]["resource_exit_deltas"][key]
                     - previous["comparison"]["resource_exit_deltas"][key])
            resource_delta[key] = resource_delta.get(key, 0) + delta
        cost = row["comparison"]["cost"]["bounded_head_prefetch"]
        if (cost["additional_register_reservation_bits"] > POLICY["new_control_bit_cap"]
                or any(cost[k] != 0 for k in ("additional_DSP", "additional_BRAM36",
                                             "additional_source_read_ports"))):
            reasons.append("prefetch_cost_gate_failed")
    if lost:
        reasons.append("M36_success_lost")
    if any(v > 0 for v in resource_delta.values()):
        reasons.append("M36_resource_exits_increased")
    if not recovered and cycles >= old_cycles:
        reasons.append("no_recovery_or_schedule_benefit")
    return result | dict(passed=not reasons, reasons=sorted(set(reasons)),
        M36_recovered=recovered, M36_lost=lost, M36_resource_exit_deltas=resource_delta,
        fine_request_elapsed_cycles=cycles, M36_fine_request_elapsed_cycles=old_cycles,
        original_success_loss_rule_unchanged=True, whole_fixed_or16MSps=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--role", choices=("development", "fixed-regression"),
                        default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    sources = old.reference.shared.sources()
    sources["scripts/head_prefetch_policy.json"] = old.reference.shared.sha(
        Path(__file__).with_name("head_prefetch_policy.json"))
    if args.split == "validation":
        if args.freeze is None or args.role != "development" or args.inputs is not None:
            parser.error("Need eligible development freeze before generating new held input")
        freeze = json.loads(args.freeze.read_text())
        if (freeze["source"] != sources or freeze["policy"] != POLICY
                or freeze["role"] != "development" or not freeze["qualification"]["passed"]):
            parser.error("Changed source/policy or rejected/seen set;no new held generated")
    elif args.freeze is not None:
        parser.error("Development does not import a held decision")
    if args.baseline is not None and args.inputs is None:
        parser.error("External baseline requires its literal SHA-frozen input manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "PRE-RUN-SOURCE-AND-POLICY.json", dict(source=sources, policy=POLICY))
    if args.inputs is None:
        rows = old.reference.make_inputs(args.output, args.split,
            specs=POLICY[args.split + "_specs"], payload_fn=payload,
            seed_bias=POLICY["seed_bias"][args.split])
        root, manifest = args.output, args.output / "inputs.json"
    else:
        root, manifest = args.inputs.parent, args.inputs
    input_snapshot = manifest.read_bytes()
    rows = json.loads(input_snapshot)
    input_sha = hashlib.sha256(input_snapshot).hexdigest()
    if args.baseline is None:
        baseline_dir = args.output / "M36-baseline"
        baseline_dir.mkdir()
        baseline = [old.compare(baseline_dir, root, row) for row in rows]
        write(baseline_dir / "results.json", baseline)
        baseline_file = baseline_dir / "results.json"
        write(baseline_dir / "INPUT-BINDING.json", dict(
            input_manifest_SHA256=input_sha,
            baseline_results_SHA256=old.reference.shared.sha(baseline_file)))
    else:
        baseline_file = args.baseline
        baseline = json.loads(baseline_file.read_bytes())
    baseline_sha = old.reference.shared.sha(baseline_file)
    binding = json.loads(baseline_file.with_name("INPUT-BINDING.json").read_text())
    if (binding["input_manifest_SHA256"] != input_sha
            or binding["baseline_results_SHA256"] != baseline_sha):
        raise ValueError("Baseline reference or frozen input metadata changed")
    results = []
    for row in rows:
        with patch.object(old, "InitializedCapturedFrameReplay", HeadPrefetchedCaptureReplay):
            result = old.compare(args.output, root, row)
        measured = json.loads((args.output / f"stream{row['id']}-original.json").read_text())
        previous = json.loads((baseline_file.parent / f"stream{row['id']}-original.json")
                              .read_text())
        if measured["result"] != previous["result"]:
            raise ValueError("Original transport counters changed;no resource delta comparison")
        results.append(result)
        write(args.output / "results.json", results)
        print(json.dumps(dict(id=row["id"], complete=result["modes"]["finite"]["complete"],
                              losses=result["comparison"]["old_success_losses"])), flush=True)
    now = old.reference.shared.sources()
    now["scripts/head_prefetch_policy.json"] = old.reference.shared.sha(
        Path(__file__).with_name("head_prefetch_policy.json"))
    if now != sources:
        raise ValueError("Source/policy changed during frozen run")
    if (old.reference.shared.sha(manifest) != input_sha
            or old.reference.shared.sha(baseline_file) != baseline_sha):
        raise ValueError("Input metadata or baseline changed during frozen run")
    qualification = qualify(results, baseline)
    write(args.output / "decision.json", dict(source=sources, policy=POLICY, role=args.role,
        qualification=qualification, input_manifest_SHA256=input_sha,
        baseline_results_SHA256=baseline_sha))
    print(json.dumps(qualification))
    return 0 if qualification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
