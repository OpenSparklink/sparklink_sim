"""Model one registered release of an already-completed whole consumed window."""

import json
from pathlib import Path

from scripts import model_fine_frame_overlap as overlap

POLICY_FILE = Path(__file__).with_name("completed_fine_retirement_policy.json")
FIXED_INPUTS = Path(__file__).with_name("completed_fine_fixed_inputs.json")
FROZEN_POLICY = json.loads(POLICY_FILE.read_text())
POLICY = FROZEN_POLICY | dict(
    seed_bias=dict(development=FROZEN_POLICY["development_seed_offset"],
                   validation=FROZEN_POLICY["validation_seed_offset"]),
    fixed_manifest_SHA256=list(json.loads(FIXED_INPUTS.read_text()).values()),
)
if POLICY["numerical_caps"] != overlap.POLICY["same_age_budgets"]:
    raise ValueError("Preserve the inherited same-age numerical limits")


def sources():
    sha = overlap.queued.old.reference.shared.sha
    return overlap.sources() | {
        str(p.relative_to(Path(__file__).parent.parent)): sha(p)
        for p in (POLICY_FILE, FIXED_INPUTS)
    }


def payload(n, ident, held):
    return bytes(
        (((((k + 1) * 0x1C69) ^ ((n + 3) * 0x9E37) ^ (ident * 283 + held * 2309))
          >> ((k % 4) * 3)) + 47 * (k // 7) + n) & 255
        for k in range(n)
    )


def checked_retirements(report):
    events = report["checks"]
    issues = [x["cycle"] for x in events
              if x["event"] in ("check", "queued_check", "completed_fine_check")]
    completed = [x for x in report["outcomes"] if x["status"] == "retired_after_fine"]
    accepted = {x["id"]: x for x in events
                if x["event"] == "completed_fine_check" and x["exclude"]}
    if len(issues) != len(set(issues)) or len(accepted) != len(completed):
        raise ValueError("Retirement compared twice per edge or lost a registered decision")
    for outcome in completed:
        match = accepted[outcome["id"]]
        if (match["cycle"] % 2 or match["cycle"] + 1 != outcome["cycle"]
                or not match["frame_active"]
                or outcome["after_samples"] > match["predicate_frontier"]
                or outcome["fine_started_cycle"] + outcome["fine_cycles"] > match["cycle"]):
            raise ValueError("Retirement needs complete work and an older consumed frontier")
    return len(issues), len(completed)


def compare(out, root, row, *, enabled):
    result = overlap.compare(out, root, row, enabled=enabled, retire_completed_fine=enabled)
    if enabled:
        report = json.loads((out / f"stream{row['id']}-finite.json").read_text())
        compare_count, retired = checked_retirements(report)
        result["comparison"]["completed_fine_retirement"] = dict(
            cost=report["result"]["completed_fine_retirement_cost"],
            compare_edges=compare_count, retired_after_fine=retired,
        )
    return result


def qualify(rows, baseline):
    ordinary = overlap.qualify(rows, baseline)
    reasons = set(ordinary["reasons"])
    for row in rows:
        cost = row["comparison"]["completed_fine_retirement"]["cost"]
        if (cost["register_bits"] > POLICY["extra_register_bits_max"]
                or cost["comparator_input_mux_bits_upper"]
                > POLICY["extra_comparator_input_mux_bits_max"]
                or cost["extra_IQ_RAM_DSP_or_sample_ports"]
                or not cost["compared_frontier_is_previous_committed"]
                or not cost["completed_worker_only"]
                or not cost["no_new_telemetry_counter_or_pending_descriptor"]):
            reasons.add("completed_fine_resource_or_causality_failed")
    return ordinary | dict(
        passed=not reasons, reasons=sorted(reasons),
        completed_fine_retirements=sum(
            r["comparison"]["completed_fine_retirement"]["retired_after_fine"] for r in rows),
        comparator_edges=sum(
            r["comparison"]["completed_fine_retirement"]["compare_edges"] for r in rows),
    )


def main():
    return overlap.queued.run_study(__doc__, POLICY, sources, payload, compare, qualify)


if __name__ == "__main__":
    raise SystemExit(main())
