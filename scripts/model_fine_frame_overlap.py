"""One fine service concurrent with a frame on independent owned banks."""

import json
from pathlib import Path

from scripts import model_queued_tail as queued

POLICY_FILE = Path(__file__).with_name("fine_frame_overlap_policy.json")
POLICY = json.loads(POLICY_FILE.read_text())
if (POLICY["same_age_budgets"] != queued.POLICY["same_age_budgets"]
        or POLICY["minimum_complete_numeric_pairs"]
        != queued.POLICY["minimum_complete_numeric_pairs"]):
    raise ValueError("Inherited numerical gates differ from overlap policy")


def sources():
    return queued.sources() | {
        "scripts/fine_frame_overlap_policy.json": queued.old.reference.shared.sha(POLICY_FILE)
    }


def compare(out, root, row, *, enabled):
    result = queued.compare(out, root, row, enabled=True, overlap_fine=enabled)
    if enabled:
        transport = json.loads((out / f"stream{row['id']}-finite.json").read_text())["result"]
        result["comparison"]["fine_frame_overlap"] = {
            k: transport[k] for k in ("fine_frame_overlap_cost", "fine_frame_overlap_cycles",
                                      "selected_result_wait_cycles")
        }
    return result


def qualify(rows, baseline):
    result = queued.qualify(rows, baseline)
    reasons = set(result["reasons"])
    lookup = {r["id"]: r for r in baseline}
    changed_negative = any(
        r["negative"] and r["modes"]["finite"]["coherent"] > lookup[r["id"]]["modes"][
            "finite"]["coherent"] for r in rows
    )
    if changed_negative:
        reasons.add("new_negative_confirmation_vs_serial")
    for row in rows:
        cost = row["comparison"]["fine_frame_overlap"]["fine_frame_overlap_cost"]
        if (cost["register_bits"] > POLICY["extra_register_cap_bits"]
                or cost["extra_descriptor_read_mux_bits_upper"]
                > POLICY["extra_metadata_read_bound_bits"]
                or cost["extra_IQ_RAM_DSP_or_sample_ports"]
                or cost["fine_workers"] != POLICY["fine_parallel_workers"]
                or cost["frame_owners"] != POLICY["frame_owners"]
                or cost["selection_credits"] != POLICY["selection_credits"]
                or not cost["slot_reuse_waits_for_installed_frame"]
                or not cost["separate_fine_and_frame_owned_banks"]):
            reasons.add("overlap_allocation_failed")
    drain = sum(r["modes"]["finite"]["known_drain_cycle"] for r in rows)
    serial_drain = sum(r["modes"]["finite"]["known_drain_cycle"] for r in baseline)
    if result["same_recall"] and drain < serial_drain:
        reasons.discard("no_recovery_or_same_recall_schedule_benefit")
    return result | dict(
        passed=not reasons, reasons=sorted(reasons), baseline="queued serial legal upper fence",
        known_drain_cycle_sum=drain, serial_known_drain_cycle_sum=serial_drain,
        fine_frame_overlap_cycles=sum(
            r["comparison"]["fine_frame_overlap"]["fine_frame_overlap_cycles"] for r in rows),
        selected_result_wait_cycles=sum(
            r["comparison"]["fine_frame_overlap"]["selected_result_wait_cycles"] for r in rows),
        whole_RX16MSps=False,
    )


def main():
    return queued.run_study(__doc__, POLICY, sources, queued.payload, compare, qualify)


if __name__ == "__main__":
    raise SystemExit(main())
