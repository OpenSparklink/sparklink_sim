"""Replay frozen SHA-bound IQ with/without actual word-head service availability.

All supplied cases are fixed regression,not a new RF holdout. Truth/expected
payload enters only the existing post-reception analysis. Return1 when charging
known work loses original payloads or increases resource drops;never hide that
behind identical surviving packets. Unknown full-chain latency remains None.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts import model_retired_capture as reference
from scripts.model_captured_frame import safe, write

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.captured_header_flow import ClockedHeaderCapturedFrameReplay


def gate_counts(observed, allowed_mcs):
    """Report actual checks independently;no expected payload or truth enters here."""
    frames = [event["frame"] for event, _ in observed[1]]
    return dict(
        head_CRC_pass=sum(row.get("header_crc") is True for row in frames),
        head_CRC_fail=sum(row.get("header_crc") is False for row in frames),
        length_pass=sum(row.get("length_ok") is True for row in frames),
        length_fail=sum(row.get("length_ok") is False for row in frames),
        permitted_MCS=sum(
            row.get("header_crc") is True
            and row.get("length_ok") is True
            and row.get("mcs") in allowed_mcs
            for row in frames
        ),
        data_CRC_pass=sum(row.get("data_crc") is True for row in frames),
        data_CRC_fail=sum(row.get("data_crc") is False for row in frames),
        MAC_packets=sum(len(mac.packets) for _, mac in observed[1]),
    )


def compare(out, root, case):
    path = root / case["file"]
    if path.parent != root or reference.shared.sha(path) != case["SHA256"]:
        raise ValueError("Frozen IQ path/hash changed")
    profile = replace(
        PSKReceiveProfile.ordinary(TxConfig(**case["config"])),
        head_backend=case["head_backend"],
        allowed_mcs=(case["config"]["mcs_index"],),
    )
    with np.load(path, allow_pickle=False) as data:
        iq, quiet = data["iq"], data["quiet"]
    old = reference.run_arm(profile, iq, quiet, True)
    with patch(
        "tests.captured_frame_reference.CapturedFrameReplay", ClockedHeaderCapturedFrameReplay
    ):
        new = reference.run_arm(profile, iq, quiet, True)
    a, ta = reference.analyze(out, case, "functional", old, len(iq))
    b, tb = reference.analyze(out, case, "clocked-head", new, len(iq))
    a["independent_gate_counts"] = gate_counts(old[0], profile.allowed_mcs)
    b["independent_gate_counts"] = gate_counts(new[0], profile.allowed_mcs)
    calendars = []
    for event, _ in new[0][1]:
        calendar = event["header_service_calendar"]
        issued, due, available = (
            calendar[k] for k in ("issue_cycle", "nominal_ready_cycle", "available_cycle")
        )
        if issued is not None:
            assert due == issued + calendar["service_edges"] and calendar["service_edges"] > 0
            if available is not None:
                assert available >= due
        else:
            assert available is None and calendar["service_edges"] == 0
        calendars.append(calendar | dict(id=event["selection_id"], status=event["frame"]["status"]))
    common = sorted(set(a["complete"]) & set(b["complete"]))
    for packet in common:
        assert ta[packet]["start"] == tb[packet]["start"]
        assert ta[packet]["score"] == tb[packet]["score"]
        np.testing.assert_array_equal(ta[packet]["centers"], tb[packet]["centers"])
    shared = old[3].keys() & new[3].keys()
    assert all(old[3][k] == new[3][k] and old[4][k] == new[4][k] for k in shared)
    for run in (old, new):
        assert run[0][0]["input_samples"] == len(iq) and run[0][4].done
        assert run[0][0]["peak_source_operations_per_primitive"] <= 2
        assert run[0][0]["complete_latency_cycles"] is None
    assert old[0][0]["live_source_statistics"] == new[0][0]["live_source_statistics"]
    resources = ("queue_drops", "history_drops", "frame_resource_drops")
    deltas = {key: new[0][0][key] - old[0][0][key] for key in resources}
    return dict(
        id=case["id"],
        input_SHA256=case["SHA256"],
        samples=len(iq),
        negative=case["negative"],
        expected_count=len(case["expected"]),
        functional=a,
        clocked_head=b,
        original_payload_losses=sorted(set(a["complete"]) - set(b["complete"])),
        recoveries=sorted(set(b["complete"]) - set(a["complete"])),
        resource_drop_deltas=deltas,
        common_survivor_trajectories_exact=True,
        common_fine_windows_integer_exact=len(shared),
        all_three_host_chunks_exact=True,
        head_calendars=calendars,
        header_clock_stats=new[0][3].header_clock_stats.copy(),
        owner_cost=new[0][3].estimate(),
        source_unchanged=True,
        full_cycle_or16MSps_accepted=False,
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-id", type=int, action="append")
    args = parser.parse_args()
    root = args.input_dir.resolve()
    manifest = root / "inputs.json"
    if reference.shared.sha(manifest) != args.manifest_sha256:
        parser.error("Frozen manifest SHA mismatch;do not substitute a newer index")
    cases = json.loads(manifest.read_text())
    if args.case_id is not None:
        if len(set(args.case_id)) != len(args.case_id) or not set(args.case_id) <= {
            row["id"] for row in cases
        }:
            parser.error("Need distinct existing case IDs")
        cases = [row for row in cases if row["id"] in args.case_id]
    if not cases:
        parser.error("Need at least one explicit input")
    source = reference.shared.sources()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in cases:
        row = safe(compare(args.output, root, case))
        rows.append(row)
        write(args.output / f"case{case['id']}.json", row)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in ("id", "original_payload_losses", "recoveries", "resource_drop_deltas")
                }
            ),
            flush=True,
        )
    assert source == reference.shared.sources()
    resource_pass = all(
        not row["original_payload_losses"]
        and all(delta <= 0 for delta in row["resource_drop_deltas"].values())
        for row in rows
    )
    result = dict(
        input_manifest_SHA256=args.manifest_sha256,
        input_root=str(root),
        source=source,
        rows=rows,
        all_fixed_regression=True,
        new_RF_holdout=False,
        signal_algorithm_changed=False,
        old_numeric_and16KiB_failures_preserved=True,
        complete=[
            sum(len(row[arm]["complete"]) for row in rows) for arm in ("functional", "clocked_head")
        ],
        resource_comparison_passed=resource_pass,
        expected_exit=0 if resource_pass else 1,
        full_chain_latency_cycles=None,
        all_four_RX_gates_passed=False,
    )
    write(args.output / "model.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "source")}))
    return result["expected_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
