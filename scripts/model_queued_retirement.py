"""Registered ready-window release,with fresh paired inputs and prospective freeze.

Only the queue lifetime changes. The known positive word-head service remains
charged in both arms;carrier,tracking and payload arithmetic remain off clock.
This is not full fixed-RX,legal-MAC-load or continuous16MSps acceptance.
"""

import argparse
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts import model_retired_capture as reference
from scripts.model_captured_frame import safe, write
from scripts.model_captured_header import gate_counts

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.captured_header_flow import ClockedHeaderCapturedFrameReplay

POLICY = dict(
    version=1,
    development_specs=[
        [2, 8, 4, [149, 389], 0, "clean"],
        [4, 12, 8, [61, 181], 12, "tone"],
        [3, 8, 0, [233, 47], 4, "wrong-sync"],
        [2, 12, 16, [257, 109], 0, "noise"],
        [4, 8, 4, [17, 311], 24, "step"],
        [3, 12, 16, [97, 251], 8, "burst"],
    ],
    validation_specs=[
        [2, 12, 8, [197, 337], 4, "clean"],
        [3, 8, 16, [83, 271], 0, "tone"],
        [4, 12, 4, [151, 29], 16, "wrong-sync"],
        [2, 8, 0, [193, 53], 12, "noise"],
        [3, 12, 8, [11, 283], 0, "step"],
        [4, 8, 16, [229, 71], 20, "burst"],
    ],
    seed_bias={"development": 1000003, "validation": 2000087},
    payload="three new families:Gray/ramp,7byte blocks,seeded uniform bytes;see payload()",
    baseline="actual word-head availability + registered pre-fine retirement",
    candidate="only registered ready-head release against prior-edge consumed frontier",
    whole_window_exclusion=True,
    history_samples=4096,
    owned_windows=3,
    additional_control_bits=66,
    incremental_node_cap_bits=128,
    combined_retirement_state_bits=132,
    comparator_bits=63,
    additional_operand_mux2_bits=63,
    registered_check_edges=1,
    comparisons_per_active_ready_clock_upper=1,
    additional_IQ_RAM_DSP_sample_ports=0,
    source_ports_upper=2,
    clocks_MHz=[100, 160, 200],
    original_complete_losses=0,
    additional_negative_confirmations=0,
    false_or_duplicate_MAC=0,
    common_source_fine_words_and_survivor_trajectories="exact",
    fine_candidates_and_signal_parameters_changed=False,
    development_benefit="at least one full packet recovered;not relabeling stale drops",
    held_rule="no original success losses or extra negative confirmations;no retune",
    false_positive_observation="four negative families,8192samples each per split",
    full_chain_unknown_work_cycles=None,
    old_numerical_and_16KiB_failures_preserved=True,
    full_RX_16MSps_or_physical_acceptance=False,
    default_enabled=False,
)


def payload(n, ident, held):
    if ident % 3 == 0:
        return bytes(((k ^ (k >> 1)) + 43 * ident + (0xA7 if held else 0x63)) & 255
                     for k in range(n))
    if ident % 3 == 1:
        return bytes((0x36, 0xC9, 0x72, 0x8D, 0xE1, 0x1E, 0x54)[(k // 7 + k) % 7]
                     ^ ((k // 23 + 61 * ident + 19 * held) & 255) for k in range(n))
    return np.random.default_rng(2407711 + 257 * ident + 104729 * held).integers(
        0, 256, n, dtype=np.uint8
    ).tobytes()


def control_checks(run):
    observed, checks, *_ = run
    pending, prior = None, {}
    for row in checks:
        kind = row["event"]
        if kind == "check":
            prior[row["id"]] = row
            assert not row["frame_active"]
        elif kind in ("fine_start", "retired"):
            issue = prior[row["id"]]
            assert row["cycle"] == issue["cycle"] + 1
            assert row["retired_through"] == issue["retired_through"]
            assert not row["frame_active"]
        elif kind == "queued_check":
            assert pending is None and row["frame_active"]
            assert row["copied_cycle"] < row["cycle"]
            assert row["predicate_frontier"] <= row["retired_through"]
            assert row["exclude"] == (row["source_end"] <= row["predicate_frontier"])
            pending = row
        elif kind in ("queued_retired", "queued_kept"):
            assert pending is not None and pending["id"] == row["id"]
            assert row["cycle"] == pending["cycle"] + 1
            assert row["exclude"] is pending["exclude"]
            assert (kind == "queued_retired") is row["exclude"]
            pending = None
    assert pending is None
    queued = {r["cycle"] for r in checks if r["event"] == "queued_check"}
    assert queued.isdisjoint(r["cycle"] for r in checks if r["event"] == "check")
    result = observed[0]
    assert result["requests"] == sum(result.get(k, 0) for k in (
        "completed", "queue_drops", "history_drops", "startup_drops", "source_epoch_drops",
        "pre_fine_retired", "queued_retired",
    ))
    return True


def compare(out, root, row):
    path = root / row["file"]
    if path.parent != root or reference.shared.sha(path) != row["SHA256"]:
        raise ValueError("Frozen IQ path/hash changed")
    profile = replace(
        PSKReceiveProfile.ordinary(TxConfig(**row["config"])),
        head_backend=row["head_backend"], allowed_mcs=(row["config"]["mcs_index"],),
    )
    with np.load(path, allow_pickle=False) as data:
        iq, quiet = data["iq"], data["quiet"]
    runs = []
    original_receive = reference.receive
    with patch("tests.captured_frame_reference.CapturedFrameReplay", ClockedHeaderCapturedFrameReplay):
        for enabled in (False, True):
            with patch.object(reference, "receive", partial(original_receive, retire_queued=enabled)):
                runs.append(reference.run_arm(profile, iq, quiet, True))
    summaries, traces = [], []
    for label, run in zip(("baseline", "queued"), runs, strict=True):
        summary, trace = reference.analyze(out, row, label, run, len(iq))
        summary["independent_gates"] = gate_counts(run[0], profile.allowed_mcs)
        summary["queued_checks"] = sum(r["event"] == "queued_check" for r in run[1])
        summary["queued_retired"] = run[0][0].get("queued_retired", 0)
        summary["check_count_includes_continuous_polling"] = True
        assert control_checks(run)
        summaries.append(summary)
        traces.append(trace)
    old, new = runs
    common = old[3].keys() & new[3].keys()
    exact = all(old[3][k] == new[3][k] and old[4][k] == new[4][k] for k in common)
    a, b = summaries
    survivors_exact = all(
        traces[0][k]["start"] == traces[1][k]["start"]
        and traces[0][k]["score"] == traces[1][k]["score"]
        and traces[0][k]["centers"].dtype == traces[1][k]["centers"].dtype
        and traces[0][k]["centers"].tobytes() == traces[1][k]["centers"].tobytes()
        for k in traces[0].keys() & traces[1].keys()
    )
    source_exact = all(old[0][0][key] == new[0][0][key] for key in (
        "live_source_statistics", "live_coarse_statistics", "requests", "input_samples",
    ))
    def descriptors(run):
        return sorted((e["id"], e["after_samples"], e["offered_cycle"]) for e in run[0][5])
    source_exact &= descriptors(old) == descriptors(new)
    return dict(
        id=row["id"], input_SHA256=row["SHA256"], negative=row["negative"], samples=len(iq),
        expected_count=len(row["expected"]), baseline=a, queued=b,
        original_success_losses=sorted(set(a["complete"]) - set(b["complete"])),
        recoveries=sorted(set(b["complete"]) - set(a["complete"])),
        same_source_descriptors=bool(source_exact), common_fine_windows=len(common),
        common_fine_integer_and_words_exact=exact, survivor_trajectories_exact=survivors_exact,
        registered_lifetime_and_accounting_verified=True,
        all_input_words_and_three_host_chunkings_exact=True,
        additional_control_bits=new[0][0]["queued_retirement_gate_cost"]["register_bits"],
        complete_latency_cycles=None, full_RX_gates_passed=False,
    )


def qualify(rows):
    if len(rows) != 10 or {r["id"] for r in rows} != set(range(10)):
        return False
    for row in rows:
        if row["original_success_losses"] or row["additional_control_bits"] != 66:
            return False
        if not all(row[k] is True for k in (
            "same_source_descriptors", "common_fine_integer_and_words_exact",
            "survivor_trajectories_exact", "registered_lifetime_and_accounting_verified",
            "all_input_words_and_three_host_chunkings_exact",
        )):
            return False
        a, b = row["baseline"], row["queued"]
        if b["false_MAC"] or b["duplicate_MAC"]:
            return False
        if row["negative"] and b["coherent"] > a["coherent"]:
            return False
        if not b["all_pulses_accounted"] or b["max_ports"] > 2 or b["max_owned"] > 3:
            return False
    return True


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    source = reference.shared.sources()
    if args.split == "validation":
        if args.freeze is None:
            parser.error("Need prospective development freeze before independent validation")
        frozen = json.loads(args.freeze.read_text())
        if frozen["policy"] != POLICY or frozen["source"] != source or not frozen["development_pass"]:
            parser.error("Source/policy changed or development rejected;do not retune validation")
    elif args.freeze is not None:
        parser.error("Development cannot import a holdout freeze")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "pre-run-policy.json", dict(policy=POLICY, source=source))
    inputs = reference.make_inputs(
        args.output, args.split, specs=POLICY[args.split + "_specs"], payload_fn=payload,
        seed_bias=POLICY["seed_bias"][args.split],
    )
    rows = []
    for row in inputs:
        value = safe(compare(args.output, args.output, row))
        rows.append(value)
        write(args.output / f"case{row['id']}.json", value)
        print(json.dumps({k: value[k] for k in (
            "id", "recoveries", "original_success_losses", "negative",
        )}), flush=True)
    assert reference.shared.sources() == source
    paired = qualify(rows)
    benefit = sum(len(r["recoveries"]) for r in rows) > 0
    accepted = paired and (benefit or args.split == "validation")
    result = dict(
        policy=POLICY, source=source, split=args.split,
        input_manifest_SHA256=reference.shared.sha(args.output / "inputs.json"), rows=rows,
        paired_constraints_passed=paired, recovered_packets=sum(len(r["recoveries"]) for r in rows),
        original_success_losses=sum(len(r["original_success_losses"]) for r in rows),
        bounded_queue_candidate_passed=accepted, full_RX_gates_passed=False,
        total_unique_negative_samples=sum(r["samples"] for r in rows if r["negative"]),
        previous_pressure_61_of_96_not_re_evaluated=True,
        expected_exit=0 if accepted else 1,
    )
    write(args.output / "model.json", result)
    write(args.output / "freeze.json", dict(
        policy=POLICY, source=source, development_pass=accepted and args.split == "development",
        input_manifest_SHA256=result["input_manifest_SHA256"],
        model_SHA256=reference.shared.sha(args.output / "model.json"),
    ))
    print(json.dumps({k: v for k, v in result.items() if k not in ("source", "rows", "policy")}))
    return result["expected_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
