"""Paired float/finite/clocked confirmation on the same finite acquisition cache.

This baseline is not an entirely floating receiver. Existing RRC/interpolation
rejections remain. New held inputs are created only after the numerical node's
prospective development freeze;clock/resource failures are separately retained.
"""

import argparse
import json
import math
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
from nearlink_sdr.sim.coherent_captured_frame import WordConfirmedCapturedFrameReplay

POLICY = dict(
    version=1,
    development_specs=[
        [2, 12, 4, [127, 43], 0, "clean"],
        [3, 8, 8, [59, 157], 20, "tone"],
        [4, 12, 0, [71, 211], 4, "wrong-sync"],
        [2, 8, 16, [193, 89], 8, "noise"],
        [3, 12, 4, [13, 229], 0, "step"],
        [4, 8, 8, [167, 31], 16, "burst"],
    ],
    validation_specs=[
        [3, 12, 16, [163, 19], 4, "clean"],
        [4, 8, 4, [109, 239], 8, "tone"],
        [2, 12, 8, [83, 173], 0, "wrong-sync"],
        [3, 8, 0, [149, 61], 28, "noise"],
        [4, 12, 16, [7, 257], 12, "step"],
        [2, 8, 4, [227, 107], 0, "burst"],
    ],
    seed_bias={"development": 3100019, "validation": 4100039},
    payload="new rotating counter,seven-bit block words,PCG64DXSM bytes;not seed-only",
    baseline="same finite RRC/coarse/fine cache;float confirmation;actual word-head wait",
    functional="actual finite confirmation resolved off-clock for numerical isolation",
    clocked="same finite values executed at actual positive service clocks",
    threshold=[9, 20],
    angle_turn40_circular_error_rad=1e-6,
    phase_increment20_circular_error_rad=3.1e-6,
    quality_absolute_error=0.005,
    minimum_quality_pairs_per_split=4,
    old_source_fine_words_and_successful_trajectories="exact",
    three_host_chunkings="exact per mode",
    original_functional_success_loss=0,
    additional_negative_confirmations=0,
    false_and_duplicate_MAC=0,
    word_gate_control_cap_bits=8192,
    max_source_ports=2,
    history=4096,
    owned=3,
    new_IQ_RAM=0,
    dedicated_DSP25x18=1,
    dedicated_coefficient_RAM18=1,
    comparison_bits=110,
    existing_CORDIC_and_ROM_macro_assumptions_not_synthesis=True,
    unproved_cross_module_sharing_not_credited=True,
    changing_template_configuration_service=None,
    clock_source_MHz=200,
    other_clock_Fmax_or_full_RX_services_not_verified=True,
    numeric_node_and_clock_resource_gates_separate=True,
    clock_resource_rule="zero original functional success loss;no extra queue/history/frame resource exits",
    resource_losses_do_not_become_numerical_losses=True,
    validation_requires_numerical_development_pass=True,
    old_numeric_and_16KiB_rejections_retained=True,
    MAC_gap_conformance_not_claimed=True,
    full_RX_16MSps_or_hardware_acceptance=False,
    default_enabled=False,
)


def payload(n, ident, held):
    if ident % 3 == 0:
        data = []
        for k in range(n):
            value, shift = (19 * k + 73 * ident + 105 * held) & 255, (k // 5) % 8
            data.append(((value << shift) | (value >> ((8 - shift) % 8))) & 255)
        return bytes(data)
    if ident % 3 == 1:
        return bytes(((k * 11) ^ (k >> 2) ^ (ident * 43) ^ (0xE7 if held else 0x96)) & 127
                     ^ (128 if (k // 13) % 2 else 0) for k in range(n))
    return np.random.Generator(np.random.PCG64DXSM(5728849 + ident * 151 + held * 9187)).integers(
        0, 256, n, dtype=np.uint8
    ).tobytes()


class _ObserveConfirmation:
    def _terminal(self, event):
        event["observed_confirmation"] = None if self.rx is None else self.rx._confirmation
        return super()._terminal(event)


class _FloatOwner(_ObserveConfirmation, ClockedHeaderCapturedFrameReplay):
    pass


class _WordOwner(_ObserveConfirmation, WordConfirmedCapturedFrameReplay):
    pass


def _metrics(out, row, label, run, samples, allowed):
    summary, traces = reference.analyze(out, row, label, run, samples)
    summary["independent_gates"] = gate_counts(run[0], allowed)
    summary["word_computation_edges"] = sum(
        event.get("coherent_word_service", {}).get("computed_edges") or 0
        for event, _ in run[0][1]
    )
    summary["native_products"] = sum(
        (event.get("coherent_word_service", {}).get("statistics") or {}).get("native_products", 0)
        for event, _ in run[0][1]
    )
    return summary, traces


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
    runs, summaries, traces = {}, {}, {}
    modes = dict(floating=_FloatOwner, functional=partial(_WordOwner, clocked=False),
                 clocked=partial(_WordOwner, clocked=True))
    for label, owner in modes.items():
        with patch("tests.captured_frame_reference.CapturedFrameReplay", owner):
            runs[label] = reference.run_arm(profile, iq, quiet, True)
        summaries[label], traces[label] = _metrics(
            out, row, label, runs[label], len(iq), profile.allowed_mcs
        )
    base, functional = runs["floating"], runs["functional"]
    shared = base[3].keys() & functional[3].keys()
    exact = all(base[3][k] == functional[3][k] and base[4][k] == functional[4][k] for k in shared)
    source_exact = all(
        run[0][0][key] == base[0][0][key]
        for run in runs.values()
        for key in ("requests", "input_samples", "live_source_statistics", "live_coarse_statistics")
    )
    surviving = traces["floating"].keys() & traces["functional"].keys()
    trajectories = all(
        traces["floating"][k]["start"] == traces["functional"][k]["start"]
        and traces["floating"][k]["centers"].dtype == traces["functional"][k]["centers"].dtype
        and traces["floating"][k]["centers"].tobytes() == traces["functional"][k]["centers"].tobytes()
        for k in surviving
    )
    corresponding_arrays_exact = True
    function_events = {e["selection_id"]: e for e, _ in functional[0][1]}
    for event, _ in base[0][1]:
        other = function_events.get(event["selection_id"])
        if other is None or not (event.get("observed_confirmation") or {}).get("accepted", False):
            continue
        if not (other.get("observed_confirmation") or {}).get("accepted", False):
            continue
        keys = {k for k, v in event.items() if isinstance(v, np.ndarray)}
        corresponding_arrays_exact &= keys == {
            k for k, v in other.items() if isinstance(v, np.ndarray)
        }
        corresponding_arrays_exact &= all(
            event[k].dtype == other[k].dtype and event[k].shape == other[k].shape
            and event[k].tobytes() == other[k].tobytes() for k in keys
        )
    original = {e["selection_id"]: e for e, _ in base[0][1]}
    quality, clock_services = [], []
    for label in ("functional", "clocked"):
        for event, _ in runs[label][0][1]:
            node = event.get("coherent_word_service")
            if node is None or not node["actual_finite_values"]:
                continue
            if label == "clocked":
                assert node["ready_cycle"] == node["issued_cycle"] + node["computed_edges"]
                assert node["computed_edges"] > 0
                clock_services.append(node)
                continue
            old = original.get(event["selection_id"])
            if old is None or old["observed_confirmation"] is None:
                continue
            a, b = old["upstream_selection"], event["upstream_selection"]
            keys = ("source_end", "window", "start_tick", "exponent", "words", "correlation")
            assert all(a[k] == b[k] for k in keys)
            phase = math.atan2(*a["correlation"][::-1])
            angle_error = math.remainder(node["angle_turn40"] * math.tau / 2**40 - phase, math.tau)
            step_error = math.remainder(node["step_word20"] * math.tau / 2**20 - phase, math.tau)
            before = old["observed_confirmation"]
            quality.append(dict(
                id=event["selection_id"], float_quality=before["quality"],
                word_quality=node["word_quality"],
                absolute_error=abs(before["quality"] - node["word_quality"]),
                angle_circular_error_rad=angle_error, step_circular_error_rad=step_error,
                float_decision=before["accepted"], word_decision=node["word_decision"],
            ))
    a, b, c = (summaries[k] for k in ("floating", "functional", "clocked"))
    cost = runs["clocked"][0][0]["frame_replay_cost"]
    return dict(
        id=row["id"], input_SHA256=row["SHA256"], negative=row["negative"],
        samples=len(iq), expected_count=len(row["expected"]),
        floating=a, functional=b, clocked=c,
        functional_success_losses=sorted(set(a["complete"]) - set(b["complete"])),
        functional_recoveries=sorted(set(b["complete"]) - set(a["complete"])),
        clock_resource_success_losses=sorted(set(b["complete"]) - set(c["complete"])),
        clock_recoveries=sorted(set(c["complete"]) - set(b["complete"])),
        quality_pairs=quality, clock_services=clock_services,
        common_fine_integer_and_words_exact=exact, source_values_and_requests_exact=bool(source_exact),
        functional_survivor_trajectories_exact=trajectories,
        corresponding_accepted_symbol_arrays_exact=bool(corresponding_arrays_exact),
        three_chunkings_and_registered_source_words_exact=True,
        gate_control_bits=cost["word_gate_control_node_bits"],
        combined_frame_control_bits=cost["combined_owner_register_bits"],
        clock_resource_drop_deltas={key: runs["clocked"][0][0][key] - functional[0][0][key]
                                    for key in ("queue_drops", "history_drops", "frame_resource_drops")},
        old_numeric_and_resource_failures_retained=True, full_RX_gate=False,
    )


def qualify_node(rows):
    if len(rows) != 10 or {r["id"] for r in rows} != set(range(10)):
        return False
    pairs = sum(len(r["quality_pairs"]) for r in rows)
    if pairs < POLICY["minimum_quality_pairs_per_split"]:
        return False
    for row in rows:
        if row["functional_success_losses"] or row["gate_control_bits"] > 8192:
            return False
        if not all(row[k] is True for k in (
            "common_fine_integer_and_words_exact", "source_values_and_requests_exact",
            "functional_survivor_trajectories_exact", "three_chunkings_and_registered_source_words_exact",
            "corresponding_accepted_symbol_arrays_exact",
        )):
            return False
        for pair in row["quality_pairs"]:
            if (pair["absolute_error"] > .005 or abs(pair["angle_circular_error_rad"]) > 1e-6
                    or abs(pair["step_circular_error_rad"]) > 3.1e-6):
                return False
        a, b = row["floating"], row["functional"]
        if b["false_MAC"] or b["duplicate_MAC"]:
            return False
        if row["negative"] and b["coherent"] > a["coherent"]:
            return False
        if any(not row[k]["all_pulses_accounted"] or row[k]["max_ports"] > 2
               or row[k]["max_owned"] > 3 for k in ("floating", "functional", "clocked")):
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
            parser.error("Need prospective numerical development freeze")
        frozen = json.loads(args.freeze.read_text())
        if frozen["source"] != source or frozen["policy"] != POLICY or not frozen["numeric_node_pass"]:
            parser.error("Source/policy changed or numerical development rejected")
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
            "id", "functional_success_losses", "clock_resource_success_losses",
        )}), flush=True)
    assert reference.shared.sources() == source
    numeric = qualify_node(rows)
    clock_preserved = all(not r["clock_resource_success_losses"] for r in rows)
    clock_resource = all(
        not r["clock_resource_success_losses"]
        and all(v <= 0 for v in r["clock_resource_drop_deltas"].values()) for r in rows
    )
    result = dict(
        source=source, policy=POLICY, split=args.split, rows=rows,
        input_manifest_SHA256=reference.shared.sha(args.output / "inputs.json"),
        numeric_node_pass=numeric, clock_path_no_original_success_losses=clock_preserved,
        clock_resource_comparison_pass=clock_resource,
        actual_resource_and_full_RX_acceptance=False,
        new_default=False, pure_float_full_chain_baseline=False,
        unique_negative_samples=sum(r["samples"] for r in rows if r["negative"]),
        expected_exit=0 if numeric and clock_resource else 1,
    )
    write(args.output / "model.json", result)
    write(args.output / "freeze.json", dict(
        source=source, policy=POLICY, numeric_node_pass=numeric and args.split == "development",
        model_SHA256=reference.shared.sha(args.output / "model.json"),
        input_manifest_SHA256=result["input_manifest_SHA256"],
    ))
    print(json.dumps({k: v for k, v in result.items() if k not in ("source", "policy", "rows")}))
    return result["expected_exit"]


if __name__ == "__main__":
    raise SystemExit(main())
