"""Actual capture→finite init/head bridge; explicit remaining off-clock payload.

Independent analytic pulses/channel, unchanged fine selection and coherent gate.
Expected bytes and truth are posthoc diagnostics, never receiver arguments.
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
from tests.initialized_capture_reference import (
    existing_at_copy_edges,
    observed_receipt,
    same_age_confirmed_float,
)

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.coherent_interleaved import ScheduledCoherentCapturedFrameReplay
from nearlink_sdr.sim.fine_selection import FineSelection
from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay

POLICY = dict(
    version=1,
    development_specs=[
        [2, 8, 4, [13, 71], 20, "clean"],
        [3, 12, 16, [61, 173], 36, "tone"],
        [4, 8, 0, [47, 113], 12, "wrong-sync"],
        [2, 12, 8, [233, 29], 28, "noise"],
        [3, 8, 8, [127, 43], 16, "step"],
        [4, 12, 4, [83, 251], 8, "burst"],
    ],
    validation_specs=[
        [4, 12, 8, [103, 17], 16, "clean"],
        [2, 8, 16, [53, 181], 28, "tone"],
        [3, 12, 4, [199, 37], 8, "wrong-sync"],
        [4, 8, 16, [67, 157], 24, "noise"],
        [2, 12, 0, [149, 31], 12, "step"],
        [3, 8, 0, [109, 277], 20, "burst"],
    ],
    seed_bias={"development": 11075113, "validation": 12075379},
    independent_channel="analytic RRC pulses, direct CFO/time resampling; existing field CRC",
    unchanged="energy/prescreen/fine grid/scores/waiting/coherent9/20/Gardner gains/CRC",
    private_words=1024, private_BRAM36=2, coordinator_cap_bits=2048,
    extra_global_read_ports=0, prefetch_words=8, disable_final_symbols=3,
    source_history=4096, owned_windows=3, clock_MHz=200, input_MSps=16,
    new_feedback_age=1, original_captured_age=0,
    same_age_comparison="finite bootstrap/interpolation shared; float detector/PI/normalization",
    center_samples=.001, rate_ppm=.1, normalized_symbol_RMS=.0001,
    normalized_symbol_peak=.001, structural_copy_edges="exact",
    original_success_loss_max=0, false_MAC_max=0, new_negative_confirmation_max=0,
    aggregate_resource_exit_delta_max=0, minimum_numeric_pairs=4,
    minimum_original_complete=3, minimum_candidate_complete=3,
    adoption="experimental bridge only if every gate passes; otherwise rejected, no held tuning",
    new_default=False, full_fixed=False, complete_cycle_evidence=False,
    descriptor_score_angle_cycles=None, payload_cycles=None,
    old_full12_numeric_and16KiB_and18DSP_failures_retained=True,
)
FIELDS = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")
TRACES = ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error")


def payload(n, ident, held):
    # Three distinct bit/byte structures, not only another AWGN seed.
    if ident % 3 == 0:
        return bytes((((k*k*13 + ident*41) ^ (k >> 1)) + held*113) & 255 for k in range(n))
    if ident % 3 == 1:
        return bytes(((0xE1, 0x39, 0x86, 0x5A, 0xC7)[(k//7+k) % 5]
                      ^ ((k//11 + ident*17 + held*83) & 255)) for k in range(n))
    return np.random.Generator(np.random.Philox(1775911+ident*263+held*1021)).integers(
        0, 256, n, dtype=np.uint8).tobytes()


def load_input(root, row):
    path = root / row["file"]
    if path.parent != root or reference.shared.sha(path) != row["SHA256"]:
        raise ValueError("Frozen input path/SHA changed")
    with np.load(path, allow_pickle=False) as z:
        return z["iq"], z["quiet"]


def numeric_pair(event, floating):
    result = dict(frame_fields_exact=all(event["frame"].get(k) == floating["frame"].get(k)
                                        for k in FIELDS), same_shape=True)
    for key in TRACES:
        if event[key].shape != floating[key].shape:
            return result | dict(same_shape=False, passed=False)
    for key, name in [("symbol_centers_filtered", "center_samples"),
                      ("timing_rate_ppm", "rate_ppm")]:
        result[name] = float(np.max(abs(event[key]-floating[key]), initial=0))
    scale = max(float(np.mean(abs(floating["symbols"])**2))**.5, 1e-30)
    difference = abs(event["symbols"]-floating["symbols"])/scale
    result["normalized_symbol_RMS"] = float(np.mean(difference**2))**.5
    result["normalized_symbol_peak"] = float(np.max(difference, initial=0))
    result["passed"] = result["frame_fields_exact"] and all(
        result[k] <= POLICY[k] for k in ("center_samples", "rate_ppm", "normalized_symbol_RMS",
                                        "normalized_symbol_peak"))
    return result


def compare(
    out, root, row, *, retire_queued=False, overlap_fine=False, retire_completed_fine=False,
):
    if (type(retire_queued) is not bool or type(overlap_fine) is not bool
            or (overlap_fine and not retire_queued)):
        raise ValueError("Need an explicit queued-retirement experiment choice")
    if type(retire_completed_fine) is not bool or (retire_completed_fine and not overlap_fine):
        raise ValueError("Completed fine retirement requires overlapping ownership")
    iq, quiet = load_input(root, row)
    profile = replace(PSKReceiveProfile.ordinary(TxConfig(**row["config"])),
                      head_backend=row["head_backend"], allowed_mcs=(row["config"]["mcs_index"],))
    modes = dict(original=partial(ScheduledCoherentCapturedFrameReplay,
                                  structure="interleaved-signed22x18"),
                 finite=partial(InitializedCapturedFrameReplay, payload_mode="functional"),
                 stop=partial(InitializedCapturedFrameReplay, payload_mode="stop"))
    summaries, observed, all_words = {}, {}, None
    for name, factory in modes.items():
        words = []

        def observe(*values):
            word = pack_filter_word(*values)
            words.append(word)
            return word

        with patch("tests.captured_frame_reference.CapturedFrameReplay", factory), patch(
                "tests.captured_frame_reference.pack_filter_word", observe):
            options = dict(retire_queued=True) if retire_queued and name != "original" else {}
            if overlap_fine and name != "original":
                options["overlap_fine"] = True
            if retire_completed_fine and name != "original":
                options["retire_completed_fine"] = True
            run = reference.run_arm(profile, iq, quiet, True, **options)
        n = len(iq)
        assert len(words) == 3*n and words[:n] == words[n:2*n] == words[2*n:]
        if all_words is None:
            all_words = words[:n]
            np.save(out / f"words-{row['id']}.npy", np.asarray(all_words, dtype="<u8"))
        else:
            assert all_words == words[:n]
        summaries[name], _ = reference.analyze(out, row, name, run, n)
        observed[name] = run[0]
    original, finite, stop = (observed[k] for k in ("original", "finite", "stop"))
    pairs, excluded = [], []
    for event, _ in finite[1]:
        selected = FineSelection(**event["upstream_selection"])
        detail = event["initialized_capture_bridge"]
        if event["frame"]["status"] not in ("valid", "bad_data_crc", "bad_header_crc",
                                               "invalid_length", "unsupported_mcs"):
            excluded.append(dict(id=selected.ident, status=event["frame"]["status"],
                                 reason="no complete same service transaction; not numeric pass"))
            continue
        reads = [r for r in finite[2] if r["id"] == selected.ident]
        receipt = observed_receipt(profile, selected, event["coherent_word_service"])
        structural = existing_at_copy_edges(
            profile, selected, reads, detail["actual_terminal_cycle"], receipt=receipt)
        exact = event["frame"] == structural.event["frame"] and all(
            np.array_equal(event[k], structural.event[k]) for k in TRACES)
        origin = selected.source_end-selected.window
        floating = same_age_confirmed_float(profile, all_words[origin:],
            dict(selected.reference_selection(), sample_limit=2**20, coherent_threshold=None),
            receipt).event
        pairs.append(dict(id=selected.ident, structural_exact=exact,
                          numeric=numeric_pair(event, floating)))
    costs = finite[3].estimate()
    deltas = {k: finite[0][k]-original[0][k] for k in
              ("queue_drops", "history_drops", "frame_resource_drops")}
    decision = dict(
        old_success_losses=sorted(set(summaries["original"]["complete"])
                                  -set(summaries["finite"]["complete"])),
        recovered=sorted(set(summaries["finite"]["complete"])
                         -set(summaries["original"]["complete"])),
        resource_exit_deltas=deltas, numeric_pairs=pairs, unpaired=excluded,
        cost=costs, stop_never_delivered=not any(m.packets for _, m in stop[1]),
        same_raw_words_all_arms=True, all_three_RRC_input_chunkings_exact=True,
        original_vs_finite_is_algorithm_age_and_interpolation_change=True,
    )
    return dict(id=row["id"], input_SHA256=row["SHA256"], negative=row["negative"],
                legal_packets=len(row["expected"]), samples=len(iq), modes=summaries,
                comparison=safe(decision))


def qualify(rows):
    reasons = []
    if len(rows) != 10 or {r["id"] for r in rows} != set(range(10)):
        reasons.append("incomplete_stream_set")
    for label in ("original", "finite"):
        if sum(len(r["modes"][label]["complete"]) for r in rows) < POLICY[
                "minimum_"+("original" if label == "original" else "candidate")+"_complete"]:
            reasons.append(label+"_insufficient_positive")
    pairs = [p for r in rows for p in r["comparison"]["numeric_pairs"]]
    if len(pairs) < POLICY["minimum_numeric_pairs"]:
        reasons.append("insufficient_numeric_pairs")
    if any(not p["structural_exact"] or not p["numeric"]["passed"] for p in pairs):
        reasons.append("structural_or_numerical_failure")
    if any(r["comparison"]["old_success_losses"] for r in rows):
        reasons.append("original_success_lost")
    for key in ("queue_drops", "history_drops", "frame_resource_drops"):
        if sum(r["comparison"]["resource_exit_deltas"][key] for r in rows) > 0:
            reasons.append(key+"_increased")
    for r in rows:
        d, m = r["comparison"], r["modes"]["finite"]
        if m["false_MAC"] or m["duplicate_MAC"] or not d["stop_never_delivered"]:
            reasons.append("false_or_unknown_service_delivery")
        if r["negative"] and m["coherent"] > r["modes"]["original"]["coherent"]:
            reasons.append("new_negative_confirmation")
        if (d["cost"]["new_bridge_control_bits"] > POLICY["coordinator_cap_bits"]
                or d["cost"]["new_private_source_BRAM36"] != POLICY["private_BRAM36"]
                or d["cost"]["actual_source_read_ports_added"] != 0
                or not m["all_pulses_accounted"] or m["max_ports"] > 2 or m["max_owned"] > 3):
            reasons.append("component_resource_or_input_failure")
    return dict(passed=not reasons, reasons=sorted(set(reasons)), full_receiver_acceptance=False)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument(
        "--inputs", type=Path, help="replay SHA-frozen input manifest;never regenerate")
    args = parser.parse_args()
    sources = reference.shared.sources()
    if args.split == "validation":
        if args.freeze is None:
            parser.error("Need successful frozen development before new held evaluation")
        frozen = json.loads(args.freeze.read_text())
        if (frozen["source"] != sources or frozen["policy"] != POLICY
                or not frozen["qualification"]["passed"]):
            parser.error("Source/policy changed or development rejected;no held input generated")
    elif args.freeze is not None:
        parser.error("Development does not import a held decision")
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "PRE-RUN-SOURCE-AND-POLICY.json", dict(source=sources, policy=POLICY))
    if args.inputs is None:
        rows = reference.make_inputs(args.output, args.split, specs=POLICY[args.split+"_specs"],
                                     payload_fn=payload, seed_bias=POLICY["seed_bias"][args.split])
        root = args.output
    else:
        root, rows = args.inputs.parent, json.loads(args.inputs.read_text())
    results = []
    for row in rows:
        result = compare(args.output, root, row)
        results.append(result)
        write(args.output / "results.json", results)
        print(json.dumps(dict(id=row["id"], complete={k:v["complete"]
                             for k,v in result["modes"].items()},
                              losses=result["comparison"]["old_success_losses"])), flush=True)
    if reference.shared.sources() != sources:
        raise ValueError("Source changed during frozen run")
    qualification = qualify(results)
    write(args.output / "decision.json", dict(source=sources, policy=POLICY,
                                              qualification=qualification))
    print(json.dumps(qualification))
    return 0 if qualification["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
