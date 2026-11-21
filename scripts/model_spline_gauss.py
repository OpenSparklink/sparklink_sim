"""Exact conjugate Gauss fine score:explicit node study,not RF/16MSps acceptance."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts import model_coarse_ingress as ingress
from tests.maximum_reference import channel, packet

from nearlink_sdr.phy.acquisition_maximum import RollingMaximumRankedBank
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions

POLICY = dict(
    difference="exact three22x23 real products;three signed23 preadds;signed45 final",
    reference="unchanged full-growth four-product score,all semantic events exact",
    known_baseline="existing power reuse,defer-all-future,two-ALU word fabric",
    new_state_cap_bits=2048,
    existing_word_context_cap_bits=24576,
    minimum_saved_percent=3,
    minimum_scored_windows=6,
    all_frame_types_required=[2, 3, 4],
    maximum_enabled_queries_per_stream=4,
    prefix_samples=3072,
    source_selection="unchanged raw-energy OR and normalized scaled10 bank,every32samples",
    old_numeric_and_resource_failures_retained=True,
    interpolation_grid_threshold_and_loop_changed=False,
    new_RF_held=False,
    new_default=False,
    whole_RX_accepted=False,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    return ingress.sources()


def payload(n, family):
    if family == 0:
        return bytes(((j * 53) ^ (j >> 2) ^ 0xB7) & 255 for j in range(n))
    if family == 1:
        palette = (0x17, 0xD2, 0x4B, 0x90, 0x6E)
        return bytes(palette[(j // 19 + j % 3) % 5] for j in range(n))
    return bytes((int(f"{(j * 71 + 39) & 255:08b}"[::-1], 2) ^ 0x6A) for j in range(n))


def inputs(out, split):
    held = split == "validation"
    specs = (
        [(4, 12, 8, 223, "step"), (2, 8, 4, 457, "noise"),
         (3, 12, 0, 1019, "tone"), (4, 8, 16, 347, "clean"),
         (2, 12, 16, 181, "wrong-sync"), (3, 8, 8, 653, "burst")]
        if held else
        [(2, 8, 16, 113, "tone"), (3, 12, 4, 389, "noise"),
         (4, 8, 0, 73, "clean"), (2, 12, 8, 617, "burst"),
         (3, 8, 16, 929, "step"), (4, 12, 4, 157, "wrong-sync")]
    )
    rows = []
    for j, (ft, mcs, pilot, size, family) in enumerate(specs):
        cfg = ingress.shared.configuration(ft, mcs, pilot, size)
        profile = PSKReceiveProfile.ordinary(cfg)
        data = payload(size, (j + int(held)) % 3)
        params = dict(
            offset=[0.213, 0.687, 0.433, 0.971][(j + int(held)) % 4],
            ppm=[-100, 25, 75, -50][(j + 2 * int(held)) % 4],
            cfo=[-73000, 39000, 113000, -19000][(j + int(held)) % 4],
            prefix=37 + 23 * j + 41 * int(held),
            amplitude=[0.071, 0.29, 0.117, 0.043][(j + int(held)) % 4],
            family=family,
            seed=841103 + 79 * j + 601 * int(held),
            snr_db=12.5 if family == "noise" else None,
        )
        iq, quiet, _, _, metadata = channel(packet(profile, data), **params)
        if family == "wrong-sync":
            wrong = ingress.shared.configuration(ft, mcs, pilot, 523 + 62 * int(held))
            wrong.pid = (cfg.pid + 1) % 6
            other, _, _, _, _ = channel(
                packet(PSKReceiveProfile.ordinary(wrong), payload(523 + 62 * int(held), 2)),
                offset=0.351, ppm=-25, cfo=71000, prefix=13, amplitude=0.019,
                family="clean", seed=841991,
            )
            iq = np.pad(iq, (0, max(0, len(other) - len(iq))))
            iq[:len(other)] += other
            metadata["wrong_profile_PID"] = wrong.pid
        file = out / f"rf-{j}-input.npz"
        np.savez_compressed(file, iq=iq, quiet=quiet)
        config = asdict(cfg)
        config.pop("_mcs", None)
        rows.append(dict(
            id=j, config=config, family=family, artifact=file.name, input_SHA256=sha(file),
            payload_SHA256=hashlib.sha256(data).hexdigest(), payload_family=(j + int(held)) % 3,
            parameters=params, channel_metadata=metadata,
        ))
    write(out / "inputs.json", rows)
    return rows


def compare(out, row):
    profile, flags, words, triples, prefix = ingress.prepare(out, row)
    window = {2: 256, 3: 376, 4: 632}[profile.config.frame_type]
    reference = RollingMaximumRankedBank(profile.sync_symbols(), 4, window, origin=-40)
    template = SyncSignTemplate.from_symbols(profile.sync_symbols())
    records = []
    for tag, triple in enumerate(triples):
        reference.advance_words(*(np.asarray([v]) for v in triple))
        if (tag + 1) % 32 or tag + 1 < window or not any(flags[tag + 1 - window:tag + 1]):
            continue
        enabled, _, chosen = reference.query_best(
            tag + 2 - 40 - window, tag - 40 - reference.k * 4
        )
        if not enabled:
            continue
        peak = chosen - (tag + 1 - window - 40)
        selected = [tuple(int(v[k]) for v in words) for k in range(tag + 1 - window, tag + 1)]
        arms = []
        for gauss in (False, True):
            trace = []

            def observe(event):
                if event["event"] in ("sample", "score", "selected_sample"):
                    trace.append({k: v for k, v in event.items() if k != "cycle"})

            flow = SplineScoreFlow(
                SplineScoreAssumptions(window, reuse_symbol_power=True, defer_all_future=True),
                word_fabric=WordFabricAssumptions(2), gauss_difference=gauss,
            )
            result = flow.run(selected, template, (peak,), minimum_tick=8,
                              lookahead_ticks=128, sink=observe)
            name = f"stream{row['id']}-query{len(records)}-gauss{int(gauss)}.json"
            write(out / name, dict(result=result, semantic_trace=trace, cost=flow.estimate()))
            arms.append((result, trace, flow.estimate(), name))
        (old, before, _, _), (new, after, cost, _) = arms
        assert before == after and old["best"] == new["best"]
        assert not old["resource_drop"] and not new["resource_drop"]
        for key in ("numerical_rejection", "numerical_pass", "prepared_source_words",
                    "prepared_second_words", "common_exponent", "scoring_reads"):
            assert old[key] == new[key], key
        scored = old["best"] is not None
        records.append(dict(
            input_SHA256=row["input_SHA256"], prefix=prefix, frame_type=profile.config.frame_type,
            source_window_end=tag + 1, data_selected_peak=peak,
            baseline_cycles=old["cycles"], candidate_cycles=new["cycles"], scored=scored,
            numerical_rejected=old["numerical_rejection"],
            numerical_evaluation_unknown=old["numerical_pass"] is None,
            every_sample_score_rank_and_selected_word_exact=True,
            result_files=[arm[3] for arm in arms],
            incremental_register_bits=cost["gauss_difference"]["incremental_register_bits"],
            total_word_context_bits=cost["additional_word_and_context_bits"],
            baseline_tiles=old["operation_counts"].get("tile", 0),
            candidate_tiles=new["operation_counts"].get("tile", 0),
            new_RF_gain_measured=False,
        ))
        if len(records) == POLICY["maximum_enabled_queries_per_stream"]:
            break
    return dict(id=row["id"], input_SHA256=row["input_SHA256"], records=records,
                no_enabled_coarse_query=not records)


def qualify(streams):
    if (len(streams) != 6 or {s["id"] for s in streams} != set(range(6))
            or len({s["input_SHA256"] for s in streams}) != 6):
        return False
    rows = [r for stream in streams for r in stream["records"]]
    if len({(r["input_SHA256"], r["source_window_end"]) for r in rows}) != len(rows):
        return False
    if any(
        r["input_SHA256"] != stream["input_SHA256"]
        or any(type(r[k]) is not int or r[k] <= 0
               for k in ("baseline_cycles", "candidate_cycles"))
        or any(type(r[k]) is not int or r[k] < 0
               for k in ("incremental_register_bits", "total_word_context_bits"))
        or (r["scored"] and (r["numerical_rejected"] or r["numerical_evaluation_unknown"]))
        for stream in streams for r in stream["records"]
    ):
        return False
    scored = [r for r in rows if r["scored"]]
    if len(scored) < POLICY["minimum_scored_windows"] or {
        r["frame_type"] for r in scored
    } != set(POLICY["all_frame_types_required"]):
        return False
    old = sum(r["baseline_cycles"] for r in scored)
    new = sum(r["candidate_cycles"] for r in scored)
    return (
        old > 0 and 100 * new <= (100 - POLICY["minimum_saved_percent"]) * old
        and all(r["every_sample_score_rank_and_selected_word_exact"]
                and r["incremental_register_bits"] <= POLICY["new_state_cap_bits"]
                and r["total_word_context_bits"] <= POLICY["existing_word_context_cap_bits"]
                for r in rows)
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=sources())
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen eligible development")
        prior = json.loads(args.freeze.read_text())
        model = json.loads((args.freeze.parent / "model.json").read_text())
        end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        if (prior != frozen or end != frozen["sources"] or
                model.get("component_qualified") is not True or
                model.get("freeze_SHA256") != sha(args.freeze)):
            raise ValueError("Need exact frozen source and eligible development")
    args.output.mkdir(exist_ok=False)
    write(args.output / "freeze.json", frozen)
    rows = inputs(args.output, args.split)
    streams = [compare(args.output, row) for row in rows]
    assert sources() == frozen["sources"]
    write(args.output / "source-bindings-at-end.json", sources())
    result = dict(
        split=args.split, streams=streams, component_qualified=qualify(streams),
        all_semantic_values_exact=True, freeze_SHA256=sha(args.output / "freeze.json"),
        ordinary_profile=True, new_RF_held=False, CRC_or_truth_used_to_select=False,
        old_numeric_and_resource_failures_retained=True, new_default=False,
        whole_RX_accepted=False, physical_timing_verified=False,
    )
    write(args.output / "model.json", result)
    print(json.dumps(dict(component_qualified=result["component_qualified"],
                          node_windows=sum(len(s["records"]) for s in streams),
                          new_RF_gain_measured=False, whole_RX_accepted=False)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
