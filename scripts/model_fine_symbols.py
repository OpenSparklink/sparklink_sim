"""Finite exact fine-symbol pipeline;node experiment,not an RF packet-rate test."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts import model_coarse_ingress as ingress
from scripts import model_spline_gauss as shared

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions

POLICY = dict(
    exact_semantics=True, minimum_saved_percent=25, minimum_scored=6,
    frame_types=[2, 3, 4], additional_DSP_cap=16,
    additional_external_register_bits_cap=12288, new_RAM_and_sample_ports=0,
    baseline="power reuse,defer-all-future,Gauss,2ALU",
    new_RF_holdout=False, new_default=False, full_RX_accepted=False,
    old_budgets_and_failures_retained=True,
)


def fixtures(out, split):
    held = split == "validation"
    rows = []
    frames = [4, 2, 3, 4, 3, 2] if held else [2, 3, 4, 2, 3, 4]
    for k, ft in enumerate(frames):
        n = {2: 256, 3: 376, 4: 632}[ft]
        family = (k + 2 * int(held)) % 3
        rng = np.random.default_rng(960173 + 43 * k + 511 * int(held))
        words = []
        for j in range(n):
            if family == 0:
                a = (j * (38113 + 232 * int(held)) + 297 * k) % 262139 - 131069
                b = (j * 7127 + (j // 11) * 17 + 571 * k) % 231143 - 115571
                e = -4
            elif family == 1:
                palette = (-131072, 131071, -81719, 46333, -127)
                a, b = palette[(j + k) % 5], palette[(j // 3 + 1) % 5]
                e = (j // (31 if held else 47)) % 4 - 3
            else:
                a, b = map(int, rng.integers(-65536, 65536, 2))
                if (j + k) % (67 if held else 53) < 2:
                    a, b = -131072, 131071
                e = -7 + int(j > n // 2)
            words.append((a, b, e))
        filename = f"input{k}.npz"
        np.savez_compressed(out / filename, words=np.asarray(words, dtype=np.int64))
        first, second = (13 + 7 * k + 3 * int(held)), (72 + 8 * k)
        peaks = (first, second) if k % 3 != 1 else (first,)
        cfg = ingress.shared.configuration(ft, 8 if k % 2 else 12,
                                            (0, 4, 8, 16)[(k + int(held)) % 4], 71 + 97 * k)
        cfg.pid = (2 * k + int(held)) % 6
        config = asdict(cfg)
        config.pop("_mcs")
        rows.append(dict(id=k, file=filename, SHA256=shared.sha(out / filename),
                         frame_type=ft, sync_PID=cfg.pid, config=config, family=family,
                         peaks=peaks, minimum_tick=first * 8 + 4 if k % 3 == 1 else 8,
                         lookahead_ticks=128 if k % 2 else 0,
                         threshold=(0, 1) if k == 5 and held else (17, 20)))
    shared.write(out / "inputs.json", rows)
    return rows


def compare(out, row):
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    assert shared.sha(out / row["file"]) == row["SHA256"]
    with np.load(out / row["file"]) as data:
        words = [tuple(map(int, w)) for w in data["words"]]
    profile = PSKReceiveProfile.ordinary(TxConfig(**row["config"]))
    template = SyncSignTemplate.from_symbols(profile.sync_symbols())
    arms = []
    for pipelined in (False, True):
        events = []

        def sink(event):
            if event["event"] in ("sample", "score", "selected_sample"):
                events.append({k: v for k, v in event.items() if k != "cycle"})

        flow = SplineScoreFlow(
            SplineScoreAssumptions(len(words), reuse_symbol_power=True, defer_all_future=True),
            word_fabric=WordFabricAssumptions(2), gauss_difference=True,
            symbol_pipeline=pipelined,
        )
        result = flow.run(words, template, tuple(row["peaks"]),
                          minimum_tick=row["minimum_tick"], lookahead_ticks=row["lookahead_ticks"],
                          threshold=tuple(row["threshold"]), sink=sink)
        filename = f"window{row['id']}-pipeline{int(pipelined)}.json"
        shared.write(out / filename, dict(result=result, events=events, cost=flow.estimate()))
        arms.append((result, events, flow.estimate(), filename))
    (old, before, _, _), (new, after, cost, _) = arms
    assert old["best"] == new["best"] and before == after
    assert not old["resource_drop"] and not new["resource_drop"]
    for key in ("prepared_source_words", "prepared_second_words", "common_exponent",
                "scoring_reads", "preparation_cycles", "numerical_pass", "numerical_rejection"):
        assert old[key] == new[key], key
    return dict(id=row["id"], input_SHA256=row["SHA256"], frame_type=row["frame_type"],
                all_semantic_values_exact=True, scored=old["best"] is not None,
                numerical_pass=old["numerical_pass"],
                numerical_rejection=old["numerical_rejection"],
                baseline_cycles=old["cycles"], candidate_cycles=new["cycles"],
                new_DSP=cost["fine_symbol_pipeline"]["dedicated_DSP25x18"],
                new_register_bits=cost["additional_symbol_pipeline_register_bits"],
                extra_RAM_and_sample_ports=0, result_files=[arm[3] for arm in arms],
                new_RF_gain_claimed=False)


def qualify(rows):
    if (len(rows) != 6 or {r["id"] for r in rows} != set(range(6))
            or len({r["input_SHA256"] for r in rows}) != 6):
        return False
    if any(not r["all_semantic_values_exact"]
           or any(type(r[k]) is not int or r[k] <= 0
                  for k in ("baseline_cycles", "candidate_cycles"))
           or any(type(r[k]) is not int or r[k] < 0 for k in
                  ("new_DSP", "new_register_bits", "extra_RAM_and_sample_ports"))
           or r["new_DSP"] > POLICY["additional_DSP_cap"]
           or r["new_register_bits"] > POLICY["additional_external_register_bits_cap"]
           or r["extra_RAM_and_sample_ports"] != 0 for r in rows):
        return False
    scored = [r for r in rows if r["scored"]]
    if (len(scored) < POLICY["minimum_scored"]
            or {r["frame_type"] for r in scored} != set(POLICY["frame_types"])
            or any(r["numerical_pass"] is not True or r["numerical_rejection"] for r in scored)):
        return False
    return 100 * sum(r["candidate_cycles"] for r in scored) <= (
        100 - POLICY["minimum_saved_percent"]
    ) * sum(r["baseline_cycles"] for r in scored)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=shared.sources())
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen eligible development before node validation")
        prior = json.loads(args.freeze.read_text())
        model = json.loads((args.freeze.parent / "model.json").read_text())
        end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        if (prior != frozen or end != frozen["sources"] or not model["component_qualified"]
                or model["freeze_SHA256"] != shared.sha(args.freeze)):
            raise ValueError("Need unchanged source,policy and eligible development")
    args.output.mkdir(exist_ok=False)
    shared.write(args.output / "freeze.json", frozen)
    rows = [compare(args.output, row) for row in fixtures(args.output, args.split)]
    assert shared.sources() == frozen["sources"]
    shared.write(args.output / "source-bindings-at-end.json", shared.sources())
    result = dict(rows=rows, split=args.split, component_qualified=qualify(rows),
                  all_semantic_values_exact=True,
                  freeze_SHA256=shared.sha(args.output / "freeze.json"),
                  new_RF_holdout=False, new_default=False, whole_RX_accepted=False,
                  full_chain_fixed_or_continuous16MSps_passed=False,
                  actual_synthesis_or_Fmax=False, old_numeric_and_resource_failures_retained=True)
    shared.write(args.output / "model.json", result)
    print(json.dumps(dict(component_qualified=result["component_qualified"], windows=len(rows),
                          whole_RX_accepted=False)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
