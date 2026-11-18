"""Single-factor guard-before-alignment experiment;full receiver gates stay open."""

import argparse
import json
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import bridge
from model_initial_reuse import calendar, cases, exact, verify_manifest
from model_interpolation_word import comparison, word_execute
from model_selected_frame import TRACE_KEYS, execute, metrics, normalized
from tests.test_interpolation_word import exact_reference

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    interpolate_filter_word,
    unpack_interpolated_word,
)
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.interpolation_flow import LagrangeInterpolationFlow

POLICY = Path("tests/data/alignment_guard_policy.json")
OLD = LagrangeWordFormat(16, 2, 22)
NEW = LagrangeWordFormat(16, 2, 22, alignment_guard_first=True)


def numeric(reference, event, budgets):
    observed = bool(len(reference.get("symbols", ())))
    if not observed:
        return dict(observed=False, passed=None, reason="no float symbols;not a numerical pass")
    result = comparison(reference, event)
    result.update(observed=True)
    result["passed"] = result["shape_exact"] and all(
        result.get(k, float("inf")) <= v for k, v in budgets.items()
    )
    return result


def evaluate(out, name, profile, words, selection, payload, policy, truth=None):
    values = np.array([physical_filter_word(w) for w in words])
    reference, _ = execute(
        out, name + "-float", profile, values, selection, capacity=len(values), chunking="whole"
    )
    events, records = {}, {}
    truth = truth or {}
    for label, fmt in (("old", OLD), ("guard", NEW)):
        event, memory = word_execute(
            out, name + "-" + label, profile, words, selection, fmt, [32], reuse=True
        )
        if label == "guard":
            for kind, pattern in (
                ("fixed", [7, 31]),
                ("random", np.random.default_rng(944119).integers(1, 33, 23)),
            ):
                other, _ = word_execute(
                    out, name + "-" + kind, profile, words, selection, fmt, pattern, reuse=True
                )
                exact(event, other)
        events[label] = event
        records[label] = dict(
            metrics=metrics(event, profile, expected=payload, **truth),
            numeric=numeric(reference, event, policy["budgets"]),
            interpolation=event["word_interpolation"],
            memory=memory,
            consumed=event["delivered_after_filtered_samples"],
            header_consumed=event.get("header_decision_after_filtered_samples"),
        )
    row = dict(
        id=name,
        profile=normalized(asdict(profile)),
        selection=normalized({k: v for k, v in selection.items() if k != "selected_sync"}),
        reference=metrics(reference, profile, expected=payload, **truth),
        **records,
        success_loss=records["old"]["metrics"]["payload_exact"]
        and not records["guard"]["metrics"]["payload_exact"],
        added_false_delivery=records["guard"]["metrics"]["false_delivery"]
        and not records["old"]["metrics"]["false_delivery"],
        new_false_delivery=records["guard"]["metrics"]["false_delivery"],
        frame_decisions_equal_float=normalized(reference["frame"])
        == normalized(events["guard"]["frame"]),
        partitions_exact=True,
        truth_postdecision_only=True,
    )
    return row, events


def fresh(out, split, policy):
    rows = []
    for meta, profile, iq, payload in cases(split, policy):
        name = split + "-" + str(meta["id"])
        filtered = np.convolve(iq, rrc_filter(0.4, 4))
        words, quantization = bridge(filtered)
        np.savez_compressed(
            out / (name + "-input.npz"),
            iq=iq,
            filtered=filtered,
            words=words,
            payload=np.frombuffer(payload or b"", np.uint8),
        )
        row = dict(
            id=name,
            input=meta,
            input_sha256=sha(out / (name + "-input.npz")),
            quantization=quantization,
        )
        found = search_psk_sync(
            np.array([physical_filter_word(w) for w in words]), 4, profile.sync_symbols(), 8
        )
        if found is None:
            row["upstream_no_candidate"] = True
        else:
            selection = dict(
                start_tick=round(8 * (found["phase"] + 4 * found["offset"])),
                delta=found["delta"],
                differential_score=found["score"],
                coherent_threshold=0.45,
                selected_sync=found["phase_symbols"][
                    found["offset"] : found["offset"] + len(profile.sync_symbols())
                ],
            )
            np.savez_compressed(
                out / (name + "-selection.npz"), selected_sync=selection["selected_sync"]
            )
            pair, _ = evaluate(
                out,
                name,
                profile,
                words,
                selection,
                payload,
                policy,
                dict(
                    truth_start=meta.get("truth_start_filtered"), truth_step=meta.get("truth_step")
                )
                if not meta.get("negative")
                else None,
            )
            row.update(pair)
        rows.append(row)
    return rows


def fixed(out, args, policy):
    parents = {
        str(p): verify_manifest(p) for p in (args.selected_root, args.capture_root, args.word_root)
    }
    records = json.loads((args.selected_root / "regressions.json").read_text())
    metas = {r["id"]: r["input"] for r in json.loads((args.capture_root / "rows.json").read_text())}
    rows = []
    for record in records:
        name = record["id"]
        if record.get("selection_not_available"):
            rows.append(dict(id=name, selection_not_available=True, original_request=record))
            continue
        meta = metas[record["stream"]]
        assert meta["input_sha256"] == record["original_archive_sha256"]
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        selection = json.loads((args.selected_root / (name + "-full.json")).read_text())[
            "selection"
        ]
        for k in ("sample_limit", "timing_backend", "history_capacity"):
            selection.pop(k, None)
        with np.load(args.selected_root / (name + "-full-trajectory.npz")) as z:
            selection["selected_sync"] = z["selected_sync"].copy()
        with np.load(args.capture_root / (record["stream"] + "-filtered.npz")) as z:
            words = z["words"][record["origin"] :, record["request"]["path"]]
        with np.load(args.capture_root / (record["stream"] + "-input.npz")) as z:
            payload = z["payload"].tobytes() if meta["positive"] else None
        row, events = evaluate(
            out,
            name,
            profile,
            words,
            selection,
            payload,
            policy,
            dict(truth_start=meta["sync_start"] + 40 - record["origin"], truth_step=meta["step"])
            if meta["positive"]
            else None,
        )
        old = json.loads((args.word_root / (name + "-word.json")).read_text())
        assert old["frame"] == normalized(events["old"]["frame"])
        with np.load(args.word_root / (name + "-word-trajectory.npz")) as z:
            assert set(z.files) == {k for k in TRACE_KEYS if k in events["old"]}
            for k in z.files:
                np.testing.assert_array_equal(z[k], events["old"][k])
        row.update(
            input=meta,
            original_request=record,
            original_input_sha256=record["original_archive_sha256"],
            original_window_sha256=record["window_sha256"],
            baseline_original_frames_and_trajectories_exact=True,
        )
        rows.append(row)
    return rows, parents


def nodes(out, split, policy):
    rng = np.random.default_rng(policy["node_seeds"][split])
    queries = []
    bounds, changed, erased = [], 0, [0, 0]
    for k in range(policy["node_queries"]):
        exponents = [-128, 127, -17, 0] if k % 4 == 0 else rng.integers(-128, 128, 4)
        words = tuple(
            pack_filter_word(int(r), int(i), int(e))
            for r, i, e in zip(
                rng.integers(-262144, 262144, 4),
                rng.integers(-262144, 262144, 4),
                exponents,
                strict=True,
            )
        )
        phase = int(rng.integers(0, 65537))
        queries.append((words, phase))
        a, ai = interpolate_filter_word(words, phase, OLD)
        b, bi = interpolate_filter_word(words, phase, NEW)
        assert not ai["output_saturations"] and not bi["output_saturations"]
        changed += a != b
        erased = [v + i["aligned_components_erased"] for v, i in zip(erased, (ai, bi), strict=True)]
        r, i, e = unpack_interpolated_word(b, NEW)
        error = max(
            abs(Fraction(v) * Fraction(2) ** e - y)
            for v, y in zip((r, i), exact_reference(words, phase, NEW), strict=True)
        )
        bound = Fraction(11, 8) * Fraction(2) ** (bi["common_exponent"] - 2)
        bound += Fraction(1, 2) * Fraction(2) ** e
        assert error <= bound
        bounds.append(float(error / bound))
    flow, issued, outputs = LagrangeInterpolationFlow(NEW), 0, []
    while len(outputs) < len(queries):
        if not 45 <= flow.cycle < 311:
            row = flow.take()
            if row is not None:
                assert row["tag"] == len(outputs)
                assert (row["word"], row["info"]) == interpolate_filter_word(
                    *queries[row["tag"]], NEW
                )
                outputs.append(row)
        if issued < len(queries) and flow.issue(*queries[issued], tag=issued):
            issued += 1
        flow.advance()
        assert flow.cycle < 5000 and flow.live <= 7
    np.savez_compressed(
        out / "node-inputs.npz",
        words=np.array([q[0] for q in queries], np.uint64),
        phase=np.array([q[1] for q in queries], np.uint32),
    )
    write(out / "node-outputs.json", outputs)
    calendars = []
    for n in (32, 62, 126):
        words = [
            pack_filter_word(int(r), int(i), int(e))
            for r, i, e in zip(
                rng.integers(-131072, 131072, 4 * n + 1024),
                rng.integers(-131072, 131072, 4 * n + 1024),
                rng.integers(-28, 9, 4 * n + 1024),
                strict=True,
            )
        ]
        np.savez_compressed(out / f"service-{n}-words.npz", words=np.array(words, np.uint64))
        for hz in (100_000_000, 160_000_000, 200_000_000):
            for retire, blocked in ((True, False), (False, False), (True, True)):
                calendars.append(calendar(words, n, hz, retire=retire, blocked=blocked, fmt=NEW))
    write(out / "calendars.json", calendars)
    return dict(
        queries=len(queries),
        changed_outputs=changed,
        independent_rational_bound=True,
        bound_max_ratio=max(bounds),
        erased_components_old_new=erased,
        bitexact_value_cycle=True,
        cost=flow.estimate(),
        reference_cost=LagrangeInterpolationFlow(OLD).estimate(),
        stats=flow.stats,
    )


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY.read_text())
    files = [
        POLICY,
        Path("uv.lock"),
        *Path("src").rglob("*.py"),
        *Path("tests").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    ]
    source = {str(p): sha(p) for p in sorted(set(files))}
    freeze = json.loads(args.freeze.read_text()) if args.freeze else None
    if freeze is not None:
        assert freeze["source"] == source and freeze["policy"] == policy
    rf = args.split == "development" or freeze["eligible"]
    write(
        out / "protocol.json",
        dict(
            policy=policy,
            source=source,
            split=args.split,
            RF_evaluated=rf,
            native_word_interpolation=False,
            only_alignment_order_changed=True,
            other_RX_service_cycles=None,
            upstream_global_maximum_is_fixture=True,
        ),
    )
    rows = fresh(out, args.split, policy) if rf else []
    regressions, parents = fixed(out, args, policy) if args.split == "development" else ([], {})
    node = nodes(out, args.split, policy)
    actual = [r for r in rows + regressions if "guard" in r]
    cost, budget = node["cost"], policy["node_budget"]
    resource = all(
        cost[k] <= budget[b]
        for k, b in (
            ("DSP25x18", "DSP25x18_max"),
            ("pipeline_register_bound_bits", "pipeline_register_bound_bits_max"),
            ("alignment_RNE_width", "alignment_RNE_width_max"),
            ("no_stall_II", "II_max"),
            ("no_stall_latency", "no_stall_latency_max"),
        )
    )
    observed = [r for r in actual if r["guard"]["numeric"]["observed"]]
    eligible = (
        bool(observed)
        and resource
        and all(r["guard"]["numeric"]["passed"] for r in observed)
        and all(
            r["frame_decisions_equal_float"]
            and not r["success_loss"]
            and not r["added_false_delivery"]
            and not r["new_false_delivery"]
            for r in actual
        )
    )
    model = dict(
        stage=policy["stage"],
        split=args.split,
        eligible=eligible if rf else None,
        RF_evaluated=rf,
        RF_validation_skipped_due_rejection=not rf,
        new_reference_full=sum(r.get("reference", {}).get("payload_exact", False) for r in rows),
        new_old_full=sum(
            r.get("old", {}).get("metrics", {}).get("payload_exact", False) for r in rows
        ),
        new_guard_full=sum(
            r.get("guard", {}).get("metrics", {}).get("payload_exact", False) for r in rows
        ),
        fixed_old_full=sum(
            r.get("old", {}).get("metrics", {}).get("payload_exact", False) for r in regressions
        ),
        fixed_guard_full=sum(
            r.get("guard", {}).get("metrics", {}).get("payload_exact", False) for r in regressions
        ),
        old_numeric_failures=[r["id"] for r in actual if r["old"]["numeric"]["passed"] is False],
        guard_numeric_failures=[
            r["id"] for r in actual if r["guard"]["numeric"]["passed"] is False
        ],
        success_losses=[r["id"] for r in actual if r["success_loss"]],
        added_false_deliveries=[r["id"] for r in actual if r["added_false_delivery"]],
        observed_numeric_pairs=len(observed),
        provisional_node_resource_gate=resource,
        node=node,
        parent_manifests=parents,
        negative_observation_samples=sum(policy["negative_samples"][args.split]) if rf else 0,
        false_deliveries_observed=sum(
            r.get("guard", {}).get("metrics", {}).get("false_delivery", False) for r in rows
        ),
        old_RRC_and16KiB_failures_preserved=True,
        default_RX_changed=False,
        full_RX_gates={k: False for k in ("streaming", "fixed", "system_resources", "16MSps")},
    )
    for filename, value in (
        ("rows.json", rows),
        ("fixed-regressions.json", regressions),
        ("model.json", model),
    ):
        write(out / filename, value)
    write(
        out / "failure-associations.json",
        dict(
            numeric=[r for r in actual if r["guard"]["numeric"]["passed"] is False],
            prior_numeric=[r for r in actual if r["old"]["numeric"]["passed"] is False],
            receive=[
                r
                for r in actual
                if not (
                    r.get("input", {}).get("negative")
                    or r.get("input", {}).get("positive") is False
                )
                and not r["guard"]["metrics"]["payload_exact"]
            ]
            + [r for r in rows if r.get("upstream_no_candidate")],
            regressions=[r for r in actual if r["success_loss"] or r["added_false_delivery"]],
            geometry=[r for r in regressions if r.get("selection_not_available")],
            classification=(
                "Observed failures not theory of irrecoverability;no truth replay or CRC selection"
            ),
        ),
    )
    if args.split == "development":
        write(
            out / "freeze.json",
            dict(
                source=source,
                policy=policy,
                eligible=eligible,
                new_RF_validation_not_generated=True,
                default_RX_not_promoted=True,
            ),
        )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(model, indent=2))
    return 1  # Algorithm-node experiment cannot close the four full-RX gates.


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--split", choices=("development", "validation"), required=True)
    p.add_argument("--selected-root", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--word-root", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.split == "development" and any(
        x is None for x in (args.selected_root, args.capture_root, args.word_root)
    ):
        p.error("Development requires three original frozen roots")
    if args.split == "validation" and args.freeze is None:
        p.error(
            "Validation requires source/policy freeze;rejected choice runs arithmetic nodes only"
        )
    raise SystemExit(run(args))
