"""Isolated word Lagrange,actual selected frames and finite query pipeline;exit1."""

import argparse
import json
from collections import deque
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import bridge
from model_selected_frame import TRACE_KEYS, execute, fresh_cases, metrics, normalized
from tests.test_interpolation_word import exact_reference

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    interpolate_filter_word,
    unpack_interpolated_word,
)
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import WordInterpolatedPSKReceiver, physical_filter_word
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.interpolation_flow import LagrangeInterpolationFlow

POLICY = Path("tests/data/interpolation_word_policy.json")


def word_execute(out, name, profile, words, selection, fmt, pattern, *, reuse=False):
    rx = WordInterpolatedPSKReceiver(
        profile,
        **selection,
        sample_limit=len(words),
        word_format=fmt,
        reuse_sync_interpolants=reuse,
    )
    events, pos, k = [], 0, 0
    while pos < len(words) and not rx.done:
        n = int(pattern[k % len(pattern)])
        events.extend(rx.feed_words(words[pos : pos + n]))
        pos += rx.last_input_consumed
        k += 1
    events.extend(rx.finish())
    (event,) = events
    write(
        out / (name + ".json"), normalized({k: v for k, v in event.items() if k not in TRACE_KEYS})
    )
    np.savez_compressed(
        out / (name + "-trajectory.npz"), **{k: event[k] for k in TRACE_KEYS if k in event}
    )
    return event, rx.reference_memory()


def comparison(reference, event):
    keys = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")
    decision = {k: reference["frame"].get(k) for k in keys} == {
        k: event["frame"].get(k) for k in keys
    }
    if len(reference["symbols"]) != len(event["symbols"]):
        return dict(decisions_exact=decision, shape_exact=False, numerical=False)
    if not len(reference["symbols"]):
        return dict(
            decisions_exact=decision,
            shape_exact=True,
            symbol_normalized_rms=0.0,
            symbol_normalized_peak=0.0,
            center_peak_samples=0.0,
            rate_peak_ppm=0.0,
        )
    power = max(float(np.mean(abs(reference["symbols"]) ** 2)), 1e-30)
    difference = abs(reference["symbols"] - event["symbols"]) / np.sqrt(power)
    return dict(
        decisions_exact=decision,
        shape_exact=True,
        symbol_normalized_rms=float(np.sqrt(np.mean(difference**2))) if len(difference) else 0.0,
        symbol_normalized_peak=float(np.max(difference, initial=0)),
        center_peak_samples=float(
            np.max(
                abs(reference["symbol_centers_filtered"] - event["symbol_centers_filtered"]),
                initial=0,
            )
        ),
        rate_peak_ppm=float(
            np.max(abs(reference["timing_rate_ppm"] - event["timing_rate_ppm"]), initial=0)
        ),
    )


def node_cases(out, fmt, split):
    rng = np.random.default_rng(941117 if split == "development" else 941333)
    queries = []
    for k in range(64):
        exps = rng.integers(-96, 49, 4) if k % 3 else [-96, 48, -13, 0]
        words = tuple(
            pack_filter_word(int(r), int(i), int(e))
            for r, i, e in zip(
                rng.integers(-(2**18), 2**18, 4),
                rng.integers(-(2**18), 2**18, 4),
                exps,
                strict=True,
            )
        )
        queries.append((words, int(rng.integers(0, 2**fmt.phase_bits + 1))))
    bounds = []
    for words, phase in queries:
        word, info = interpolate_filter_word(words, phase, fmt)
        r, i, e = unpack_interpolated_word(word, fmt)
        exact = exact_reference(words, phase, fmt)
        error = max(
            abs(Fraction(v) * Fraction(2) ** e - y) for v, y in zip((r, i), exact, strict=True)
        )
        bound = Fraction(5, 8) * Fraction(2) ** info["common_exponent"]
        bound += Fraction(3, 4) * Fraction(2) ** (info["common_exponent"] - fmt.guard_bits)
        bound += Fraction(1, 2) * Fraction(2) ** e
        assert error <= bound and not info["output_saturations"]
        bounds.append(float(error / bound))
    flow = LagrangeInterpolationFlow(fmt)
    offered = 0
    outputs = []
    while len(outputs) < len(queries):
        row = None if 60 <= flow.cycle < 260 else flow.take()
        if row is not None:
            assert row["tag"] == len(outputs)
            assert (row["word"], row["info"]) == interpolate_filter_word(*queries[row["tag"]], fmt)
            outputs.append(row)
        if offered < len(queries) and flow.issue(*queries[offered], tag=offered):
            offered += 1
        flow.advance()
        assert flow.cycle < 3000 and flow.live <= 7
    name = f"node-p{fmt.phase_bits}g{fmt.guard_bits}o{fmt.output_bits}"
    np.savez_compressed(
        out / (name + "-inputs.npz"),
        words=np.array([q[0] for q in queries], np.uint64),
        phase=np.array([q[1] for q in queries], np.uint32),
    )
    write(
        out / (name + ".json"),
        dict(
            format=asdict(fmt),
            cost=flow.estimate(),
            outputs=outputs,
            stats=flow.stats,
            independent_error_bound_max_ratio=max(bounds),
        ),
    )
    return dict(
        format=asdict(fmt),
        bitexact_cycle=True,
        independent_rational_bound=True,
        cost=flow.estimate(),
    )


def receiver_case(out, meta, profile, iq, payload, formats, budgets):
    name = meta["split"] + "-" + str(meta["id"])
    filtered = np.convolve(iq, rrc_filter(0.4, 4))
    words, quantization = bridge(filtered)
    values = np.array([physical_filter_word(w) for w in words])
    np.savez_compressed(
        out / (name + "-input.npz"),
        iq=iq,
        filtered=filtered,
        words=words,
        payload=np.frombuffer(payload or b"", np.uint8),
    )
    found = search_psk_sync(values, 4, profile.sync_symbols(), 8)
    if found is None:
        return dict(
            id=name,
            input=meta,
            upstream_no_candidate=True,
            input_sha256=sha(out / (name + "-input.npz")),
        )
    offset = found["offset"]
    selection = dict(
        start_tick=round(8 * (found["phase"] + 4 * offset)),
        delta=found["delta"],
        differential_score=found["score"],
        coherent_threshold=0.45,
        selected_sync=found["phase_symbols"][offset : offset + len(profile.sync_symbols())],
        timing_backend="python",
        history_capacity=512,
    )
    reference, _ = execute(
        out, name + "-float", profile, values, selection, capacity=len(values), chunking="whole"
    )
    before = metrics(
        reference,
        profile,
        expected=payload,
        truth_start=meta.get("truth_start_filtered"),
        truth_step=meta.get("truth_step"),
    )
    records = []
    for fmt in formats:
        label = name + f"-p{fmt.phase_bits}g{fmt.guard_bits}o{fmt.output_bits}"
        event, memory = word_execute(out, label, profile, words, selection, fmt, [32])
        row = comparison(reference, event)
        row["numerical"] = row["shape_exact"] and all(
            row.get(k, float("inf")) <= v for k, v in budgets.items()
        )
        row.update(
            format=asdict(fmt),
            metrics=metrics(
                event,
                profile,
                expected=payload,
                truth_start=meta.get("truth_start_filtered"),
                truth_step=meta.get("truth_step"),
            ),
            memory=memory,
        )
        for kind, pattern in (
            ("fixed", [7, 31]),
            ("random", np.random.default_rng(941118).integers(1, 33, 19)),
        ):
            alternate, _ = word_execute(
                out, label + "-" + kind, profile, words, selection, fmt, pattern
            )
            assert alternate["frame"] == event["frame"]
            assert (
                alternate["delivered_after_filtered_samples"]
                == event["delivered_after_filtered_samples"]
            )
            for key in TRACE_KEYS:
                assert (key in alternate) == (key in event)
                if key in event:
                    np.testing.assert_array_equal(alternate[key], event[key])
        row["partitions_exact"] = True
        records.append(row)
    return dict(
        id=name,
        input=meta,
        input_sha256=sha(out / (name + "-input.npz")),
        quantization=quantization,
        selection=normalized({k: v for k, v in selection.items() if k != "selected_sync"}),
        reference=before,
        candidates=records,
    )


def paired_calendar(fmt, clock, blocked=False):
    """Synthetic query-port load only;raw reads/PI/head are not free service."""
    flow = LagrangeInterpolationFlow(fmt)
    words = tuple(pack_filter_word(v, -v, -17) for v in (131071, -131072, 71, 19))
    queue, outputs, drops, pending = deque(), [], [], 0
    peak = 0
    for cycle in range(6000):
        if not (blocked and 60 <= cycle < 360):
            row = flow.take()
            if row is not None:
                assert (
                    row["word"]
                    == interpolate_filter_word(
                        words, row["tag"] * 911 % (2**fmt.phase_bits + 1), fmt
                    )[0]
                )
                outputs.append(row)
        while pending < 32 and (pending * clock + 3999999) // 4000000 <= cycle:
            for tag in (2 * pending, 2 * pending + 1):
                if len(queue) == 2:
                    drops.append(tag)
                else:
                    queue.append(tag)
            pending += 1
        peak = max(peak, len(queue))
        if queue and flow.issue(words, queue[0] * 911 % (2**fmt.phase_bits + 1), tag=queue[0]):
            queue.popleft()
        flow.advance()
        if pending == 32 and not queue and not flow.live:
            break
    assert len(outputs) + len(drops) == 64 and peak <= 2
    pairs = {}
    for row in outputs:
        pairs.setdefault(row["tag"] // 2, []).append(row["emitted_cycle"])
    margins = [
        ((symbol + 1) * clock + 3999999) // 4000000 - max(cycles)
        for symbol, cycles in pairs.items()
        if len(cycles) == 2
    ]
    return dict(
        clock=clock,
        blocked=blocked,
        queries_offered=64,
        queries_emitted=len(outputs),
        query_resource_drops=drops,
        queue_capacity=2,
        queue_peak=peak,
        queue_payload_bits=2 * (184 + fmt.phase_bits + 1 + 16),
        queue_control_bits=4,
        minimum_interpolation_only_feedback_margin_cycles=min(margins, default=None),
        interpolation_only_deadline_failures=sum(m < 0 for m in margins),
        macro_assumptions=flow.estimate()["macro_assumptions"],
        source="two interpolation queries per4Msymbol/s;all4input words assumed available",
        raw_read_initial_fit_Gardner_head_decoder_cycles=None,
        whole_receiver16msps=False,
        outputs=outputs,
    )


def fixed_regressions(args, out, fmt, budgets):
    """Original hashes and ONE selection;no newly searched/CRC-selected answer."""
    for folder in (args.selected_root, args.capture_root):
        manifest = json.loads((folder / "manifest.json").read_text())
        assert all(sha(folder / k) == v for k, v in manifest.items())
    metas = {r["id"]: r["input"] for r in json.loads((args.capture_root / "rows.json").read_text())}
    results = []
    for record in json.loads((args.selected_root / "regressions.json").read_text()):
        if record.get("selection_not_available"):
            results.append(dict(id=record["id"], selection_not_available=True))
            continue
        name = record["id"]
        meta = metas[record["stream"]]
        assert meta["input_sha256"] == record["original_archive_sha256"]
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        with np.load(args.capture_root / (record["stream"] + "-filtered.npz")) as ar:
            words = ar["words"][record["origin"] :, record["request"]["path"]]
        saved = json.loads((args.selected_root / (name + "-full.json")).read_text())
        with np.load(args.selected_root / (name + "-full-trajectory.npz")) as ar:
            selection = saved["selection"] | dict(
                selected_sync=ar["selected_sync"].copy(),
                timing_backend="python",
                history_capacity=512,
            )
        selection.pop("sample_limit", None)
        values = np.array([physical_filter_word(w) for w in words])
        reference, _ = execute(
            out, name + "-float", profile, values, selection, capacity=len(values), chunking="whole"
        )
        keys = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")
        assert {k: normalized(reference["frame"].get(k)) for k in keys} == {
            k: saved["frame"].get(k) for k in keys
        }
        with np.load(args.capture_root / (record["stream"] + "-input.npz")) as ar:
            payload = ar["payload"].tobytes() if meta["positive"] else None
        event, memory = word_execute(out, name + "-word", profile, words, selection, fmt, [32])
        comparison_row = comparison(reference, event)
        comparison_row["numerical"] = comparison_row["shape_exact"] and all(
            comparison_row.get(k, float("inf")) <= v for k, v in budgets.items()
        )
        before = metrics(reference, profile, expected=payload)
        after = metrics(event, profile, expected=payload)
        for suffix, pattern in (
            ("fixed", [7, 31]),
            ("random", np.random.default_rng(941119).integers(1, 33, 19)),
        ):
            split, _ = word_execute(
                out, name + "-" + suffix, profile, words, selection, fmt, pattern
            )
            assert split["frame"] == event["frame"]
            for key in TRACE_KEYS:
                if key in event:
                    np.testing.assert_array_equal(event[key], split[key])
        results.append(
            dict(
                id=name,
                input_archive_sha256=record["original_archive_sha256"],
                source_window_sha256=record["window_sha256"],
                reference=before,
                word=after,
                comparison=comparison_row,
                memory=memory,
                success_loss=before["payload_exact"] and not after["payload_exact"],
            )
        )
    write(out / "fixed-regressions.json", results)
    return dict(
        requests=len(results),
        scored=sum("word" in r for r in results),
        reference_full=sum(r.get("reference", {}).get("payload_exact", False) for r in results),
        full=sum(r.get("word", {}).get("payload_exact", False) for r in results),
        success_losses=[r["id"] for r in results if r.get("success_loss")],
        numeric_or_decision_failures=[
            r["id"]
            for r in results
            if "comparison" in r
            and (not r["comparison"]["numerical"] or not r["comparison"]["decisions_exact"])
        ],
        scope="fixed viewed regression,not independent held",
    )


def run(args):
    if args.split == "development" and (args.selected_root is None or args.capture_root is None):
        raise ValueError("Development adoption needs both frozen parent datasets")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY.read_text())
    paths = [
        Path(__file__),
        POLICY,
        Path("uv.lock"),
        *Path("src").rglob("*.py"),
        *Path("tests").rglob("*.py"),
        Path("scripts/model_selected_frame.py"),
        Path("scripts/model_capture_fine.py"),
    ]
    sources = {str(p): sha(p) for p in paths}
    if args.split in ("validation", "node-validation"):
        if args.freeze is None:
            raise ValueError("Validation needs frozen development decision")
        freeze = json.loads(args.freeze.read_text())
        assert freeze["source"] == sources
        if args.split == "validation" and freeze["selected"] is None:
            raise ValueError("No eligible development candidate;do not evaluate held alternatives")
        formats = (
            [LagrangeWordFormat(**freeze["selected"])]
            if args.split == "validation"
            else [LagrangeWordFormat(*f) for f in policy["formats"]]
        )
    else:
        formats = [LagrangeWordFormat(*f) for f in policy["formats"]]
    write(
        out / "protocol.json",
        dict(
            stage=policy["stage"],
            source=sources,
            policy=policy,
            split=args.split,
            interpolation_only=True,
            other_arithmetic_cycles=None,
            whole_receiver16msps=False,
            default_RX_changed=False,
        ),
    )
    nodes = [node_cases(out, fmt, args.split) for fmt in formats]
    rows = (
        []
        if args.split == "node-validation"
        else [
            receiver_case(out, *case, formats, policy["budgets"])
            for case in fresh_cases(args.split, policy)
        ]
    )
    gates = []
    for fmt in formats:
        actual = [
            (r, c) for r in rows for c in r.get("candidates", []) if c["format"] == asdict(fmt)
        ]
        losses = [
            r["id"]
            for r, c in actual
            if r["reference"]["payload_exact"] and not c["metrics"]["payload_exact"]
        ]
        false = [
            r["id"]
            for r, c in actual
            if c["metrics"]["false_delivery"] and not r["reference"]["false_delivery"]
        ]
        failures = [r["id"] for r, c in actual if not c["numerical"] or not c["decisions_exact"]]
        gates.append(
            dict(
                format=asdict(fmt),
                eligible=None if not actual else not (losses or false or failures),
                reception_evaluated=bool(actual),
                success_losses=losses,
                added_false_deliveries=false,
                numeric_or_decision_failures=failures,
                reference_full=sum(r["reference"]["payload_exact"] for r, c in actual),
                full=sum(c["metrics"]["payload_exact"] for r, c in actual),
                cost=LagrangeInterpolationFlow(fmt).estimate(),
            )
        )
    selected = None
    regression = None
    candidate = None
    calendars = []
    if args.split == "development":
        eligible = [r for r in gates if r["eligible"]]
        if eligible:
            selected = min(
                eligible,
                key=lambda r: (r["cost"]["DSP25x18"], r["cost"]["pipeline_register_bound_bits"]),
            )["format"]
        candidate = selected
        if selected is not None and args.selected_root is not None:
            regression = fixed_regressions(
                args, out, LagrangeWordFormat(**selected), policy["budgets"]
            )
            if regression["success_losses"] or regression["numeric_or_decision_failures"]:
                selected = None
        write(
            out / "freeze.json",
            dict(
                source=sources,
                selected=selected,
                development_candidate=candidate,
                fixed_regressions=regression,
                frozen_formats=policy["formats"],
                adoption_rule=policy["adoption"],
                new_validation_not_generated=True,
            ),
        )
    elif args.split == "validation":
        selected = asdict(formats[0])
    for fmt in formats:
        for clock in (100000000, 160000000, 200000000):
            for blocked in (False, True):
                calendars.append(dict(format=asdict(fmt), **paired_calendar(fmt, clock, blocked)))
    write(out / "paired-calendars.json", calendars)
    write(out / "rows.json", rows)
    write(out / "gates.json", gates)
    model = dict(
        stage=policy["stage"],
        split=args.split,
        selected=selected,
        development_candidate=candidate,
        gates=gates,
        node_cases=nodes,
        fixed_regressions=regression,
        paired_query_scenarios=len(calendars),
        default_RX_changed=False,
        full_fixed_point_receiver=False,
        whole_receiver16msps=False,
        actual_capture_gain=None,
        RF_validation_skipped_due_rejection=selected is None,
        standalone_node_validation_only=args.split == "node-validation",
        old2percent_and16KiB_failures_preserved=True,
        unresolved=[
            "raw window/word query port coupling",
            "initialfit/Gardner/Polar service",
            "full-system budgets",
        ],
    )
    write(out / "model.json", model)
    write(
        out / "failure-associations.json",
        dict(
            fixed_regressions=regression,
            query_port_failures=[
                {k: v for k, v in c.items() if k != "outputs"}
                for c in calendars
                if c["query_resource_drops"] or c["interpolation_only_deadline_failures"]
            ],
            fresh=[
                r
                for r in rows
                if r.get("upstream_no_candidate")
                or any(
                    (not c["metrics"]["payload_exact"] and not r["input"].get("negative"))
                    or not c["numerical"]
                    or not c["decisions_exact"]
                    for c in r.get("candidates", [])
                )
            ],
        ),
    )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(dict(selected=selected, gates=gates), indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--split", choices=("development", "validation", "node-validation"), required=True
    )
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--selected-root", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
