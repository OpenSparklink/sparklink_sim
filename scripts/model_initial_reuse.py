"""Exact initial-word reuse and finite source/cache service;full-RX gates stay open."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import bridge
from model_interpolation_word import word_execute
from model_selected_frame import TRACE_KEYS, fresh_cases, independent_packet, metrics, normalized
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy.interpolation_word import LagrangeWordFormat, interpolate_filter_word
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.initial_interpolant_flow import InitialStencilService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

POLICY = Path("tests/data/initial_reuse_policy.json")
FMT = LagrangeWordFormat(16, 2, 22)
FS = 16_000_000


def verify_manifest(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert all(sha(folder / k) == v for k, v in manifest.items())
    return sha(folder / "manifest.json")


def exact(a, b):
    assert normalized(a["frame"]) == normalized(b["frame"])
    for key in (
        "delivered_after_filtered_samples",
        "header_decision_after_filtered_samples",
        "sync_start_filtered",
        "cfo_hz",
        "gain_codes",
        "timing_loop",
    ):
        assert a.get(key) == b.get(key), key
    for key in TRACE_KEYS:
        assert (key in a) == (key in b), key
        if key in a:
            np.testing.assert_array_equal(a[key], b[key])


def pair(out, name, profile, words, selection, payload, *, truth=None):
    np.savez_compressed(out / (name + "-selection.npz"), selected_sync=selection["selected_sync"])
    before, bm = word_execute(out, name + "-original", profile, words, selection, FMT, [32])
    after, am = word_execute(out, name + "-reuse", profile, words, selection, FMT, [32], reuse=True)
    exact(before, after)
    for label, pattern in (
        ("fixed", [7, 31]),
        ("random", np.random.default_rng(942119).integers(1, 33, 23)),
    ):
        split, _ = word_execute(
            out, name + "-" + label, profile, words, selection, FMT, pattern, reuse=True
        )
        exact(after, split)
    cache = am["initial_interpolant_cache"]
    saved = bm["interpolation_stats"]["queries"] - am["interpolation_stats"]["queries"]
    assert saved == cache["reused"] == cache["written"]
    assert cache["valid_words"] == 0
    truth = truth or {}
    a, b = (
        metrics(
            e,
            profile,
            expected=payload,
            truth_start=truth.get("start"),
            truth_step=truth.get("step"),
        )
        for e in (before, after)
    )
    return dict(
        id=name,
        profile=normalized(asdict(profile)),
        selection=normalized({k: v for k, v in selection.items() if k != "selected_sync"}),
        selection_archive_sha256=sha(out / (name + "-selection.npz")),
        initial_CFO_error_Hz=(
            after.get("cfo_hz") - truth["cfo_hz"]
            if after.get("cfo_hz") is not None and "cfo_hz" in truth
            else None
        ),
        truth_is_postdecision_diagnostic_only=True,
        reference=a,
        reuse=b,
        decisions_and_trajectories_exact=True,
        interpolation_calls_saved=saved,
        raw_reads_saved=4 * saved,
        old_memory=bm,
        reuse_memory=am,
        success_loss=a["payload_exact"] and not b["payload_exact"],
        added_false_delivery=b["false_delivery"] and not a["false_delivery"],
    )


def cases(split, policy):
    p = policy | dict(negative_samples_each_split=policy["negative_samples"][split])
    neg = policy["negative_templates"][split]
    for meta, profile, iq, payload in fresh_cases(split, p):
        if meta.get("negative"):
            if meta["kind"] == "tone":
                iq = np.exp(
                    1j * (2 * np.pi * neg["tone_frequency"] * np.arange(len(iq)) + neg["phase"])
                )
            elif meta["kind"] == "wrong_sync":
                wrong = PSKReceiveProfile.ordinary(
                    TxConfig(mcs_index=8, pid=neg["wrong_pid"], symbol_rate_mhz=4)
                )
                wave, _, _, _ = iq_record(
                    independent_packet(wrong, neg["wrong_payload"].encode()),
                    cfo=policy[split]["cfo_hz"][0],
                    offset=policy[split]["offset"],
                )
                iq = np.resize(wave, len(iq))
            meta = meta | dict(negative_template=neg, observation_samples=len(iq))
        yield meta, profile, iq, payload


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
        # One existing differential maximum;no truth/payload/CRC selection or retry.
        from nearlink_sdr.phy.selected_word import physical_filter_word

        found = search_psk_sync(
            np.array([physical_filter_word(w) for w in words]), 4, profile.sync_symbols(), 8
        )
        row = dict(
            id=name,
            input=meta,
            input_sha256=sha(out / (name + "-input.npz")),
            quantization=quantization,
        )
        if found is None:
            row["upstream_no_candidate"] = True
        else:
            start = found["phase"] + 4 * found["offset"]
            selection = dict(
                start_tick=round(8 * start),
                delta=found["delta"],
                differential_score=found["score"],
                coherent_threshold=0.45,
                selected_sync=found["phase_symbols"][
                    found["offset"] : found["offset"] + len(profile.sync_symbols())
                ],
            )
            row.update(
                pair(
                    out,
                    name,
                    profile,
                    words,
                    selection,
                    payload,
                    truth=dict(
                        start=meta.get("truth_start_filtered"),
                        step=meta.get("truth_step"),
                        cfo_hz=meta.get("cfo_hz"),
                    )
                    if not meta.get("negative")
                    else None,
                )
            )
        rows.append(row)
    return rows


def fixed(out, args):
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
        saved = json.loads((args.selected_root / (name + "-full.json")).read_text())
        with np.load(args.selected_root / (name + "-full-trajectory.npz")) as z:
            selection = saved["selection"] | dict(selected_sync=z["selected_sync"].copy())
        selection.pop("sample_limit", None)
        selection.pop("timing_backend", None)
        selection.pop("history_capacity", None)
        with np.load(args.capture_root / (record["stream"] + "-filtered.npz")) as z:
            words = z["words"][record["origin"] :, record["request"]["path"]]
        with np.load(args.capture_root / (record["stream"] + "-input.npz")) as z:
            payload = z["payload"].tobytes() if meta["positive"] else None
        row = pair(
            out,
            name,
            profile,
            words,
            selection,
            payload,
            truth=dict(start=meta["sync_start"] + 40 - record["origin"], step=meta["step"])
            if meta["positive"]
            else None,
        )
        row["input"] = meta
        # Original selected frames and exact old word trajectories are fixed regressions.
        old = json.loads((args.word_root / (name + "-word.json")).read_text())
        new = json.loads((out / (name + "-original.json")).read_text())
        assert old["frame"] == new["frame"]
        with (
            np.load(args.word_root / (name + "-word-trajectory.npz")) as a,
            np.load(out / (name + "-original-trajectory.npz")) as b,
        ):
            assert a.files == b.files
            for k in a.files:
                np.testing.assert_array_equal(a[k], b[k])
        with (
            np.load(args.word_root / (name + "-float-trajectory.npz")) as a,
            np.load(out / (name + "-reuse-trajectory.npz")) as b,
        ):
            error = abs(a["symbols"] - b["symbols"])
            observed = bool(len(error))
            rms = (
                float(np.sqrt(np.mean(error**2) / max(np.mean(abs(a["symbols"]) ** 2), 1e-30)))
                if observed
                else None
            )
        row.update(
            input_archive_sha256=record["original_archive_sha256"],
            source_window_sha256=record["window_sha256"],
            old_float_word_symbol_RMS=rms,
            numeric_observation=observed,
            old_float_numeric_budget_passed=(rms <= 0.0001 if observed else None),
        )
        rows.append(row)
    return rows, parents


def calendar(words, symbols, hz, *, retire=True, blocked=False):
    source = ReceiveHistoryPort()
    node = InitialStencilService(source, FMT, symbols, 1, 32768, retire_after_staging=retire)
    cursor = 0
    activation = ((node.activation_end - 1) * hz + FS - 1) // FS
    status = None
    overflow = None
    while node.cycle < activation + 10000:
        c = node.cycle
        if cursor < len(words) and c == (cursor * hz + FS - 1) // FS:
            if not source.write(cursor, int(words[cursor]), c):
                status = "source_overflow_candidate_abort"
                overflow = dict(index=cursor, cycle=c)
                disposition = node.abort()
                break
            cursor += 1
        node.step(output_ready=not (blocked and activation <= c < activation + 2000))
        if node.ready:
            status = "cache_ready_fit_service_unknown"
            break
        if cursor == len(words) and not node.enabled:
            status = "EOF_missing_sync_future"
            node.abort()
            break
    assert status is not None, "No unbounded model wait"
    ready = node.ready_cycle
    if node.ready:
        assert source.stats["read_returns"] == 4 * symbols
        for k in range(symbols):
            expected = interpolate_filter_word(
                tuple(map(int, words[4 * k : 4 * k + 4])), 32768, FMT
            )[0]
            assert node.output.cache.words[k] == expected
    free = source.capacity - source.live
    first_full_clock = ((cursor + free) * hz + FS - 1) // FS if ready is not None else None
    return dict(
        symbols=symbols,
        clock_hz=hz,
        retire_after_staging=retire,
        blocked=blocked,
        status=status,
        activation_cycle=activation,
        cache_ready_cycle=ready,
        source_consumed=cursor,
        source=source.stats,
        source_live=source.live,
        future_wait_cycles=node.future_wait_cycles,
        abort_disposition=(disposition if overflow is not None else None),
        overflow=overflow,
        unconsumed_stimulus_words=len(words) - cursor,
        post_ready_no_retirement_headroom_cycles=(first_full_clock - ready - 1 if ready else None),
        initial_fit_PI_head_decoder_cycles=None,
        cache=node.output.cache.diagnostics(),
        aborted_interpolation_contexts=node.aborted_inflight,
        cost=node.estimate(),
        counterfactual_retention_is_not_full_old_RX=not retire,
        synthetic_word_stimulus_not_RF=True,
        whole_receiver16msps=False,
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
    if args.split == "validation":
        freeze = json.loads(args.freeze.read_text())
        assert freeze["source"] == source and freeze["reuse_eligible"]
    write(
        out / "protocol.json",
        dict(
            stage=policy["stage"],
            source=source,
            policy=policy,
            split=args.split,
            actual_timing_backend="python",
            native_word_interpolation=False,
            only_reuse_changed=True,
            all_other_receiver_service_cycles=None,
        ),
    )
    rows = fresh(out, args.split, policy)
    regressions, parents = fixed(out, args) if args.split == "development" else ([], {})
    rng = np.random.default_rng(943117 if args.split == "development" else 943333)
    calendars = []
    for n in (32, 62, 126):
        words = [
            pack_filter_word(int(r), int(i), int(e))
            for r, i, e in zip(
                rng.integers(-(2**17), 2**17, 4 * n + 1024),
                rng.integers(-(2**17), 2**17, 4 * n + 1024),
                rng.integers(-28, 9, 4 * n + 1024),
                strict=True,
            )
        ]
        np.savez_compressed(out / f"service-{n}-words.npz", words=np.array(words, np.uint64))
        for hz in (100_000_000, 160_000_000, 200_000_000):
            for retire, blocked in ((True, False), (False, False), (True, True)):
                calendars.append(calendar(words, n, hz, retire=retire, blocked=blocked))
    cost = calendars[0]["cost"]
    budget = policy["provisional_incremental_node_budget"]
    resource = (
        cost["cache"]["payload_bits"] <= budget["payload_bits_max"]
        and cost["cache"]["bram36"] <= budget["cache_BRAM36_max"]
        and cost["incremental_control_bits"] <= budget["cache_and_source_service_control_bits_max"]
        and cost["cache"]["extra_DSP"] == budget["additional_DSP"]
        and cost["source_read_ports"] - 1 <= budget["extra_source_read_ports"]
    )
    actual = [r for r in rows + regressions if "reuse" in r]
    assert actual and all(r["decisions_and_trajectories_exact"] for r in actual)
    assert all(not r["success_loss"] and not r["added_false_delivery"] for r in actual)
    active = sum(r["interpolation_calls_saved"] > 0 for r in actual)
    eligible = resource and active > 0
    model = dict(
        stage=policy["stage"],
        split=args.split,
        reuse_eligible=eligible,
        reference_full=sum(r.get("reference", {}).get("payload_exact", False) for r in rows),
        reuse_full=sum(r.get("reuse", {}).get("payload_exact", False) for r in rows),
        fixed_reference_full=sum(
            r.get("reference", {}).get("payload_exact", False) for r in regressions
        ),
        fixed_reuse_full=sum(r.get("reuse", {}).get("payload_exact", False) for r in regressions),
        old_numeric_failures=[
            r["id"] for r in regressions if r.get("old_float_numeric_budget_passed") is False
        ],
        active_pairs=active,
        all_decisions_and_trajectories_exact=True,
        success_losses=[],
        added_false_deliveries=[],
        interpolation_calls_saved=sum(r["interpolation_calls_saved"] for r in actual),
        interpolation_calls_before=sum(
            r["old_memory"]["interpolation_stats"]["queries"] for r in actual
        ),
        interpolation_calls_after=sum(
            r["reuse_memory"]["interpolation_stats"]["queries"] for r in actual
        ),
        source_raw_reads_saved=sum(r["raw_reads_saved"] for r in actual),
        calendars=len(calendars),
        provisional_node_resource_gate=resource,
        default_RX_changed=False,
        full_word_receiver_adopted=False,
        whole_receiver16msps=False,
        full_RX_gates={k: False for k in ("streaming", "fixed", "system_resources", "16MSps")},
        negative_observation_samples=sum(policy["negative_samples"][args.split]),
        false_deliveries_observed_before=sum(
            r.get("reference", {}).get("false_delivery", False) for r in rows
        ),
        false_deliveries_observed_after=sum(
            r.get("reuse", {}).get("false_delivery", False) for r in rows
        ),
        frozen_parent_manifests=parents,
    )
    write(out / "rows.json", rows)
    write(out / "fixed-regressions.json", regressions)
    write(out / "calendars.json", calendars)
    write(out / "model.json", model)
    write(
        out / "failure-associations.json",
        dict(
            positive_receive_failures=[
                r
                for r in rows + regressions
                if not (
                    r.get("input", {}).get("negative")
                    or r.get("input", {}).get("positive") is False
                )
                and (
                    r.get("upstream_no_candidate")
                    or ("reuse" in r and not r["reuse"]["payload_exact"])
                )
            ],
            negative_false_deliveries=[
                r for r in rows + regressions if r.get("reuse", {}).get("false_delivery")
            ],
            correct_negative_rejections=[
                dict(id=r["id"], status=r.get("reuse", {}).get("status", "upstream_no_candidate"))
                for r in rows + regressions
                if (
                    r.get("input", {}).get("negative")
                    or r.get("input", {}).get("positive") is False
                )
                and not r.get("reuse", {}).get("false_delivery")
            ],
            numeric_failures=[
                r for r in regressions if r.get("old_float_numeric_budget_passed") is False
            ],
            unscored_geometry=[r for r in regressions if r.get("selection_not_available")],
            resources=[r for r in calendars if r["overflow"] is not None],
        ),
    )
    if args.split == "development":
        write(
            out / "freeze.json",
            dict(
                source=source,
                reuse_eligible=eligible,
                policy=policy,
                old_word_receiver_numeric_rejection_remains=True,
                new_validation_not_generated=True,
            ),
        )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(model, indent=2))
    return 1  # Open full-RX/numeric/resource boundaries are not converted to overall success.


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
        p.error("Development requires all three frozen parent roots")
    if args.split == "validation" and args.freeze is None:
        p.error("Validation requires frozen development")
    raise SystemExit(run(args))
