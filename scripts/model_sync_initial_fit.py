"""Single-factor finite known-sync initialization;four whole-RX gates remain open."""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import bridge
from model_initial_reuse import cases, exact, verify_manifest
from model_interpolation_word import comparison
from model_selected_frame import TRACE_KEYS, metrics, normalized
from tests.test_initial_interpolant_flow import input_words

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import WordInterpolatedPSKReceiver, physical_filter_word
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, TURN, SyncFitFormat
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from nearlink_sdr.sim.sync_fit_flow import SyncFitWordFlow

POLICY = Path("tests/data/sync_fit_policy.json")


class ObservedReceiver(WordInterpolatedPSKReceiver):
    """Host-only evidence copy,not a hardware buffer or an answer-selection path."""

    def _fit_selected_sync(self, raw, delta):
        self.observed_sync = raw.copy()
        return super()._fit_selected_sync(raw, delta)


def execute(out, name, profile, words, selection, peak, pattern):
    rx = ObservedReceiver(
        profile,
        **selection,
        sample_limit=len(words),
        word_format=INPUT_FORMAT,
        reuse_sync_interpolants=True,
        sync_fit_format=None if peak is None else SyncFitFormat(peak),
    )
    pos = k = 0
    events = []
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
        out / (name + "-trajectory.npz"),
        **{k: event[k] for k in TRACE_KEYS if k in event},
        initial_sync=getattr(rx, "observed_sync", np.empty(0, complex)),
    )
    return event, rx


def initializer_error(rx, selection):
    raw = getattr(rx, "observed_sync", None)
    if raw is None:
        return dict(observed=False, passed=None)
    result = rx._sync_fit_result
    if not result["valid"]:
        return dict(observed=True, passed=False, reason=result["reason"])
    # Independent NumPy least-squares reference on the SAME actual interpolated words.
    t = np.arange(len(raw))
    corrected = raw * np.exp(-1j * selection["delta"] * t)
    slope, phase = np.polyfit(t, np.unwrap(np.angle(corrected * rx.reference.conj())), 1)
    gain = max(float(np.mean(abs(corrected))), 1e-30)
    power = float(np.mean(abs(raw) ** 2))
    return dict(
        observed=True,
        initial_phase_rad=float(
            abs(np.angle(np.exp(1j * (result["phase_turn40"] * math.tau / TURN - phase))))
        ),
        initial_CFO_Hz=abs(
            result["frequency_turn40"] * math.tau / TURN - slope - selection["delta"]
        )
        * 4_000_000
        / math.tau,
        gain_relative=abs(math.ldexp(*result["gain"]) - gain) / gain,
        power_relative=abs(math.ldexp(*result["power"]) - power) / max(power, 1e-300),
    )


def evaluate(out, name, profile, words, selection, payload, policy, peaks, truth=None):
    baseline, _ = execute(out, name + "-baseline", profile, words, selection, None, [32])
    records = {}
    truth = truth or {}
    for peak in peaks:
        label = "p" + str(peak)
        event, rx = execute(out, name + "-" + label, profile, words, selection, peak, [32])
        for partition, pattern in (
            ("fixed", [7, 31]),
            ("random", np.random.default_rng(949129).integers(1, 33, 23)),
        ):
            other, _ = execute(
                out, name + "-" + label + "-" + partition, profile, words, selection, peak, pattern
            )
            exact(event, other)
        node = initializer_error(rx, selection)
        if node["observed"] and "reason" not in node:
            node["passed"] = all(node[k] <= v for k, v in policy["budgets"].items())
        delta = comparison(baseline, event)
        delta["observed"] = bool(len(baseline["symbols"]))
        delta["passed"] = (
            delta["shape_exact"]
            and delta["decisions_exact"]
            and all(
                delta.get(k, float("inf")) <= v for k, v in policy["receiver_delta_budgets"].items()
            )
        )
        records[label] = dict(
            metrics=metrics(event, profile, expected=payload, **truth),
            initializer=node,
            receiver_delta=delta,
            partitions_exact=True,
            result=rx._sync_fit_result,
            memory=rx.reference_memory(),
            consumed=rx.input_samples,
            header_consumed=rx.header_decision_after_filtered_samples,
            success_loss=baseline["frame"]["payload"] == payload
            and payload is not None
            and event["frame"]["payload"] != payload,
            added_false_delivery=payload is None and event["frame"]["status"] == "valid",
        )
    return dict(
        id=name,
        profile=normalized(asdict(profile)),
        selection=normalized({k: v for k, v in selection.items() if k != "selected_sync"}),
        baseline=metrics(baseline, profile, expected=payload, **truth),
        **records,
    ), baseline


def fresh(out, split, policy, peaks):
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
            result, _ = evaluate(
                out,
                name,
                profile,
                words,
                selection,
                payload,
                policy,
                peaks,
                dict(truth_start=meta["truth_start_filtered"], truth_step=meta["truth_step"])
                if not meta.get("negative")
                else None,
            )
            row.update(result)
        rows.append(row)
    return rows


def fixed(out, args, policy, peaks):
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
        row, baseline = evaluate(out, name, profile, words, selection, payload, policy, peaks)
        old = json.loads((args.word_root / (name + "-word.json")).read_text())
        assert old["frame"] == normalized(baseline["frame"])
        with np.load(args.word_root / (name + "-word-trajectory.npz")) as z:
            for k in z.files:
                np.testing.assert_array_equal(z[k], baseline[k])
        row.update(
            input=meta,
            original_request=record,
            original_input_sha256=meta["input_sha256"],
            baseline_original_frames_and_trajectories_exact=True,
        )
        rows.append(row)
    return rows, parents


def calendar(n, hz, blocked):
    history = ReceiveHistoryPort()
    ft = {32: 2, 62: 3, 126: 4}[n]
    profile = validate_profiles([(ft, 8, 0, 13)], 0)[0]
    from nearlink_sdr.phy.sync_fit_word import compile_sync_phases

    service = InitialSyncFitService(
        history, compile_sync_phases(profile.sync_symbols()), 12345, 1, 32768, fmt=SyncFitFormat(30)
    )
    words = input_words(4 * n + 2048)
    offered = 0
    overflow = None
    while service.cycle < 20000:
        c = service.cycle
        if offered < len(words) and c == math.ceil(offered * hz / 16_000_000):
            if not history.write(offered, words[offered], c):
                overflow = dict(cycle=c, index=offered)
                service.abort()
                break
            offered += 1
        service.step(output_ready=not blocked or c >= 14000)
        if service.ready:
            break
    assert service.ready or overflow is not None
    return dict(
        symbols=n,
        clock_Hz=hz,
        input_Hz=16_000_000,
        blocked=blocked,
        status="initial_fit_ready" if service.ready else "resource_overflow",
        overflow=overflow,
        source=dict(head=history.head, end=history.end, live=history.live, stats=history.stats),
        cache_ready=service.stencil.ready_cycle,
        fit_ready=service.fit.ready_cycle,
        stats=service.fit.stats,
        cost=service.estimate(),
        full_RX16MSps=False,
        remaining_PI_carrier_head_decoder_cycles=None,
    )


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY.read_text())
    sources = [
        POLICY,
        Path("uv.lock"),
        *Path("src").rglob("*.py"),
        *Path("tests").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    ]
    source = {str(p): sha(p) for p in sorted(set(sources))}
    freeze = json.loads(args.freeze.read_text()) if args.freeze else None
    if freeze is not None:
        assert freeze["source"] == source and freeze["policy"] == policy
    peaks = policy["CORDIC_peak_bits"] if freeze is None else [freeze["selected"]]
    evaluate_rf = freeze is None or freeze["selected"] is not None
    write(
        out / "protocol.json",
        dict(
            policy=policy,
            source=source,
            split=args.split,
            RF_evaluated=evaluate_rf,
            upstream_global_maximum_is_fixture=True,
            only_initializer_changed=True,
            native_word_fit=False,
        ),
    )
    rows = fresh(out, args.split, policy, peaks) if evaluate_rf else []
    regressions, parents = fixed(out, args, policy, peaks) if freeze is None else ([], {})
    active = [r for r in rows + regressions if "baseline" in r]
    eligible = []
    costs = {
        str(p): SyncFitWordFlow([0] * 126, 0, SyncFitFormat(p)).estimate()
        for p in peaks
        if p is not None
    }
    budget = policy["provisional_node_budget"]
    resources = {
        p: c["additional_DSP25x18"] <= budget["additional_DSP_max"]
        and c["register_state_upper_bits"] <= budget["state_register_bits_max"]
        and c["logical_template_and_constant_bits"]
        <= budget["logical_template_and_constant_bits_max"]
        and c["additional_source_read_ports"] == budget["additional_source_read_ports"]
        for p, c in costs.items()
    }
    for peak in peaks:
        if peak is None:
            continue
        label = "p" + str(peak)
        if (
            resources[str(peak)]
            and active
            and any(r[label]["initializer"]["observed"] for r in active)
            and all(
                r[label]["initializer"]["passed"] is not False
                and r[label]["receiver_delta"]["passed"]
                and not r[label]["success_loss"]
                and not r[label]["added_false_delivery"]
                for r in active
            )
        ):
            eligible.append(peak)
    selected = min(eligible) if eligible else None
    calendars = [
        calendar(n, hz, blocked)
        for n in (32, 62, 126)
        for hz in (100_000_000, 160_000_000, 200_000_000)
        for blocked in (False, True)
    ]
    model = dict(
        stage=policy["stage"],
        split=args.split,
        selected=selected,
        eligible=eligible,
        RF_evaluated=evaluate_rf,
        reference_full=sum(r.get("baseline", {}).get("payload_exact", False) for r in rows),
        fixed_reference_full=sum(
            r.get("baseline", {}).get("payload_exact", False) for r in regressions
        ),
        parent_manifests=parents,
        counts={
            "p" + str(p): dict(
                full=sum(
                    r.get("p" + str(p), {}).get("metrics", {}).get("payload_exact", False)
                    for r in rows
                ),
                fixed_full=sum(
                    r.get("p" + str(p), {}).get("metrics", {}).get("payload_exact", False)
                    for r in regressions
                ),
            )
            for p in peaks
            if p is not None
        },
        negative_observation_samples=sum(policy["negative_samples"][args.split])
        if evaluate_rf
        else 0,
        default_RX_changed=False,
        whole_receiver16MSps=False,
        full_RX_gates={k: False for k in ("streaming", "fixed", "system_resources", "16MSps")},
        old_RRC_word_and16KiB_failures_preserved=True,
        provisional_node_resources=resources,
        costs=costs,
    )
    write(out / "rows.json", rows)
    write(out / "fixed-regressions.json", regressions)
    write(out / "calendars.json", calendars)
    write(out / "model.json", model)
    write(
        out / "failure-associations.json",
        dict(
            algorithm=[
                r
                for r in active
                if any(
                    not r["p" + str(p)]["metrics"]["payload_exact"] for p in peaks if p is not None
                )
            ],
            numerical=[
                r
                for r in active
                if any(
                    r["p" + str(p)]["initializer"]["passed"] is False
                    or not r["p" + str(p)]["receiver_delta"]["passed"]
                    for p in peaks
                    if p is not None
                )
            ],
            resource=[c for c in calendars if c["overflow"]],
            unavailable=[
                r
                for r in rows + regressions
                if r.get("upstream_no_candidate") or r.get("selection_not_available")
            ],
            interpretation="observed boundary;not proof of theoretical irrecoverability",
        ),
    )
    if freeze is None:
        write(
            out / "freeze.json",
            dict(
                source=source,
                policy=policy,
                selected=selected,
                validation_not_yet_generated=True,
                default_not_promoted=True,
            ),
        )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(model, indent=2))
    return 1  # All four whole-RX gates remain open,even if this module passes.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=("development", "validation"), required=True)
    parser.add_argument("--selected-root", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--word-root", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split == "development" and any(
        p is None for p in (args.selected_root, args.capture_root, args.word_root)
    ):
        parser.error("Development requires three original frozen roots")
    if args.split == "validation" and args.freeze is None:
        parser.error("Validation requires preexisting source/parameter/selection freeze")
    raise SystemExit(run(args))
