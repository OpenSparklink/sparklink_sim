"""Actual search requests → bounded raw windows → measured integer fine schedule.

All signal inputs are already-viewed frozen regressions. This is not an RX
promotion,closed-loop packet-performance experiment,or full16MS/s acceptance.
"""

import argparse
import hashlib
import json
import math
import zipfile
from pathlib import Path
from unittest.mock import patch

import audit_carrier_loop as loop_audit
import audit_carrier_parallel as frozen_generator
import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles

from nearlink_sdr.phy import sync_candidates
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.tone_notch import CausalToneNotch
from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_transport import (
    CaptureRequest,
    CaptureTransport,
    CaptureTransportAssumptions,
    FineService,
)
from nearlink_sdr.sim.spline_score import SyncSignTemplate

INPUTS = Path("tests/data/capture_fine_inputs.json")
MEMBERS = Path("tests/data/capture_fine_members.json")
BUDGET = Path("tests/data/spline_fixed_format.json")
GEOMETRY = {2: (32, 256, 384), 3: (62, 376, 504), 4: (126, 632, 760)}


def verify_input_container(path, expected, *, require_original):
    """Exact source-member bytes,with separate original/derived archive hashes.

    Compression streams can differ across platforms. Frozen-root mode still
    requires the original archive itself;generated mode is explicitly derived.
    No array casts,tolerances,header rewriting or payload changes are allowed.
    """
    actual = sha(path)
    original = expected["original_archive_sha256"]
    if require_original and actual != original:
        raise ValueError("Frozen original archive SHA mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(expected["members"]) or set(names) != set(expected["members"]):
            raise ValueError("Input archive has missing,duplicate or unexpected members")
        for name, member in expected["members"].items():
            if archive.getinfo(name).file_size != member["bytes"]:
                raise ValueError(f"Input member size mismatch:{name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != member["sha256"]:
                raise ValueError(f"Input member SHA mismatch:{name}")
    return dict(
        original_archive_sha256=original,
        current_archive_sha256=actual,
        original_archive_hash_matches=actual == original,
        exact_original_member_hashes_verified=True,
        members=expected["members"],
    )


def image_sha(words):
    return hashlib.sha256(np.asarray(words, dtype="<u8").tobytes()).hexdigest()


def bridge(values):
    """Explicit reference boundary; no hardware latency/resource claim."""
    words = []
    maximum_error = 0.0
    adjustments = 0
    for value in values:
        peak = max(abs(float(value.real)), abs(float(value.imag)))
        exponent = math.frexp(peak)[1] - 18 if peak else 0
        if peak and round(math.ldexp(peak, -exponent)) >= 2**18:
            exponent += 1
            adjustments += 1
        real, imag = (round(math.ldexp(float(c), -exponent)) for c in (value.real, value.imag))
        words.append(pack_filter_word(real, imag, exponent))
        if peak:
            error = (
                max(
                    abs(math.ldexp(real, exponent) - value.real),
                    abs(math.ldexp(imag, exponent) - value.imag),
                )
                / peak
            )
            maximum_error = max(maximum_error, error)
    assert maximum_error <= 2**-18
    return np.asarray(words, np.uint64), dict(
        words=len(words),
        exponent_adjustments=adjustments,
        max_component_error_relative_to_word_peak=maximum_error,
        independent_analytic_error_bound=2**-18,
        hardware_cost_and_latency_verified=False,
    )


def observe(folder, meta, profile, iq, quiet, payload):
    """Observer only forwards each selected function once; noCRC/truth in hooks."""
    raw, tone, requests = [], [], []
    active = {}
    original_notch = CausalToneNotch.advance
    original_path = PSKStreamReceiver._search_path
    original_search = sync_candidates.search_candidate_grid

    def notch(self, values):
        result = original_notch(self, values)
        raw.append(np.array(values, copy=True))
        tone.append(np.array(result, copy=True))
        return result

    def path(self):
        active["rx"] = self
        return original_path(self)

    def search(*args, **kwargs):
        rx = active["rx"]
        result = original_search(*args, **kwargs)
        peaks = kwargs["coarse_starts"]
        assert all(float(v).is_integer() for v in peaks)
        requests.append(
            dict(
                after_samples=int(rx._processed),
                path=int(rx.integer_screen_backend.startswith("tone-")),
                window_samples=len(args[0]),
                sync_symbols=len(args[2]),
                origin_raw=int(rx._origin + len(rx._filtered) - len(args[0])),
                peaks=[int(v) for v in peaks],
                minimum_tick=math.ceil(8 * kwargs["min_start"]),
                lookahead_ticks=round(8 * kwargs["lookahead"]),
                float_diagnostic=dict(kwargs["diagnostics"]),
                window=np.array(args[0], copy=True),
            )
        )
        return result

    with (
        patch.object(CausalToneNotch, "advance", notch),
        patch.object(PSKStreamReceiver, "_search_path", path),
        patch.object(sync_candidates, "search_candidate_grid", search),
    ):
        row, arrays = loop_audit.run_arm(folder, meta, profile, iq, quiet, payload, "medium-head")
    return row, arrays, np.column_stack((np.concatenate(raw), np.concatenate(tone))), requests


def load_cases(args, out):
    plan = json.loads(INPUTS.read_text())
    member_index = json.loads(MEMBERS.read_text())
    if member_index["source_index_sha256"] != sha(INPUTS):
        raise ValueError("Original input index does not match frozen member provenance")
    generated = {}
    containers = {}
    if args.frozen_root is None:
        for held in (False, True):
            for meta, _p, iq, q, payload in frozen_generator.cases(held):
                generated[("validation" if held else "development", meta["id"])] = (iq, q, payload)
    for case in plan["cases"]:
        split, meta = case["split"], case["meta"]
        name = split + "-" + meta["id"]
        target = out / (name + "-input.npz")
        if args.frozen_root is not None:
            source = args.frozen_root / (split + "-first") / (meta["id"] + "-input.npz")
            assert sha(source) == meta["input_sha256"]
            target.write_bytes(source.read_bytes())
        else:
            iq, q, payload = generated[(split, meta["id"])]
            np.savez_compressed(target, iq=iq, quiet=q, payload=np.frombuffer(payload, np.uint8))
        expected_container = member_index["entries"][name]
        if expected_container["original_archive_sha256"] != meta["input_sha256"]:
            raise ValueError("Original archive provenance mismatch")
        containers[name] = verify_input_container(
            target, expected_container, require_original=args.frozen_root is not None
        )
        write(out / "input-containers.json", containers)
        with np.load(target) as ar:
            iq, q, payload = ar["iq"], ar["quiet"], ar["payload"].tobytes()
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        yield name, meta, profile, iq, q, payload, case["reference"]


def replay(out, name, source, requests, services, clock, history, window, *, stalls=()):
    a = CaptureTransportAssumptions(clock_hz=clock, window=window, history=history)
    events = []

    def measured(event, words):
        ident = event["id"]
        cached = services[ident]
        assert event["geometry"] == requests[ident]
        assert image_sha(words) == cached["source_sha256"]
        return FineService(**cached["service"])

    def sink(event, words):
        if words is not None:
            assert image_sha(words) == services[event["id"]]["source_sha256"]
        ident = event["id"]
        event.update(shadow_disposition=services[ident]["service"]["disposition"])
        events.append(event)

    result = CaptureTransport(a).simulate(
        source, requests, fine_worker=measured, sink=sink, stalls=stalls
    )
    resource = [e for e in events if e["status"] != "completed"]
    ready = [
        e
        for e in events
        if e["status"] == "completed" and e["disposition"] in ("ready", "confirm_pending")
    ]
    last_arrival = ((len(source) - 1) * clock + 16000000 - 1) // 16000000
    required = max(
        (e["continuous_count_after_edge"] - e["after_samples"] for e in ready), default=0
    )
    projection = dict(
        required_history_on_this_trace=max(window, required),
        source_history_BRAM36_capacity_only=4 * math.ceil(max(window, required) / 1024),
        not_adopted_or_replayed=True,
        does_not_fix_queue_or_replay_port_gates=True,
    )
    report = dict(
        assumptions=vars(a),
        result=result,
        events=events,
        unavailable_requests=len(resource),
        unavailable_shadow_confirmable=sum(
            e["shadow_disposition"] in ("ready", "confirm_pending") for e in resource
        ),
        potential_continuation_retention_lost=sum(e["continuation_gap_samples"] > 0 for e in ready),
        potential_continuation_retention_necessary_pass=sum(
            e["continuation_gap_samples"] == 0 for e in ready
        ),
        potential_completed_after_source_eof=sum(e["cycle"] > last_arrival for e in ready),
        max_required_continuation_history_samples=required,
        projection=projection,
        all_borrowed_images_exact=True,
        service_microtraces_replayed_after_exact_input_hash=True,
        resource_feedback_to_receiver=False,
        whole_receiver_pass=False,
    )
    path = out / f"{name}-schedule-{clock}{'-blocked' if stalls else ''}.json"
    write(path, report)
    return dict(
        file=path.name,
        sha256=sha(path),
        clock=clock,
        requests=len(requests),
        completed=result["completed"],
        queue_drops=result["queue_drops"],
        history_drops=result["history_drops"],
        startup_drops=result["startup_drops"],
        potential_continuation_retention_lost=report["potential_continuation_retention_lost"],
        potential_continuation_retention_necessary_pass=report[
            "potential_continuation_retention_necessary_pass"
        ],
        unavailable_shadow_confirmable=report["unavailable_shadow_confirmable"],
        potential_completed_after_source_eof=report["potential_completed_after_source_eof"],
        max_required_continuation_history_samples=required,
        max_latency_cycles=result["max_latency_cycles"],
    )


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    index = json.loads(INPUTS.read_text())
    write(
        out / "protocol.json",
        dict(
            stage="SYS-C1",
            input_index_sha256=sha(INPUTS),
            parent_evidence_sha256=index["parent_index_sha256"],
            input_mode="frozen"
            if args.frozen_root
            else "derived-container-original-members-verified",
            original_archive_provenance_preserved=True,
            generated_container_hashes_separately_recorded=True,
            input_members_sha256=sha(MEMBERS),
            input_container_ledger="input-containers.json",
            geometry=GEOMETRY,
            clocks=[100000000, 160000000, 200000000],
            owned=3,
            paths=2,
            word_bits=46,
            transport_BRAM36_study_cap=12,
            source={
                str(p): sha(p)
                for p in (
                    Path(__file__),
                    INPUTS,
                    MEMBERS,
                    BUDGET,
                    Path("scripts/audit_carrier_loop.py"),
                    Path("scripts/audit_carrier_parallel.py"),
                    Path("scripts/audit_rx_soft_capture.py"),
                    Path("scripts/audit_spline_fixed.py"),
                    Path("scripts/rx_timing_audit.py"),
                    Path("tests/test_strict_receive.py"),
                    Path("tests/test_psk_field_layout.py"),
                    Path("pyproject.toml"),
                    Path("uv.lock"),
                    *Path("src").rglob("*.py"),
                )
            },
            viewed_fixed_regressions_only=True,
            new_signal_held_generated=False,
            runtime_truth_or_CRC_selection=False,
            default_changed=False,
            input_converter_and_upstream_RRC_scheduling_verified=False,
        ),
    )
    rows = []
    costs = {}
    for name, meta, p, iq, q, payload, expected in load_cases(args, out):
        direct, baseline = loop_audit.run_arm(
            out / (name + "-direct"), meta, p, iq, q, payload, "medium-head"
        )
        row, arrays, filtered, offered = observe(
            out / (name + "-observed"), meta, p, iq, q, payload
        )
        assert row == direct == expected, name
        assert baseline.keys() == arrays.keys() and all(
            np.array_equal(v, arrays[k]) for k, v in baseline.items()
        ), name
        assert len(offered) == row["stats"]["full_searches"]
        count, window, history = GEOMETRY[meta["frame_type"]]
        assert count == len(p.sync_symbols())
        packed = []
        bridge_stats = []
        for path in (0, 1):
            words, stats = bridge(filtered[:, path])
            packed.append(words)
            bridge_stats.append(stats)
        source = np.column_stack(packed)
        np.savez_compressed(out / (name + "-filtered.npz"), reference=filtered, words=source)
        template = SyncSignTemplate.from_symbols(p.sync_symbols())
        worker = CaptureFineWorker(template, window, confirmation_required=True)
        costs[str(meta["frame_type"])] = dict(
            transport=CaptureTransport(
                CaptureTransportAssumptions(window=window, history=history)
            ).estimate(measured_worker=True),
            fine=worker.estimate(),
        )
        assert costs[str(meta["frame_type"])]["transport"]["transport_bram36"] <= 12
        requests = []
        services = []
        numeric = []
        for j, offer in enumerate(offered):
            after, path = offer["after_samples"], offer["path"]
            assert offer["window_samples"] == window and after <= len(source)
            assert np.array_equal(offer["window"], filtered[after - window : after, path])
            words = source[after - window : after, path]
            request = CaptureRequest(
                after, path, tuple(offer["peaks"]), offer["minimum_tick"], offer["lookahead_ticks"]
            )
            requests.append(request)
            result = []
            worker.sink = lambda event, r: result.append(r)
            service = worker(
                dict(id=j, after_samples=after, path=path, window_samples=window, geometry=request),
                words,
            )
            record = dict(
                id=j,
                source_sha256=image_sha(words),
                request=vars(request),
                service=vars(service),
                microtrace=result[0],
                float_diagnostic=offer["float_diagnostic"],
            )
            write(out / f"{name}-fine-{j}.json", record)
            services.append(record)
            best = result[0]["best"]
            diag = offer["float_diagnostic"]
            score = (
                math.sqrt(best["numerator"] / best["denominator"])
                if best and best["denominator"]
                else 0.0
            )
            start = best["start_tick"] / 8 if best else None
            score_error = (
                abs(score - diag["best_score"]) if diag.get("best_score") is not None else None
            )
            start_error = (
                abs(start - diag["best_start"])
                if start is not None and "best_start" in diag
                else None
            )
            same_presence = (best is None) == (diag.get("best_start") is None)
            same_disposition = best is None or best["disposition"] == diag["disposition"]
            budget = json.loads(BUDGET.read_text())
            numerical_pass = (
                not result[0].get("numerical_rejection")
                and same_presence
                and same_disposition
                and (score_error is None or score_error <= budget["score_limit"])
                and (start_error is None or start_error <= budget["start_limit"])
            )
            numeric.append(
                dict(
                    id=j,
                    score_error=score_error,
                    start_error=start_error,
                    passed=numerical_pass,
                    same_best_presence=same_presence,
                    same_disposition=same_disposition,
                    service=vars(service),
                    source_sha256=image_sha(words),
                )
            )
        schedules = [
            replay(out, name, source, requests, services, c, history, window)
            for c in (100000000, 160000000, 200000000)
        ]
        row = dict(
            id=name,
            input=meta,
            reference=row,
            observer_arrays_exact=True,
            requests=len(requests),
            raw_samples=len(iq),
            filtered_words=len(source),
            EOF_filter_flush_words=len(source) - len(iq),
            bridge=bridge_stats,
            numerical=numeric,
            schedules=schedules,
        )
        rows.append(row)
        write(out / "rows.json", rows)
        print(name, len(requests), "requests", flush=True)
    total = dict(
        streams=len(rows),
        samples=sum(r["raw_samples"] for r in rows),
        requests=sum(r["requests"] for r in rows),
        observer_arrays_exact=True,
        functional_reference_capture=sum(r["reference"]["captured"] for r in rows),
        functional_reference_full=sum(r["reference"]["correct"] for r in rows),
        functional_reference_heads=sum(r["reference"]["head_crc"] for r in rows),
        functional_reference_data_crc=sum(r["reference"]["data_crc"] for r in rows),
        functional_reference_bit_errors=sum(r["reference"]["bit_errors"] for r in rows),
        functional_reference_ber_bits=sum(r["reference"]["ber_bits"] for r in rows),
        numerical_failures=[
            dict(stream=r["id"], **n) for r in rows for n in r["numerical"] if not n["passed"]
        ],
        schedules={
            str(c): {
                k: sum(s[k] for r in rows for s in r["schedules"] if s["clock"] == c)
                for k in (
                    "requests",
                    "completed",
                    "queue_drops",
                    "history_drops",
                    "startup_drops",
                    "potential_continuation_retention_lost",
                    "potential_continuation_retention_necessary_pass",
                    "unavailable_shadow_confirmable",
                    "potential_completed_after_source_eof",
                )
            }
            for c in (100000000, 160000000, 200000000)
        },
        costs=costs,
        old2percent16KiB_changed=False,
        default_changed=False,
        new_signal_held_generated=False,
        whole_receiver16msps=False,
        resource_feedback_to_receiver=False,
        retention_pass_is_necessary_only=True,
        upstream_RRC_or_integer_bank_or_converter_schedule_verified=False,
    )
    write(out / "model.json", total)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            {k: v for k, v in total.items() if k not in ("costs", "numerical_failures")}, indent=2
        )
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path)
    raise SystemExit(run(parser.parse_args()))
