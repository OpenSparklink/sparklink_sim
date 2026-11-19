"""Selected filtered words → realCRC frame boundary;exit1 retains full-system gaps.

Fixed requests are exogenous replay,not a closed-loop acquisition improvement.
Fresh waveform tests exercise this port after one automatic whole-record maximum.
Truth and expected bytes are passed only to postdecision metrics.
"""

import argparse
import json
import math
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from audit_rx_soft_capture import diagnostic_bits
from audit_spline_fixed import independent_packet
from model_capture_fine import GEOMETRY
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import unpack_filter_word

POLICY = Path("tests/data/selected_frame_policy.json")
TRACE_KEYS = ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error")


def normalized(value):
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, np.generic):
        return value.item()
    return value


def execute(out, label, profile, source, selection, *, capacity, chunking):
    """No expected data/truth enters receiver;one candidate,no retry after failure."""
    pattern = {
        "whole": [len(source)],
        "fixed": [37],
        "random": np.random.default_rng(911118).integers(1, 257, 31),
    }[chunking]
    rx = SelectedPSKReceiver(profile, **(selection | {"sample_limit": capacity}))
    events, pos, j = [], 0, 0
    while pos < len(source) and not rx.done:
        n = int(pattern[j % len(pattern)])
        events.extend(rx.feed_filtered(source[pos : pos + n]))
        pos += rx.last_input_consumed
        j += 1
    events.extend(rx.finish())
    (event,) = events
    metadata = {k: v for k, v in event.items() if k not in TRACE_KEYS}
    metadata["selection"] = {k: v for k, v in selection.items() if k != "selected_sync"}
    metadata["profile"] = asdict(profile)
    write(out / (label + ".json"), normalized(metadata))
    arrays = {key: event[key] for key in TRACE_KEYS if key in event}
    np.savez_compressed(
        out / (label + "-trajectory.npz"), selected_sync=selection["selected_sync"], **arrays
    )
    return event, dict(
        consumed_filtered=rx.input_samples,
        header_decision_filtered=rx.header_decision_after_filtered_samples,
        frame_end_symbol=rx.frame_end_symbol,
        memory=rx.reference_memory(),
        eof_padding=False,
    )


def metrics(event, profile, *, expected, truth_start=None, truth_step=None):
    """Read-only metrics AFTER terminal decision;never selects or retries anything."""
    frame = event["frame"]
    row = dict(
        status=frame["status"],
        header_crc=frame["header_crc"],
        data_crc=frame["data_crc"],
        length_ok=frame["length_ok"],
        mcs=frame.get("mcs"),
        payload_exact=expected is not None and frame["payload"] == expected,
        false_delivery=expected is None and frame["status"] == "valid",
        bit_errors=None,
        ber_bits=0,
        timing_error_samples_max=None,
    )
    bits = diagnostic_bits(event, profile)
    if bits is not None and expected is not None:
        reference = np.unpackbits(
            np.frombuffer(expected, np.uint8), bitorder=profile.payload_bitorder
        )
        if len(bits) == len(reference):
            row.update(bit_errors=int(np.count_nonzero(bits != reference)), ber_bits=len(bits))
    if truth_start is not None and len(event.get("symbols", ())):
        centers = event["symbol_centers_filtered"]
        row["timing_error_samples_max"] = float(
            np.max(abs(centers - (truth_start + np.arange(len(centers)) * truth_step)))
        )
    return row


def compare_events(a, b):
    assert a["frame"] == b["frame"]
    assert a["delivered_after_filtered_samples"] == b["delivered_after_filtered_samples"]
    for key in TRACE_KEYS:
        if key in a:
            np.testing.assert_array_equal(a[key], b[key])


def run_selection(
    out, name, profile, source, selection, *, cap, expected, truth_start=None, truth_step=None
):
    full, full_info = execute(
        out, name + "-full", profile, source, selection, capacity=len(source), chunking="whole"
    )
    bounded, info = execute(
        out,
        name + "-prefix",
        profile,
        source,
        selection,
        capacity=min(cap, len(source)),
        chunking="whole",
    )
    for pattern in ("fixed", "random"):
        event, other = execute(
            out,
            name + "-" + pattern,
            profile,
            source,
            selection,
            capacity=min(cap, len(source)),
            chunking=pattern,
        )
        compare_events(bounded, event)
        assert other["consumed_filtered"] == info["consumed_filtered"]
    available = bounded["frame"]["status"] != "retention_limit"
    if available:
        compare_events(full, bounded)
    args = dict(expected=expected, truth_start=truth_start, truth_step=truth_step)
    return dict(
        id=name,
        capacity=min(cap, len(source)),
        full=metrics(full, profile, **args),
        prefix=metrics(bounded, profile, **args),
        full_info=full_info,
        prefix_info=info,
        partition_exact=True,
        available_frame_result_exact=available,
        scope="first-prefix availability only;not actual RAM/calendar feedback",
    )


def fresh_cases(split, policy):
    spec = policy[split]
    rng = np.random.default_rng(spec["seed"])
    for j, (ft, mcs, pilot, length) in enumerate(spec["profiles"]):
        profile = validate_profiles([(ft, mcs, pilot, length)], spec["pid"])[0]
        values = (
            (np.arange(length) * 83 + j * 41) % 256
            if j % 3 == 0
            else np.resize([255, 0, 13, 128, 1, 3, 239], length)
            if j % 3 == 1
            else rng.integers(0, 256, length)
        )
        if 'payload_transform' in spec:
            a, b, mask = spec['payload_transform']
            values = ((np.asarray(values, np.int64) * a + b) % 256) ^ mask
        payload = np.asarray(values, np.uint8).tobytes()
        cfo, ppm = spec["cfo_hz"][j % 2], spec["sro_ppm"][j % 2]
        clean, _, start, step = iq_record(
            independent_packet(profile, payload), cfo=cfo, ppm=ppm, offset=spec["offset"]
        )
        prefix = (127 + 23 * j) if split == "development" else (313 + 31 * j)
        n = prefix + len(clean) + 401
        t = np.arange(n)
        kind = spec["kinds"][j]
        sigma = 0 if kind == "clean" else 0.7 if kind == "noise" else 0.25
        iq = sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        if "tone" in kind:
            iq += (1.7 if split == "development" else 2.3) * np.exp(2j * np.pi * 0.04371 * t)
        if "two_tone" in kind:
            iq += 1.1 * np.exp(-2j * np.pi * 0.07931 * t)
        if "burst" in kind:
            for a in range(337, n, 1937 if split == "development" else 1421):
                count = min(9 if split == "development" else 5, n - a)
                iq[a : a + count] += 24 * np.exp(1j * rng.uniform(-np.pi, np.pi, count))
        if "wrong" in kind:
            wrong = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, pid=7, symbol_rate_mhz=4))
            x, _, _, _ = iq_record(independent_packet(wrong, b"new-wrong-template-content"))
            iq += 0.08 * np.resize(x, n)
        amplitude = (0.2, 0.35, 0.7)[j % 3]
        clean *= amplitude
        if kind == "amplitude_step":
            clean[len(clean) // 2 :] *= 0.55
        iq[prefix : prefix + len(clean)] += clean
        meta = dict(
            split=split,
            id=j,
            frame_type=ft,
            mcs=mcs,
            pilot=pilot,
            length=length,
            kind=kind,
            cfo_hz=cfo,
            sro_ppm=ppm,
            amplitude=amplitude,
            truth_start_filtered=prefix + start + 40,
            truth_step=step,
        )
        yield meta, profile, iq, payload
    p = validate_profiles([(2, 8, 0, 0)], spec["pid"])[0]
    for j, (n, kind) in enumerate(
        zip(policy["negative_samples_each_split"], policy["negative_kinds"], strict=True)
    ):
        iq = np.zeros(n, complex)
        if kind == "noise":
            iq = rng.normal(size=n) + 1j * rng.normal(size=n)
        elif kind == "tone":
            iq = np.exp(2j * np.pi * 0.07197 * np.arange(n))
        elif kind == "wrong_sync":
            wrong = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, pid=7, symbol_rate_mhz=4))
            x, _, _, _ = iq_record(independent_packet(wrong, b"different-negative-payload"))
            iq = np.resize(x, n)
        yield dict(split=split, id=8 + j, kind=kind, negative=True), p, iq, None


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY.read_text())
    sources = [Path(__file__), POLICY, Path("uv.lock"), *Path("src").rglob("*.py")]
    source = {str(p): sha(p) for p in sources}
    write(
        out / "protocol.json",
        dict(
            stage="SYS-C2E1",
            policy=policy,
            source=source,
            floating_selected_port_only=True,
            interpolator="existing four-point local cubic/Gardner",
            upstream_fresh="whole-record differential maximum;no streaming-capture-rate claim",
            frozen_requests="exogenous replay of already-viewed requests;noCRC retry",
            cycles=dict(interpolator=None, carrier=None, head=None, Polar=None),
            raw_relay_II4_is_not_RX_service=True,
            default_RX_changed=False,
        ),
    )
    rows = []
    for split in ("development", "validation"):
        for meta, profile, iq, payload in fresh_cases(split, policy):
            name = split + "-" + str(meta["id"])
            values = np.convolve(iq, rrc_filter(0.4, 4))
            np.savez_compressed(
                out / (name + "-input.npz"),
                iq=iq,
                filtered=values,
                payload=np.frombuffer(payload or b"", np.uint8),
            )
            found = search_psk_sync(values, 4, profile.sync_symbols(), 8)
            # No legal maximum on zero energy is itself an explicit pre-port rejection.
            if found is None:
                rows.append(dict(id=name, input=meta, upstream_no_candidate=True))
                continue
            offset = found["offset"]
            selection = dict(
                start_tick=round(8 * (found["phase"] + 4 * offset)),
                delta=found["delta"],
                differential_score=found["score"],
                selected_sync=found["phase_symbols"][offset : offset + len(profile.sync_symbols())],
                coherent_threshold=policy["coherent_Q"],
            )
            row = run_selection(
                out,
                name,
                profile,
                values,
                selection,
                cap=632 + policy["tail_words"],
                expected=payload,
                truth_start=meta.get("truth_start_filtered"),
                truth_step=meta.get("truth_step"),
            )
            rows.append(row | dict(input=meta, input_sha256=sha(out / (name + "-input.npz"))))
        write(out / (split + ".json"), [r for r in rows if r["input"]["split"] == split])
        if split == "development":
            write(
                out / "postdevelopment-freeze.json",
                dict(
                    source=source,
                    policy=policy,
                    no_parameter_changes=True,
                    new_validation_not_generated=True,
                    rule="functional exactness/explicit finite failure;no RF or cadence promotion",
                ),
            )
    regressions = []
    for folder in (args.capture_root, args.score_root):
        index = json.loads((folder / "manifest.json").read_text())
        assert all(sha(folder / k) == v for k, v in index.items())
    fixture = json.loads(Path("tests/data/spline_geometry_regression.json").read_text())["cases"]
    metadata = {
        r["id"]: r["input"] for r in json.loads((args.capture_root / "rows.json").read_text())
    }
    for entry in fixture:
        name = entry["file"][:-5]
        fine = json.loads((args.score_root / (name + "-alu2.json")).read_text())
        assert fine["source_sha256"] == entry["source_sha256"]
        if fine["service"]["disposition"] not in ("ready", "confirm_pending"):
            regressions.append(dict(id=name, service=fine["service"], selection_not_available=True))
            continue
        meta = metadata[entry["stream"]]
        assert meta["input_sha256"] == entry["raw_original_archive_sha256"]
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        with np.load(args.capture_root / (entry["stream"] + "-filtered.npz")) as ar:
            packed = ar["words"][:, fine["request"]["path"]]
        window = GEOMETRY[meta["frame_type"]][1]
        origin = fine["request"]["after_samples"] - window
        triples = [unpack_filter_word(w) for w in packed[origin:]]
        values = np.array([complex(math.ldexp(r, e), math.ldexp(i, e)) for r, i, e in triples])
        best = fine["microtrace"]["best"]
        exp = fine["microtrace"]["common_exponent"]
        sync = np.array(
            [
                complex(math.ldexp(e["word"][0], exp), math.ldexp(e["word"][1], exp))
                for e in fine["trace"]
                if e["event"] == "selected_sample"
            ]
        )
        assert len(sync) == len(profile.sync_symbols())
        corr = best["correlation"]
        selection = dict(
            start_tick=best["start_tick"],
            delta=math.atan2(corr[1], corr[0]),
            differential_score=min(1.0, math.sqrt(best["numerator"] / best["denominator"])),
            selected_sync=sync,
            coherent_threshold=policy["coherent_Q"],
        )
        with np.load(args.capture_root / (entry["stream"] + "-input.npz")) as ar:
            payload = ar["payload"].tobytes() if meta["positive"] else None
        row = run_selection(
            out,
            name,
            profile,
            values,
            selection,
            cap=window + policy["tail_words"],
            expected=payload,
            truth_start=None if payload is None else meta["sync_start"] + 40 - origin,
            truth_step=meta.get("step"),
        )
        regressions.append(
            row
            | dict(
                stream=entry["stream"],
                request=fine["request"],
                origin=origin,
                original_archive_sha256=meta["input_sha256"],
                window_sha256=entry["source_sha256"],
                filtered_archive_sha256=sha(
                    args.capture_root / (entry["stream"] + "-filtered.npz")
                ),
                old_C1_reference_used_different_carrier="medium-head PI;not directly comparable",
            )
        )
    write(out / "regressions.json", regressions)
    evaluated = [r for r in regressions if not r.get("selection_not_available")]
    summary = dict(
        fixed_requests=len(regressions),
        scored=len(evaluated),
        accepted=sum(r["full"]["status"] != "sync_confirmation_rejected" for r in evaluated),
        full_header_crc=sum(r["full"]["header_crc"] is True for r in evaluated),
        full_data_crc=sum(r["full"]["data_crc"] is True for r in evaluated),
        full_payload=sum(r["full"]["payload_exact"] for r in evaluated),
        prefix_payload=sum(r["prefix"]["payload_exact"] for r in evaluated),
        prefix_limits=sum(r["prefix"]["status"] == "retention_limit" for r in evaluated),
        prefix_sufficient=sum(r["available_frame_result_exact"] for r in evaluated),
        partition_exact=True,
        RF_capture_gain=None,
        whole_receiver16msps=False,
        full_system_resources_accepted=False,
        default_RX_changed=False,
        scope="availability reference only;no actual C2D early release or port/service proof",
        fresh=[
            dict(
                split=s,
                rows=sum(r["input"]["split"] == s for r in rows),
                selected_full=sum(
                    r.get("full", {}).get("payload_exact", False)
                    for r in rows
                    if r["input"]["split"] == s
                ),
                selected_prefix=sum(
                    r.get("prefix", {}).get("payload_exact", False)
                    for r in rows
                    if r["input"]["split"] == s
                ),
                false_deliveries=sum(
                    r.get("full", {}).get("false_delivery", False)
                    for r in rows
                    if r["input"]["split"] == s
                ),
            )
            for s in ("development", "validation")
        ],
    )
    write(out / "model.json", summary)
    write(
        out / "failure-associations.json",
        dict(
            fixed=[
                r
                for r in regressions
                if r.get("selection_not_available") or not r.get("prefix", {}).get("payload_exact")
            ],
            fresh=[
                r
                for r in rows
                if not r["input"].get("negative") and not r.get("full", {}).get("payload_exact")
            ],
            negative_controls=[r for r in rows if r["input"].get("negative")],
            interpretation=(
                "diagnostic failure is not theoretical impossibility;negative inputs separate"
            ),
        ),
    )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return 1  # Functional reference is not full resource/cadence acceptance.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
