"""Paired finite pre-fine retirement study;not full RX/FPGA or16MSps acceptance."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scripts import model_coarse_ingress as shared
from scripts.model_captured_frame import safe, write
from tests.captured_frame_reference import packet, receive
from tests.maximum_reference import channel

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig

POLICY = dict(
    development_specs=[
        [2, 8, 0, [23, 57], 48, "clean"],
        [3, 12, 4, [41, 73], 16, "tone"],
        [4, 8, 8, [17, 61], 32, "clean"],
        [2, 12, 16, [91, 37], 8, "noise"],
        [3, 8, 0, [67, 139], 24, "step"],
        [4, 12, 4, [97, 19], 64, "burst"],
    ],
    validation_specs=[
        [4, 12, 16, [31, 101], 12, "clean"],
        [2, 8, 8, [79, 13], 40, "tone"],
        [3, 12, 0, [113, 47], 56, "wrong-sync"],
        [4, 8, 4, [11, 83], 20, "step"],
        [2, 12, 0, [151, 29], 28, "burst"],
        [3, 8, 16, [53, 173], 72, "noise"],
    ],
    symbol_MHz=4,
    SPS=4,
    clock_MHz=200,
    history=4096,
    owned=3,
    coherent_threshold=0.45,
    additional_control_cap_bits=128,
    new_IQ_RAM_DSP_or_ports=0,
    registered_check_cycles=1,
    score_abs_budget=0.005,
    start_samples_budget=0.125,
    trajectory_samples_budget=0.25,
    same_window_integer_and_input_words="exact",
    same_backend_chunks="byte exact",
    negative_families=["noise", "tone", "wrong-sync", "burst"],
    negative_samples_each=8192,
    original_complete_losses=0,
    negative_additional_confirmations=0,
    false_MAC_and_duplicate_limit=0,
    diagnostic_BER_association_radius_samples=2,
    preamble_symbols={"2": 40, "3": 48, "4": 64},
    new_default=False,
    old_numeric_and16KiB_failures_preserved=True,
    full_cycle_or_fixed_or16MSps=False,
    required_development_benefit="positive aggregate reduction of fine+registered-check work",
    minimum_original_complete=1,
    aggregate_queue_and_history_drop_delta_max=0,
    MAC_gap_conformance_not_claimed=True,
)


def payload(n, ident, held):
    if ident % 2:
        return bytes(
            (0x39, 0xE6, 0x1B, 0x94, 0x67, 0xAD)[(k // 3 + k) % 6]
            ^ ((ident * 37 + k // 17 + 53 * held) & 255)
            for k in range(n)
        )
    return bytes(
        ((k * 37 + 71 * ident) ^ (k >> 3) ^ (0xD5 if held else 0x4B)) & 255 for k in range(n)
    )


def make_inputs(out, split, *, specs=None, payload_fn=None, seed_bias=0):
    """Independent analytic generator;defaults preserve the original frozen bytes."""
    if type(seed_bias) is not int or not 0 <= seed_bias < 2**31:
        raise ValueError("Need a finite explicit generation seed bias")
    held = split == "validation"
    specs = POLICY["validation_specs" if held else "development_specs"] if specs is None else specs
    data_fn = payload if payload_fn is None else payload_fn
    rows = []
    for j in range(10):
        negative = j >= 6
        ft, mcs, pilots, lengths, idle, family = specs[j % 6]
        if negative:
            ft, mcs, pilots = (2 + j % 3), (8 if j % 2 else 12), (0 if held else 8)
            lengths, idle, family = [53, 101], 24, POLICY["negative_families"][j - 6]
        cfg = shared.shared.configuration(ft, mcs, pilots, lengths[0])
        cfg.pid = (j + 2 + 3 * held) % 6
        profile = replace(
            PSKReceiveProfile.ordinary(cfg),
            allowed_mcs=(mcs,),
            head_backend="word-dense" if ft == 2 else "word-depth-parity",
        )
        data = [data_fn(n, 2 * j + k, held) for k, n in enumerate(lengths)]
        wave = [packet(profile, d) for d in data]
        offsets = [0, len(wave[0]) + idle]
        symbols = np.r_[wave[0], np.zeros(idle, complex), wave[1]]
        params = dict(
            offset=((0.431, 0.827, 0.193, 0.659) if held else (0.217, 0.583, 0.761, 0.349))[j % 4],
            ppm=((100, -50, 75, -25) if held else (-75, 25, 50, -100))[j % 4],
            cfo=((83000, -57000, 129000, -23000) if held else (-41000, 67000, -93000, 17000))[
                j % 4
            ],
            prefix=(79 + 43 * j) if held else (41 + 53 * j),
            amplitude=((0.089, 0.157, 0.061, 0.137) if held else (0.119, 0.073, 0.181, 0.097))[
                j % 4
            ],
            seed=((1329113 + 211 * j) if held else (1288401 + 179 * j)) + seed_bias,
        )
        rng = np.random.default_rng(params["seed"] + 103)
        truth = []
        if negative and family != "wrong-sync":
            n = POLICY["negative_samples_each"]
            iq = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.002
            quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.002
            if family == "tone":
                iq += 0.9 * np.exp(2j * np.pi * (-317000 if held else 193000) * np.arange(n) / 16e6)
            elif family == "burst":
                iq[1841:1854] += 3 - 2j
            metadata = dict(no_legal_packet=True)
        else:
            wire_profile = profile
            if negative:
                wire_profile = replace(profile, config=replace(cfg, pid=(cfg.pid + 1) % 6))
                part = packet(wire_profile, data[0])
                symbols = np.tile(np.r_[part, np.zeros(idle, complex)], 4)
            iq, quiet, legacy_truth, step, _ = channel(symbols, family="clean", **params)
            origin = legacy_truth - 40 * step
            extra = 1792 if held else 1536
            iq = np.r_[iq, np.zeros(extra, complex)]
            lo, hi = int(np.ceil(origin)), int(np.floor(origin + len(symbols) * step))
            rms = float(np.sqrt(np.mean(abs(iq[lo:hi]) ** 2)))
            metadata = dict(
                clean_active_RMS=rms,
                clean_active_interval=[lo, hi],
                includes_interpacket_idle=True,
                step=step,
            )
            if not negative:
                truth = [
                    dict(
                        preamble=origin + v * step,
                        sync=origin + (v + POLICY["preamble_symbols"][str(ft)]) * step + 40,
                        step=step,
                    )
                    for v in offsets
                ]
            if family == "noise":
                snr = 10.8 if held else 12.3
                sigma = rms / np.sqrt(2 * 10 ** (snr / 10))
                iq += (rng.normal(size=len(iq)) + 1j * rng.normal(size=len(iq))) * sigma
                quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * sigma
                metadata.update(snr_db=snr, noise_std_component=float(sigma))
            elif family == "tone":
                ratio, hz = (0.45, -317000) if held else (0.25, 193000)
                iq += rms * ratio * np.exp(2j * np.pi * hz * np.arange(len(iq)) / 16e6)
                metadata.update(tone_to_clean_RMS=ratio, tone_Hz=hz)
            elif family == "step":
                index = int(np.ceil(origin + offsets[1] * step))
                iq[index:] *= 0.3125 if held else 0.25
                metadata.update(step_index=index, factor=0.3125 if held else 0.25)
            elif family == "burst":
                index = lo + int((hi - lo) * (0.41 if held else 0.57))
                size = 13 if held else 9
                iq[index : index + size] += rms * (3 - 2j)
                metadata.update(burst_index=index, burst_samples=size, burst_to_RMS=[3, -2])
            elif family == "wrong-sync" and not negative:
                wrong = replace(profile, config=replace(cfg, pid=(cfg.pid + 1) % 6))
                other = [
                    packet(wrong, data_fn(n, 2 * j + k + 23, not held))
                    for k, n in enumerate(lengths)
                ]
                other_symbols = np.tile(
                    np.r_[other[0], np.zeros(idle), other[1], np.zeros(idle)], 3
                )
                interference, _, _, _, _ = channel(
                    other_symbols, family="clean", **(params | dict(cfo=params["cfo"] - 21700))
                )
                assert len(interference) >= len(iq)
                iq += 0.43 * interference[: len(iq)]
                metadata.update(wrong_sync_ratio=0.43, wrong_sync_CFO_delta_Hz=-21700)
            if negative:
                iq = np.pad(iq, (0, max(0, 8192 - len(iq))))[:8192]
        file = out / f"input{j}.npz"
        np.savez_compressed(file, iq=iq, quiet=quiet)
        config = vars(cfg).copy()
        config.pop("_mcs")
        rows.append(
            dict(
                id=j,
                negative=negative,
                file=file.name,
                SHA256=shared.sha(file),
                config=config,
                head_backend=profile.head_backend,
                family=family,
                params=params,
                metadata=metadata,
                truth_diagnostic_only=truth,
                expected=[] if negative else [d.hex() for d in data],
            )
        )
    write(out / "inputs.json", rows)
    return rows


def event_metadata(frames):
    return [
        safe({k: v for k, v in e.items() if not isinstance(v, np.ndarray)})
        | dict(MAC=[p.mac_payload.hex() for p in mac.packets])
        for e, mac in frames
    ]


def run_arm(profile, iq, quiet, enabled):
    checks, decoded, fine, windows, coherence = [], [], {}, {}, []

    def fine_result(descriptor, result):
        best = result["best"]
        critical = (
            None
            if best is None
            else {
                k: best[k]
                for k in ("start_tick", "correlation", "numerator", "energy", "threshold_pass")
            }
        )
        geometry = asdict(descriptor["geometry"])
        fine[descriptor["id"]] = dict(
            geometry=geometry,
            best=safe(critical),
            common_exponent=result.get("common_exponent"),
            cycles=result["cycles"],
        )

    def window_result(event, data):
        windows[event["id"]] = hashlib.sha256(np.asarray(data, dtype="<u8").tobytes()).hexdigest()

    observed = receive(
        profile,
        iq,
        quiet,
        retire_before_fine=enabled,
        retirement_observer=checks.append if enabled else None,
        decoder_observer=decoded.append,
        fine_observer=fine_result,
        copied_window_observer=window_result,
        coherence_observer=coherence.append,
    )
    result, frames, reads, _, _, _ = observed
    baseline_metadata = event_metadata(frames)
    for groups in ((1, 31, 127), tuple(np.random.default_rng(193279).integers(1, 258, 19))):
        other_checks, other_coherence = [], []
        other = receive(
            profile,
            iq,
            quiet,
            groups=groups,
            retire_before_fine=enabled,
            retirement_observer=other_checks.append if enabled else None,
            coherence_observer=other_coherence.append,
        )
        assert other[0] == result and other[2] == reads and checks == other_checks
        assert other_coherence == coherence
        assert event_metadata(other[1]) == baseline_metadata
        assert len(other[1]) == len(frames)
        for (a, _), (b, _) in zip(frames, other[1], strict=True):
            assert {k for k, v in a.items() if isinstance(v, np.ndarray)} == {
                k for k, v in b.items() if isinstance(v, np.ndarray)
            }
            for k, v in a.items():
                if isinstance(v, np.ndarray):
                    assert (
                        v.dtype == b[k].dtype
                        and v.shape == b[k].shape
                        and v.tobytes() == b[k].tobytes()
                    )
    return observed, checks, decoded, fine, windows, coherence


def analyze(out, row, label, run, samples):
    observed, checks, decoded, fine, windows, coherence = run
    result, frames, reads, owner, source, outcomes = observed
    expected = [bytes.fromhex(d) for d in row["expected"]]
    complete, false, events, traces = [], 0, [], {}
    accepted = {r["id"]: r["accepted"] for r in coherence}
    assert len(accepted) == len(coherence)
    assert sum(accepted.values()) == owner.stats["coherent_accepted"]
    for k, (event, mac) in enumerate(frames):
        file = f"trace-{row['id']}-{label}-{k}.npz"
        arrays = {a: v for a, v in event.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(out / file, **arrays)
        origin, s = event["source_window_origin"], event["upstream_selection"]
        start = origin + s["start_tick"] / 8
        truth = row["truth_diagnostic_only"]
        nearest = (
            min(range(len(truth)), key=lambda q: abs(truth[q]["sync"] - start)) if truth else None
        )
        associated = (
            nearest if nearest is not None and abs(truth[nearest]["sync"] - start) <= 2 else None
        )
        timing = None
        if associated is not None and len(arrays.get("symbol_centers_filtered", ())):
            target = truth[associated]
            err = (
                origin
                + arrays["symbol_centers_filtered"]
                - (
                    target["sync"]
                    + np.arange(len(arrays["symbol_centers_filtered"])) * target["step"]
                )
            )
            timing = dict(
                max_abs=float(max(abs(err))),
                RMS=float(np.sqrt(np.mean(err**2))),
                head_mid_tail_RMS=[float(np.sqrt(np.mean(v**2))) for v in np.array_split(err, 3)],
                diagnostic_only=True,
            )
        for p in mac.packets:
            assert accepted[event["selection_id"]] is True
            q = expected.index(p.mac_payload) if p.mac_payload in expected else None
            frame = event["frame"]
            if (
                q is not None
                and frame.get("header_crc") is True
                and frame.get("data_crc") is True
                and frame.get("length_ok") is True
                and frame.get("mcs") == row["config"]["mcs_index"]
                and frame.get("payload_length_bytes") == len(expected[q])
            ):
                complete.append(q)
                traces[q] = dict(
                    start=start,
                    score=np.sqrt(s["numerator"] / (s["energy"] * s["template_energy"])),
                    centers=origin + arrays["symbol_centers_filtered"],
                )
            else:
                false += 1
        events.append(
            dict(
                metadata=event_metadata([(event, mac)])[0],
                trace=file,
                SHA256=shared.sha(out / file),
                association=associated,
                coherent_accepted=accepted[event["selection_id"]],
                timing=timing,
            )
        )
    bit_reports = []
    for d in decoded:
        e = next(e for e in events if e["metadata"]["selection_id"] == d["id"])
        q = e["association"]
        reference = None if q is None else np.unpackbits(np.frombuffer(expected[q], np.uint8))
        file = f"bits-{row['id']}-{label}-{d['id']}.npz"
        np.savez_compressed(out / file, bits=d["bits"])
        bit_reports.append(
            dict(
                file=file,
                SHA256=shared.sha(out / file),
                id=d["id"],
                association=q,
                bits=len(d["bits"]),
                CRC=bool(d["CRC"]),
                MCS=d["MCS"],
                errors=None
                if reference is None or len(reference) != len(d["bits"])
                else int(np.count_nonzero(reference != d["bits"])),
            )
        )
    data = dict(
        result=result,
        checks=checks,
        coherent_decisions=coherence,
        fine=fine,
        windows=windows,
        frames=events,
        decoded=bit_reports,
        reads=reads,
        outcomes=[
            {k: safe(v) for k, v in r.items() if k not in ("strict_frame", "MAC_packets")}
            for r in outcomes
        ],
    )
    write(out / f"stream{row['id']}-{label}.json", data)
    work = sum(r["fine_cycles"] or 0 for r in outcomes)
    summary = dict(
        complete=sorted(complete),
        captured_packet_associations_diagnostic=sorted(
            {
                e["association"]
                for e in events
                if e["association"] is not None and e["coherent_accepted"]
            }
        ),
        false_MAC=false,
        duplicate_MAC=len(complete) - len(set(complete)),
        fine_cycles=work,
        check_cycles=sum(r["event"] == "check" for r in checks),
        fine_calls=len(fine),
        pre_fine_retired=result.get("pre_fine_retired", 0),
        queue_drops=result["queue_drops"],
        history_drops=result["history_drops"],
        coherent=owner.stats["coherent_accepted"],
        head_CRC=owner.stats["header_valid"],
        data_CRC=sum(e["frame"].get("data_crc") is True for e, _ in frames),
        BER_errors=sum(r["errors"] for r in bit_reports if r["errors"] is not None),
        BER_bits=sum(r["bits"] for r in bit_reports if r["errors"] is not None),
        source_lossless=source.flow.lossless,
        all_pulses_accounted=result["input_samples"] == samples,
        max_ports=result["peak_source_operations_per_primitive"],
        max_owned=result["peak_owned"],
        known_drain_cycle=result["drain_cycle"],
        complete_latency_cycles=None,
    )
    return summary, traces


def evaluate(out, row):
    assert shared.sha(out / row["file"]) == row["SHA256"]
    p = replace(
        PSKReceiveProfile.ordinary(TxConfig(**row["config"])),
        head_backend=row["head_backend"],
        allowed_mcs=(row["config"]["mcs_index"],),
    )
    with np.load(out / row["file"]) as z:
        iq, quiet = z["iq"], z["quiet"]
    old, new = (run_arm(p, iq, quiet, enabled) for enabled in (False, True))
    a, ta = analyze(out, row, "old", old, len(iq))
    b, tb = analyze(out, row, "new", new, len(iq))
    shared_ids = old[3].keys() & new[3].keys()
    exact = all(old[3][k] == new[3][k] and old[4][k] == new[4][k] for k in shared_ids)
    source_exact = (
        old[0][0]["live_source_statistics"] == new[0][0]["live_source_statistics"]
        and old[0][0]["live_coarse_statistics"] == new[0][0]["live_coarse_statistics"]
    )

    def descriptors(r):
        return sorted((e["id"], e["after_samples"], e["offered_cycle"]) for e in r[5])

    source_exact &= descriptors(old[0]) == descriptors(new[0])
    comparisons = []
    for q in ta.keys() & tb.keys():
        x, y = ta[q], tb[q]
        same_length = len(x["centers"]) == len(y["centers"])
        error = float(max(abs(x["centers"] - y["centers"]))) if same_length else None
        comparisons.append(
            dict(
                packet=q,
                start_abs=abs(x["start"] - y["start"]),
                score_abs=float(abs(x["score"] - y["score"])),
                trajectory_max=error,
            )
        )
    checks = {e["id"]: e for e in new[1] if e["event"] == "check"}
    stable = all(
        not e["frame_active"]
        and e["retired_through"] == checks[e["id"]]["retired_through"]
        and (
            e["event"] not in ("fine_start", "retired")
            or e["cycle"] == checks[e["id"]]["cycle"] + 1
        )
        for e in new[1]
    )
    accounted = all(
        r[0][0]["requests"]
        == sum(
            r[0][0].get(k, 0)
            for k in (
                "completed",
                "queue_drops",
                "history_drops",
                "startup_drops",
                "source_epoch_drops",
                "pre_fine_retired",
            )
        )
        for r in (old, new)
    )
    cost = new[0][0]["retirement_gate_cost"]
    return dict(
        id=row["id"],
        input_SHA256=row["SHA256"],
        negative=row["negative"],
        samples=len(iq),
        old=a,
        new=b,
        original_success_losses=sorted(set(a["complete"]) - set(b["complete"])),
        recoveries=sorted(set(b["complete"]) - set(a["complete"])),
        same_window_fine_integer_exact=exact,
        common_fine_windows=len(shared_ids),
        source_descriptors_exact=bool(source_exact),
        original_trajectory_comparisons=comparisons,
        registered_frontier_stable=stable,
        all_requests_accounted=accounted,
        all_words_and_chunks_exact=True,
        additional_control_bits=cost["register_bits"],
        new_IQ_RAM_DSP_or_ports=cost["new_IQ_RAM_DSP_or_sample_ports"],
        whole_RX_accepted=False,
    )


def qualify(rows):
    if len(rows) != 10 or {r["id"] for r in rows} != set(range(10)):
        return False
    if len({r["input_SHA256"] for r in rows}) != 10:
        return False
    for r in rows:
        if (
            type(r["id"]) is not int
            or type(r["samples"]) is not int
            or r["samples"] <= 0
            or type(r["negative"]) is not bool
            or r["negative"] != (r["id"] >= 6)
            or r["whole_RX_accepted"] is not False
            or r["original_success_losses"]
            or any(
                r[k] is not True
                for k in (
                    "same_window_fine_integer_exact",
                    "source_descriptors_exact",
                    "registered_frontier_stable",
                    "all_requests_accounted",
                    "all_words_and_chunks_exact",
                )
            )
            or type(r["additional_control_bits"]) is not int
            or not 0 <= r["additional_control_bits"] <= POLICY["additional_control_cap_bits"]
            or r["new_IQ_RAM_DSP_or_ports"] != 0
        ):
            return False
        for arm in (r["old"], r["new"]):
            captured = arm["captured_packet_associations_diagnostic"]
            if (
                not isinstance(captured, list)
                or any(type(q) is not int or q not in (0, 1) for q in captured)
                or len(set(captured)) != len(captured)
                or len(captured) > arm["coherent"]
                or not set(arm["complete"]) <= set(captured)
                or (r["negative"] and captured)
            ):
                return False
            counters = (
                "false_MAC",
                "duplicate_MAC",
                "fine_cycles",
                "check_cycles",
                "fine_calls",
                "pre_fine_retired",
                "queue_drops",
                "history_drops",
                "coherent",
                "head_CRC",
                "data_CRC",
                "BER_errors",
                "BER_bits",
                "max_ports",
                "max_owned",
                "known_drain_cycle",
            )
            if any(type(arm[k]) is not int or arm[k] < 0 for k in counters):
                return False
            if (
                arm["false_MAC"]
                or arm["duplicate_MAC"]
                or arm["source_lossless"] is not True
                or arm["all_pulses_accounted"] is not True
                or arm["max_ports"] > 2
                or arm["max_owned"] > 3
                or arm["complete_latency_cycles"] is not None
            ):
                return False
            if (
                not isinstance(arm["complete"], list)
                or any(type(q) is not int or q not in (0, 1) for q in arm["complete"])
                or len(set(arm["complete"])) != len(arm["complete"])
                or len(arm["complete"]) > arm["data_CRC"]
                or not arm["data_CRC"] <= arm["head_CRC"] <= arm["coherent"] <= arm["fine_calls"]
                or (r["negative"] and arm["complete"])
            ):
                return False
        if r["negative"] and r["new"]["coherent"] > r["old"]["coherent"]:
            return False
        if len(r["original_trajectory_comparisons"]) != len(set(r["old"]["complete"])):
            return False
        for c in r["original_trajectory_comparisons"]:
            if any(
                type(c[k]) not in (int, float) or not np.isfinite(c[k]) or c[k] < 0
                for k in ("start_abs", "score_abs", "trajectory_max")
            ):
                return False
            if (
                c["start_abs"] > POLICY["start_samples_budget"]
                or c["score_abs"] > POLICY["score_abs_budget"]
                or c["trajectory_max"] is None
                or c["trajectory_max"] > POLICY["trajectory_samples_budget"]
            ):
                return False
    return (
        sum(len(r["old"]["complete"]) for r in rows) >= POLICY["minimum_original_complete"]
        and sum(r["old"]["fine_cycles"] for r in rows)
        > sum(r["new"]["fine_cycles"] + r["new"]["check_cycles"] for r in rows)
        and sum(r["new"]["queue_drops"] for r in rows) <= sum(r["old"]["queue_drops"] for r in rows)
        and sum(r["new"]["history_drops"] for r in rows)
        <= sum(r["old"]["history_drops"] for r in rows)
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=shared.sources())
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen eligible development")
        old = json.loads(args.freeze.read_text())
        summary = json.loads((args.freeze.parent / "model.json").read_text())
        end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        if old != frozen or end != frozen["sources"] or not summary["component_qualified"]:
            raise ValueError("Need unchanged source/policy and eligible development")
        if summary["freeze_SHA256"] != shared.sha(args.freeze):
            raise ValueError("Need exact frozen development envelope")
    args.output.mkdir(exist_ok=False)
    write(args.output / "freeze.json", frozen)
    rows = [evaluate(args.output, r) for r in make_inputs(args.output, args.split)]
    assert shared.sources() == frozen["sources"]
    write(args.output / "source-bindings-at-end.json", shared.sources())
    result = dict(
        split=args.split,
        rows=rows,
        component_qualified=qualify(rows),
        freeze_SHA256=shared.sha(args.output / "freeze.json"),
        new_default=False,
        old_numeric_and_resource_failures_preserved=True,
        all_four_RX_gates_passed=False,
        actual_synthesis=False,
        complete_arithmetic_cycles=None,
        full16MSps=False,
    )
    write(args.output / "model.json", result)
    print(
        json.dumps(
            dict(
                component_qualified=result["component_qualified"],
                old_full=sum(len(r["old"]["complete"]) for r in rows),
                new_full=sum(len(r["new"]["complete"]) for r in rows),
                all_four_RX_gates_passed=False,
            )
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
