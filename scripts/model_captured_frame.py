"""Live finite capture→real strict frame consumer;remaining arithmetic is off-clock."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scripts import model_coarse_ingress as shared
from tests.captured_frame_reference import delivered, packet, receive
from tests.maximum_reference import channel

from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.fine_selection import FineSelection

POLICY = dict(
    clock_MHz=200,
    history=4096,
    owned=3,
    coherent_threshold=0.45,
    formats=["A2", "B1"],
    MCS=[8, 12],
    symbol_MHz=4,
    SPS=4,
    preamble_symbols={"2": 40, "3": 48, "4": 64},
    new_control_cap_bits=4096,
    new_selected_cache_BRAM36_cap=1,
    private_full_tail_banks=0,
    float_consumer_persistent_NumPy_plus_cache_cap_bytes=524288,
    raw_consumer_history=512,
    minimum_complete_positive=1,
    negative_samples_each=4096,
    negative_families=["noise", "tone", "wrong-sync", "burst"],
    input_and_same_backend_sampling_tolerance="exact",
    new_default=False,
    fixed_and_full16MSps=False,
    unknown_confirmation_head_carrier_decoder_cycles=None,
    retirement="reject selected start before actual consumed prefix,regardless of CRC",
    previous_three_tail_16KiB_and_numeric_failures_retained=True,
    CRC_reference=(
        "existing previously verified encoder;not a new independent CRC conformance claim"
    ),
)


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def safe(value):
    if isinstance(value, bytes):
        return dict(hex=value.hex())
    if isinstance(value, dict):
        return {k: safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def payload(size, ident, held):
    if ident % 2:
        return bytes(
            (0xC3, 0x58, 0x2D, 0x96, 0x0F, 0x71)[(j // 5 + j) % 6] ^ ((j // 37 + held * 29) & 255)
            for j in range(size)
        )
    return bytes(((j * 83 + 0x65) ^ (j >> 2) ^ (held * 0xA7)) & 255 for j in range(size))


def make_inputs(out, split):
    held = split == "validation"
    specs = (
        [
            (4, 12, 16, 43, "clean"),
            (2, 8, 8, 83, "tone"),
            (3, 12, 0, 127, "wrong-sync"),
            (4, 8, 4, 11, "step"),
            (2, 12, 0, 197, "burst"),
            (3, 8, 16, 37, "noise"),
        ]
        if held
        else [
            (2, 8, 0, 17, "clean"),
            (3, 12, 4, 53, "tone"),
            (4, 8, 8, 31, "clean"),
            (2, 12, 16, 113, "noise"),
            (3, 8, 0, 67, "step"),
            (4, 12, 4, 149, "burst"),
        ]
    )
    rows = []
    for j in range(10):
        negative = j >= 6
        ft, mcs, pilots, size, family = specs[j % 6]
        if negative:
            ft, mcs, pilots, size = 2 + (j % 3), 8 if j % 2 else 12, (0, 8)[j % 2], 59 + j
            family = POLICY["negative_families"][j - 6]
        cfg = shared.shared.configuration(ft, mcs, pilots, size)
        cfg.pid = (2 * j + 3 * held) % 6
        p = replace(
            PSKReceiveProfile.ordinary(cfg),
            allowed_mcs=(mcs,),
            head_backend="word-dense" if ft == 2 else "word-depth-parity",
        )
        data = payload(size, j, held)
        params = dict(
            offset=((0.259, 0.607, 0.331, 0.791) if held else (0.137, 0.463, 0.719, 0.887))[j % 4],
            ppm=((100, -25, 50, -75) if held else (-50, 75, -100, 25))[j % 4],
            cfo=((73000, -31000, 113000, -57000) if held else (-47000, 53000, -89000, 19000))[
                j % 4
            ],
            prefix=23 + j * 47 + held * 31,
            amplitude=(0.11, 0.073, 0.193, 0.139)[(j + held) % 4],
            family=family,
            seed=1108021 + j * 137 + held * 1103,
            snr_db=11.25 if family == "noise" else None,
        )
        rng = np.random.default_rng(params["seed"] + 73)
        truth, step = None, None
        if negative and family != "wrong-sync":
            n = POLICY["negative_samples_each"]
            iq = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.003
            if family == "tone":
                iq += 0.19 * np.exp(2j * np.pi * (271000 + held * 47000) * np.arange(n) / 16e6)
            elif family == "burst":
                iq[701:708] += 3 - 5j
            quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.003
            metadata = dict(no_legal_packet=True)
        else:
            wire_profile = p
            if family == "wrong-sync":
                wire_profile = replace(p, config=replace(cfg, pid=(cfg.pid + 1) % 6))
            iq, quiet, truth, step, metadata = channel(
                packet(wire_profile, data),
                **(params | dict(family="clean" if family == "wrong-sync" else family)),
            )
            if family == "wrong-sync" and not negative:
                target, _, truth, step, _ = channel(
                    packet(p, data), **(params | dict(family="clean", snr_db=None))
                )
                iq = target + 0.55 * iq[: len(target)]
            if negative:
                iq = np.pad(iq, (0, max(0, 4096 - len(iq))))[:4096]
            else:
                extra = (1111 + j * 113) if held else (777 + j * 127)
                sigma = metadata["noise_std_per_component"] or 0.001
                tail = (rng.normal(size=extra) + 1j * rng.normal(size=extra)) * sigma
                if family == "tone":
                    tail += (
                        params["amplitude"]
                        * 0.4
                        * np.exp(2j * np.pi * 213000 * np.arange(len(iq), len(iq) + extra) / 16e6)
                    )
                iq = np.r_[iq, tail]
        file = out / f"input{j}.npz"
        np.savez_compressed(file, iq=iq, quiet=quiet)
        config = vars(cfg).copy()
        config.pop("_mcs")
        rows.append(
            dict(
                id=j,
                file=file.name,
                SHA256=shared.sha(file),
                negative=negative,
                config=config,
                head_backend=p.head_backend,
                params=params,
                preamble_symbols=POLICY["preamble_symbols"][str(ft)],
                metadata=metadata,
                true_first_preamble=truth,
                true_step=step,
                expected_payload=None if negative else data.hex(),
                expected_payload_SHA256=None if negative else hashlib.sha256(data).hexdigest(),
            )
        )
    write(out / "inputs.json", rows)
    return rows


def choice(raw):
    return FineSelection(
        **(raw | dict(words=tuple(map(tuple, raw["words"])), correlation=tuple(raw["correlation"])))
    )


def reference(event, reads, p):
    """Same existing strict consumer on exactly the actual delivered port words."""
    if event["frame"]["status"] in (
        "source_already_consumed",
        "tail_history_expired",
        "source_epoch_aborted",
    ):
        return None
    s = choice(event["upstream_selection"])
    rx = SelectedPSKReceiver(
        p,
        **s.reference_selection(),
        coherent_threshold=0.45,
        sample_limit=2**20,
        timing_backend="python",
        history_capacity=512,
    )
    data = [physical_filter_word(r["word"]) for r in reads if r["id"] == s.ident]
    actual = rx.feed_filtered(np.asarray(data, dtype=complex))
    if not actual:
        actual = rx.finish()
    assert len(actual) == 1 and actual[0]["frame"] == event["frame"]
    for k in ("symbols", "symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        if k in actual[0]:
            assert actual[0][k].dtype == event[k].dtype
            assert actual[0][k].tobytes() == event[k].tobytes(), k
    return True


def evaluate(out, row):
    assert shared.sha(out / row["file"]) == row["SHA256"]
    cfg = TxConfig(**row["config"])
    p = replace(
        PSKReceiveProfile.ordinary(cfg),
        allowed_mcs=(cfg.mcs_index,),
        head_backend=row["head_backend"],
    )
    with np.load(out / row["file"]) as z:
        iq, quiet = z["iq"], z["quiet"]
    decoded = []
    result, frames, reads, owner, source, outcomes = receive(
        p, iq, quiet, decoder_observer=decoded.append
    )
    # Reference and chunks never choose a different fine maximum or CRC answer.
    references = [reference(event, reads, p) for event, _ in frames]
    random_groups = tuple(np.random.default_rng(180271 + row["id"]).integers(1, 258, 19))
    for groups in [(1, 31, 127), random_groups]:
        other = receive(p, iq, quiet, groups=groups)
        assert other[0] == result and other[2] == reads
        assert delivered(other[1]) == delivered(frames)
        assert len(other[1]) == len(frames)
        for (a, _), (b, _) in zip(frames, other[1], strict=True):
            assert a["frame"] == b["frame"] and a["upstream_selection"] == b["upstream_selection"]
            for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm"):
                if key in a:
                    assert a[key].tobytes() == b[key].tobytes()
    expected = None if row["negative"] else bytes.fromhex(row["expected_payload"])
    bits_ref = None if expected is None else np.unpackbits(np.frombuffer(expected, np.uint8))
    errors = []
    for d in decoded:
        file = f"decoded-{row['id']}-{d['id']}.npz"
        np.savez_compressed(out / file, bits=d["bits"])
        errors.append(
            dict(
                id=d["id"],
                file=file,
                SHA256=shared.sha(out / file),
                CRC=bool(d["CRC"]),
                MCS=d["MCS"],
                bits=len(d["bits"]),
                errors=None
                if bits_ref is None or len(bits_ref) != len(d["bits"])
                else int(np.count_nonzero(d["bits"] != bits_ref)),
            )
        )
    events = []
    for j, (e, mac) in enumerate(frames):
        arrays = {k: v for k, v in e.items() if isinstance(v, np.ndarray)}
        trace = f"frame-{row['id']}-{j}.npz"
        np.savez_compressed(out / trace, **arrays)
        centers = e.get("symbol_centers_filtered")
        timing = None
        if centers is not None and len(centers) and row["true_step"] is not None:
            start = row["true_first_preamble"] + row["preamble_symbols"] * row["true_step"] + 40
            err = (
                e["source_window_origin"]
                + centers
                - (start + np.arange(len(centers)) * row["true_step"])
            )
            timing = dict(
                symbols=len(err),
                max_abs_samples=float(np.max(abs(err))),
                RMS_samples=float(np.sqrt(np.mean(err**2))),
                head_mid_tail_RMS=[
                    float(np.sqrt(np.mean(v**2))) for v in np.array_split(err, 3) if len(v)
                ],
                diagnostic_truth_only=True,
            )
        s = choice(e["upstream_selection"])
        cfo_error = None
        if row["true_step"] is not None:
            omega = s.reference_selection()["delta"]
            cfo_error = omega * 16e6 / (2 * np.pi * row["true_step"]) - row["params"]["cfo"]
        events.append(
            safe({k: v for k, v in e.items() if not isinstance(v, np.ndarray)})
            | dict(
                trace=trace,
                trace_SHA256=shared.sha(out / trace),
                MAC=[v.mac_payload.hex() for v in mac.packets],
                timing=timing,
                coarse_differential_CFO_error_Hz=cfo_error,
                full_carrier_error_convergence_slips=None,
            )
        )
    packets = delivered(frames)
    cache = result["selected_cache_cost"]
    control = (
        cache["scalar_register_bits"]
        + cache["ordered_request_state_bits"]
        + owner.estimate()["register_bits_upper"]
        + result["frame_transport_control_bits"]
    )
    storage = owner.peak_reference_bytes + cache["selected_cache_bits"] // 8
    checked = [e["frame"] for e, _ in frames if e["frame"].get("header_crc") is True]
    lengths = sum(
        e.get("payload_length_bytes") == len(expected) for e in checked if expected is not None
    )
    mcs_matches = sum(e.get("mcs") == cfg.mcs_index for e in checked)
    data_CRCs = sum(e["frame"].get("data_crc") is True for e, _ in frames)
    full = (
        expected is not None
        and packets == [expected]
        and lengths == 1
        and mcs_matches == 1
        and data_CRCs == 1
    )
    summary = dict(
        id=row["id"],
        input_SHA256=row["SHA256"],
        samples=len(iq),
        negative=row["negative"],
        source_lossless=source.flow.lossless,
        requests=result["requests"],
        fine_complete=result["completed"],
        queue_drops=result["queue_drops"],
        source_epoch_drops=result["source_epoch_drops"],
        frame_resource_drops=result["frame_resource_drops"],
        coherent_accepted=owner.stats["coherent_accepted"],
        head_CRC_and_length_valid=owner.stats["header_valid"],
        actual_packets=len(packets),
        full_correct=int(full),
        expected_length_matches=None if expected is None else lengths,
        expected_MCS_matches=mcs_matches,
        data_CRC_passes=data_CRCs,
        same_read_reference_checked=sum(v is True for v in references),
        resource_or_retired_reference_not_compared=sum(v is None for v in references),
        false_MAC=sum(expected is None or a != expected for a in packets),
        duplicate_MAC=max(0, len(packets) - 1) if expected is not None else 0,
        statuses=[e["frame"]["status"] for e, _ in frames],
        decoded=errors,
        additional_control_bits=control,
        new_selected_cache_BRAM36=cache["selected_cache_BRAM36"],
        reference_NumPy_plus_cache_bytes=storage,
        all_original_words_and_chunks_exact=True,
        old_strict_consumer_on_same_actual_reads_exact=True,
        no_private_full_tail_banks=result["no_private_full_tail_banks"],
        max_source_primitive_operations=result["peak_source_operations_per_primitive"],
        complete_cycle_evidence_valid=False,
        complete_latency_cycles=None,
    )
    write(
        out / f"stream{row['id']}.json",
        safe(
            dict(
                summary=summary,
                result=result,
                events=events,
                request_outcomes=[
                    {k: v for k, v in e.items() if k not in ("strict_frame", "MAC_packets")}
                    for e in outcomes
                ],
                reads=reads,
            )
        ),
    )
    return summary


def qualify(rows):
    if (
        len(rows) != 10
        or {r["id"] for r in rows} != set(range(10))
        or len({r["input_SHA256"] for r in rows}) != 10
    ):
        return False
    for r in rows:
        if (
            type(r["negative"]) is not bool
            or r["negative"] != (r["id"] >= 6)
            or r["source_lossless"] is not True
            or r["all_original_words_and_chunks_exact"] is not True
            or r["old_strict_consumer_on_same_actual_reads_exact"] is not True
            or r["no_private_full_tail_banks"] is not True
            or r["complete_cycle_evidence_valid"] is not False
            or r["complete_latency_cycles"] is not None
        ):
            return False
        counts = (
            "samples",
            "requests",
            "fine_complete",
            "queue_drops",
            "source_epoch_drops",
            "frame_resource_drops",
            "coherent_accepted",
            "head_CRC_and_length_valid",
            "actual_packets",
            "full_correct",
            "false_MAC",
            "duplicate_MAC",
            "additional_control_bits",
            "new_selected_cache_BRAM36",
            "reference_NumPy_plus_cache_bytes",
            "max_source_primitive_operations",
        )
        if any(type(r[k]) is not int or r[k] < 0 for k in counts):
            return False
        if (
            not r["samples"]
            or r["false_MAC"]
            or r["duplicate_MAC"]
            or r["full_correct"] > 1
            or r["additional_control_bits"] > POLICY["new_control_cap_bits"]
            or r["new_selected_cache_BRAM36"] > POLICY["new_selected_cache_BRAM36_cap"]
            or r["reference_NumPy_plus_cache_bytes"]
            > POLICY["float_consumer_persistent_NumPy_plus_cache_cap_bytes"]
            or r["max_source_primitive_operations"] > 2
            or (r["negative"] and r["coherent_accepted"])
            or r["requests"] != r["fine_complete"] + r["queue_drops"] + r["source_epoch_drops"]
            or r["actual_packets"] > r["data_CRC_passes"]
            or r["data_CRC_passes"] > r["head_CRC_and_length_valid"]
            or r["head_CRC_and_length_valid"] > r["coherent_accepted"]
            or r["coherent_accepted"] > r["fine_complete"]
            or (
                r["full_correct"]
                and (
                    r["negative"]
                    or r["actual_packets"] != 1
                    or r["expected_length_matches"] != 1
                    or r["expected_MCS_matches"] != 1
                )
            )
        ):
            return False
    return (
        sum(r["full_correct"] for r in rows if not r["negative"])
        >= POLICY["minimum_complete_positive"]
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=shared.sources())
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen eligible development")
        old = json.loads(args.freeze.read_text())
        model = json.loads((args.freeze.parent / "model.json").read_text())
        end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        if (
            old != frozen
            or end != frozen["sources"]
            or not model["component_qualified"]
            or model["freeze_SHA256"] != shared.sha(args.freeze)
        ):
            raise ValueError("Need unchanged source,policy and eligible development")
    args.output.mkdir(exist_ok=False)
    write(args.output / "freeze.json", frozen)
    rows = [evaluate(args.output, row) for row in make_inputs(args.output, args.split)]
    assert shared.sources() == frozen["sources"]
    write(args.output / "source-bindings-at-end.json", shared.sources())
    result = dict(
        split=args.split,
        rows=rows,
        component_qualified=qualify(rows),
        freeze_SHA256=shared.sha(args.output / "freeze.json"),
        all_four_RX_gates_passed=False,
        actual_synthesis=False,
        new_default=False,
        old_numeric_and_resource_failures_preserved=True,
        full16MSps_or_full_fixed_point=False,
        actual_functional_unknown_start_path=True,
        remaining_arithmetic_cycles=None,
    )
    write(args.output / "model.json", result)
    print(
        json.dumps(
            dict(
                component_qualified=result["component_qualified"],
                streams=10,
                full_correct=sum(r["full_correct"] for r in rows),
                all_four_RX_gates_passed=False,
            )
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
