"""Exact Q12 serial structures with finite coarse calendars;not full-RX acceptance."""

import argparse
import hashlib
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts import model_banked_scaled as banked
from scripts import model_scaled_maximum as shared
from tests.maximum_reference import channel, packet

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.rank_serial import Q12Serial

POLICY = dict(
    version=1,
    scope="normalization only;functional IQ context and clocked coarse prefixes separate",
    developer_variants=["pipeline", "serial1", "serial2"],
    validation_variants=["pipeline", "serial2"],
    exact_arithmetic="C/E/N/gate/Q12 and normalized query order/ties;no tolerance",
    serial_register_cap_bits=640,
    source_discard_counter_bits=64,
    source_word_rate=16000000,
    symbols_per_second=4000000,
    samples_per_symbol=4,
    clocks_MHz=[100, 160, 200],
    preferred_lanes={"2": 8, "3": 8, "4": 16},
    finite_core_queues=dict(input=16, difference=64, fold=4, CE=4, query=2, output=1),
    query_positions=131,
    query_stride_committed_words=32,
    query_timing="same committed-tag snapshots;not raw input energy trigger timing",
    word_prefix_samples=2560,
    research_DSP_cap=18,
    research_core_bits=131072,
    no_default_promotion=True,
    adoption=(
        "serial2 only if exact and <=640bits,all preferred200MHz legal word loads pass "
        "without new query losses;no retry after held failure;functional RF gains not claimed"
    ),
    functional_RF="old word bank vs banked functional reference only;not clocked IQ to MAC",
    frame_context_rule="all lengths,MCS,payload,CRC checked;no truth/CRC selects sync",
    full_RX_16MSps=False,
    inherited_numeric_failures_passed=False,
    system_budget_approved=False,
    RTL_synthesis=None,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    root = Path(__file__).resolve().parents[1]
    for fn, relative in [
        (Q12Serial.step, "src/nearlink_sdr/sim/rank_serial.py"),
        (banked.timed, "scripts/model_banked_scaled.py"),
        (channel, "tests/maximum_reference.py"),
        (packet, "tests/maximum_reference.py"),
    ]:
        if Path(fn.__code__.co_filename).resolve() != root / relative:
            raise ValueError("Need exact local model and independent reference sources")
    paths = [root / "uv.lock", root / "pyproject.toml", root / ".github/workflows/ci.yml"]
    for name in ["src", "tests", "scripts"]:
        paths.extend((root / name).rglob("*.py"))
    return {str(p.relative_to(root)): sha(p) for p in sorted(set(paths))}


def payload(length, family):
    if family == 0:
        return bytes(((j ^ (j >> 1)) * 19 + 0x6D) & 255 for j in range(length))
    if family == 1:
        return bytes(
            (0xC7, 0x0B, 0xA1, 0x3E, 0xF0, 0x62)[(j // 3) % 6] ^ (j & 31) for j in range(length)
        )
    return bytes(((j // 17) * 53 + (j % 17) * 11) & 255 for j in range(length))


def inputs(out, split):
    held = split == "validation"
    specs = [
        (2, 12, 0, 43, "tone"),
        (3, 8, 4, 137, "noise"),
        (4, 12, 8, 311, "wrong-sync-weak"),
        (2, 8, 16, 521, "burst"),
        (3, 12, 0, 79, "step"),
        (4, 8, 4, 683, "clean"),
        (2, 12, 8, 197, "noise"),
        (3, 8, 0, 347, "burst"),
        (4, 12, 16, 61, "bad_head"),
        (3, 12, 4, 211, "bad_data"),
        (2, 8, 8, 431, "truncated"),
    ]
    if held:
        specs = [
            (2, 8, 4, 101, "burst"),
            (4, 12, 0, 281, "tone"),
            (3, 12, 16, 521, "wrong-sync-weak"),
            (4, 8, 8, 97, "noise"),
            (2, 12, 16, 419, "step"),
            (3, 8, 4, 743, "clean"),
            (4, 12, 4, 167, "burst"),
            (2, 8, 0, 337, "noise"),
            (2, 12, 8, 89, "bad_head"),
            (4, 8, 0, 227, "bad_data"),
            (3, 12, 16, 503, "truncated"),
        ]
    rows = []
    for j, (ft, mcs, pilot, length, family) in enumerate(specs):
        cfg = shared.configuration(ft, mcs, pilot, 119 + j + 67 * held)
        profile = PSKReceiveProfile.ordinary(cfg)
        expected = [payload(length, (j + held) % 3)]
        wire = packet(
            profile, expected[0], bad_head=family == "bad_head", bad_data=family == "bad_data"
        )
        parameters = dict(
            family=family,
            offset=(0.381 + 0.137 * j + 0.221 * held) % 1,
            ppm=[100, -75, 50, -25][j % 4],
            cfo=[-113000, 57000, 83000, -29000][j % 4],
            amplitude=[0.071, 0.23, 0.11, 0.49, 0.087][j % 5],
            prefix=[47, 173, 269, 383][j % 4] + 59 * held,
            snr_db=({1: 12, 6: 9} if not held else {3: 10, 7: 11}).get(j),
            seed=938171 + 79 * j + 1709 * held,
        )
        x, quiet, truth, step, metadata = channel(wire, **parameters)
        truths = [(truth, step)]
        if family == "wrong-sync-weak":
            wrong = shared.configuration(ft, mcs, pilot, 119 + j + 67 * held)
            wrong.pid = (wrong.pid + 2) % 6
            cp = dict(
                family="clean",
                offset=0.617,
                ppm=-50,
                cfo=31000,
                amplitude=0.19,
                prefix=0,
                seed=938207,
            )
            bg, _, _, _, _ = channel(
                packet(PSKReceiveProfile.ordinary(wrong), payload(991, 2)), **cp
            )
            x += bg[: len(x)]
            metadata["wrong_sync_background"] = dict(parameters=cp, PID=wrong.pid, length=991)
        if j == 0:
            second = payload(251 if held else 307, 2)
            cp = dict(
                family="clean",
                offset=0.931,
                ppm=-100,
                cfo=73000,
                amplitude=0.27,
                prefix=0,
                seed=938257,
            )
            tail, _, t, st, md = channel(packet(profile, second), **cp)
            truths.append((len(x) + 48 + t, st))
            x = np.r_[x, np.zeros(48), tail]
            expected.append(second)
            metadata["dense_second"] = dict(parameters=cp, metadata=md)
        cut = None
        if family == "truncated":
            cut = math.floor(truth + (len(wire) - 40 - 67) * step)
            x = x[:cut]
        path = out / f"rf-{j}-input.npz"
        np.savez_compressed(
            path,
            iq=x,
            quiet=quiet,
            truths=np.asarray(truths),
            **{f"expected{n}": np.frombuffer(v, np.uint8) for n, v in enumerate(expected)},
        )
        config = asdict(cfg)
        config.pop("_mcs", None)
        rows.append(
            dict(
                id=j,
                config=config,
                family=family,
                legal=True,
                legal_packets=len(expected),
                expected=[v.hex() for v in expected],
                artifact=path.name,
                input_SHA256=sha(path),
                parameters=parameters,
                truth_diagnostic_only=dict(truths=truths, cut=cut),
                metadata=metadata,
            )
        )
    for j, family in enumerate(["silence", "noise-only", "tone-only", "wrong-sync"], 11):
        cfg = shared.configuration(2 if j % 2 else 4, 8, 0, j + 119 + 67 * held)
        rng = np.random.default_rng(938351 + j + 977 * held)
        quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.0017
        x = np.zeros(20480 if held else 16384, complex)
        parameters = dict(family=family, samples=len(x))
        if family == "noise-only":
            x = (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * 0.0017
        elif family == "tone-only":
            x = 0.119 * np.exp(1j * (2 * np.pi * 419000 * np.arange(len(x)) / 16e6 + 0.713))
            parameters.update(amplitude=0.119, frequency_Hz=419000, phase_rad=0.713)
        elif family == "wrong-sync":
            wrong = shared.configuration(cfg.frame_type, 8, 0, j + 119 + 67 * held)
            wrong.pid = (wrong.pid + 1) % 6
            cp = dict(
                family="clean",
                offset=0.271,
                ppm=100,
                cfo=-97000,
                amplitude=0.31,
                prefix=63,
                seed=938431,
            )
            v, _, _, _, _ = channel(
                packet(PSKReceiveProfile.ordinary(wrong), payload(281, 1)), **cp
            )
            x[: len(v)] = v
            parameters.update(wrong_PID=wrong.pid, channel=cp)
        path = out / f"rf-{j}-input.npz"
        np.savez_compressed(path, iq=x, quiet=quiet, truths=np.empty((0, 2)))
        config = asdict(cfg)
        config.pop("_mcs", None)
        rows.append(
            dict(
                id=j,
                config=config,
                family=family,
                legal=False,
                legal_packets=0,
                expected=[],
                artifact=path.name,
                input_SHA256=sha(path),
                parameters=parameters,
            )
        )
    shared.write(out / "inputs.json", rows)
    return rows


def calendar_cases(out, rows, split):
    variants = POLICY["developer_variants" if split == "development" else "validation_variants"]
    representative = {
        ft: next(r for r in rows if r["config"]["frame_type"] == ft) for ft in [2, 3, 4]
    }
    specs = [
        (ft, clock, POLICY["preferred_lanes"][str(ft)], False, False)
        for ft in [2, 3, 4]
        for clock in [100, 160, 200]
    ]
    specs += [(4, 200, 8, False, False), (4, 200, 16, True, False), (4, 100, 16, False, True)]
    results = []
    for ft, clock, lanes, blocked, armed in specs:
        row = representative[ft]
        for variant in variants:
            label = f"FT{ft}-{clock}-{lanes}-{int(blocked)}-{int(armed)}-{variant}"
            folder = out / "clock" / label
            folder.mkdir(parents=True)
            shutil.copyfile(out / row["artifact"], folder / row["artifact"])
            result = banked.timed(
                folder,
                row,
                clock,
                lanes,
                block_queries=blocked,
                armed=armed,
                normalizer=variant,
                continue_aborted_source=True,
            )
            assert result["source_tags_presented"] == 2560
            result.update(case=label, frame_type=ft)
            results.append(result)
    return results


def qualify(calendars):
    preferred = [
        r
        for r in calendars
        if r["clock_MHz"] == 200
        and not r["blocked"]
        and r["lanes"] == POLICY["preferred_lanes"][str(r["frame_type"])]
    ]
    checks = []
    for ft in [2, 3, 4]:
        a = next(r for r in preferred if r["frame_type"] == ft and r["normalization"] == "pipeline")
        b = next(r for r in preferred if r["frame_type"] == ft and r["normalization"] == "serial2")
        checks.append(
            bool(
                a["word_stream_pass"]
                and b["word_stream_pass"]
                and a["statistics"]["query_dropped"] == b["statistics"]["query_dropped"] == 0
                and a["statistics"]["query_expired"] == b["statistics"]["query_expired"] == 0
                and a["statistics"]["query_completed"] == b["statistics"]["query_completed"]
                and b["cost"]["normalization"]["research_register_cap_pass"]
                and b["source_discard_controller"]["composed_logical_bits"] <= 131072
            )
        )
    return all(checks)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=sources())
    if args.split == "validation":
        message = "Need frozen eligible development with exact source,parameters and budgets"
        if args.freeze is None:
            raise ValueError(message)
        try:
            prior = json.loads(args.freeze.read_text())
            model = json.loads((args.freeze.parent / "model.json").read_text())
            end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(message) from error
        if (
            prior != frozen
            or model.get("development_qualified") is not True
            or model.get("freeze_SHA256") != sha(args.freeze)
            or end != frozen["sources"]
        ):
            raise ValueError(message)
    args.output.mkdir(exist_ok=False)
    shared.write(args.output / "freeze.json", frozen)
    rows = inputs(args.output, args.split)
    # Existing untimed receiver context is explicitly separate from timed words.
    RF = banked.rf(args.output, rows)
    calendars = calendar_cases(args.output, rows, args.split)
    whole = [r for r in RF if r["chunks"] == "whole"]
    losses = sum(r["original_success_losses"] for r in whole)
    false_MAC = sum(r["false_MAC"] for r in whole if not r["legal"])
    eligible = qualify(calendars) and losses == false_MAC == 0
    negatives = sum(len(np.load(args.output / r["artifact"])["iq"]) for r in rows if not r["legal"])
    result = dict(
        split=args.split,
        development_qualified=eligible,
        freeze_SHA256=sha(args.output / "freeze.json"),
        integer_contract_pass=all(r["every_committed_point_exact"] for r in calendars),
        verdict="experimental-qualified" if eligible else "rejected",
        calendars=calendars,
        source_inputs=rows,
        functional_RF_context=dict(
            scope="untimed old bank vs banked reference;not serial clocked IQ to MAC",
            RF_streams=len(rows),
            RF_runs=len(RF),
            captured=sum(r["captured"] for r in whole),
            head_CRC=sum(r["head_CRC"] for r in whole),
            data_CRC=sum(r["data_CRC"] for r in whole),
            full=sum(r["complete"] for r in whole),
            original_success_losses=losses,
            false_MAC=false_MAC,
            negative_samples=negatives,
            negative_seconds=negatives / 16e6,
            normalization_RF_gain_not_measured=True,
        ),
        clock_query_scope=POLICY["query_timing"],
        upstream_RRC_raw_detection_and_rearm_cycles=None,
        fine_packet_decoder_cycles=None,
        inherited_numeric_failures_passed=False,
        new_default=False,
        whole_RX_gates_passed=False,
        actual_synthesis=False,
    )
    assert sources() == frozen["sources"], "Source changed during evaluation"
    shared.write(args.output / "source-bindings-at-end.json", sources())
    shared.write(args.output / "model.json", result)
    print(
        json.dumps(
            dict(
                split=args.split,
                verdict=result["verdict"],
                integer_contract_pass=result["integer_contract_pass"],
                calendars=len(calendars),
                functional_RF_context=result["functional_RF_context"],
                whole_RX_gates_passed=False,
            )
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
