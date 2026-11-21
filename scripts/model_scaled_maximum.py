"""Exact incremental scaled-bank maxima;retain open whole-RX/resource gates."""

import argparse
import hashlib
import json
from collections import Counter, deque
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from tests.maximum_reference import channel, packet

from nearlink_sdr.phy.acquisition_maximum import (
    ExponentWindow,
    RollingMaximumRankedBank,
    RollingMaximumScaledBank,
)
from nearlink_sdr.phy.coarse_rank import RankedCoarseBank
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import ToneNotchConfig
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.maximum_flow import ExponentMaximumService

POLICY = dict(
    version=1,
    terms=[31, 61, 125],
    SPS=4,
    mantissa_bits=10,
    exponents=[-96, 64],
    empty_max=-96,
    counter_stores=["register2r2w", "serial1r1w"],
    latency_edges=[4, 7],
    II=[5, 8],
    initialization_edges=[1, 645],
    node_research_cap_bits=8192,
    state_and_outputs_exact=True,
    no_score_or_search_change=True,
    no_default_change=True,
    full_scaled_bank_cycles=None,
    full_RX_cycles=None,
    old_numeric_and_resource_failures_retained=True,
    RF_search="existing bounded-grid one-candidate Q12;unchanged thresholds and receiver loops",
    selection="conditional serial counter-store preference only if exact and bounded;not default",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, complex):
        return dict(real=value.real, imag=value.imag)
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, sort_keys=True) + "\n")


def sources():
    root = Path(__file__).resolve().parents[1]
    if Path(ExponentWindow.advance.__code__.co_filename).resolve() != (
        root / "src/nearlink_sdr/phy/acquisition_maximum.py"
    ):
        raise ValueError("Need the exact local archived node source")
    files = [
        *sorted((root / "src").rglob("*.py")),
        Path(__file__).resolve(),
        root / "tests/maximum_reference.py",
        root / "tests/test_psk_field_layout.py",
        root / "tests/test_rx_timing_audit.py",
        root / "scripts/rx_timing_audit.py",
        root / "uv.lock",
    ]
    return {str(path.relative_to(root)): digest(path) for path in files}


def configuration(frame, mcs, pilot, index):
    return TxConfig(
        frame_type=frame,
        mcs_index=mcs,
        pilot_interval=pilot,
        symbol_rate_mhz=4,
        ctrl_bits_len=28 if frame == 2 else 27,
        head_llid=None if frame == 2 else 0x351701 + index,
        crc_len=24 if index % 2 else 32,
        crc_seed=0x172501 + index * 31,
        whitening_seed=(0x51 + index * 7) & 127,
        pid=index % 6,
    )


def numeric(output, split):
    rows, last_trace = [], None
    for index, frame in enumerate([2, 3, 4, 2, 3, 4, 3, 4]):
        length = (829 + 71 * index) if split == "development" else (1051 + 43 * index)
        rng = np.random.default_rng((381401 if split == "development" else 482513) + index)
        real, imag = [rng.integers(-512, 512, length) for _ in range(2)]
        exponents = rng.integers(-48, 33, length)
        if index == 0:
            real[:] = imag[:] = 0
        elif index == 1:
            real[:] = np.resize([-512, 511, -1, 0, 1, 256], length)
            imag[:] = np.resize([511, -512, 1, -1, 0], length)
            exponents[:] = np.resize([-48, 32, -47, 31], length)
        else:
            first = length // (3 if split == "development" else 5)
            real[first : first + 161] = imag[first : first + 161] = 0
            exponents[:first] = -48 if index % 2 else 32
            exponents[first + 161 :] = 32 if index % 2 else -48
        cfg = configuration(frame, 8, 0, index)
        ref = PSKReceiveProfile.ordinary(cfg).sync_symbols()
        input_artifact = output / f"numeric-{index}-input.npz"
        np.savez_compressed(
            input_artifact, real=real, imag=imag, exponents=exponents, reference=ref
        )
        window = {2: 256, 3: 376, 4: 632}[frame]
        old, new = [
            cls(ref, 4, window, origin=-40) for cls in (RankedCoarseBank, RollingMaximumRankedBank)
        ]
        history = [deque([(0, False)] * new.k) for _ in range(4)]
        trace, old_points, new_points, queries = [], [], [], []
        advance = new.maximum_state.advance

        def observe(tag, outgoing, incoming):
            phase = tag % 4
            assert history[phase].popleft() == outgoing
            history[phase].append(incoming)
            expected = max((e for e, nz in history[phase] if nz), default=-96)
            result = advance(tag, outgoing, incoming)
            assert result == expected
            trace.append((tag, *outgoing, *incoming, result))
            return result

        new.maximum_state.advance = observe
        for first in range(0, length, 17):
            args = [a[first : first + 17] for a in (real, imag, exponents)]
            old_points.extend(old.advance_words(*args, inspect=True))
            new_points.extend(new.advance_words(*args, inspect=True))
            hi = -40 + old.samples - 4 * old.k - 1
            a, b = old.query_best(hi - 15, hi), new.query_best(hi - 15, hi)
            assert a == b and old.stats == new.stats
            queries.append(plain(a))
        assert old_points == new_points
        values = np.array(
            [[p.start, p.numerator, p.energy, p.coefficient_energy] for p in new_points], np.int64
        ).reshape(-1, 4)
        artifact = output / f"numeric-{index}.npz"
        np.savez_compressed(
            artifact,
            real=real,
            imag=imag,
            exponents=exponents,
            reference=ref,
            points=values,
            trace=np.array(trace, np.int64),
            ranked=old._rank_scores,
        )
        rows.append(
            dict(
                id=index,
                frame_type=frame,
                samples=length,
                artifact=artifact.name,
                artifact_SHA256=digest(artifact),
                input_artifact=input_artifact.name,
                input_SHA256=digest(input_artifact),
                exact_common_scores_ranks_queries=True,
                points=len(values),
                queries=queries,
                old_state_bits=old.logical_storage_bits,
                new_state_bits=new.logical_storage_bits,
                actual_maximum_events=new.maximum_state.stats["events"],
            )
        )
        last_trace = trace
    return rows, last_trace


def calendar(trace, clock, store, blocked, *, from_reset=False):
    node = ExponentMaximumService(125, counter_store=store)
    base = 1 if from_reset else node.estimate()["initialization_edges"] + 1
    offered = trace[:512]
    arrivals = {base + (j * clock + 15) // 16: row for j, row in enumerate(offered)}
    end = max(arrivals) + node.latency + 2
    emitted, dropped, failure = [], [], None
    init_writes = reset_words = 0
    for edge in range(1, end + 1):
        initializing = node.mode == "initializing"
        node.step()
        init_writes += node.last_ports[1] if initializing else 0
        reset_words += node.last_reset_words
        stopped = blocked and base + 80 <= edge < base + 240
        if node.ready and not stopped:
            value = node.take()
            assert value[1] == trace[value[0]][5]
            emitted.append(value)
        if edge in arrivals:
            tag, old, old_nz, new, new_nz, _ = arrivals[edge]
            if failure:
                dropped.append(int(tag))
            elif not node.issue(int(tag), (int(old), bool(old_nz)), (int(new), bool(new_nz))):
                failure = "uninitialized_or_busy_without_upstream_FIFO"
                dropped.append(int(tag))
                node.abort("upstream-resource")
    assert node.stats["accepted"] + len(dropped) == len(offered)
    aborted_accepted = node.stats["accepted"] - len(emitted)
    assert aborted_accepted >= 0
    if failure is None:
        assert node.mode == "idle" and aborted_accepted == 0 and len(emitted) == len(offered)
        node.finish()
    return dict(
        clock_MHz=clock,
        counter_store=store,
        output_blocked=blocked,
        armed_after_initialization=not from_reset,
        event_rate=16_000_000,
        offered=len(offered),
        accepted=node.stats["accepted"],
        emitted=emitted,
        resource_rejected_tags=dropped,
        aborted_accepted=aborted_accepted,
        failure=failure,
        component_continuous_pass=failure is None,
        initialization_write_counter_observation=init_writes,
        parallel_reset_words=reset_words,
        cost=node.estimate(),
        difference_history_RRC_score_cycles=None,
        full_receiver16MSps=False,
        EOF_finished=node.mode == "finished",
    )


def payload(length, family):
    if family == 0:
        return bytes(length)
    if family == 1:
        return bytes(np.resize([0x55, 0xAA], length).astype(np.uint8))
    if family == 2:
        return bytes((j * 17 + j // 7) & 255 for j in range(length))
    return bytes((j * j + 31 * j + 0xA7) & 255 for j in range(length))


def rf_inputs(output, split):
    validation = split == "validation"
    specs = [
        (2, 8, 0, 19, "clean"),
        (2, 12, 8, 61, "noise"),
        (3, 12, 4, 129, "tone"),
        (3, 8, 16, 211, "burst"),
        (4, 10, 4, 47, "noise"),
        (4, 12, 0, 255, "step"),
        (2, 8, 0, 43, "bad_head"),
        (3, 12, 8, 71, "bad_data"),
        (4, 12, 4, 67, "truncated"),
    ]
    if validation:
        specs = [
            (2, 12, 4, 7, "tone"),
            (2, 8, 0, 83, "step"),
            (3, 8, 8, 173, "clean"),
            (3, 12, 0, 303, "noise"),
            (4, 12, 16, 31, "burst"),
            (4, 10, 8, 383, "noise"),
            (3, 8, 4, 59, "bad_head"),
            (2, 12, 0, 101, "bad_data"),
            (2, 8, 8, 97, "truncated"),
        ]
    rows = []
    for index, (ft, mcs, pilot, length, family) in enumerate(specs):
        cfg = configuration(ft, mcs, pilot, index + 19 * validation)
        profile = PSKReceiveProfile.ordinary(cfg)
        data = payload(length, (index + validation) % 4)
        wire = packet(profile, data, bad_head=family == "bad_head", bad_data=family == "bad_data")
        channel_parameters = dict(
            offset=(0.137 + index * 0.113 + validation * 0.07) % 1,
            ppm=[75, -25, 50, -100, 100, -50][index % 6],
            cfo=[-42000, 57000, -31000, 83000, 17000, -69000][index % 6],
            prefix=[13, 67, 149, 251][index % 4],
            amplitude=[0.03, 0.2, 0.12, 1.0, 0.4, 0.8][index % 6],
            family=family,
            seed=381701 + index + 37 * validation,
            snr_db=({1: 10, 4: 8} if not validation else {3: 9, 5: 11}).get(index),
        )
        x, quiet, truth, step, channel_metadata = channel(wire, **channel_parameters)
        expected, truths = [data], [(truth, step)]
        if index == (2 if validation else 0):
            second = payload(length - 3, (index + validation + 1) % 4)
            tail_parameters = dict(
                offset=0.671,
                ppm=-50,
                cfo=-27000,
                prefix=0,
                amplitude=0.3,
                family="clean",
                seed=381799,
            )
            tail, _, next_truth, next_step, tail_metadata = channel(
                packet(profile, second),
                **tail_parameters,
            )
            truths.append((len(x) + 96 + next_truth, next_step))
            x = np.r_[x, np.zeros(96), tail]
            expected.append(second)
            channel_metadata["second_packet"] = dict(
                parameters=tail_parameters, metadata=tail_metadata
            )
        if family == "truncated":
            # Independent sender time deliberately removes data,not just pulse tails.
            cut = int(np.floor(truth + (len(wire) - 40 - 13) * step))
            x = x[:cut]
        artifact = output / f"rf-{index}-input.npz"
        np.savez_compressed(
            artifact,
            iq=x,
            quiet=quiet,
            truths=np.array(truths),
            **{f"expected{j}": np.frombuffer(p, np.uint8) for j, p in enumerate(expected)},
        )
        config = asdict(cfg)
        config.pop("_mcs", None)
        rows.append(
            dict(
                id=index,
                config=config,
                family=family,
                legal=True,
                legal_packets=len(expected),
                artifact=artifact.name,
                input_SHA256=digest(artifact),
                channel_parameters=channel_parameters,
                channel_metadata=channel_metadata,
                EOF_cut_from_transmitted_time=(cut if family == "truncated" else None),
            )
        )
    for j, family in enumerate(["silence", "noise-only", "tone-only", "wrong-sync"], 9):
        cfg = configuration(2 if j % 2 else 4, 8, 0, j + validation * 19)
        rng = np.random.default_rng(381901 + j + validation * 31)
        sigma = 0.0017 if validation else 0.001
        quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * sigma
        x = np.zeros(11264 if validation else 8192, complex)
        parameters = dict(
            samples=len(x), quiet_std_per_component=sigma, seed=381901 + j + validation * 31
        )
        if family == "noise-only":
            x = (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * sigma
            parameters["input_noise_std_per_component"] = sigma
        elif family == "tone-only":
            frequency = -183000 if validation else 237000
            amplitude = 0.057 if validation else 0.1
            phase = 0.319 if validation else 0.0
            x = amplitude * np.exp(1j * (2 * np.pi * frequency * np.arange(len(x)) / 16e6 + phase))
            parameters.update(frequency_Hz=frequency, amplitude=amplitude, phase_rad=phase)
        elif family == "wrong-sync":
            wrong = configuration(cfg.frame_type, 8, 0, j + validation * 19)
            wrong.pid = (wrong.pid + (2 if validation else 1)) % 6
            wire = packet(
                PSKReceiveProfile.ordinary(wrong),
                payload(53 if validation else 37, 2 if validation else 3),
            )
            wrong_parameters = dict(
                offset=0.711 if validation else 0.331,
                ppm=-75 if validation else 25,
                cfo=-39000 if validation else 17000,
                prefix=43 if validation else 17,
                amplitude=0.19 if validation else 0.3,
                family="clean",
                seed=381903,
            )
            source, _, _, _, _ = channel(
                wire,
                **wrong_parameters,
            )
            x[: len(source)] = source
            parameters.update(
                wrong_pid=wrong.pid,
                channel=wrong_parameters,
                payload_length=53 if validation else 37,
            )
        artifact = output / f"rf-{j}-input.npz"
        np.savez_compressed(artifact, iq=x, quiet=quiet, truths=np.empty((0, 2)))
        config = asdict(cfg)
        config.pop("_mcs", None)
        rows.append(
            dict(
                id=j,
                config=config,
                family=family,
                legal=False,
                legal_packets=0,
                artifact=artifact.name,
                input_SHA256=digest(artifact),
                negative_parameters=parameters,
            )
        )
    return rows


def receive(row, data, chunks, candidate):
    with ExitStack() as context:
        if candidate:
            for module, cls in [
                ("acquisition_scaled.ScaledDifferentialBank", RollingMaximumScaledBank),
                ("coarse_rank.RankedCoarseBank", RollingMaximumRankedBank),
            ]:
                context.enter_context(patch("nearlink_sdr.phy." + module, cls))
        profile = PSKReceiveProfile.ordinary(TxConfig(**row["config"]))
        rx = MacStreamReceiver(
            profile,
            data["quiet"],
            quiet_batching=False,
            defer_quiet_fir=False,
            adaptive_quiet=False,
            rrc_backend="word-full12-c18",
            correlation_backend="scaled",
            correlation_mantissa_bits=10,
            sync_search_method="bounded-grid",
            normalized_coarse=True,
            tone_notch=ToneNotchConfig(radius=31 / 32),
            timing_backend="python",
            integer_screen_backend="python",
        )
        if candidate:
            assert isinstance(rx.phy.correlation_bank, RollingMaximumRankedBank)
        rng = np.random.default_rng(381977)
        events, packets, pos = [], [], 0
        while pos < len(data["iq"]):
            size = (
                len(data["iq"])
                if chunks == "whole"
                else 257
                if chunks == "fixed"
                else int(rng.integers(1, 258))
            )
            result = rx.feed(data["iq"][pos : pos + size])
            events.extend(result.diagnostics["events"])
            packets.extend(p.mac_payload for p in result.packets)
            pos += size
            assert len(rx.phy._filtered) <= rx.phy.history_limit
            assert len(rx.phy._tone_filtered) <= rx.phy.history_limit
        result = rx.finish()
        events.extend(result.diagnostics["events"])
        packets.extend(p.mac_payload for p in result.packets)
        frames = [event["frame"] for event in events]
        expected = [data[key].tobytes() for key in data.files if key.startswith("expected")]
        remaining = Counter(expected)
        content_remaining = Counter(expected)
        whole = 0
        for packet_data in packets:
            if remaining[packet_data]:
                remaining[packet_data] -= 1
                whole += 1
        content_checks = content_matches = 0
        for frame in frames:
            decoded = frame.get("payload")
            if isinstance(decoded, bytes):
                content_checks += 1
                if content_remaining[decoded]:
                    content_remaining[decoded] -= 1
                    content_matches += 1
        arrays, errors, bits, timing = {}, 0, 0, []
        for j, event in enumerate(events):
            for key in [
                "symbols",
                "symbol_centers_raw",
                "timing_rate_ppm",
                "timing_detector_error",
            ]:
                arrays[f"{j}-{key}"] = event[key]
            if row["legal"] and len(data["truths"]):
                center = event["symbol_centers_raw"]
                t, step = min(data["truths"], key=lambda pair: abs(pair[0] - center[0]))
                timing.append(float(np.max(abs(center - (t + np.arange(len(center)) * step)))))
            decoded = event["frame"].get("payload")
            if (
                isinstance(decoded, bytes)
                and len(expected) == 1
                and len(decoded) == len(expected[0])
            ):
                delta = np.frombuffer(decoded, np.uint8) ^ np.frombuffer(expected[0], np.uint8)
                errors += int(np.unpackbits(delta).sum())
                bits += len(decoded) * 8
        summary = dict(
            frames=frames,
            MAC_payloads=packets,
            captured=len(frames),
            head_CRC=sum(f.get("header_crc") is True for f in frames),
            data_CRC=sum(f.get("data_crc") is True for f in frames),
            length_ok=sum(f.get("length_ok") is True for f in frames),
            MCS=[f.get("mcs") for f in frames],
            complete=whole,
            payload_content_checks=content_checks,
            payload_content_matches=content_matches,
            false_MAC=len(packets) if not row["legal"] else len(packets) - whole,
            BER=(errors / bits if bits else None),
            BER_errors=errors,
            BER_bits=bits,
            timing_error_max=timing,
            timing_truth_diagnostic_only=True,
            capture_stats=rx.phy.capture_stats,
            bank_stats=rx.phy.correlation_bank.stats,
        )
        return plain(summary), arrays


def rf_compare(output, inputs):
    rows = []
    for row in inputs:
        path = output / row["artifact"]
        assert digest(path) == row["input_SHA256"]
        canonical = None
        with np.load(path) as data:
            for chunks in ["whole", "fixed", "random"]:
                old, a = receive(row, data, chunks, False)
                new, b = receive(row, data, chunks, True)
                stem = f"rf-{row['id']}-{chunks}"
                write(output / (stem + "-baseline.json"), old)
                write(output / (stem + "-candidate.json"), new)
                np.savez_compressed(output / (stem + "-baseline-trace.npz"), **a)
                np.savez_compressed(output / (stem + "-candidate-trace.npz"), **b)
                exact = (
                    old == new
                    and a.keys() == b.keys()
                    and all(
                        a[key].dtype == b[key].dtype and a[key].tobytes() == b[key].tobytes()
                        for key in a
                    )
                )
                semantic = {key: value for key, value in new.items() if key != "capture_stats"}
                if canonical is None:
                    canonical = semantic, b
                chunk_exact = (
                    semantic == canonical[0]
                    and b.keys() == canonical[1].keys()
                    and all(
                        b[key].dtype == canonical[1][key].dtype
                        and b[key].tobytes() == canonical[1][key].tobytes()
                        for key in b
                    )
                )
                rows.append(
                    dict(
                        id=row["id"],
                        chunks=chunks,
                        legal=row["legal"],
                        legal_packets=row["legal_packets"],
                        input_SHA256=row["input_SHA256"],
                        comparisons_exact=exact,
                        whole_chunk_semantics_and_trajectories_exact=chunk_exact,
                        captured=new["captured"],
                        head_CRC=new["head_CRC"],
                        data_CRC=new["data_CRC"],
                        length_ok=new["length_ok"],
                        MCS=new["MCS"],
                        complete=new["complete"],
                        baseline_complete=old["complete"],
                        recovered=max(0, new["complete"] - old["complete"]),
                        original_success_losses=max(0, old["complete"] - new["complete"]),
                        payload_content_checks=new["payload_content_checks"],
                        payload_content_matches=new["payload_content_matches"],
                        statuses=[frame["status"] for frame in new["frames"]],
                        false_MAC=new["false_MAC"],
                        BER=new["BER"],
                        BER_errors=new["BER_errors"],
                        BER_bits=new["BER_bits"],
                        timing_error_max=new["timing_error_max"],
                        failure_records=[stem + "-baseline.json", stem + "-candidate.json"],
                    )
                )
    return rows


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=sources())
    if args.split == "validation" and (
        args.freeze is None or json.loads(args.freeze.read_text()) != frozen
    ):
        raise ValueError(
            "Need frozen exact algorithm,parameters,budgets and source before validation"
        )
    args.output.mkdir(exist_ok=False)
    write(args.output / "freeze.json", frozen)
    numeric_rows, trace = numeric(args.output, args.split)
    calendars = [
        calendar(trace, clock, store, block)
        for store in POLICY["counter_stores"]
        for clock in [100, 160, 200]
        for block in [False, True]
    ]
    calendars += [
        calendar(trace, 200, store, False, from_reset=True) for store in POLICY["counter_stores"]
    ]
    inputs = rf_inputs(args.output, args.split)
    write(args.output / "inputs.json", inputs)
    RF = rf_compare(args.output, inputs)
    write(args.output / "RF-rows.json", RF)
    cost = [
        ExponentMaximumService(k, counter_store=s).estimate()
        for k in POLICY["terms"]
        for s in POLICY["counter_stores"]
    ]
    failures = [
        dict(
            kind="resource",
            clock=row["clock_MHz"],
            store=row["counter_store"],
            blocked=row["output_blocked"],
            armed=row["armed_after_initialization"],
            reason=row["failure"],
        )
        for row in calendars
        if row["failure"]
    ]
    failures += [
        dict(kind="RF", **row)
        for row in RF
        if row["chunks"] == "whole" and row["legal"] and row["complete"] < row["legal_packets"]
    ]
    whole_rows = [row for row in RF if row["chunks"] == "whole"]
    negative_samples = sum(
        len(np.load(args.output / row["artifact"])["iq"]) for row in inputs if not row["legal"]
    )
    result = dict(
        split=args.split,
        component_exact=all(
            row["comparisons_exact"] and row["whole_chunk_semantics_and_trajectories_exact"]
            for row in RF
        ),
        numeric_rows=numeric_rows,
        costs=cost,
        calendars=calendars,
        failure_links=failures,
        RF_streams=len(inputs),
        RF_runs=len(RF),
        legal_packets=sum(row["legal_packets"] for row in inputs),
        captured=sum(row["captured"] for row in whole_rows if row["legal"]),
        complete=sum(row["complete"] for row in whole_rows),
        success_losses=sum(row["original_success_losses"] for row in whole_rows),
        numeric_or_frame_mismatches=sum(not row["comparisons_exact"] for row in RF),
        false_MAC=sum(row["false_MAC"] for row in whole_rows if not row["legal"]),
        negative_samples=negative_samples,
        negative_exposure_seconds=negative_samples / 16e6,
        negative_candidates_per_million=(
            sum(row["captured"] for row in whole_rows if not row["legal"]) * 1e6 / negative_samples
        ),
        independent_RF_validation=args.split == "validation",
        capture_gain=0 if all(row["comparisons_exact"] for row in RF) else None,
        new_receiver_default=False,
        inherited_quantization_failures_passed=False,
        preferred_counter_store="serial1r1w;conditional component layout only",
        full_scaled_bank_cycles=None,
        full_RX_cycles=None,
        four_RX_gates_passed=False,
    )
    final_sources = sources()
    write(args.output / "source-bindings-at-end.json", final_sources)
    if final_sources != frozen["sources"]:
        raise ValueError("Source changed during evaluation;result is not frozen evidence")
    write(args.output / "model.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "split",
                    "component_exact",
                    "RF_streams",
                    "RF_runs",
                    "legal_packets",
                    "captured",
                    "complete",
                    "false_MAC",
                    "four_RX_gates_passed",
                ]
            }
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
