"""Bounded selected raw history and real-word ports;exit1 preserves whole-RX gaps."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_selected_frame import (
    TRACE_KEYS,
    execute,
    fresh_cases,
    metrics,
    normalized,
)

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.sim.capture_fine import unpack_filter_word
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

POLICY = Path("tests/data/sliding_history_policy.json")


def word(index, salt):
    return ((index * 0x31297 + salt * 0x17093) ^ (index << 27)) & (2**46 - 1)


def port_case(out, name, *, n, capacity, clock, read_ii, block, salt):
    """Pauseable replay source,not an ADC;one external held source word is counted."""
    p = ReceiveHistoryPort(capacity)
    accepted = returned = 0
    due = next_read = pause_offset = 0
    repeats = 1 + salt % 3
    output = []
    held_cycles = 0
    for cycle in range(n * max(32, read_ii * repeats) + block[1] + 2000):
        blocked = block[0] <= cycle < block[1]
        taken = p.take(cycle) if not blocked else None
        if taken is not None:
            index, value = taken
            assert index == returned // repeats and value == word(index, salt)
            returned += 1
            output.append(value)
            if returned % repeats == 0:
                p.retire_before(returned // repeats, cycle)
        if accepted < n and cycle >= due:
            if p.write(accepted, word(accepted, salt), cycle):
                # Exact rational16MS/s when unstalled;one held source word when blocked.
                pause_offset += max(0, cycle - due)
                accepted += 1
                due = pause_offset + (accepted * clock + 15999999) // 16000000
            else:
                held_cycles += 1
        if (
            not blocked
            and cycle >= next_read
            and returned // repeats < p.end
            and p.request(returned // repeats, cycle)
        ):
            next_read = cycle + read_ii
        if returned == n * repeats:
            break
    assert returned == n * repeats and accepted == n
    assert p.live == 0 and p.stats["peak_live"] <= capacity
    np.savez_compressed(out / (name + "-words.npz"), words=np.asarray(output, np.uint64))
    return dict(
        id=name,
        words=n,
        clock=clock,
        capacity=capacity,
        read_ii=read_ii,
        repeated_reads_each_word=repeats,
        source_rate_hz_without_stalls=16000000,
        blocked=block,
        elapsed_cycles=cycle + 1,
        producer_held_cycles=held_cycles,
        stats=p.stats,
        cost=p.estimate(),
        external_producer_hold_bits=64 + 46 + 1,
        source="pauseable word replay,not nonstop ADC",
        full_receiver16msps=False,
        arithmetic_latency_unknown=True,
    )


def startup_live_failure(clock):
    """Optimistic RAM-only necessary test;unknown arithmetic earns no pass credit.

    Current initializer needs every sync sample before tracking/retirement.
    A direct-live consumer bank is distinct from borrowing an upstream snapshot.
    """
    p = ReceiveHistoryPort(512)

    def period(i):
        return (i * clock + 15999999) // 16000000

    # start=1+1/8,Nsync126:505 arrivals needed by the existing waiting rule.
    for index in range(505):
        assert p.write(index, word(index, 127), period(index))
    start = period(504)
    offered = 505
    returned = 0
    for cycle in range(start, start + 2000):
        value = p.take(cycle)
        if value is not None:
            assert value == (returned, word(returned, 127))
            returned += 1
        if cycle >= period(offered):
            if not p.write(offered, word(offered, 127), cycle):
                assert returned < 504 and offered == 512
                return dict(
                    clock=clock,
                    status="direct_live_startup_overflow",
                    first_lost_absolute_index=offered,
                    cycles_since_initial_ready=cycle - start,
                    initial_scan_returns_before_loss=returned,
                    initial_scan_required=504,
                    peak_live=p.stats["peak_live"],
                    stats=p.stats,
                    arithmetic_cycles=None,
                    optimistic_memory_only_failure=True,
                    scope=(
                        "current no-early-retire initializer,direct live bank;"
                        "not upstream borrowed window"
                    ),
                    remedy_unverified=[
                        "borrow existing owned window",
                        "cache/reuse initial sync samples",
                    ],
                    whole_receiver16msps=False,
                )
            offered += 1
        if returned < 504:
            p.request(returned, cycle)
    raise AssertionError("Expected direct-live capacity failure was not observed")


def ring_case(
    out, name, profile, values, selection, policy, expected, reference, truth_start=None, step=None
):
    result = None
    info = None
    for pattern in ("whole", "fixed", "random"):
        event, current = execute(
            out,
            name + "-" + pattern,
            profile,
            values,
            {**selection, "history_capacity": policy["history_capacity"]},
            capacity=policy["input_horizon"],
            chunking=pattern,
        )
        if result is None:
            result, info = event, current
        else:
            assert normalized(event["frame"]) == normalized(result["frame"])
            assert (
                event["delivered_after_filtered_samples"]
                == result["delivered_after_filtered_samples"]
            )
            for key in TRACE_KEYS:
                if key in result:
                    np.testing.assert_array_equal(event[key], result[key])
        assert normalized(event["frame"]) == reference["frame"]
        assert (
            event["delivered_after_filtered_samples"]
            == reference["delivered_after_filtered_samples"]
        )
        with np.load(reference["trajectory"]) as ar:
            for key in TRACE_KEYS:
                if key in event:
                    np.testing.assert_array_equal(event[key], ar[key])
        h = current["memory"]["raw_history"]
        assert h["peak_samples"] <= policy["startup_existing_wait_peak_max"]
        assert current["memory"]["retained_filtered_bytes"] == 2 * 512 * 16
    centers = result.get("symbol_centers_filtered", np.empty(0))
    n = len(profile.sync_symbols())
    if len(centers) > n:
        current = centers[n:]
        previous = centers[n - 1 : -1]
        low = np.floor((previous + current) / 2).astype(np.int64) - 1
        high = np.floor(current).astype(np.int64) + 2
        needed = np.ceil(current + 3).astype(np.int64) - low
        steady_peak = int(np.max(needed))
        assert steady_peak <= policy["steady_history_max"]
        stencil = dict(
            steady_history_peak_from_actual_centers=steady_peak,
            current_and_midpoint_unique_read_min=int(np.min(high - low + 1)),
            current_and_midpoint_unique_read_max=int(np.max(high - low + 1)),
            logical_current_and_midpoint_reads=8 * (len(centers) - n),
            logical_sync_initialization_and_hold_reads=8 * n,
            reference_dependency_trace_only=True,
            actual_compute_calendar_verified=False,
        )
    else:
        stencil = dict(
            steady_history_peak_from_actual_centers=0,
            reference_dependency_trace_only=True,
            actual_compute_calendar_verified=False,
        )
    return dict(
        interpolation_dependencies=stencil,
        id=name,
        metrics=metrics(
            result, profile, expected=expected, truth_start=truth_start, truth_step=step
        ),
        info=info,
        same_full_reference_and_partitions=True,
        upstream_retention_or_service_verified=False,
    )


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = json.loads(POLICY.read_text())
    sources = [Path(__file__), POLICY, Path("uv.lock"), *Path("src").rglob("*.py")]
    source = {str(p): sha(p) for p in sources}
    write(
        out / "protocol.json",
        dict(
            stage="SYS-C2E2A",
            source=source,
            policy=policy,
            raw_history="512 logical samples;host contiguous adapter duplicates counted",
            port="46bit ring with one read/write per cycle,registered read1 and held reply",
            cycles=dict(interpolation=None, initial_carrier_fit=None, head=None, decoder=None),
            no_zero_cycle_credit=True,
            whole_receiver16msps=False,
            default_RX_changed=False,
        ),
    )
    nodes = []
    node_specs = [
        (257, 8, 100000000, 1, (30, 79)),
        (619, 16, 160000000, 4, (100, 900)),
        (1027, 512, 200000000, 16, (317, 4317)),
        (101, 8, 200000000, 3, (0, 0)),
    ]
    val_specs = [
        (397, 16, 200000000, 1, (127, 309)),
        (733, 8, 100000000, 16, (71, 701)),
        (2051, 512, 160000000, 4, (211, 6211)),
        (513, 32, 200000000, 2, (0, 0)),
    ]
    rows = []
    for split, specs in (("development", node_specs), ("validation", val_specs)):
        for k, (n, capacity, clock, read_ii, block) in enumerate(specs):
            nodes.append(
                port_case(
                    out,
                    split + "-port-" + str(k),
                    n=n,
                    capacity=capacity,
                    clock=clock,
                    read_ii=read_ii,
                    block=block,
                    salt=317 + k if split == "development" else 911 + k,
                )
            )
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
            if found is None:
                rows.append(dict(id=name, input=meta, upstream_no_candidate=True))
                continue
            offset = found["offset"]
            selection = dict(
                start_tick=round(8 * (found["phase"] + 4 * offset)),
                delta=found["delta"],
                differential_score=found["score"],
                coherent_threshold=0.45,
                selected_sync=found["phase_symbols"][offset : offset + len(profile.sync_symbols())],
            )
            reference, _ = execute(
                out,
                name + "-reference",
                profile,
                values,
                selection,
                capacity=len(values),
                chunking="whole",
            )
            ref = dict(
                frame=normalized(reference["frame"]),
                delivered_after_filtered_samples=reference["delivered_after_filtered_samples"],
                trajectory=out / (name + "-reference-trajectory.npz"),
            )
            row = ring_case(
                out,
                name,
                profile,
                values,
                selection,
                policy,
                payload,
                ref,
                meta.get("truth_start_filtered"),
                meta.get("truth_step"),
            )
            rows.append(row | dict(input=meta, input_sha256=sha(out / (name + "-input.npz"))))
        write(out / (split + ".json"), [r for r in rows if r["input"]["split"] == split])
        if split == "development":
            write(
                out / "postdevelopment-freeze.json",
                dict(
                    source=source,
                    policy=policy,
                    new_validation_not_generated=True,
                    initial505bound_preserves_original_wait=True,
                    rule=(
                        "same decisions and trajectories;bounded history;"
                        "exact words;not hardware16MS"
                    ),
                ),
            )
    # Read original selected answers;never rerun fine search or choose by the following CRC.
    for folder in (args.selected_root, args.capture_root):
        index = json.loads((folder / "manifest.json").read_text())
        assert all(sha(folder / k) == v for k, v in index.items())
    records = json.loads((args.selected_root / "regressions.json").read_text())
    metas = {r["id"]: r["input"] for r in json.loads((args.capture_root / "rows.json").read_text())}
    regressions = []
    for record in records:
        if record.get("selection_not_available"):
            regressions.append(record)
            continue
        name = record["id"]
        meta = metas[record["stream"]]
        assert meta["input_sha256"] == record["original_archive_sha256"]
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        with np.load(args.capture_root / (record["stream"] + "-filtered.npz")) as ar:
            packed = ar["words"][record["origin"] :, record["request"]["path"]]
        triples = [unpack_filter_word(w) for w in packed]
        values = np.array([complex(math.ldexp(r, e), math.ldexp(i, e)) for r, i, e in triples])
        saved = json.loads((args.selected_root / (name + "-full.json")).read_text())
        with np.load(args.selected_root / (name + "-full-trajectory.npz")) as ar:
            selection = saved["selection"] | dict(selected_sync=ar["selected_sync"].copy())
        saved["trajectory"] = args.selected_root / (name + "-full-trajectory.npz")
        with np.load(args.capture_root / (record["stream"] + "-input.npz")) as ar:
            expected = ar["payload"].tobytes() if meta["positive"] else None
        row = ring_case(
            out,
            name,
            profile,
            values,
            selection,
            policy,
            expected,
            saved,
            None if expected is None else meta["sync_start"] + 40 - record["origin"],
            meta.get("step"),
        )
        regressions.append(
            row
            | dict(
                stream=record["stream"],
                source_window_sha256=record["window_sha256"],
                original_archive_sha256=record["original_archive_sha256"],
                previous_prefix_metrics=record["prefix"],
                previous_full_metrics=record["full"],
            )
        )
    write(out / "regressions.json", regressions)
    write(out / "ports.json", nodes)
    startup = [startup_live_failure(clock) for clock in (100000000, 160000000, 200000000)]
    write(out / "startup-live-failures.json", startup)
    evaluated = [r for r in regressions if "metrics" in r]
    summary = dict(
        fixed_requests=len(regressions),
        scored=len(evaluated),
        heads=sum(r["metrics"]["header_crc"] is True for r in evaluated),
        full=sum(r["metrics"]["payload_exact"] for r in evaluated),
        conditional_prefix_recoveries=sum(
            r["metrics"]["payload_exact"] and not r["previous_prefix_metrics"]["payload_exact"]
            for r in evaluated
        ),
        full_reference_losses=sum(
            r["previous_full_metrics"]["payload_exact"] and not r["metrics"]["payload_exact"]
            for r in evaluated
        ),
        history_resource_failures=sum(
            r["metrics"]["status"] in ("retention_limit", "history_overflow") for r in evaluated
        ),
        maximum_history_words=max(
            r["info"]["memory"]["raw_history"]["peak_samples"] for r in evaluated
        ),
        host_history_bytes=2 * 512 * 16,
        word_port_cost=ReceiveHistoryPort().estimate(),
        partition_and_reference_exact=True,
        direct_live_startup_resource_failures=len(startup),
        actual_pipeline_recoveries=None,
        actual_pipeline_deadlines_verified=False,
        whole_receiver16msps=False,
        default_RX_changed=False,
        fresh=[
            dict(
                split=s,
                positives=sum(
                    not r["input"].get("negative") for r in rows if r["input"]["split"] == s
                ),
                full=sum(
                    r.get("metrics", {}).get("payload_exact", False)
                    for r in rows
                    if r["input"]["split"] == s
                ),
                false_deliveries=sum(
                    r.get("metrics", {}).get("false_delivery", False)
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
                r for r in regressions if "metrics" not in r or not r["metrics"]["payload_exact"]
            ],
            fresh=[
                r
                for r in rows
                if not r["input"].get("negative") and not r.get("metrics", {}).get("payload_exact")
            ],
            negative_controls=[r for r in rows if r["input"].get("negative")],
            interpretation="conditional availability only;not actual acquisition/queue recovery",
        ),
    )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--selected-root", type=Path, required=True)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(p.parse_args()))
