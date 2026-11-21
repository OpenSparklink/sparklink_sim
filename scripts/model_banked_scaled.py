"""Finite banked coarse-word core;retain numeric/resource and whole-RX failures."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scripts import model_scaled_maximum as baseline
from tests.maximum_reference import channel, packet

from nearlink_sdr.phy.acquisition_maximum import RollingMaximumRankedBank
from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat
from nearlink_sdr.phy.banked_words import BankedRankedReference, BankedScaledReference
from nearlink_sdr.phy.rrc_bridge import rrc_to_scaled_words
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.banked_scaled_core import BankedScaledCore


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(length, family):
    if family == 0:
        return bytes((j * 31 + (j >> 3) * 17 + 0xD3) & 255 for j in range(length))
    if family == 1:
        return bytes(np.resize([0, 255, 0x13, 0xE7, 0x55], length).astype(np.uint8))
    return bytes(((j * j * 7) ^ (j << 2) ^ 0x96) & 255 for j in range(length))


def inputs(out, split):
    validation = split == "validation"
    specs = [
        (2, 8, 0, 23, "clean"),
        (2, 12, 4, 91, "noise"),
        (3, 8, 8, 179, "wrong-sync-weak"),
        (3, 12, 0, 293, "burst"),
        (4, 10, 16, 57, "tone"),
        (4, 12, 4, 397, "step"),
        (2, 8, 8, 247, "burst"),
        (4, 8, 0, 511, "noise"),
        (3, 12, 16, 103, "bad_head"),
        (2, 12, 0, 67, "bad_data"),
        (4, 8, 8, 131, "truncated"),
    ]
    if validation:
        specs = [
            (2, 12, 16, 37, "tone"),
            (3, 10, 0, 89, "noise"),
            (4, 8, 4, 157, "wrong-sync-weak"),
            (2, 8, 0, 331, "burst"),
            (3, 12, 16, 419, "step"),
            (4, 12, 8, 487, "clean"),
            (2, 12, 4, 59, "noise"),
            (3, 8, 4, 233, "burst"),
            (4, 8, 0, 211, "bad_head"),
            (3, 12, 8, 143, "bad_data"),
            (2, 8, 16, 177, "truncated"),
        ]
    rows = []
    for j, (ft, mcs, pilot, length, family) in enumerate(specs):
        cfg = baseline.configuration(ft, mcs, pilot, j + 47 + 71 * validation)
        profile = PSKReceiveProfile.ordinary(cfg)
        expected = [data(length, (j + validation) % 3)]
        wire = packet(
            profile, expected[0], bad_head=family == "bad_head", bad_data=family == "bad_data"
        )
        params = dict(
            offset=(0.219 + 0.173 * j + 0.097 * validation) % 1,
            ppm=[-100, 75, -50, 25][j % 4],
            cfo=[91000, -73000, 41000, -125000, 19000][j % 5],
            prefix=[29, 113, 197, 311][j % 4] + 73 * validation,
            amplitude=[0.055, 0.17, 0.13, 0.67, 0.09, 0.31][j % 6],
            family=family,
            seed=541709 + 53 * j + 913 * validation,
            snr_db=({1: 11, 6: 10} if validation else {1: 12, 7: 9}).get(j),
        )
        x, quiet, truth, step, metadata = channel(wire, **params)
        truths = [(truth, step)]
        if family == "wrong-sync-weak":
            wrong = baseline.configuration(ft, mcs, pilot, j + 47 + 71 * validation)
            wrong.pid = (wrong.pid + 3) % 6
            background_params = dict(
                offset=0.789,
                ppm=100,
                cfo=-37000,
                prefix=0,
                amplitude=0.175,
                family="clean",
                seed=541381,
            )
            interference, _, _, _, _ = channel(
                packet(PSKReceiveProfile.ordinary(wrong), data(701, 2)), **background_params
            )
            x += interference[: len(x)]
            metadata["wrong_sync_background"] = dict(
                parameters=background_params, wrong_PID=wrong.pid, length=701
            )
        if j == 0:
            second = data(53 if validation else 41, 2)
            params2 = dict(
                offset=0.839,
                ppm=100,
                cfo=-69000,
                prefix=0,
                amplitude=0.21,
                family="clean",
                seed=541127,
            )
            tail, _, t, st, md = channel(packet(profile, second), **params2)
            truths.append((len(x) + 32 + t, st))
            x = np.r_[x, np.zeros(32), tail]
            expected.append(second)
            metadata["dense_second"] = dict(parameters=params2, metadata=md)
        cut = None
        if family == "truncated":
            cut = math.floor(truth + (len(wire) - 40 - 31) * step)
            x = x[:cut]
        f = out / f"rf-{j}-input.npz"
        np.savez_compressed(
            f,
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
                artifact=f.name,
                input_SHA256=digest(f),
                channel_parameters=params,
                channel_metadata=metadata,
                EOF_cut=cut,
            )
        )
    for j, family in enumerate(["silence", "noise-only", "tone-only", "wrong-sync"], 11):
        cfg = baseline.configuration(2 if j % 2 else 4, 8, 0, j + 47 + 71 * validation)
        rng = np.random.default_rng(541853 + j + 127 * validation)
        quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.0013
        x = np.zeros(14336 if validation else 12288, complex)
        params = dict(samples=len(x), seed=541853 + j + 127 * validation, quiet_sigma=0.0013)
        if family == "noise-only":
            x = (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * 0.0013
        if family == "tone-only":
            x = 0.073 * np.exp(1j * (2 * np.pi * -317000 * np.arange(len(x)) / 16e6 + 0.417))
            params.update(amplitude=0.073, frequency_Hz=-317000, phase_rad=0.417)
        if family == "wrong-sync":
            wrong = baseline.configuration(cfg.frame_type, 8, 0, j + 47 + 71 * validation)
            wrong.pid = (wrong.pid + 3) % 6
            cp = dict(
                offset=0.873,
                ppm=-100,
                cfo=79000,
                prefix=37,
                amplitude=0.27,
                family="clean",
                seed=541903,
            )
            v, _, _, _, _ = channel(packet(PSKReceiveProfile.ordinary(wrong), data(229, 1)), **cp)
            x[: len(v)] = v
            params.update(wrong_PID=wrong.pid, channel=cp)
        f = out / f"rf-{j}-input.npz"
        np.savez_compressed(f, iq=x, quiet=quiet, truths=np.empty((0, 2)))
        config = asdict(cfg)
        config.pop("_mcs", None)
        rows.append(
            dict(
                id=j,
                config=config,
                family=family,
                legal=False,
                legal_packets=0,
                artifact=f.name,
                input_SHA256=digest(f),
                parameters=params,
            )
        )
    baseline.write(out / "inputs.json", rows)
    return rows


def rf(out, rows):
    old_receive = baseline.receive

    def receive(row, values, chunks, candidate):
        if not candidate:
            return old_receive(row, values, chunks, False)
        with (
            patch.object(baseline, "RollingMaximumScaledBank", BankedScaledReference),
            patch.object(baseline, "RollingMaximumRankedBank", BankedRankedReference),
        ):
            return old_receive(row, values, chunks, True)

    with patch.object(baseline, "receive", receive):
        comparisons = baseline.rf_compare(out, rows)
    baseline.write(out / "RF-comparisons.json", comparisons)
    assert all(
        r["comparisons_exact"] and r["whole_chunk_semantics_and_trajectories_exact"]
        for r in comparisons
    )
    return comparisons


def timed(out, row, clock, lanes, *, block_queries=False, armed=False,
          normalizer="pipeline", continue_aborted_source=False):
    profile = PSKReceiveProfile.ordinary(TxConfig(**row["config"]))
    with np.load(out / row["artifact"]) as z:
        # Finite real RRC output;no alternate TX/RX interpolation identity proof.
        iq = z["iq"][:2560]
        filt = FullMantissaRRC(FixedRRCFormat(12, 18))
        words = []
        for pos in range(0, len(iq), 32):
            values = filt.feed_output_words(iq[pos : pos + 32])
            r, i, e = rrc_to_scaled_words(*values, ScaledBankFormat(10), stats={})
            words.extend(zip(r.tolist(), i.tolist(), e.tolist(), strict=True))
    ref = profile.sync_symbols()
    oracle = RollingMaximumRankedBank(ref, 4, {31: 256, 61: 376, 125: 632}[len(ref) - 1])
    points, queries, query_available, query_delivered = [], [], [], []
    expected_queries = {}

    def observe(event):
        if event["kind"] == "query":
            query_available.append(dict(event))
        if event["kind"] != "point":
            return
        tag = event["tag"]
        assert oracle.samples == tag
        r, i, e = words[tag]
        v = oracle.advance_words([r], [i], [e], inspect=True)
        if event["ready"]:
            assert len(v) == 1
            assert (event["numerator"], event["energy"]) == (v[0].numerator, v[0].energy)
            assert event["rank"] == int(oracle._rank_scores[tag % oracle.rank_history])
            assert event["hit"] == bool(oracle._hits[tag % oracle.window_samples])
        points.append(
            [
                tag,
                event["real"],
                event["imag"],
                event["energy"],
                event["numerator"],
                event["rank"],
                event["hit"],
                event["cycle"],
            ]
        )

    core = BankedScaledCore(ref, lanes=lanes, sink=observe, normalizer=normalizer)
    if armed:
        for _ in range(645):
            core.step()
    input_base = core.cycle
    sent, last_query, discarded, first_abort_cycle = 0, -1, 0, None
    for _ in range(len(words) * clock // 16 + 30000):
        if sent < len(words) and core.cycle >= input_base + math.ceil(sent * clock / 16):
            if core.failure:
                discarded += 1
            elif not core.offer(sent, *words[sent]):
                core.input_overrun(sent)
                if not continue_aborted_source:
                    break
                discarded += 1
                first_abort_cycle = core.cycle
            sent += 1
        core.step()
        if core.failure:
            if continue_aborted_source and sent == len(words):
                break
            continue
        n = core.committed_count - 1
        if n >= core.k * 4 and n % 32 == 0 and n != last_query:
            upper = n - core.k * 4
            qid = core.next_query_id
            answer = oracle.query_best(upper - 130, upper)
            accepted = core.request_query(upper - 130, upper)
            if accepted:
                expected_queries[qid] = answer
            queries.append(dict(id=qid, accepted=accepted, tag=n, accepted_cycle=core.cycle))
            last_query = n
        if not block_queries:
            while core.candidates:
                result = core.take_candidate()
                query_delivered.append(dict(result, consumed_cycle=core.cycle))
                expected = expected_queries.pop(result["request_id"])
                if result["reason"] is None:
                    assert (result["enabled"], result["count"], result["chosen"]) == expected
                else:
                    assert result["reason"] == "history-expired"
        if sent == len(words) and core.committed_count == sent and core.data_idle:
            break
    input_end_state = dict(
        query_output_held=bool(core.candidates),
        query_active=core.query_active is not None,
        queued_queries=len(core.query_queue),
        cycle=core.cycle,
    )
    drain_results = []
    if not core.failure:
        for _ in range(8192):
            while core.candidates:
                value = core.take_candidate()
                query_delivered.append(dict(value, consumed_cycle=core.cycle))
                expected = expected_queries.pop(value["request_id"])
                if value["reason"] is None:
                    assert (value["enabled"], value["count"], value["chosen"]) == expected
                else:
                    assert value["reason"] == "history-expired"
                drain_results.append(value)
            if core.data_idle and not core.query_queue and core.query_active is None:
                break
            core.step()
        assert core.finish() and not expected_queries
    remaining = core.input_count - core.committed_count
    tag = (
        f"clock-{row['id']}-{clock}-{lanes}-"
        f"{'blocked' if block_queries else 'open'}-{'armed' if armed else 'reset'}"
    )
    baseline.write(out / (tag + "-points.json"), points)
    baseline.write(out / (tag + "-queries.json"), queries)
    np.savez_compressed(out / (tag + "-words.npz"), words=np.asarray(words, np.int64))

    def distribution(values):
        if not values:
            return dict(count=0, minimum=None, p50=None, p95=None, p99=None, maximum=None)
        return dict(
            count=len(values),
            minimum=min(values),
            maximum=max(values),
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
            p99=float(np.percentile(values, 99)),
        )

    arrivals = {r["id"]: r["accepted_cycle"] for r in queries if r["accepted"]}
    word_latencies = [r[-1] - (input_base + math.ceil(r[0] * clock / 16)) for r in points]
    available_latencies = [
        r["available_cycle"] - arrivals[r["request_id"]] for r in query_available
    ]
    delivered_latencies = [r["consumed_cycle"] - arrivals[r["request_id"]] for r in query_delivered]
    baseline.write(out / (tag + "-query-delivery.json"), query_delivered)
    result = dict(
        word_latency_cycles=distribution(word_latencies),
        last512_word_latency_cycles=distribution(word_latencies[-512:]),
        query_available_latency_cycles=distribution(available_latencies),
        query_delivery_latency_cycles=distribution(delivered_latencies),
        unfinished_accepted_queries=sorted(
            set(arrivals) - {r["request_id"] for r in query_delivered}
        ),
        native_product_counts=dict(prefix=core.prefix.stats, score=core.score.native.stats),
        fold_operations=core.fold.stats,
        id=row["id"],
        input_SHA256=row["input_SHA256"],
        clock_MHz=clock,
        armed_after_paid_initialization=armed,
        initialization_edges_before_input=input_base,
        input_rate=16000000,
        lanes=lanes,
        blocked=block_queries,
        offered=len(words),
        accepted=core.input_count,
        committed=core.committed_count,
        unretired_accepted=remaining,
        failure=core.failure,
        first_lost_tag=core.first_lost_tag,
        high_water=core.high_water,
        statistics=core.stats,
        input_end_snapshot=input_end_state,
        released_results=drain_results,
        cost=core.estimate(),
        every_committed_point_exact=True,
        word_stream_pass=core.failure is None and remaining == 0,
        RRC_cycles=None,
        fine_search_cycles=None,
        full_RX_16MSps=False,
    )
    if continue_aborted_source:
        assert sent == len(words) == core.input_count + discarded
        if core.failure:
            assert not core.finish()
        result.update(
            normalization=normalizer,
            source_tags_presented=sent,
            source_epoch_discards=discarded,
            first_abort_cycle=first_abort_cycle,
            all_planned_source_arrivals_processed=True,
            automatic_rearm=False,
            EOF_epoch_closed=core.closed,
            source_discard_controller=dict(
                additional_counter_bits=64, increment_units64=1,
                input_valid_and_existing_failure_status=True,
                additional_FIFO_or_IQ_state=0,
                composed_register_bits=core.estimate()["fabric_register_bits_upper"] + 64,
                composed_logical_bits=core.estimate()["logical_state_bits_upper"] + 64,
                actual_synthesis_or_timing=False,
                upstream_RRC_arrivals_and_rearm_cycles=None,
            ),
        )
    baseline.write(out / (tag + ".json"), result)
    return result


POLICY = dict(
    version=1,
    scope="one scaled10 coarse word path;not ADC to MAC clock closure",
    symbols_per_second=4000000,
    samples_per_symbol=4,
    input_rate=16000000,
    terms=[31, 61, 125],
    mantissa_bits=10,
    exponent_range=[-48, 32],
    studied_lanes=[8, 16],
    preferred_lanes={"2": 8, "3": 8, "4": 16},
    clocks_MHz=[100, 160, 200],
    input_words_per_clock_case=2560,
    input_FIFO=16,
    differences=64,
    folds=4,
    CE=4,
    queries=2,
    candidate_output=1,
    query_stride_committed_words=32,
    query_positions=131,
    snapshot_minimum_age_words=256,
    histogram="serial1r1w;645 paid initialization edges;II8",
    DSP_per_path=12,
    old_node_DSP_cap=18,
    old_node_state_cap_bits=131072,
    fixed_state_and_counters=True,
    exact_words_gate_Q12_and_query_required=True,
    physical_PRNG_or_CRC_answer_selection=False,
    search_and_timing_parameters_unchanged=True,
    full_bank_work="every input,all k terms;no silence skipping",
    adoption=(
        "explicit experimental node only if exact,no success loss,no new false MAC;not default"
    ),
    reference="existing scaled word bank,independent integer checks;parent float loss remains",
    old_2percent_RRC_failure_retained=True,
    whole_system_budget_approved=False,
    whole_RX_cycles=None,
    whole_RX_16MSps=False,
    RTL_synthesis=None,
)


def sources():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *root.glob("src/**/*.py"),
        *root.glob("tests/**/*.py"),
        root / "scripts/model_scaled_maximum.py",
        root / "scripts/rx_timing_audit.py",
        Path(__file__).resolve(),
        root / "uv.lock",
    ]
    return {str(f.relative_to(root)): digest(f) for f in sorted(set(paths))}


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
            "Need frozen exact algorithm,parameters,budgets and adoption before validation"
        )
    out = args.output
    out.mkdir(exist_ok=False)
    baseline.write(out / "freeze.json", frozen)
    rows = inputs(out, args.split)
    comparisons = rf(out, rows)
    cases = (0, 1, 2) if args.split == "validation" else (0, 2, 4)
    calendars = [
        timed(out, rows[j], clock, lanes)
        for j in cases
        for clock in POLICY["clocks_MHz"]
        for lanes in POLICY["studied_lanes"]
    ]
    calendars += [
        timed(out, rows[j], 100, lanes, armed=True)
        for j in cases
        for lanes in POLICY["studied_lanes"]
    ]
    calendars.append(timed(out, rows[cases[2]], 200, 16, block_queries=True))
    costs = []
    for j in cases:
        profile = PSKReceiveProfile.ordinary(TxConfig(**rows[j]["config"]))
        for lanes in POLICY["studied_lanes"]:
            node = BankedScaledCore(profile.sync_symbols(), lanes=lanes)
            costs.append(
                dict(frame_type=profile.config.frame_type, lanes=lanes, cost=node.estimate())
            )
    whole = [r for r in comparisons if r["chunks"] == "whole"]
    component = all(
        r["comparisons_exact"] and r["whole_chunk_semantics_and_trajectories_exact"]
        for r in comparisons
    )
    component &= all(r["every_committed_point_exact"] for r in calendars)
    component &= all(
        r["word_stream_pass"] for r in calendars if r["clock_MHz"] == 200 and r["lanes"] == 16
    )
    component &= all(r["original_success_losses"] == 0 and r["false_MAC"] == 0 for r in whole)
    preferred = POLICY["preferred_lanes"]
    component &= all(
        r["word_stream_pass"]
        for r in calendars
        if r["clock_MHz"] == 200
        and not r["blocked"]
        and r["lanes"] == preferred[str(rows[r["id"]]["config"]["frame_type"])]
    )
    component &= all(
        c["cost"]["logical_module_research_cap_pass"] and c["cost"]["DSP_research_cap_pass"]
        for c in costs
    )
    negative_samples = sum(len(np.load(out / r["artifact"])["iq"]) for r in rows if not r["legal"])
    failures = [
        dict(kind="RF", **r) for r in whole if r["legal"] and r["complete"] < r["legal_packets"]
    ]
    failures += [
        dict(kind="resource", **r)
        for r in calendars
        if r["failure"] or r["statistics"]["query_dropped"] or r["statistics"]["query_expired"]
    ]
    result = dict(
        split=args.split,
        component_word_and_frame_exact=bool(component),
        RF_streams=len(rows),
        RF_runs=len(comparisons),
        calendars=calendars,
        costs=costs,
        failure_links=failures,
        full=sum(r["complete"] for r in whole),
        captured=sum(r["captured"] for r in whole),
        head_CRC=sum(r["head_CRC"] for r in whole),
        data_CRC=sum(r["data_CRC"] for r in whole),
        success_losses=sum(r["original_success_losses"] for r in whole),
        false_MAC=sum(r["false_MAC"] for r in whole if not r["legal"]),
        negative_samples=negative_samples,
        negative_exposure_seconds=negative_samples / 16e6,
        capture_gain=0 if component else None,
        RF_input_truth_used_only_for_diagnostics=True,
        shared_histogram_and_prefix_not_additive=True,
        two_paths_cost_doubled=True,
        RRC_cycles=None,
        raw_arrival_join_cycles=None,
        fine_search_cycles=None,
        carrier_and_payload_cycles=None,
        whole_RX_gates_passed=False,
        old_numeric_and_resource_failures_preserved=True,
        new_default=False,
    )
    final = sources()
    baseline.write(out / "source-bindings-at-end.json", final)
    if final != frozen["sources"]:
        raise ValueError("Source changed during evaluation;not frozen evidence")
    baseline.write(out / "model.json", result)
    if not component:
        raise ValueError("Component mismatch or original-success regression;candidate rejected")
    print(
        json.dumps(
            {
                key: result[key]
                for key in [
                    "split",
                    "component_word_and_frame_exact",
                    "RF_streams",
                    "RF_runs",
                    "full",
                    "captured",
                    "head_CRC",
                    "data_CRC",
                    "success_losses",
                    "false_MAC",
                    "negative_samples",
                    "whole_RX_gates_passed",
                ]
            }
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
