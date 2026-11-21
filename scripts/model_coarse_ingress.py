"""RRC releases/raw flags/banked ingress on one clock;not full-RX acceptance."""

import argparse
import gzip
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts import model_banked_scaled as banked
from scripts import model_scaled_maximum as shared
from scripts.model_raw_gate_join import legacy_flags
from tests.maximum_reference import channel, packet

from nearlink_sdr.phy.acquisition_maximum import RollingMaximumRankedBank
from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat
from nearlink_sdr.phy.rrc_bridge import rrc_to_scaled_words
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.coarse_ingress import CoarseIngress

POLICY = dict(
    version=1,
    prefix_samples=3072,
    clocks_MHz=[100, 160, 200],
    normalizers=["pipeline", "serial2"],
    preferred_lanes={"2": 8, "3": 8, "4": 16},
    pending_query_credit=1,
    query_stride=32,
    origin=-40,
    raw_flag_latency_hypotheses=[10, 80],
    RRC="existing PairRRCFlow;4complex lanes,history128,FIFO4,products3,II12",
    own_control_research_cap_bits=1536,
    old_coarse_DSP_cap=18,
    old_coarse_logical_cap_bits=131072,
    no_extra_IQ_FIFO_or_DSP=True,
    source_word_and_query_tolerance="exact",
    adoption=(
        "200MHz preferred layouts exact and lossless;finite all-source/EOF accounting;"
        "own1536bits;explicit experimental only"
    ),
    raw_detector_arithmetic_cycles=None,
    fine_packet_decoder_cycles=None,
    no_new_default=True,
    no_system_budget_approval=True,
    old_numeric_and_resource_failures_retained=True,
    whole_RX_16MSps=False,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    root = Path(__file__).resolve().parents[1]
    for fn, name in [
        (CoarseIngress.step, "src/nearlink_sdr/sim/coarse_ingress.py"),
        (packet, "tests/maximum_reference.py"),
    ]:
        if Path(fn.__code__.co_filename).resolve() != root / name:
            raise ValueError("Need exact own compiled model and independent reference")
    files = [root / n for n in ["uv.lock", "pyproject.toml", ".github/workflows/ci.yml"]]
    files += [f for n in ["src", "tests", "scripts"] for f in (root / n).rglob("*.py")]
    return {str(f.relative_to(root)): sha(f) for f in sorted(files)}


def payload(n, family):
    if family == 0:
        return bytes(((j * 47 + 0x89) ^ (j >> 5)) & 255 for j in range(n))
    if family == 1:
        return bytes((0xAA, 0xF0, 0x4B, 0x95, 0x6E, 3)[j % 6] ^ ((j // 31) & 255) for j in range(n))
    return bytes((int(f"{j & 255:08b}"[::-1], 2) + 7 + (j // 257) * 59) & 255 for j in range(n))


def inputs(out, split):
    held = split == "validation"
    specs = [
        (2, 8, 0, 67, "tone"),
        (3, 12, 4, 231, "noise"),
        (4, 12, 16, 419, "wrong-sync-weak"),
        (2, 12, 8, 733, "burst"),
        (3, 8, 0, 149, "step"),
        (4, 8, 4, 307, "clean"),
    ]
    if held:
        specs = [
            (4, 8, 8, 121, "tone"),
            (2, 12, 4, 283, "noise"),
            (3, 8, 16, 557, "wrong-sync-weak"),
            (4, 12, 0, 97, "burst"),
            (2, 8, 16, 461, "step"),
            (3, 12, 8, 811, "clean"),
        ]
    specs += [(2, 12, 4, 89, "bad_head"), (3, 8, 8, 257, "bad_data"), (4, 12, 0, 431, "truncated")]
    rows = []
    for j, (ft, mcs, pilot, length, family) in enumerate(specs):
        cfg = shared.configuration(ft, mcs, pilot, 173 + j + 89 * held)
        profile = PSKReceiveProfile.ordinary(cfg)
        expected = [payload(length, (j + 2 * held) % 3)]
        wire = packet(
            profile, expected[0], bad_head=family == "bad_head", bad_data=family == "bad_data"
        )
        params = dict(
            family=family,
            offset=(0.143 + 0.191 * j + 0.337 * held) % 1,
            ppm=[-75, 100, -50, 25][(j + held) % 4],
            cfo=[107000, -47000, 23000, -119000][j % 4],
            prefix=[79, 151, 293, 421][j % 4] + 31 * held,
            amplitude=[0.093, 0.37, 0.063, 0.21, 0.12][j % 5],
            seed=771827 + 113 * j + 1297 * held,
            snr_db=11 if family == "noise" else None,
        )
        x, quiet, truth, step, metadata = channel(wire, **params)
        truths = [(truth, step)]
        if family == "wrong-sync-weak":
            wrong = shared.configuration(ft, mcs, pilot, 173 + j + 89 * held)
            wrong.pid = (wrong.pid + 1) % 6
            cp = dict(
                family="clean",
                offset=0.713,
                ppm=75,
                cfo=-63000,
                prefix=0,
                amplitude=0.23,
                seed=771919,
            )
            bg, _, _, _, _ = channel(
                packet(PSKReceiveProfile.ordinary(wrong), payload(1117, 1)), **cp
            )
            x += bg[: len(x)]
            metadata["wrong_sync_background"] = dict(parameters=cp, PID=wrong.pid, length=1117)
        if j == 0:
            second = payload(97 if not held else 173, 2)
            cp = dict(
                family="clean",
                offset=0.417,
                ppm=-100,
                cfo=89000,
                prefix=0,
                amplitude=0.17,
                seed=772049,
            )
            tail, _, t, st, md = channel(packet(profile, second), **cp)
            truths.append((len(x) + 40 + t, st))
            x = np.r_[x, np.zeros(40), tail]
            expected.append(second)
            metadata["dense_second"] = dict(parameters=cp, metadata=md)
        cut = None
        if family == "truncated":
            cut = math.floor(truth + (len(wire) - 40 - 89) * step)
            x = x[:cut]
        path = out / f"rf-{j}-input.npz"
        np.savez_compressed(
            path,
            iq=x,
            quiet=quiet,
            truths=np.asarray(truths),
            **{f"expected{k}": np.frombuffer(v, np.uint8) for k, v in enumerate(expected)},
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
                parameters=params,
                metadata=metadata,
                truth_diagnostic_only=dict(truths=truths, cut=cut),
            )
        )
    for j, family in enumerate(["silence", "noise-only", "tone-only", "wrong-sync"], 9):
        cfg = shared.configuration(2 if j % 2 else 4, 8, 0, 173 + j + 89 * held)
        rng = np.random.default_rng(772181 + j + 223 * held)
        quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.0019
        x = np.zeros(16384 if not held else 18432, complex)
        params = dict(family=family, samples=len(x))
        if family == "noise-only":
            x = (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * 0.0019
        elif family == "tone-only":
            x = 0.137 * np.exp(2j * np.pi * -271000 * np.arange(len(x)) / 16e6)
        elif family == "wrong-sync":
            wrong = shared.configuration(cfg.frame_type, 8, 0, 173 + j + 89 * held)
            wrong.pid = (wrong.pid + 2) % 6
            cp = dict(
                family="clean",
                offset=0.817,
                ppm=50,
                cfo=97000,
                prefix=43,
                amplitude=0.29,
                seed=772297,
            )
            v, _, _, _, _ = channel(
                packet(PSKReceiveProfile.ordinary(wrong), payload(317, 0)), **cp
            )
            x[: len(v)] = v
            params.update(channel=cp, wrong_PID=wrong.pid)
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
                parameters=params,
            )
        )
    shared.write(out / "inputs.json", rows)
    return rows


def prepare(out, row):
    with np.load(out / row["artifact"]) as z:
        iq, quiet = z["iq"][:3072], z["quiet"]
    padding = max(0, 3072 - len(iq))
    iq = np.r_[iq, np.zeros(padding)]
    profile = PSKReceiveProfile.ordinary(TxConfig(**row["config"]))
    rx = PSKStreamReceiver(profile, quiet, quiet_batching=False, defer_quiet_fir=False)
    flags = legacy_flags(rx, iq)
    versions = []
    for mode in ["whole", "fixed", "random"]:
        filt = FullMantissaRRC(FixedRRCFormat(12, 18))
        chunks = []
        rng = np.random.default_rng(772453)
        start = 0
        while start < len(iq):
            size = (
                len(iq) if mode == "whole" else 32 if mode == "fixed" else int(rng.integers(1, 48))
            )
            # Host call grouping never widens the existing1..32 word port.
            for inner in range(start, min(start + size, len(iq)), 32):
                end = min(inner + 32, start + size, len(iq))
                chunks.append(filt.feed_output_words(iq[inner:end]))
            start += size
        versions.append(tuple(np.concatenate([v[k] for v in chunks]) for k in range(3)))
    assert all(
        all(
            a.dtype == b.dtype and a.tobytes() == b.tobytes()
            for a, b in zip(versions[0], v, strict=True)
        )
        for v in versions[1:]
    )
    words = versions[0]
    triples = []
    for j in range(0, len(iq), 32):
        arrays = rrc_to_scaled_words(
            *(v[j : j + 32] for v in words), ScaledBankFormat(10), stats={}
        )
        triples.extend(tuple(map(int, v)) for v in zip(*arrays, strict=True))
    file = out / f"prefix-{row['id']}.npz"
    np.savez_compressed(
        file,
        iq=iq,
        flags=np.asarray(flags),
        rrc_words=np.asarray(words),
        scaled_words=np.asarray(triples, np.int64),
    )
    return (
        profile,
        flags,
        words,
        triples,
        dict(
            file=file.name,
            SHA256=sha(file),
            padding=padding,
            samples=3072,
            whole_fixed_random_words_exact=True,
        ),
    )


def calendar(out, row, prepared, clock, lanes, normalizer, *, latency=10, blocked=False):
    profile, flags, words, triples, prefix = prepared
    reference = RollingMaximumRankedBank(
        profile.sync_symbols(), 4, {2: 256, 3: 376, 4: 632}[profile.config.frame_type], origin=-40
    )
    expected, events = {}, []

    def observe(event):
        if event["kind"] == "point":
            tag = event["tag"]
            assert tag == reference.samples
            v = reference.advance_words(*(np.asarray([x]) for x in triples[tag]), inspect=True)
            if event["ready"]:
                assert (event["numerator"], event["energy"]) == (v[0].numerator, v[0].energy)
                assert event["rank"] == int(reference._rank_scores[tag % reference.rank_history])
                assert event["hit"] == bool(reference._hits[tag % reference.window_samples])
        elif event["kind"] == "ingress_query_request" and event["accepted"]:
            assert event["snapshot"] == reference.samples
            expected[event["request_id"]] = reference.query_best(event["lower"], event["upper"])
        events.append(dict(event))

    flow = CoarseIngress(profile.sync_symbols(), lanes=lanes, normalizer=normalizer, sink=observe)
    producer_node = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=clock * 1000000))
    producer = producer_node.simulate(len(flags), trace=True)
    released = {e["cycle"]: e for e in producer["trace"] if e["event"] == "output"}
    flag_at = {math.ceil(j * clock / 16) + latency: j for j in range(len(flags))}
    delivered = []
    for cycle in range(max(producer["cycles"], max(flag_at)) + 8192):
        if not (blocked and 3000 <= cycle < 19000):
            while flow.core.candidates:
                result = flow.take_candidate()
                answer = expected.pop(result["request_id"])
                if result["reason"] is None:
                    assert (result["enabled"], result["count"], result["chosen"]) == answer
                else:
                    assert result["reason"] == "history-expired"
                delivered.append(dict(result, consumed_cycle=cycle))
        if cycle in flag_at:
            tag = flag_at[cycle]
            flow.push_flag(tag, flags[tag])
        if cycle in released:
            event = released[cycle]
            tag = event["sample"]
            flow.push_rrc(
                tag, None if event["invalid"] else pack_filter_word(*(int(v[tag]) for v in words))
            )
        flow.step()
        if cycle > max(producer["cycles"], max(flag_at)) and flow.drained:
            break
    assert flow.drained
    passed = flow.finish()
    assert flow.stats["rrc_events"] == producer["emitted"]
    assert flow.stats["rrc_missing_tags"] == producer["dropped"]
    assert flow.join.next_tag == len(flags)
    assert flow.join.stats["flags_accepted"] + flow.join.stats["flags_discarded"] == len(flags)
    assert flow.core.input_count == flow.join.stats["words_joined"]
    assert (
        flow.stats["rrc_events"] == flow.join.bridge.stats["accepted"] + flow.stats["rrc_discarded"]
    )
    assert (
        flow.join.bridge.stats["accepted"]
        == flow.join.bridge.stats["retired"] + flow.join.bridge.stats["aborted"]
    )
    assert flow.stats["query_requests"] == sum(
        flow.stats[k]
        for k in ["query_submitted", "pending_query_dropped", "pending_query_canceled"]
    )
    assert flow.core.stats["query_completed"] == len(delivered) + flow.stats["candidate_canceled"]
    label = f"FT{profile.config.frame_type}-{clock}-{lanes}-{normalizer}-{latency}-{int(blocked)}"
    path = out / (label + "-events.jsonl.gz")
    raw = b"".join((json.dumps(e, sort_keys=True) + "\n").encode() for e in events)
    path.write_bytes(gzip.compress(raw, mtime=0))
    cost = flow.estimate()
    rrc_cost = producer_node.resource_estimate()
    completed = {e["request_id"] for e in events if e["kind"] == "query"}
    canceled = completed - {r["request_id"] for r in delivered}
    assert len(canceled) == flow.stats["candidate_canceled"] and canceled <= expected.keys()

    def distribution(values):
        return dict(
            count=len(values),
            minimum=min(values) if values else None,
            p50=float(np.percentile(values, 50)) if values else None,
            p95=float(np.percentile(values, 95)) if values else None,
            p99=float(np.percentile(values, 99)) if values else None,
            maximum=max(values) if values else None,
        )

    issued = {
        e["request_id"]: e["cycle"]
        for e in events
        if e["kind"] == "ingress_query_request" and e["accepted"]
    }
    result = dict(
        case=label,
        frame_type=profile.config.frame_type,
        clock_MHz=clock,
        lanes=lanes,
        normalizer=normalizer,
        raw_flag_latency_hypothesis=latency,
        blocked=blocked,
        input_SHA256=row["input_SHA256"],
        prefix=prefix,
        cycles=flow.cycle,
        producer={k: v for k, v in producer.items() if k != "trace"},
        source_samples=len(flags),
        RRC_emitted=flow.stats["rrc_events"],
        accepted=flow.core.input_count,
        committed=flow.core.committed_count,
        source_fault=flow.fault_reason,
        source_fault_tag=flow.fault_tag,
        control_statistics=flow.stats,
        join_statistics=flow.join.stats,
        bridge_statistics=flow.join.bridge.stats,
        core_statistics=flow.core.stats,
        high_water=flow.core.high_water,
        query_deliveries=delivered,
        canceled_completed_queries=sorted(canceled),
        unfinished_accepted_queries=sorted(expected.keys() - canceled),
        word_latency_cycles=distribution(
            [e["cycle"] - math.ceil(e["tag"] * clock / 16) for e in events if e["kind"] == "point"]
        ),
        delivered_query_latency_cycles=distribution(
            [r["consumed_cycle"] - issued[r["request_id"]] for r in delivered]
        ),
        event_file=path.name,
        event_SHA256=sha(path),
        source_query_prefix_pass=passed,
        all_committed_integer_and_query_values_exact=True,
        all_source_and_EOF_accounting_exact=True,
        cost=cost,
        RRC_cost=rrc_cost,
        dedicated_prefix_DSP_upper=rrc_cost["multiplier_units"] + cost["coarse"]["DSP"],
        known_prefix_logical_bits=rrc_cost["register_tap_logical_bits"]
        + cost["whole_prefix_excluding_RRC_and_detector_logical_bits"],
        RRC_physical_mapping_and_wrapper_timing_qualified=False,
        raw_detector_arithmetic_cycles=None,
        fine_packet_decoder_cycles=None,
        whole_RX_16MSps=False,
        actual_synthesis=False,
    )
    shared.write(out / (label + ".json"), result)
    return result


def qualify(cases):
    """All six frozen preferred layouts and numeric budgets must actually hold."""
    preferred = [
        r
        for r in cases
        if r["clock_MHz"] == 200
        and not r["blocked"]
        and r["raw_flag_latency_hypothesis"] == 10
        and r["lanes"] == POLICY["preferred_lanes"][str(r["frame_type"])]
    ]
    expected = {(ft, mode) for ft in [2, 3, 4] for mode in POLICY["normalizers"]}
    if len(preferred) != 6 or {(r["frame_type"], r["normalizer"]) for r in preferred} != expected:
        return False
    return all(
        r["source_query_prefix_pass"]
        and r["cost"]["own_register_research_cap_pass"]
        and r["cost"]["own_register_bits_upper"] <= POLICY["own_control_research_cap_bits"]
        and r["cost"]["coarse"]["DSP"] <= POLICY["old_coarse_DSP_cap"]
        and r["cost"]["coarse"]["logical_state_bits_upper"] <= POLICY["old_coarse_logical_cap_bits"]
        for r in preferred
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=sources())
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen eligible development")
        try:
            prior = json.loads(args.freeze.read_text())
            model = json.loads((args.freeze.parent / "model.json").read_text())
            end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Need complete frozen eligible development") from error
        if (
            prior != frozen
            or model.get("component_qualified") is not True
            or end != frozen["sources"]
            or model.get("freeze_SHA256") != sha(args.freeze)
        ):
            raise ValueError("Need exact frozen source,parameters,budget and eligible development")
    args.output.mkdir(exist_ok=False)
    shared.write(args.output / "freeze.json", frozen)
    rows = inputs(args.output, args.split)
    RF = banked.rf(args.output, rows)
    cases = []
    for ft in [2, 3, 4]:
        row = next(r for r in rows if r["config"]["frame_type"] == ft)
        prepared = prepare(args.output, row)
        for clock in [100, 160, 200]:
            for mode in POLICY["normalizers"]:
                cases.append(
                    calendar(
                        args.output, row, prepared, clock, POLICY["preferred_lanes"][str(ft)], mode
                    )
                )
        if ft == 4:
            for lanes, latency, blocked in [(8, 10, False), (16, 80, False), (16, 10, True)]:
                cases.append(
                    calendar(
                        args.output,
                        row,
                        prepared,
                        200,
                        lanes,
                        "serial2",
                        latency=latency,
                        blocked=blocked,
                    )
                )
    qualified = qualify(cases)
    assert sources() == frozen["sources"]
    shared.write(args.output / "source-bindings-at-end.json", sources())
    whole = [r for r in RF if r["chunks"] == "whole"]
    negative_samples = sum(
        len(np.load(args.output / r["artifact"])["iq"]) for r in rows if not r["legal"]
    )
    result = dict(
        split=args.split,
        calendars=cases,
        component_qualified=qualified,
        all_integer_and_accounting_exact=True,
        freeze_SHA256=sha(args.output / "freeze.json"),
        functional_RF_context=dict(
            scope="untimed old bank vs banked reference,not this ingress IQ-to-MAC",
            runs=len(RF),
            captured=sum(r["captured"] for r in whole),
            head_CRC=sum(r["head_CRC"] for r in whole),
            data_CRC=sum(r["data_CRC"] for r in whole),
            length_ok=sum(r["length_ok"] for r in whole),
            payload_content_checks=sum(r["payload_content_checks"] for r in whole),
            payload_content_matches=sum(r["payload_content_matches"] for r in whole),
            BER_bits=sum(r["BER_bits"] for r in whole),
            BER_errors=sum(r["BER_errors"] for r in whole),
            BER_excludes_rows_without_comparable_bits=True,
            MCS_by_row=[
                dict(
                    id=r["id"],
                    expected=rows[r["id"]]["config"]["mcs_index"],
                    observed=r["MCS"],
                    head_CRC=r["head_CRC"],
                )
                for r in whole
            ],
            true_timing_error_max_samples=max(
                (v for r in whole for v in r["timing_error_max"]), default=None
            ),
            full=sum(r["complete"] for r in whole),
            losses=sum(r["original_success_losses"] for r in whole),
            false_MAC=sum(r["false_MAC"] for r in whole if not r["legal"]),
            negative_samples=negative_samples,
            negative_seconds=negative_samples / 16e6,
        ),
        raw_detector_arithmetic_cycles=None,
        fine_packet_decoder_cycles=None,
        all_four_RX_gates_passed=False,
        new_default=False,
        actual_synthesis=False,
    )
    shared.write(args.output / "model.json", result)
    print(
        json.dumps(
            dict(
                cases=len(cases),
                component_qualified=qualified,
                all_integer_and_accounting_exact=True,
                whole_RX_accepted=False,
            )
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
