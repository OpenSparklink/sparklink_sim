"""Actual RRC words/raw flags with bounded alignment;not whole-RX cycle acceptance."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scripts.model_rrc_bridge import filter_words
from tests.test_rrc_bridge import oracle

from nearlink_sdr.phy.acquisition_scaled import ScaledDifferentialBank
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver, _fir_block
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.raw_gate_join import RawGateJoin
from nearlink_sdr.sim.rrc_bridge_flow import RRCScaledBridgeService

POLICY = dict(
    fixed_window_OR_representation="saturating_age",
    FIFO_depth=16,
    compiled_windows=[256, 376, 632],
    registered_flag_read_edges=1,
    node_research_register_budget_bits=1536,
    bit_word_tag_and_gate_identity="exact",
    raw_flag_latency_hypotheses=[10, 80],
    clock_MHz=[100, 160, 200],
    no_main_RX_default_change=True,
    no_new_RF_holdout=True,
    raw_detector_arithmetic_and_scaled_bank_cycles=None,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_bindings():
    root = Path(__file__).resolve().parents[1]
    for fn, path in (
        (filter_words, root / "scripts/model_rrc_bridge.py"),
        (oracle, root / "tests/test_rrc_bridge.py"),
        (RawGateJoin.step, root / "src/nearlink_sdr/sim/raw_gate_join.py"),
    ):
        if Path(fn.__code__.co_filename).resolve() != path:
            raise ValueError("Reference must load from this exact source tree")
    paths = [
        *sorted((root / "src").rglob("*.py")),
        Path(__file__).resolve(),
        root / "scripts/model_rrc_bridge.py",
        root / "tests/test_rrc_bridge.py",
        root / "uv.lock",
    ]
    return {str(p.relative_to(root)): sha(p) for p in paths}


def context(frame_type, quiet):
    cfg = TxConfig(
        frame_type=frame_type,
        symbol_rate_mhz=4,
        sps=4,
        mcs_index={2: 8, 3: 12, 4: 10}[frame_type],
        pilot_interval={2: 0, 3: 8, 4: 4}[frame_type],
        ctrl_bits_len=28 if frame_type == 2 else 27,
        head_llid=None if frame_type == 2 else 0x531721,
    )
    return PSKStreamReceiver(
        PSKReceiveProfile.ordinary(cfg), quiet, quiet_batching=False, defer_quiet_fir=False
    )


def legacy_flags(receiver, samples):
    """Unchanged float detector used only as an external Boolean source."""
    state, flags = np.zeros(len(receiver._box) - 1), []
    for start in range(0, len(samples), 32):
        power, state = _fir_block(receiver._box, abs(samples[start : start + 32]) ** 2, state)
        flags.extend(bool(v > receiver.threshold) for v in power)
    return flags


def expected_values(words, receiver):
    triples = [oracle(*(int(v) for v in triple), 10)[0] for triple in zip(*words, strict=True)]
    bank = ScaledDifferentialBank(receiver.reference, 4, receiver.search_samples, origin=-40)
    points, queries = {}, {}
    for start in range(0, len(triples), 32):
        part = triples[start : start + 32]
        for p in bank.advance_words(*(np.array(a) for a in zip(*part, strict=True)), inspect=True):
            points[p.start] = [int(p.numerator), int(p.energy), p.coefficient_energy]
        tag = start + len(part) - 1
        if bank.samples >= receiver.search_samples and (tag + 1) % 32 == 0:
            queries[tag] = bank.gate(tag + 2 - 40 - receiver.search_samples, tag - 40 - bank.k * 4)
    return triples, points, queries


def simulate(clock, latency, blocked, words, flags, receiver, expected):
    """Only RRC/bridge/join edges are timed;bank evaluation is explicitly off-clock."""
    triples, points, queries = expected
    producer = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=clock * 1_000_000)).simulate(
        len(flags), trace=True
    )
    outputs = {e["cycle"]: e for e in producer["trace"] if e["event"] == "output"}
    flag_at = {(i * clock + 15) // 16 + latency: i for i in range(len(flags))}
    assert len(flag_at) == len(flags) == producer["source_history_writes"]
    join = RawGateJoin(RRCScaledBridgeService(), window_samples=receiver.search_samples)
    sink = ScaledDifferentialBank(receiver.reference, 4, receiver.search_samples, origin=-40)
    emitted, rejected, invalid, scored, gate_queries = [], [], [], [], []
    peak, final_cycle = 0, max(producer["cycles"], max(flag_at), 1000 if blocked else 0) + 16
    while join.cycle <= final_cycle:
        cycle = join.cycle
        if join.ready and not (blocked and 200 <= cycle < 1000):
            row = join.take()
            tag = row["tag"]
            assert tag == len(emitted) and row["word"] == triples[tag]
            assert row["energy_trigger"] == flags[tag]
            assert row["window_energy_present"] == any(
                flags[max(0, tag + 1 - join.window) : tag + 1]
            )
            emitted.append([tag, cycle, row["energy_trigger"], row["window_energy_present"]])
            for p in sink.advance_words(*(np.array([v]) for v in row["word"]), inspect=True):
                assert points[p.start] == [int(p.numerator), int(p.energy), p.coefficient_energy]
                scored.append(
                    dict(
                        start=p.start,
                        numerator=int(p.numerator),
                        energy=int(p.energy),
                        score_available_cycle=None,
                    )
                )
            if tag in queries:
                actual = sink.gate(tag + 2 - 40 - join.window, tag - 40 - sink.k * 4)
                assert actual == queries[tag]
                gate_queries.append(
                    dict(
                        tag=tag,
                        raw_energy=row["window_energy_present"],
                        integer_gate=actual,
                        available_cycle=None,
                    )
                )
        if cycle in flag_at:
            tag = flag_at[cycle]
            join.push_flag(tag, flags[tag])
            peak = max(peak, join.count)
        if cycle in outputs:
            event = outputs[cycle]
            tag = event["sample"]
            if event["invalid"]:
                invalid.append([tag, cycle])
            else:
                word = pack_filter_word(*(int(v[tag]) for v in words))
                if not join.bridge.issue(word, tag):
                    rejected.append([tag, cycle])
        join.step()
    join.finish()
    assert len(emitted) == join.stats["words_joined"] == sink.samples
    assert join.stats["flags_accepted"] + join.stats["flags_discarded"] == len(flags)
    assert join.stats["flags_accepted"] == len(emitted) + join.stats["orphan_flags"]
    assert len(emitted) <= join.bridge.stats["accepted"]
    assert join.bridge.stats["accepted"] == (
        join.bridge.stats["retired"] + join.bridge.stats["aborted"]
    )
    assert join.bridge.stats["accepted"] + len(rejected) + len(invalid) == producer["emitted"]
    return dict(
        clock_MHz=clock,
        raw_flag_latency_hypothesis=latency,
        output_blocked=blocked,
        producer={k: v for k, v in producer.items() if k != "trace"},
        source_drop_tags=[e["sample"] for e in producer["trace"] if e["event"] == "drop_new"],
        source_invalid_outputs=invalid,
        bridge_rejected_words=rejected,
        emitted=emitted,
        functional_score_points=scored,
        functional_gate_queries=gate_queries,
        peak_flag_FIFO=peak,
        join_stats=join.stats,
        bridge_stats=join.bridge.stats,
        fault_reason=join.fault_reason,
        fault_tag=join.fault_tag,
        continuous_alignment_pass=len(emitted) == len(flags) and not join.fault_reason,
        functional_sink_off_clock=True,
        raw_detector_arithmetic_cycles=None,
        scaled_bank_cycles=None,
        capture_header_payload_MAC=None,
        four_RX_gates_passed=False,
    )


def inputs(split):
    rng = np.random.default_rng(970241 if split == "development" else 1731251)
    for family in range(8):
        n = 288 + 49 * family if split == "development" else 448 + 47 * family
        t = np.arange(n)
        quiet = (rng.normal(size=513) + 1j * rng.normal(size=513)) * 0.03
        x = np.exp(1j * (t * (0.071 if split == "development" else -0.117)))
        if family == 0:
            x *= 0
        elif family == 1:
            x *= np.where(t % 71 < 39, 0.001, 0.5)
        elif family == 2:
            x = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.013
        elif family == 3:
            x *= np.where(t < n // 3, 1e-17, 1.3)
        elif family == 4:
            x[73:98] *= 63
        elif family == 5:
            x *= 0.02 * (1 + 0.91 * np.cos(t * (0.091 if split == "development" else 0.173)))
        elif family == 6:
            x *= np.take(
                [0.001, 0.25, 0.01, 2.0], (t // (17 if split == "development" else 29)) % 4
            )
        else:
            x *= np.linspace(0.0001, 3.0, n) if split == "development" else np.linspace(2.0, 0.0, n)
        yield family, (2, 3, 4)[family % 3], x, quiet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    freeze = dict(policy=POLICY, sources=source_bindings())
    if args.split == "validation" and (
        args.freeze is None or json.loads(args.freeze.read_text()) != freeze
    ):
        raise ValueError("Need frozen exact parameters,sources and references before validation")
    args.output.mkdir(exist_ok=False)
    write(args.output / "freeze.json", freeze)
    rows = []
    for family, ft, x, quiet in inputs(args.split):
        receiver = context(ft, quiet)
        words, source_stats = filter_words(x)
        flags = legacy_flags(receiver, x)
        expected = expected_values(words, receiver)
        result = simulate(200, 10, False, words, flags, receiver, expected)
        assert result["continuous_alignment_pass"]
        file = args.output / f"value-{family}.npz"
        np.savez_compressed(
            file,
            iq=x,
            quiet=quiet,
            flags=np.array(flags),
            sync_reference=receiver.reference,
            r=words[0],
            i=words[1],
            e=words[2],
        )
        rows.append(
            dict(
                id=family,
                frame_type=ft,
                search_samples=receiver.search_samples,
                samples=len(x),
                artifact_SHA256=sha(file),
                source_statistics=source_stats,
                absolute_quantization_budget_pass=None,
                result=result,
            )
        )
    t = np.arange(1024)
    x = np.exp(1j * t * (0.071 if args.split == "development" else -0.173))
    x *= np.take([0.0001, 1.0, 0.02, 3.0], (t // (63 if args.split == "development" else 47)) % 4)
    quiet = np.full(513, 0.03 + 0.03j)
    receiver = context(4, quiet)
    words, _ = filter_words(x)
    flags = legacy_flags(receiver, x)
    expected = expected_values(words, receiver)
    file = args.output / "calendar-input.npz"
    np.savez_compressed(
        file,
        iq=x,
        quiet=quiet,
        flags=np.array(flags),
        sync_reference=receiver.reference,
        r=words[0],
        i=words[1],
        e=words[2],
    )
    calendars = [
        simulate(c, latency, blocked, words, flags, receiver, expected)
        for c in (100, 160, 200)
        for latency in (10, 80)
        for blocked in (False, True)
    ]
    costs = [
        dict(frame_type=ft, **RawGateJoin(RRCScaledBridgeService(), window_samples=w).estimate())
        for ft, w in zip((2, 3, 4), (256, 376, 632), strict=True)
    ]
    for cost, window in zip(costs, (256, 376, 632), strict=True):
        cost["full_history_counterfactual_bits"] = (
            cost["register_bits_upper"] - cost["register_fields"]["trigger_age"] + window
        )
        cost["saved_vs_full_history_bits"] = window - cost["register_fields"]["trigger_age"]
        cost["node_research_register_budget_pass"] = cost["register_bits_upper"] <= 1536
    write(
        args.output / "model.json",
        dict(
            split=args.split,
            value_cases=rows,
            calendars=calendars,
            costs=costs,
            calendar_artifact_SHA256=sha(file),
            exact_word_flag_and_query_prefixes=True,
            FT4_node_research_register_budget_pass=costs[-1]["register_bits_upper"] <= 1536,
            all_numeric_and_resource_failures_retained=True,
            new_RF_holdout=False,
            main_RX_default_changed=False,
            four_RX_gates_passed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                split=args.split,
                value_cases=len(rows),
                calendars=len(calendars),
                node_register_bits=[r["register_bits_upper"] for r in costs],
                resource_failure_cases=sum(not r["continuous_alignment_pass"] for r in calendars),
            )
        )
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
