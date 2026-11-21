"""Independent word/range checks and bounded RRC→scaled bridge calendars."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tests.test_rrc_bridge import oracle

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, quantize_scaled_words
from nearlink_sdr.phy.rrc_bridge import rrc_to_scaled_words
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.rrc_bridge_flow import RRCScaledBridgeService

POLICY = dict(
    word_identity="exact",
    register_bits_upper=1024,
    output_credits=1,
    input_contexts=1,
    stages=3,
    unstalled_II=4,
    mantissa_bits=10,
    new_whole_system_budget=False,
    new_RF_gain_claim=False,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def source_bindings():
    root = Path(__file__).resolve().parents[1]
    paths = [
        *sorted((root / "src").rglob("*.py")),
        Path(__file__).resolve(),
        root / "tests/test_rrc_bridge.py",
        root / "uv.lock",
    ]
    for function, expected in (
        (oracle, root / "tests/test_rrc_bridge.py"),
        (rrc_to_scaled_words, root / "src/nearlink_sdr/phy/rrc_bridge.py"),
        (RRCScaledBridgeService.step, root / "src/nearlink_sdr/sim/rrc_bridge_flow.py"),
    ):
        if Path(function.__code__.co_filename).resolve() != expected:
            raise ValueError("Loaded node/oracle differs from the frozen source root")
    return {str(p.relative_to(root)): sha(p) for p in paths}


def inputs(split):
    """Different value families/lengths,not a seed-only validation split."""
    n = 257 if split == "development" else 383
    k = np.arange(n)
    if split == "development":
        return [
            np.zeros(n, complex),
            np.exp(1j * k * k * 0.003) * 0.007,
            np.where(k % 2, 1e7 + 2e7j, -1e7 - 2e7j),
            np.where((k >= 101) & (k < 106), 2e6 + 3e6j, 1e-6 + 2e-6j),
            np.full(n, 1e-18 + 2e-18j),
            np.full(n, 2**49 + 1j * 2**50),
            np.exp(1j * k * 0.13) * np.where(k < n // 2, 1e8, 1e-12),
            (k - 127) * (1 + 2j) * 0.03125,
        ]
    return [
        np.where(k % 17 == 3, 0.125j, np.where(k % 19 == 7, -0.0625, 0)),
        np.take([0.03125 + 0.5j, -0.25 + 0.125j, 0.0625 - 0.25j], k % 3),
        np.cos(k * 0.031) * np.exp(1j * k * 0.291) * (k + 1) * 0.0007,
        1j * np.where(k < 127, 1e-21, 2**52),
        np.where(k < 191, 0j, -1e-17 + 0.03125j),
        np.take([1j, -1.0, -1j, 1.0], (k // 7) % 4) * np.where(k % 31 < 13, 3e8, 1e-11),
        (np.sin(k * 0.077) + 1j * np.cos(k * 0.131)) * np.where(k < 129, 1e-9, 2.5),
        np.where(k % 23 == 0, -(2**48) + 1j * 2**47, 0.007 - 0.003j),
    ]


def filter_words(values):
    model = FullMantissaRRC(FixedRRCFormat(12, 18))
    groups = [model.feed_output_words(values[j : j + 32]) for j in range(0, len(values), 32)]
    return tuple(np.concatenate([row[k] for row in groups]) for k in range(3)), model.stats


def value_case(values, out, ident):
    words, source_stats = filter_words(values)
    real, imag, exponents = words
    exact_values = np.ldexp(real.astype(float), exponents) + 1j * np.ldexp(
        imag.astype(float), exponents
    )
    old_stats, new_stats, chunks = {}, {}, []
    fmt = ScaledBankFormat(POLICY["mantissa_bits"])
    old = quantize_scaled_words(exact_values, fmt, stats=old_stats)
    for j in range(0, len(values), 32):
        chunks.append(rrc_to_scaled_words(*(v[j : j + 32] for v in words), fmt, stats=new_stats))
    new = tuple(np.concatenate([a[k] for a in chunks]) for k in range(3))
    assert old_stats == new_stats
    for a, b in zip(old, new, strict=True):
        np.testing.assert_array_equal(a, b)
    quantized = np.ldexp(new[0].astype(float), new[2]) + 1j * np.ldexp(new[1].astype(float), new[2])
    energy = float(np.vdot(exact_values, exact_values).real)
    relative_RMS = (
        float(np.linalg.norm(quantized - exact_values) / np.sqrt(energy)) if energy else None
    )
    node = RRCScaledBridgeService(fmt)
    for tag, triple in enumerate(zip(*words, strict=True)):
        triple = tuple(map(int, triple))
        assert node.issue(pack_filter_word(*triple), tag)
        for _ in range(3):
            node.step()
        word, events = oracle(*triple, fmt.mantissa_bits)
        assert node.result == dict(tag=tag, word=word, events=events)
        if tag % 7 == 0:
            node.step()  # Held output,not a free extra credit.
            assert node.result == dict(tag=tag, word=word, events=events)
        assert node.take() is not None
        node.step()
    path = out / f"value-{ident}.npz"
    np.savez_compressed(
        path,
        input=values,
        real=real,
        imag=imag,
        exponent=exponents,
        output_real=new[0],
        output_imag=new[1],
        output_exponent=new[2],
    )
    source_events = {
        k: source_stats[k]
        for k in (
            "input_saturations",
            "exponent_underflows",
            "exponent_overflows",
            "output_saturations",
            "output_underflows",
            "output_overflows",
            "intermediate_overflows",
        )
    }
    return dict(
        id=ident,
        samples=len(values),
        value_artifact_SHA256=sha(path),
        input_array_SHA256=hashlib.sha256(values.tobytes()).hexdigest(),
        input_dtype=str(values.dtype),
        input_shape=list(values.shape),
        packed_RRC_word_SHA256=hashlib.sha256(
            np.asarray(
                [pack_filter_word(*(int(v) for v in row)) for row in zip(*words, strict=True)],
                dtype="<u8",
            ).tobytes()
        ).hexdigest(),
        packed_RRC_encoding="46valid bits,padded to little-endianu64",
        word_identity=True,
        source_statistics=source_stats,
        source_error_budget_pass=None,
        source_range_events=source_events,
        source_range_pass=not any(source_events.values()),
        destination_range_events=new_stats,
        destination_range_pass=not any(new_stats.values()),
        RRC_to_scaled_relative_RMS_error=relative_RMS,
        incremental_word_error_vs_old_adapter=0,
        actual_node_statistics=node.stats,
    )


def calendar(clock, blocked, words):
    producer = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=clock * 1_000_000)).simulate(
        len(words[0]), trace=True
    )
    arrivals = {e["cycle"]: e for e in producer["trace"] if e["event"] == "output"}
    node, accepted, emitted, dropped, invalid = RRCScaledBridgeService(), [], [], [], []
    while node.cycle <= max(producer["cycles"], 1000 if blocked else 0) + 8:
        cycle = node.cycle
        if node.ready and not (blocked and 200 <= cycle < 1000):
            row = node.take()
            triple = tuple(int(v[row["tag"]]) for v in words)
            word, events = oracle(*triple, 10)
            assert row == dict(tag=row["tag"], word=word, events=events)
            emitted.append([row["tag"], cycle])
        if cycle in arrivals:
            event = arrivals[cycle]
            tag = event["sample"]
            if event["invalid"]:
                invalid.append([tag, cycle])
            else:
                word = pack_filter_word(*(int(v[tag]) for v in words))
                (accepted if node.issue(word, tag) else dropped).append([tag, cycle])
        node.step()
    assert [r[0] for r in accepted] == [r[0] for r in emitted]
    assert len(accepted) + len(dropped) + len(invalid) == producer["emitted"]
    return dict(
        clock_MHz=clock,
        output_blocked=blocked,
        producer={k: v for k, v in producer.items() if k != "trace"},
        accepted=accepted,
        emitted=emitted,
        bridge_credit_drops=dropped,
        invalid_RRC_outputs=invalid,
        peak_bridge_contexts=1,
        RRC_pops_continue_without_backpressure=True,
        raw_energy_alignment_cycles=None,
        scaled_bank_cycles=None,
        full_receiver16MSps=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    sources = source_bindings()
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen development source,policy and adoption rules")
        frozen = json.loads(args.freeze.read_text())
        if frozen != dict(policy=POLICY, sources=sources):
            raise ValueError("Source or policy changed after freeze")
    args.output.mkdir(exist_ok=False, parents=True)
    rows = [value_case(v, args.output, i) for i, v in enumerate(inputs(args.split))]
    k = np.arange(1024)
    words, _ = filter_words(np.exp(1j * k * 0.071) * np.take([1e-6, 1.0, 1e-9, 3.0], k // 256))
    calendars = [calendar(c, b, words) for c in (100, 160, 200) for b in (False, True)]
    cost = RRCScaledBridgeService().estimate()
    assert cost["register_bits_upper"] <= POLICY["register_bits_upper"]
    write(args.output / "freeze.json", dict(policy=POLICY, sources=sources))
    write(
        args.output / "model.json",
        dict(
            split=args.split,
            value_cases=rows,
            calendars=calendars,
            cost=cost,
            component_word_pass=all(r["word_identity"] for r in rows),
            component_register_budget_pass=True,
            source_and_destination_range_failures_retained=True,
            new_RF_holdout=False,
            synthetic_value_validation=True,
            four_RX_gates_passed=False,
        ),
    )
    print(
        json.dumps(
            dict(
                split=args.split,
                exact_value_cases=len(rows),
                range_failure_cases=sum(
                    not r["source_range_pass"] or not r["destination_range_pass"] for r in rows
                ),
                calendar_cases=len(calendars),
                register_bits=cost["register_bits_upper"],
                four_RX_gates_passed=False,
            )
        )
    )
    return 1  # Range/resource and unmodeled whole-chain work remain explicit.


if __name__ == "__main__":
    raise SystemExit(main())
