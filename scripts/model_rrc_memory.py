"""Replay actual RRC access calendars into finite LUTRAM images;not RX acceptance."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write

from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.sim.rrc_memory import RRCMemory


def input_words(samples):
    folder = Path(__file__).resolve().parents[1]/"tests/data"
    path = folder/"sync_carrier_tone_success.npz"
    meta = json.loads((folder/"sync_carrier_tone_success.json").read_text())
    assert sha(path) == meta["derived_npz_sha256"]
    with np.load(path) as archive:
        x = archive["iq"][:samples]
    fir = FullMantissaRRC(FixedRRCFormat(12, 18))
    fields = [fir._quantize(x[i:i+32]) for i in range(0, len(x), 32)]
    r, i, e = [np.concatenate([row[k] for row in fields]) for k in range(3)]
    words = ((r & 4095) | ((i & 4095) << 12) | ((e & 255) << 24)).astype("<u4")
    return words, dict(fixture_sha256=sha(path), original_input_sha256=meta["source_input"]["sha256"],
                       words_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
                       scope="actual full12 input adapter words;not new RF/ADC calibration")


def product_word(owner, j):
    """Independent synthetic69-bit producer;memory must preserve any bit pattern."""
    return (owner*0x7165A7B8394C27ABCDEF+j*0x6A45B193704E4F1) % 2**69


def replay(a, mode, words, stalls):
    errors, pending, expired_owners = [], [], set()
    memory = RRCMemory(mode, a.product_buffers, diagnostic_sink=errors.append)
    digest = hashlib.sha256()
    checked_raw = checked_product = 0
    current_count = 0

    def observer(event):
        nonlocal pending, current_count, checked_raw, checked_product
        n = event["source_write"]
        if n is not None:
            current_count = n+1
        result = memory.tick(event["cycle"],
            source_write=None if n is None else (n, int(words[n])),
            raw_reads=event["raw_reads"],
            product_writes=[(slot, j, product_word(owner, j))
                            for slot, j, owner in event["product_writes"]],
            product_reads=[(slot, j) for slot, j, _ in event["product_reads"]])
        assert result["products"] == pending
        checked_product += len(pending)
        pending = [(slot, j, product_word(owner, j)) for slot, j, owner in event["product_reads"]]
        for index, owner, value in zip(event["raw_reads"], event["raw_samples"],
                                       result["raw"], strict=True):
            if index >= 0 and index < current_count-128:
                assert value is None
                expired_owners.add(owner)
            elif value is not None:
                assert value == (0 if index < 0 else int(words[index]))
                checked_raw += 1
            else:
                assert mode == "dual" and memory.stats["raw_conflict_banks"] > 0
        digest.update(json.dumps(dict(access=event, response=result),
                                 sort_keys=True, separators=(",", ":")).encode())

    flow = PairRRCFlow(a)
    baseline = flow.simulate(len(words), downstream_stalls=stalls, trace=True)
    result = flow.simulate(len(words), downstream_stalls=stalls, trace=True, access_sink=observer)
    assert result == baseline and not pending and not memory.pending
    assert memory.stats["product_hazards"] == 0
    assert len(expired_owners) == result["invalid"]
    outcomes = [r for r in result.pop("trace")
                if r["event"] in ("output", "drop_new", "history_gap")]
    estimate = memory.estimate()
    return dict(assumptions=asdict(a), stalls=stalls, result=result,
        memory=memory.stats, estimate=estimate, diagnostics=errors, outcomes=outcomes,
        checked_raw_responses=checked_raw, checked_product_responses=checked_product,
        access_response_sha256=digest.hexdigest(),
        old_schedule_exact=True,
        port_candidate_valid=not(memory.stats["raw_conflict_banks"] or memory.stats["product_hazards"]),
        memory_lut_study_ceiling=1200,
        memory_luts_within_study_budget=estimate["raw_memory_luts"]+estimate["product_memory_luts"] <= 1200,
        rrc_numeric_gate=False, whole_receiver_accepted=False,
        new_signal_gain=None, product_arithmetic_integrated=False)


def run(output, samples=4096):
    if type(samples) is not int or not 512 <= samples <= 8192:
        raise ValueError("Need512..8192source samples inside the frozen fixture")
    output.mkdir(parents=True, exist_ok=False)
    words, identity = input_words(samples)
    np.save(output/"rrc-input-u32.npy", words)
    rows = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        for stalled in (False, True):
            for mode in ("copies", "dual", "quad"):
                a = PairRRCFlowAssumptions(clock_hz=clock)
                row = replay(a, mode, words, [(100, 4100)] if stalled else [])
                name = f"{clock}-{mode}-{int(stalled)}.json"
                write(output/name, row)
                rows.append(dict(file=name, sha256=sha(output/name), clock_hz=clock,
                    stalled=stalled, mode=mode, **{k: v for k, v in row.items()
                        if k not in ("diagnostics", "outcomes", "assumptions", "stalls")}))
    variations = [dict(arithmetic_ii=2), dict(coefficient_ports=3), dict(product_buffers=2),
                  dict(product_read_ports=1), dict(consumer_ii=14)]
    for index, change in enumerate(variations):
        row = replay(replace(PairRRCFlowAssumptions(), **change), "quad", words, [])
        name = f"quad-variation-{index}.json"
        write(output/name, row)
        rows.append(dict(file=name, sha256=sha(output/name), variation=change, mode="quad",
                         **{k: v for k, v in row.items() if k not in ("diagnostics", "outcomes")}))
    report = dict(scope="existing word-full12/C18/SPS4/81tap RRC memory mapping only",
        input=identity, samples=len(words), scenarios=rows,
        product_data_source="independent synthetic69bit producer;not recalculated FIR products",
        frozen_original_budget_bits=131072, original_budget_still_failed=True,
        known_capture_information_lower_with_controls_bits=148844+1280,
        known_partial_with_lutram_cells_bits=148844+1280+32768+35328-(4096+3*44*69),
        system_memory_projection=dict(baseline=73, rrc_raw_and_product_bram=0,
            capture_transport=10, independent_coherent_rom=1, five_packet_slots=5,
            subtotal=89, unapproved_reserve25=35, residual_after_proposed_reserve=16,
            excludes=["coefficient ROM/routing", "coarse bank port mapping", "fine spline state/ports",
                      "packet timing/carrier", "Polar", "TX", "realtime MAC/DMA/AES"]),
        no_credit="unimplemented packing/routing/placement/timing;no hardware or full16MS/s claim",
        whole_system_accepted=False, synthesis_evidence=False)
    write(output/"model.json", report)
    sources = [Path(__file__), Path("pyproject.toml"), Path("uv.lock"), *Path("src").rglob("*.py")]
    write(output/"sources.json", {str(p): sha(p) for p in sources})
    (output/"driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(output/"manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(scenarios=len(rows),
        quad_port_conflicts=sum(r["memory"]["raw_conflict_banks"] for r in rows if r["mode"] == "quad"),
        whole_system_accepted=False), indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.samples))
