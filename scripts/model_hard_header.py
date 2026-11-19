"""Actual normalized words→registered hard head→real CRC/geometry component.

Whole-record front-end search and payload continuation remain reference boundaries.
Neither a successful node nor this finite calendar proves complete RX16MS/s.
"""

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from model_capture_fine import bridge
from model_header_words import source_identity
from model_normalization_words import execute
from tests.test_psk_hard_word import projection_reference
from tests.test_strict_receive import iq_record, transmitted

from nearlink_sdr.common import polar
from nearlink_sdr.phy.normalization_word import (
    NormalizationFormat,
    pack_normalized_word,
    physical_normalized_word,
)
from nearlink_sdr.phy.psk import PSKDemodulator, rrc_filter
from nearlink_sdr.phy.psk_fields import demap_psk_field
from nearlink_sdr.phy.psk_hard_word import hard_psk_word
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_word import physical_filter_word
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, _decode_psk_payload
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.normalized_word_replay import NormalizedWordFrameReplay
from nearlink_sdr.sim.psk_hard_flow import HardPSKWordService, HeaderFromWordsService

POLICY = Path("tests/data/hard_header_policy.json")
KEYS = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")


def plain(value):
    return json.loads(
        json.dumps(
            value,
            default=lambda v: (
                v.hex()
                if isinstance(v, bytes)
                else v.tolist()
                if isinstance(v, np.ndarray)
                else v.item()
            ),
        )
    )


def arithmetic(output, split):
    """Independent enumeration covers full mantissa/exponent domains without float."""
    rng = np.random.default_rng(265021 if split == "development" else 265022)
    edge = [-(1 << 21), -(1 << 21) + 1, -1, 0, 1, (1 << 21) - 1]
    pairs = [(r, i) for r in edge for i in edge]
    pairs += [tuple(map(int, row)) for row in rng.integers(-(1 << 21), 1 << 21, (384, 2))]
    records, tie_differences, non_tie_errors = [], 0, 0
    for kind in ("BPSK", "QPSK"):
        demap = PSKDemodulator(kind)
        for index in (0, 1):
            for real, imag in pairs:
                expected = projection_reference(real, imag, kind, index)
                for exponent in (-1024, -21, 0, 1023):
                    word = pack_normalized_word(real, imag, exponent)
                    row = hard_psk_word(word, kind, index)
                    assert (row["label"], row["bits"], row["mathematical_tie"]) == expected
                # A separate well-scaled float comparison; no enormous exponent adapter.
                floating = tuple(
                    map(int, demap.demap_symbols(np.array([complex(real, imag) * 2**-21]), index))
                )
                different = row["bits"] != floating
                tie_differences += int(different and row["mathematical_tie"])
                non_tie_errors += int(different and not row["mathematical_tie"])
                records.append(
                    dict(
                        real=real,
                        imag=imag,
                        kind=kind,
                        index=index,
                        expected=expected,
                        float_bits=floating,
                        actual=row,
                    )
                )
    write(output / "integer-projection-vectors.json", records)
    result = dict(
        scalar_cases=len(records),
        exponent_cases=4 * len(records),
        integer_reference_bit_errors=0,
        float_non_tie_bit_errors=non_tie_errors,
        float_mathematical_tie_differences=tie_differences,
        reference="independent exhaustive labeled integer dot products",
    )
    write(output / "arithmetic.json", result)
    return result


def header_calendar(profile, records, *, layout, pattern):
    """Replay actual registered word availability; one credit, no input queue.

    Captured records are host-only evidence. The modeled owner supplies one held
    word with its absolute index. No payload/sync word is stored by this component.
    """
    trace = []
    flow = HeaderFromWordsService(profile, layout=layout, observer=trace.append)
    flow.begin()
    selected = [r for r in records if flow.start <= r["values"]["index"] < flow.end]
    origin = selected[0]["taken_edge"] if selected else 0
    position = 0
    available = 0
    group = 0
    for _ in range(1_000_000):
        if position == available and available < len(selected):
            available = min(len(selected), available + pattern[group % len(pattern)])
            group += 1
        if position < available and flow.word_input_ready:
            record = selected[position]
            if flow.cycle >= record["taken_edge"] - origin:
                assert flow.offer(record["values"]["normalized_word"], record["values"]["index"])
                position += 1
        if position == len(selected) and not flow.eof:
            flow.end_input()
        if flow.ready:
            return flow.take(), flow, trace, origin
        flow.step()
    raise RuntimeError("Finite head component exceeded its host watchdog")


def packet(output, name, profile, iq, payload, pattern):
    filtered = np.convolve(iq, rrc_filter(0.4, 4))
    words, quantization = bridge(filtered)
    input_path = output / (name + "-input.npz")
    np.savez_compressed(
        input_path,
        iq=iq,
        filtered=filtered,
        words=words,
        payload=np.frombuffer(payload or b"", np.uint8),
    )
    row = dict(id=name, input_sha256=sha(input_path), quantization=quantization)
    # This fixture's upstream whole-record differential maximum is unchanged.
    found = search_psk_sync(
        np.array([physical_filter_word(w) for w in words]), 4, profile.sync_symbols(), 8
    )
    row["candidate"] = found is not None
    if found is None:
        return row
    selection = dict(
        start_tick=round(8 * (found["phase"] + 4 * found["offset"])),
        delta=found["delta"],
        differential_score=found["score"],
        coherent_threshold=0.45,
        sample_limit=len(words),
        selected_sync=found["phase_symbols"][
            found["offset"] : found["offset"] + len(profile.sync_symbols())
        ],
    )
    np.savez_compressed(output / (name + "-selection.npz"), **selection)
    service_records = []
    event, _owner, _obs = execute(
        output,
        name + "-actual-normalization",
        profile,
        words,
        selection,
        TimingWordFormat(30, 32),
        pattern,
        flow_class=partial(
            NormalizedWordFrameReplay,
            normalization_format=NormalizationFormat(),
            service_observer=service_records.append,
        ),
    )
    records = [r for r in service_records if r["kind"] == "normalization"]
    write(output / (name + "-actual-normalization-services.json"), records)
    np.savez_compressed(
        output / (name + "-normalized-words.npz"),
        words=np.asarray([r["values"]["normalized_word"] for r in records], dtype=np.uint64),
    )
    expected = event["frame"]
    row["reference"] = {k: expected.get(k) for k in KEYS}
    row["reference_payload_exact"] = expected.get("payload") == payload
    row["header_paths"] = []
    for layout in ("dense", "depth-parity"):
        result, service, trace, origin = header_calendar(
            profile, records, layout=layout, pattern=pattern
        )
        write(output / (name + "-" + layout + "-calendar.json"), trace)
        write(output / (name + "-" + layout + "-header.json"), result)
        head_words = [
            r["values"]["normalized_word"]
            for r in records
            if service.start <= r["values"]["index"] < service.end
        ]
        floating_bits = demap_psk_field(
            np.asarray([physical_normalized_word(w) for w in head_words]),
            service.modulation,
            service.start,
            service.interval,
        )
        # Separate bit comparison, not CRC-based candidate/path selection.
        bits_exact = tuple(service.bits[: service.bit_count]) == tuple(map(int, floating_bits))
        tie_words = sum(
            hard_psk_word(w, service.modulation, service.start + i)["mathematical_tie"]
            for i, w in enumerate(head_words)
        )
        if result["status"] == "header_valid":
            decoded = _decode_psk_payload(event["symbols"], profile, result)
        else:
            decoded = result
        final = {k: decoded.get(k) for k in KEYS}
        reference = {k: expected.get(k) for k in KEYS}
        full = decoded.get("payload") == payload and decoded.get("status") == "valid"
        row["header_paths"].append(
            dict(
                layout=layout,
                head_bits_exact=bits_exact,
                mathematical_tie_words=tie_words,
                final=final,
                final_reference_exact=final == reference,
                payload_exact=full,
                success_loss=row["reference_payload_exact"] and not full,
                new_false_delivery=decoded.get("status") == "valid"
                and not full
                and not (expected.get("status") == "valid" and not row["reference_payload_exact"]),
                cycles=service.cycle,
                last_word_arrival_cycle=(
                    records[service.end - 1]["taken_edge"] - origin
                    if len(records) >= service.end
                    else None
                ),
                head_assembly_and_decode_cycles_are_actual=True,
                availability_origin=origin,
                statistics=service.stats.copy(),
                header_statistics=service.header.stats.copy(),
                cost=service.cost(),
                full_pipeline_latency=None,
                input_word_requantized=False,
            )
        )
    return plain(row)


def run(output, split, freeze=None):
    policy = json.loads(POLICY.read_text())
    identity = dict(source=source_identity(), policy=policy)
    if split == "validation" and (freeze is None or json.loads(freeze.read_text()) != identity):
        raise ValueError("Frozen hard head source/policy differs before opening validation inputs")
    output.mkdir(parents=True, exist_ok=False)
    write(output / "freeze.json", identity)
    arithmetic_row = arithmetic(output, split)
    rng = np.random.default_rng(265023 if split == "development" else 265024)
    rows = []
    for j, (ft, mcs, pilot, length, crc, seed) in enumerate(policy[split]):
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pid=j % 6,
            pilot_interval=pilot,
            crc_len=crc,
            crc_seed=0x132425 + seed,
            whitening_seed=seed,
            ctrl_bits_len=28 if ft == 2 else 27,
            symbol_rate_mhz=4,
            head_llid=0x276513 + seed if ft >= 3 else None,
        )
        profile = PSKReceiveProfile.ordinary(cfg)
        wire, payload = transmitted(profile, length=length, payload_seed=741000 + seed)
        npre = {2: 40, 3: 48, 4: 64}[ft]
        wire = np.r_[np.exp(1j * np.tile([np.pi / 4, 0], npre // 2)), wire[40:]]
        iq, _quiet, truth, step = iq_record(
            wire,
            offset=(seed * 0.039) % 4,
            ppm=(-1) ** j * (23 + 7 * j),
            cfo=(-1) ** (j + 1) * (17000 + 1300 * j),
        )
        truth += (npre - 40) * step
        amplitude = [0.05, 0.11, 0.42, 0.75, 0.085, 0.31][j]
        iq *= amplitude
        iq += (rng.normal(size=len(iq)) + 1j * rng.normal(size=len(iq))) * amplitude * 0.7
        prefix = 101 + 23 * j
        iq = np.r_[np.zeros(prefix, complex), iq, np.zeros(91 + 11 * j, complex)]
        t = np.arange(len(iq))
        if j == 1:
            iq += 0.2 * amplitude * np.exp(2j * np.pi * 0.08371 * t)
        elif j == 2:
            begin = len(iq) // 2
            iq[begin : begin + 9] += 80 * amplitude * np.exp(1j * rng.uniform(-np.pi, np.pi, 9))
        elif j == 3:
            iq[len(iq) // 2 :] *= 0.8
        elif j == 4:
            # Different legitimate sync template is interference, never a reference answer.
            other = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, pid=7, symbol_rate_mhz=4))
            wrong, _ = transmitted(other, length=19, payload_seed=771)
            z, _, _, _ = iq_record(wrong, cfo=7000, offset=0.619)
            iq += 0.002 * np.resize(z, len(iq))
        pattern = (
            [32]
            if j % 3 == 0
            else [1, 7, 31]
            if j % 3 == 1
            else list(map(int, rng.integers(1, 33, 19)))
        )
        row = packet(output, str(j), profile, iq, payload, pattern)
        row.update(
            frame_type=ft,
            mcs=mcs,
            pilot=pilot,
            length=length,
            CRC_width=crc,
            whitening_seed=seed,
            truth_used_for_selection=False,
            diagnostic_truth_start_filtered=truth + prefix + 40,
            diagnostic_truth_step=step,
            input_pattern=pattern,
        )
        rows.append(row)
        write(output / "rows.json", rows)
    # Front-end fixtures only: never multiply counts by layouts/backends/replays.
    negatives = []
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    for kind in ("noise", "tone", "wrong_sync", "burst"):
        n = policy["negative_samples_each_class"]
        noise = rng.normal(size=n) + 1j * rng.normal(size=n)
        t = np.arange(n)
        if kind == "tone":
            noise += 6 * np.exp(2j * np.pi * 0.11379 * t)
        elif kind == "burst":
            noise[n // 2 : n // 2 + 11] += 90
        elif kind == "wrong_sync":
            other = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, pid=7, symbol_rate_mhz=4))
            wire, _ = transmitted(other, length=31, payload_seed=741131)
            wrong, _, _, _ = iq_record(wire, cfo=9000, offset=0.73)
            noise += np.resize(wrong, n)
        found = search_psk_sync(np.convolve(noise, rrc_filter(0.4, 4)), 4, p.sync_symbols(), 8)
        file = output / ("negative-" + kind + ".npz")
        np.savez_compressed(file, iq=noise)
        negatives.append(
            dict(
                kind=kind,
                samples=n,
                input_sha256=sha(file),
                differential_candidate=found is not None,
                score=None if found is None else found["score"],
                MAC_false_deliveries=None,
                MAC_not_executed=True,
            )
        )
    write(output / "negative-front-end-fixtures.json", negatives)
    paths = [p for r in rows for p in r.get("header_paths", [])]
    passed = bool(paths) and arithmetic_row["float_non_tie_bit_errors"] == 0 and all(
        p["head_bits_exact"]
        and p["final_reference_exact"]
        and not p["success_loss"]
        and not p["new_false_delivery"]
        and p["cost"]["extra_candidate_capacity_met"]
        for p in paths
    )
    result = dict(
        split=split,
        actual_reference_rust=polar._HAS_RUST_ACCEL,
        arithmetic=arithmetic_row,
        input_packets=len(rows),
        captured=sum(r["candidate"] for r in rows),
        reference_full=sum(r.get("reference_payload_exact", False) for r in rows),
        component_checks=len(paths),
        component_pass=passed,
        success_losses=sum(p["success_loss"] for p in paths),
        unknown_live_capture_or_full_RX_acceptance=False,
        whole_RX_gates={k: False for k in ("streaming", "fixed", "resources", "16MSps")},
        negative_independent_samples=4 * policy["negative_samples_each_class"],
        hard_service_cost=HardPSKWordService.cost(),
    )
    write(output / "result.json", result)
    print(json.dumps(result))
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--freeze", type=Path)
    a = parser.parse_args()
    raise SystemExit(0 if run(a.output, a.split, a.freeze) else 1)
