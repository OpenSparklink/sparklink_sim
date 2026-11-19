"""Strict functional word-head regression with independent pulses and frozen policy.

No new acquisition algorithm,soft head,expected-data choice or whole-RX cycle claim.
The transmitter bit encoder is shared;independent standard matrix tests remain separate.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from tests.test_strict_receive import iq_record, transmitted

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.header_word_flow import HeaderWordService

# Predeclared new combinations,not tuned using validation results. Different lengths,
# modulation,CRC,pilot/context,channel and payload compositions across the splits.
CELLS = {
    "development": [
        (2, 8, 8, 29, 32, 24),
        (3, 6, 4, 53, 24, 90),
        (4, 9, 16, 61, 32, 1),
        (2, 10, 0, 79, 24, 7),
        (3, 12, 8, 127, 32, 39),
        (4, 0, 4, 17, 24, 17),
    ],
    "validation": [
        (2, 7, 4, 41, 24, 45),
        (3, 8, 16, 89, 32, 0),
        (4, 12, 8, 73, 24, 21),
        (2, 12, 0, 157, 32, 103),
        (3, 2, 4, 31, 24, 52),
        (4, 6, 16, 47, 32, 69),
    ],
}


def receive(profile, x, quiet, chunks):
    """Same receiver,finite caller blocks;expected data/truth never enter this helper."""
    rx = MacStreamReceiver(profile, quiet)
    packets, events, start, index = [], [], 0, 0
    rng = np.random.default_rng(440117)
    while start < len(x):
        n = (len(x) if chunks == "whole" else (1, 32, 257)[index % 3]
             if chunks == "fixed" else int(rng.integers(1, 514)))
        result = rx.feed(x[start:start + n])
        packets.extend(result.packets)
        events.extend(result.diagnostics["events"])
        start += n
        index += 1
        assert rx.phy.buffered_samples <= (
            rx.phy.history_limit + len(rx.phy.h) - 1 + len(rx.phy._box) - 1 + rx.phy.quantum)
    before = len(packets)
    result = rx.finish()
    packets.extend(result.packets)
    events.extend(result.diagnostics["events"])
    return packets, events, before, rx.phy.capture_stats


def source_identity():
    root = Path(__file__).resolve().parents[1]
    files = sorted(
        {
            *root.glob("src/**/*.py"),
            *root.glob("tests/**/*.py"),
            *root.glob("scripts/*.py"),
            root / "uv.lock",
        }
    )
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def serial(value):
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def run(output, split, freeze=None):
    identity = dict(
        source=source_identity(),
        policy=CELLS,
        numerical=dict(symbol_centers=1e-10, symbols=1e-12, rate_ppm=1e-8),
        adoption="explicit experimental;no default or fullRX promotion",
        pass_rule="identical decisions/bytes and fixed numeric budgets;no CRC-based retry",
        length_contract="standard;old viewed prototype cohorts are fixed regressions only",
    )
    identity = json.loads(json.dumps(identity))
    if split == "validation":
        if freeze is None:
            raise ValueError("Validation requires the preceding source/policy freeze")
        if json.loads(freeze.read_text()) != identity:
            raise ValueError("Frozen header source/policy differs before opening validation inputs")
    output.mkdir(parents=True, exist_ok=False)
    (output / "freeze.json").write_text(json.dumps(identity, indent=2) + "\n")
    rng = np.random.default_rng(2026112201 if split == "development" else 2026112202)
    rows = []
    for index, (ft, mcs, pilot, length, crc, seed) in enumerate(CELLS[split]):
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=0x276513 + seed if ft >= 3 else None,
            pilot_interval=pilot,
            whitening_seed=seed,
            crc_len=crc,
            crc_seed=0x132425 + seed,
            symbol_rate_mhz=4,
        )
        profile = PSKReceiveProfile.ordinary(cfg)
        wire, payload = transmitted(profile, length=length, payload_seed=721000 + seed)
        npre = {2: 40, 3: 48, 4: 64}[ft]
        wire = np.r_[np.exp(1j * np.tile([np.pi / 4, 0], npre // 2)), wire[40:]]
        x, quiet, truth, step = iq_record(
            wire,
            offset=(seed * 0.047) % 4,
            cfo=(-1) ** index * (19000 + 1000 * index),
            ppm=(-1) ** (index + 1) * (17 + index * 11),
        )
        truth += (npre - 40) * step  # Diagnostic only;reference helper uses a40-symbol prefix.
        amplitude = [0.025, 0.07, 0.23, 0.7, 0.04, 0.16][index]
        # Independent reference has an explicit100 gain;use unit-energy RRC units.
        x *= amplitude / 100
        x += (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * amplitude / 30
        if index == 2:
            # Transmitter/channel construction only;no truth is passed to receiver.
            begin = len(x) // 2
            x[begin : begin + 12] += (
                amplitude * 8 * (rng.normal(size=12) + 1j * rng.normal(size=12))
            )
        if index == 0:
            nsync = 32 if ft == 2 else 62 if ft == 3 else 126
            nhead = 32 if ft == 2 else 128 if ft == 3 else 256
            hi = 16 if ft == 2 else 4
            data_start = nsync + nhead + (nhead - int(pilot == 0)) // hi
            payload_middle = (data_start + len(wire) - npre) // 2
            begin = round(truth + step * payload_middle)
            assert truth + step * data_start <= begin < len(x) - 12
            x[begin : begin + 12] += (
                amplitude * 8 * (rng.normal(size=12) + 1j * rng.normal(size=12))
            )
        input_file = output / f"{index}-input.npz"
        np.savez_compressed(
            input_file,
            iq=x,
            quiet=quiet,
            payload=np.frombuffer(payload, dtype=np.uint8),
            diagnostic_truth=np.asarray([truth, step]),
        )
        baseline = None
        for backend in ["reference", "word-dense", "word-depth-parity"]:
            p = replace(profile, head_backend=backend)
            for chunks in ["whole", "fixed", "random"]:
                packets, events, before, statistics = receive(p, x, quiet, chunks)
                frames, traces = [], {}
                for event_index, event in enumerate(events):
                    frame = event["frame"].copy()
                    word = frame.pop("head_word_model", None)
                    frames.append(frame)
                    for key in ["symbols", "symbol_centers_raw", "timing_rate_ppm"]:
                        traces[f"{event_index}-{key}"] = event[key]
                    if word is not None:
                        assert word["demap_service_cycles"] is None
                actual = [packet.mac_payload.hex() for packet in packets]
                if baseline is None:
                    baseline = (frames, actual, traces)
                else:
                    assert frames == baseline[0] and actual == baseline[1]
                    for key, values in traces.items():
                        np.testing.assert_allclose(
                            values,
                            baseline[2][key],
                            rtol=0,
                            atol=identity["numerical"][
                                {
                                    "symbols": "symbols",
                                    "symbol_centers_raw": "symbol_centers",
                                    "timing_rate_ppm": "rate_ppm",
                                }[key.split("-", 1)[1]]
                            ],
                        )
                tag = f"{index}-{backend}-{chunks}"
                np.savez_compressed(output / (tag + "-trace.npz"), **traces)
                ber = None
                if len(packets) == 1 and len(packets[0].mac_payload) == len(payload):
                    error = np.bitwise_xor(
                        np.frombuffer(packets[0].mac_payload, np.uint8),
                        np.frombuffer(payload, np.uint8),
                    )
                    errors = int(np.sum(np.unpackbits(error)))
                    ber = dict(errors=errors, bits=len(payload) * 8)
                report = dict(
                    id=index,
                    backend=backend,
                    chunks=chunks,
                    input_sha256=hashlib.sha256(input_file.read_bytes()).hexdigest(),
                    frames=frames,
                    delivered=actual,
                    expected_payload=payload.hex(),
                    full_correct=actual == [payload.hex()],
                    before_EOF=before,
                    statistics=statistics,
                    timing_error_limitations="truth retained for diagnostic only;not a pass gate",
                    original_IQ_modified=False,
                    BER=ber,
                    BER_limitation="Strict MAC gate does not expose failed decoded bits",
                )
                (output / (tag + ".json")).write_text(
                    json.dumps(report, indent=2, default=serial) + "\n"
                )
                rows.append(report)
    negatives = []
    for kind in ["noise", "tone", "wrong_sync", "burst"]:
        n = 16384
        q = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.001
        x = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.001
        if kind == "tone":
            x += 0.3 * np.exp(2j * np.pi * 0.137 * np.arange(n))
        elif kind == "wrong_sync":
            # FT3/4 use one of six m-sequence indices;FT2 uses a24-bit link ID.
            wrong = replace(profile, config=replace(profile.config,
                                                    pid=(profile.config.pid + 1) % 6))
            assert not np.array_equal(wrong.sync_bits(), profile.sync_bits())
            wire, _ = transmitted(wrong, length=31, payload_seed=722093)
            npre = {2: 40, 3: 48, 4: 64}[wrong.config.frame_type]
            wire = np.r_[np.exp(1j * np.tile([np.pi / 4, 0], npre // 2)), wire[40:]]
            wave = iq_record(wire, offset=.413, cfo=-17000, ppm=23)[0] * .3 / 100
            assert len(wave) < n
            x[:len(wave)] += wave
        elif kind == "burst":
            for begin in range(137, n - 11, 977):
                x[begin:begin + 11] += rng.normal(size=11) + 1j * rng.normal(size=11)
        path = output / (kind + "-negative-input.npz")
        np.savez_compressed(path, iq=x, quiet=q)
        for backend in ["reference", "word-dense", "word-depth-parity"]:
            p = replace(profile, head_backend=backend)
            packets, events, _, statistics = receive(p, x, q, "random")
            assert packets == []
            negatives.append(
                dict(
                    kind=kind,
                    backend=backend,
                    samples=n,
                    events=len(events),
                    delivered=len(packets),
                    statistics=statistics,
                    input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
    costs = {
        backend: HeaderWordService(
            replace(profile, head_backend=backend), layout=backend.removeprefix("word-")
        ).cost()
        for backend in ["word-dense", "word-depth-parity"]
    }
    result = dict(
        split=split,
        rows=len(rows),
        input_cases=6,
        full_by_backend={
            b: sum(r["full_correct"] for r in rows if r["backend"] == b and r["chunks"] == "whole")
            for b in ["reference", "word-dense", "word-depth-parity"]
        },
        all_frames_trajectories_and_deliveries_match=True,
        negative_samples=65536,
        negatives=negatives,
        costs=costs,
        default_changed=False,
        full_RX_gates_passed=False,
        no_Rust_word_implementation=True,
        whole_RX_latency_cycles=None,
        quantitative_RF_gain_not_claimed=True,
    )
    (output / "model.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--split", choices=list(CELLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    result = run(args.output, args.split, args.freeze)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ["split", "full_by_backend", "negative_samples", "full_RX_gates_passed"]
            }
        )
    )


if __name__ == "__main__":
    main()
