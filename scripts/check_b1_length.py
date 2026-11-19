"""Replay original SHA-frozen length counterexamples; no RF holdout or hardware claim."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig


def receive(profile, iq, quiet, pattern):
    receiver = MacStreamReceiver(profile, quiet)
    packets, events, pos, j = [], [], 0, 0
    while pos < len(iq):
        n = int(pattern[j % len(pattern)])
        result = receiver.feed(iq[pos:pos + n])
        packets.extend(result.packets)
        events.extend(result.diagnostics["events"])
        pos += n
        j += 1
    result = receiver.finish()
    packets.extend(result.packets)
    events.extend(result.diagnostics["events"])
    return packets, events, receiver.phy


def run(output, expected_rust):
    root = Path(__file__).resolve().parents[1]
    inputs = root / "tests/data/b1-wire-length"
    original = json.loads((inputs / "baseline.json").read_text())
    for row in original["rows"]:
        path = inputs / f"ft{row['frame_type']}-crc{row['CRC_width']}-input.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["input_sha256"]
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, str(root / "scripts/check_receiver_backend.py"),
                    "--expected-rust", expected_rust, "--output", str(output / "backend.json")],
                   check=True, stdout=subprocess.DEVNULL)
    rows = []
    for old in original["rows"]:
        ft, width = old["frame_type"], old["CRC_width"]
        key = f"ft{ft}-crc{width}"
        with np.load(inputs / (key + "-input.npz")) as data:
            cfg = TxConfig(frame_type=ft, mcs_index=8, ctrl_bits_len=28 if ft == 2 else 27,
                           head_llid=0x173925 if ft != 2 else None, crc_len=width,
                           crc_seed=0x123456, pilot_interval=8, symbol_rate_mhz=4)
            profile = PSKReceiveProfile.ordinary(cfg)
            aligned = decode_psk_frame(data["symbols"], profile)
            assert aligned["status"] == "valid" and aligned["data_crc"] is True
            assert aligned["frame_end_symbol"] == old["actual_frame_end"]
            expected = bytes.fromhex(old["expected_payload"])
            assert aligned["payload"] == expected == data["payload"].tobytes()
            patterns = {"whole": (len(data["iq"]),), "fixed": (1, 32, 257),
                        "random": tuple(np.random.default_rng(181391).integers(1, 514, 37))}
            traces, reports = {}, []
            for name, pattern in patterns.items():
                packets, events, rx = receive(profile, data["iq"], data["quiet"], pattern)
                assert [p.mac_payload for p in packets] == [expected] and len(events) == 1
                event = events[0]
                assert event["frame"]["header_crc"] and event["frame"]["data_crc"]
                assert event["timing_backend"] == ("rust" if expected_rust == "yes" else "python")
                for field in ("symbols", "symbol_centers_raw", "timing_rate_ppm",
                              "timing_detector_error"):
                    traces[name + "-" + field] = event[field]
                    np.testing.assert_array_equal(event[field], traces["whole-" + field])
                truth = old["known_transmitter_sync_time_diagnostic_only"] + np.arange(
                    len(event["symbols"])) * 4
                reports.append(dict(chunks=name, statuses=[event["frame"]["status"]],
                                    samples=len(data["iq"]), packet_correct=True, payload_BER=0.,
                                    full_header=event["frame"]["header"],
                                    payload_bytes=event["frame"]["payload_length_bytes"],
                                    frame_end=event["frame"]["frame_end_symbol"],
                                    timing_error_max_samples=float(np.max(abs(
                                        event["symbol_centers_raw"] - truth))),
                                    timing_backend=event["timing_backend"],
                                    search_funnel=rx.capture_stats,
                                    float_frame_capacity_bytes=sum(getattr(rx, n).nbytes for n in (
                                        "_values", "_centers", "_rates", "_errors")),
                                    resource_schedule_drops=None))
            np.savez_compressed(output / (key + "-traces.npz"), **traces)
            rows.append(dict(id=key, original=old, after=reports,
                             parameters=dict(frame_type=ft, data_CRC_bits=width, MCS=8,
                                             head_LLID=cfg.head_llid, PID=0, whitening_seed=0,
                                             data_CRC_seed=0x123456, pilot_interval=8, SPS=4,
                                             symbol_rate_MHz=4, length_contract="standard")))
    negatives = []
    rng = np.random.default_rng(182016)
    iq_noise = (rng.normal(size=32768) + 1j * rng.normal(size=32768)) * .001
    quiet = iq_noise[:1024].copy()
    for name, iq in (("silence", np.zeros(4096, complex)), ("noise", iq_noise),
                     ("tone", .1 * np.exp(2j * np.pi * .081 * np.arange(32768)))):
        packets, events, rx = receive(profile, iq, quiet, (37, 257, 1))
        np.savez_compressed(output / (name + "-input.npz"), iq=iq, quiet=quiet)
        negatives.append(dict(id=name, samples=len(iq), deliveries=len(packets),
                              profile_frame_type=4, data_CRC_bits=32,
                              statuses=[e["frame"]["status"] for e in events],
                              search_funnel=rx.capture_stats))
        assert packets == []
    result = dict(stage="B1 standard length correction", rows=rows, negatives=negatives,
                  baseline_MAC_correct=2, after_MAC_correct=len(rows),
                  old_failures_preserved=True, previously_viewed_fixed_regression=True,
                  new_RF_holdout_evaluation=False, parameters_selected_from_payload_or_CRC=False,
                  within_backend_chunk_tolerance="exact", cross_backend_center_tolerance=1e-10,
                  cross_backend_symbol_tolerance=1e-12, cross_backend_rate_tolerance=1e-8,
                  full_RX_gates_passed=False, full_receiver_16MSps=False,
                  arithmetic_cost_estimate={"once_per_verified_head":
                                            "11-bit zero/CRC-width compare, constant subtraction",
                                            "cycle_model": None, "synthesis": None},
                  source={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*root.glob("src/**/*.py"), Path(__file__).resolve(),
                                    inputs / "baseline.json", root / "uv.lock"]})
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rust", choices=("yes", "no"), required=True)
    args = parser.parse_args()
    result = run(args.output.resolve(), args.expected_rust)
    print(json.dumps({k: result[k] for k in ("baseline_MAC_correct", "after_MAC_correct",
                                             "full_RX_gates_passed", "negatives")}))
