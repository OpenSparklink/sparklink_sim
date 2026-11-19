"""Actual finite geometry words and scalar CRC-gated handoff;no full-RX cycle credit."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL
from nearlink_sdr.phy.psk_fields import psk_field_layout
from nearlink_sdr.phy.rx_pipeline import payload_coded_length
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.header_geometry_flow import HeaderGeometryService

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "tests/data/header_geometry_policy.json"


def complete(service, length, mcs, limit):
    begin = service.cycle
    assert service.issue(length, mcs)
    for _ in range(limit):
        if service.ready:
            break
        service.step()
    assert service.ready, "Actual finite geometry failed to drain"
    return service.take(), service.cycle - begin


def run(output):
    policy = json.loads(POLICY.read_text())
    assert policy["output_integer_tolerance"] == 0 and not policy["old_receiver_budgets_changed"]
    output.mkdir(parents=True, exist_ok=False)
    rows, arrays = [], {}
    for ft in policy["frame_types"]:
        for crc in policy["data_CRC_bits"]:
            for pilot in policy["pilot_intervals"]:
                contracts = ["standard"] if ft == 2 else ["standard", "legacy-b1-payload-length"]
                for contract in contracts:
                    node_peaks = {}
                    def observe(event):
                        if event["event"] == "register":
                            name, value = event["name"], event["value"]
                        elif event["event"] == "engine":
                            name, value = "engine-" + event["opcode"], event["value"]
                        else:
                            return
                        old = node_peaks.get(name, (value, value))
                        node_peaks[name] = (min(old[0], value), max(old[1], value))
                    service = HeaderGeometryService(ft, crc, pilot, length_contract=contract,
                                                    observer=observe)
                    maximum, peak_registers, peaks, samples = 0, 0, {}, []
                    for mcs in policy["MCS"]:
                        cfg = TxConfig(frame_type=ft, mcs_index=mcs, crc_len=crc)
                        lengths = ([0, crc // 8, crc // 8 + 1, 37, 127, 511, 2047]
                                   if ft != 2 and contract == "standard" else
                                   [0, 1, 31, 37, 127, 511, 2047])
                        for length in lengths:
                            result, latency = complete(service, length, mcs,
                                                       policy["max_drain_edges"])
                            present = not (ft != 2 and contract == "standard" and length == 0)
                            information = (max(0, length - crc // 8)
                                           if ft != 2 and contract == "standard" else length)
                            coded = payload_coded_length(information * 8, cfg) if present else 0
                            positions, wire = psk_field_layout(
                                coded, cfg._mcs.bits_per_symbol, pilot)
                            assert result["coded_bits"] == coded
                            assert result["data_symbols"] == len(positions)
                            assert result["wire_data_symbols"] == wire
                            assert result["frame_end_symbol"] == result["data_start_symbol"] + wire
                            assert result["payload_length_bytes"] == information
                            assert result["data_crc_present"] is present
                            maximum = max(maximum, latency)
                            peak_registers = max(peak_registers, len(service.registers))
                            for name, value in service.registers.items():
                                old = peaks.get(name, (value, value))
                                peaks[name] = (min(old[0], value), max(old[1], value))
                            samples.append([mcs, length, latency, information, int(present),
                                            coded, len(positions), wire,
                                            result["frame_end_symbol"]])
                    tag = f"ft{ft}-crc{crc}-pilot{pilot}-{contract}"
                    arrays[tag] = np.asarray(samples, dtype=np.int32)
                    assert peak_registers <= policy["register_slots"]
                    assert len(service.program) <= policy["max_program_instructions"]
                    cost = service.cost()
                    assert cost["DSP"] == policy["DSP"] and cost["RAMB18"] == policy["RAMB18"]
                    rows.append(dict(id=tag, scalar_cases=len(samples), latency_max_edges=maximum,
                                     max_active_registers=peak_registers,
                                     observed_final_ranges=peaks,
                                     observed_write_and_engine_ranges=node_peaks,
                                     cost=cost, statistics=service.stats,
                                     clock_sensitivity=[dict(MHz=hz, service_max_us=maximum / hz,
                                                             placement_timing_proven=False)
                                                        for hz in policy["clock_sensitivity_MHz"]]))
    np.savez_compressed(output / "all-scalar-words.npz", **arrays)
    original = json.loads((ROOT / "tests/data/b1-wire-length/baseline.json").read_text())
    handoff = []
    for old in original["rows"]:
        ft, crc = old["frame_type"], old["CRC_width"]
        path = ROOT / f"tests/data/b1-wire-length/ft{ft}-crc{crc}-input.npz"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == old["input_sha256"]
        cfg = TxConfig(frame_type=ft, mcs_index=8, crc_len=crc, symbol_rate_mhz=4,
                       ctrl_bits_len=28 if ft == 2 else 27, head_llid=0x173925 if ft != 2 else None,
                       pilot_interval=8, crc_seed=0x123456)
        with np.load(path) as data:
            reference = decode_psk_frame(data["symbols"], PSKReceiveProfile.ordinary(cfg))
        assert reference["header_crc"] is True
        events = []
        service = HeaderGeometryService(ft, crc, 8, observer=events.append)
        result, latency = complete(service, reference["header"]["data_length"], reference["mcs"],
                                   policy["max_drain_edges"])
        assert result["frame_end_symbol"] == old["actual_frame_end"]
        for name, value in result.items():
            assert reference[name] == value
        (output / f"ft{ft}-crc{crc}-calendar.json").write_text(json.dumps(events) + "\n")
        handoff.append(dict(input_SHA256=old["input_sha256"], result=result, edges=latency,
                            expectation_used_for_receiver_selection=False,
                            head_CRC_service_cycles=None, live_stream_handoff=False))
    report = dict(stage=policy["stage"], policy=policy, actual_rust_reference=_HAS_RUST_ACCEL,
                  rows=rows, original_length_handoff=handoff, word_errors=0,
                  new_RF_holdout=False, complete_RX_gates=False, receiver16MSps=False,
                  adopted_as_default=False, previous_failures_retained=True,
                  operation_and_capacity_estimates_only=True, actual_synthesis=False,
                  source={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), POLICY,
                                    ROOT / "src/nearlink_sdr/sim/header_geometry_flow.py")})
    (output / "model.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.output.resolve())
    print(json.dumps(dict(configurations=len(report["rows"]), word_errors=report["word_errors"],
                          original_handoffs=len(report["original_length_handoff"]),
                          receiver16MSps=False)))
