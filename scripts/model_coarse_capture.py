"""Real RRC/coarse/window/fine calendar;not a complete packet receiver."""

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scripts import model_coarse_ingress as ingress
from scripts import model_spline_gauss as shared
from tests.maximum_reference import channel, packet

from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from nearlink_sdr.sim.coarse_ingress import CoarseIngress
from nearlink_sdr.sim.spline_score import SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions

POLICY = dict(clock_MHz=200, lanes_by_FT={"2": 8, "3": 8, "4": 16}, normalizer="serial2",
              history=4096, owned=3, prefix_cap=3072, EOF_padding=0,
              additional_register_cap=2048, new_IQ_RAM_DSP_or_sample_ports=0,
              raw_detector_service_cycles=None, carrier_frame_decoder_cycles=None,
              new_default=False, new_RF_packet_rate_claim=False, all_RX_gates_passed=False)


def inputs(out, split):
    held = split == "validation"
    specs = (
        [(4, 12, 16, 421, "clean"), (2, 8, 0, 263, "tone"),
         (3, 12, 8, 857, "clean"), (4, 8, 4, 149, "step"),
         (2, 12, 8, 587, "burst"), (3, 8, 16, 61, "wrong-sync")]
        if held else
        [(2, 8, 4, 173, "clean"), (3, 12, 0, 541, "tone"),
         (4, 8, 8, 367, "clean"), (2, 12, 16, 743, "noise"),
         (3, 8, 4, 97, "clean"), (4, 12, 0, 619, "burst")]
    )
    rows = []
    for j, (ft, mcs, pilots, size, family) in enumerate(specs):
        cfg = ingress.shared.configuration(ft, mcs, pilots, size)
        cfg.pid = (j + 2 * int(held)) % 6
        profile = PSKReceiveProfile.ordinary(cfg)
        if j % 2:
            payload = bytes((0xE1, 0x39, 0x76, 0xA4, 0x0B)[(k // 7 + k) % 5]
                            ^ ((k // 61) & 255) for k in range(size))
        else:
            payload = bytes(((k * 109 + 43) ^ (k >> 3) ^ (0xB2 if held else 0x57)) & 255
                            for k in range(size))
        params = dict(offset=(.183, .529, .817, .341)[(j + int(held)) % 4],
                      ppm=(-75, 50, 100, -25)[(j + int(held)) % 4],
                      cfo=(23000, -61000, 89000, -13000)[(j + int(held)) % 4],
                      prefix=19 + j * 41 + 29 * int(held),
                      amplitude=(.097, .23, .061, .157)[(j + int(held)) % 4],
                      family=family, seed=972601 + j * 131 + int(held) * 701,
                      snr_db=11.25 if family == "noise" else None)
        iq, quiet, _, _, metadata = channel(packet(profile, payload), **params)
        if family == "wrong-sync":
            wrong = ingress.shared.configuration(ft, mcs, pilots, size + 139)
            wrong.pid = (cfg.pid + 1) % 6
            other, _, _, _, _ = channel(packet(PSKReceiveProfile.ordinary(wrong), payload),
                                        **(params | dict(family="clean", snr_db=None)))
            other = np.pad(other, (0, max(0, len(iq) - len(other))))[:len(iq)]
            iq = iq + .8 * other
        file = out / f"input{j}.npz"
        np.savez_compressed(file, iq=iq, quiet=quiet)
        config = asdict(cfg)
        config.pop("_mcs")
        rows.append(dict(id=j, config=config, params=params, metadata=metadata,
                         payload_SHA256=hashlib.sha256(payload).hexdigest(),
                         file=file.name, SHA256=shared.sha(file)))
    shared.write(out / "inputs.json", rows)
    return rows


def prepare(out, row):
    with np.load(out / row["file"]) as data:
        iq, quiet = data["iq"][:POLICY["prefix_cap"]], data["quiet"]
    profile = PSKReceiveProfile.ordinary(TxConfig(**row["config"]))
    detector = PSKStreamReceiver(profile, quiet, quiet_batching=False, defer_quiet_fir=False)
    flags = ingress.legacy_flags(detector, iq)
    versions = []
    for mode in ("whole", "fixed", "random"):
        filt, chunks, start = FullMantissaRRC(FixedRRCFormat(12, 18)), [], 0
        rng = np.random.default_rng(972779)
        while start < len(iq):
            size = len(iq) if mode == "whole" else 32 if mode == "fixed" else int(
                rng.integers(1, 68))
            end = min(start + size, len(iq))
            for j in range(start, end, 32):
                chunks.append(filt.feed_output_words(iq[j:min(j + 32, end)]))
            start = end
        versions.append(tuple(np.concatenate([v[k] for v in chunks]) for k in range(3)))
    assert all(all(a.tobytes() == b.tobytes() and a.dtype == b.dtype
                   for a, b in zip(versions[0], other, strict=True)) for other in versions[1:])
    words = [pack_filter_word(*(int(v[j]) for v in versions[0])) for j in range(len(iq))]
    file = out / f"prefix{row['id']}.npz"
    np.savez_compressed(file, words=np.asarray(words, np.uint64), flags=np.asarray(flags))
    return profile, words, flags, dict(file=file.name, SHA256=shared.sha(file),
                                      samples=len(iq), EOF_padding=0,
                                      whole_fixed_random_words_exact=True)


def calendar(out, row):
    profile, words, flags, prefix = prepare(out, row)
    n, ft = len(words), profile.config.frame_type
    producer_node = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=200000000))
    producer = producer_node.simulate(n, trace=True)
    release = [(e["cycle"], e["sample"], None if e["invalid"] else words[e["sample"]])
               for e in producer["trace"] if e["event"] == "output"]
    flag_events = [(math.ceil(j * 200 / 16) + 10, j, bool(flag)) for j, flag in enumerate(flags)]
    events, query_rows, windows = [], [], []

    def observe(event):
        if event["kind"] == "query":
            query_rows.append(dict(event))

    source = CoarseCaptureSource(
        profile.sync_symbols(), iter(release), iter(flag_events), sample_limit=n,
        source_horizon=producer["cycles"] + 8192,
        lanes=POLICY["lanes_by_FT"][str(ft)], sink=observe)
    worker = CaptureFineWorker(
        SyncSignTemplate.from_symbols(profile.sync_symbols()), source.flow.core.window,
        confirmation_required=True, reuse_symbol_power=True, defer_all_future=True,
        word_fabric=WordFabricAssumptions(2), gauss_difference=True, symbol_pipeline=True)

    def checked_worker(event, owned):
        end = event["after_samples"]
        assert list(owned) == words[end - source.flow.core.window:end]
        service = worker(event, owned)
        windows.append(dict(after_samples=end, geometry=asdict(event["geometry"]),
                            owned_SHA256=hashlib.sha256(
                                np.asarray(owned, np.uint64).tobytes()).hexdigest(),
                            service=asdict(service), all_source_words_exact=True))
        return service

    transport = CaptureTransport(CaptureTransportAssumptions(
        paths=1, history=4096, window=source.flow.core.window))
    result = transport.simulate(None, (), source_owner=source, released_requests=True,
                                fine_worker=checked_worker, sink=lambda e, _: events.append(e))
    # Independent old compiled mode on the same physical ports;only metadata differs.
    old_rows = []
    old = CoarseIngress(profile.sync_symbols(), lanes=POLICY["lanes_by_FT"][str(ft)],
                        normalizer="serial2",
                        sink=lambda e: old_rows.append(dict(e)) if e["kind"] == "query" else None)
    r, f = iter(release), iter(flag_events)
    rp, fp = next(r, None), next(f, None)
    for cycle in range(source.flow.cycle):
        old.take_candidate()
        if fp is not None and fp[0] == cycle:
            old.push_flag(fp[1], fp[2])
            fp = next(f, None)
        if rp is not None and rp[0] == cycle:
            old.push_rrc(rp[1], rp[2])
            rp = next(r, None)
        old.step()
    assert old_rows == [{k: v for k, v in event.items() if k != "source_end"}
                        for event in query_rows]
    old.take_candidate()
    assert old.drained
    old.finish()
    assert source.flow.lossless and source.done and source.stats["rrc_pulses"] == n
    assert source.stats["flag_pulses"] == n and result["input_samples"] == n
    assert result["requests"] == result["completed"] + result["queue_drops"]
    extra = source.estimate()["own_register_bits_upper"] + 256 + 193
    shared.write(out / f"calendar{row['id']}.json",
                 dict(result=result, producer=producer,
                      producer_cost=producer_node.resource_estimate(),
                      source_cost=source.estimate(), transport_cost=transport.estimate(
                          measured_worker=True, released_requests=True),
                      worker_cost=worker.estimate(), events=events, fine_windows=windows,
                      coarse_query_rows=query_rows, old_compiled_queries=old_rows))
    return dict(id=row["id"], input_SHA256=row["SHA256"], frame_type=ft, prefix=prefix,
                old_new_coarse_decisions_exact=True, actual_fine_windows=len(windows),
                all_actual_source_windows_exact=True, all_pulses_and_outcomes_accounted=True,
                additional_register_bits=extra, extra_IQ_RAM_DSP_or_sample_ports=0,
                requests=result["requests"], completed_fine=result["completed"],
                resource_queue_drops=result["queue_drops"],
                source_fault=result["live_coarse_failure"], frame_header_data_CRC_BER=None,
                all_four_RX_gates_passed=False)


def qualify(rows):
    return (len(rows) == 6 and len({r["input_SHA256"] for r in rows}) == 6
            and {r["id"] for r in rows} == set(range(6))
            and {r["frame_type"] for r in rows if r["actual_fine_windows"]} == {2, 3, 4}
            and all(r["old_new_coarse_decisions_exact"] and r["all_actual_source_windows_exact"]
                    and r["all_pulses_and_outcomes_accounted"] and not r["source_fault"]
                    and type(r["additional_register_bits"]) is int
                    and 0 <= r["additional_register_bits"] <= POLICY["additional_register_cap"]
                    and r["extra_IQ_RAM_DSP_or_sample_ports"] == 0
                    and all(type(r[k]) is int and r[k] >= 0 for k in
                            ("requests", "completed_fine", "resource_queue_drops",
                             "actual_fine_windows"))
                    and r["requests"] == r["completed_fine"] + r["resource_queue_drops"]
                    for r in rows))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    frozen = dict(policy=POLICY, sources=shared.sources())
    if args.split == "validation":
        if args.freeze is None:
            raise ValueError("Need frozen eligible development")
        prior = json.loads(args.freeze.read_text())
        model = json.loads((args.freeze.parent / "model.json").read_text())
        end = json.loads((args.freeze.parent / "source-bindings-at-end.json").read_text())
        if (prior != frozen or end != frozen["sources"] or not model["component_qualified"]
                or model["freeze_SHA256"] != shared.sha(args.freeze)):
            raise ValueError("Need unchanged source and eligible development")
    args.output.mkdir(exist_ok=False)
    shared.write(args.output / "freeze.json", frozen)
    rows = [calendar(args.output, row) for row in inputs(args.output, args.split)]
    assert shared.sources() == frozen["sources"]
    shared.write(args.output / "source-bindings-at-end.json", shared.sources())
    result = dict(split=args.split, rows=rows, component_qualified=qualify(rows),
                  freeze_SHA256=shared.sha(args.output / "freeze.json"), new_default=False,
                  new_RF_packet_rate_claim=False, all_four_RX_gates_passed=False,
                  raw_detector_and_frame_decoder_service_cycles=None, actual_synthesis=False,
                  old_numeric_and_resource_failures_preserved=True)
    shared.write(args.output / "model.json", result)
    print(json.dumps(dict(component_qualified=result["component_qualified"], streams=6,
                          all_four_RX_gates_passed=False)))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
