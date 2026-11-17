"""Read frozen original IQ identities and compare actual factory-profile MAC paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import rx_stream_frozen_replay as frozen
from audit_capture_hardware import sha, write
from profile_rx_stream import number
from replay_acquisition_regression import checked_manifest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def run(args):
    source = checked_manifest(args.frozen_reference, "results.jsonl")
    reference = [json.loads(line) for line in source.read_text().splitlines()]
    original = args.evidence_root / "rx-timing-tracking/final"
    plan_file = checked_manifest(original, "frozen-plan.json")
    plan = json.loads(plan_file.read_text())
    profile = PSKReceiveProfile.ws63_ft2_a3(sync_word=number(plan["sync_word"]),
                                           data_crc_seed=number(plan["crc_seed"]), pilot_interval=0)
    quiet = {r["input"]["input"]["sample_id"]: r["input"]["input"]
             for r in reference if r["input"]["source"] == "quiet"}
    old_quiet = {r["sample_id"]: r for r in plan["quiet_inputs"]}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence").mkdir()
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(args.output / "plan.json", dict(reference_sha256=sha(source),
          factory_plan_sha256=sha(plan_file), sources=sources, driver_sha256=sha(Path(__file__)),
          normalization="original ADC codes; explicit factory sync/CRC/pilot0; no unknown prefix stripping",
          whole_system_accepted=False))
    rows, regressions = [], []
    for row in reference:
        meta = row["input"]
        if meta["source"] != "real":
            continue
        x, q = frozen.load_input(meta, args.evidence_root / "rx-chain-hardening/parser-replay",
                                 original, args.dataset, args.quiet_dataset, old_quiet, quiet)
        arms = {}
        for name, options in [("reference", {}), ("bounded", dict(carrier_weight_kappa=2))]:
            rx = MacStreamReceiver(profile, q, quiet_batching=False, defer_quiet_fir=False,
                                   correlation_backend="scaled", correlation_mantissa_bits=10,
                                   tone_notch=ToneNotchConfig(radius=31/32), **options)
            events, packets = [], []
            for start in range(0, len(x), 4096):
                result = rx.feed(x[start:start + 4096])
                events += result.diagnostics["events"]
                packets += result.packets
            before_eof = len(packets)
            result = rx.finish()
            events += result.diagnostics["events"]
            packets += result.packets
            frames = frozen.score(meta, events)
            assert len(packets) == sum(f["status"] == "valid" for f in frames)
            evidence = f"{row['id']}-{name}.npz"
            np.savez_compressed(args.output / "evidence" / evidence, **{
                f"{i}-{key}": e[key] for i, e in enumerate(events) for key in
                ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]})
            arms[name] = dict(frames=frames, delivered=[p.mac_payload.hex() for p in packets],
                              before_eof=before_eof, evidence=evidence, stats=rx.phy.capture_stats)
        old, new = arms["reference"], arms["bounded"]
        assert sum(f["whole_packet_ok"] for f in old["frames"]) == row["correct"]
        for f in old["frames"]:
            if f["whole_packet_ok"] and not any(
                    g["whole_packet_ok"] and abs(g["sync_start_raw"] - f["sync_start_raw"]) <= .125
                    for g in new["frames"]):
                regressions.append(dict(id=row["id"], position=f["sync_start_raw"]))
        rows.append(dict(id=row["id"], input=meta, samples=len(x), arms=arms))
        write(args.output / "results.json", rows)
    report = dict(inputs=len(rows), original_success_losses=regressions,
                  captured={arm: sum(len(r["arms"][arm]["frames"]) for r in rows)
                            for arm in ["reference", "bounded"]},
                  correct={arm: sum(f["whole_packet_ok"] for r in rows
                                    for f in r["arms"][arm]["frames"])
                           for arm in ["reference", "bounded"]},
                  actual_mac_deliveries={arm: sum(len(r["arms"][arm]["delivered"]) for r in rows)
                                         for arm in ["reference", "bounded"]},
                  unknown="true transmitted packet count, precise timing/SRO, OEM sample loss",
                  whole_system_accepted=False)
    write(args.output / "summary.json", report)
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {str(p.relative_to(args.output)): sha(p)
                                          for p in args.output.rglob("*") if p.is_file()})
    print(json.dumps(report, indent=2))
    return int(bool(regressions))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["evidence-root", "frozen-reference", "dataset", "quiet-dataset", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
