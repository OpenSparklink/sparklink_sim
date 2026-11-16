"""Fixed attenuation/noise probes of immutable hardware IQ; no claimed timing/SNR truth."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from audit_rx_boundary import score
from profile_rx_stream import load, number

from nearlink_sdr.phy.psk_sync import PSKCaptureConfig
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    def rational(v):
        if isinstance(v, Fraction):
            return str(v)
        raise TypeError(f"Cannot serialize {type(v)}")

    p.write_text(json.dumps(value, indent=2, default=rational) + "\n")


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    (out / "evidence").mkdir()
    original = json.loads((args.original / "frozen-plan.json").read_text())
    previous = {
        r["id"]: r
        for r in [json.loads(s) for s in (args.previous / "results.jsonl").read_text().splitlines()]
        if r["source"] == "real"
    }
    box = np.ones(32) / 32
    prepared = []
    if args.inputs:
        frozen = json.loads((args.inputs / "frozen-plan.json").read_text())
        prepared = frozen["inputs"]
        for meta in prepared:
            assert sha(Path(meta["input_path"])) == meta["input_sha256"]
            for source in [meta["source"], *meta["calibration_sources"]]:
                assert sha(args.dataset / source["path"]) == source["sha256"]
    for mode in () if args.inputs else ("mode-057", "mode-065"):
        dev = next(
            r for r in original["real_inputs"] if r["sample_id"] == f"series-02/{mode}/rx-00"
        )
        dev_x = load(args.dataset, dev)
        quiet_meta = [
            r for r in original["quiet_inputs"] if r["sample_id"] in dev["quiet_control_ids"]
        ]
        quiet = np.concatenate([load(args.dataset, r) for r in quiet_meta])
        noise_power = float(np.mean(abs(quiet) ** 2))
        quiet_quantile = float(np.quantile(np.convolve(abs(quiet) ** 2, box, "same"), 0.999))
        burst_quantile = float(np.quantile(np.convolve(abs(dev_x) ** 2, box, "same"), 0.99))
        for target in (2.5, 5.0):
            # Recipe depends only on development energy statistics, never sync/CRC/content.
            alpha = float(np.sqrt(max(target * quiet_quantile - noise_power, 0.0) / burst_quantile))
            for meta in [r for r in original["real_inputs"] if f"/{mode}/" in r["sample_id"]]:
                split = "development" if meta["sample_id"].endswith("rx-00") else "held-out"
                index = int(meta["sample_id"][-2:])
                seed = [
                    224016 if split == "development" else 234016,
                    int(mode[-3:]),
                    int(target * 10),
                    index,
                ]
                rng = np.random.default_rng(np.random.SeedSequence(seed))

                def noise(n):
                    return np.sqrt(noise_power / 2) * (rng.normal(size=n) + 1j * rng.normal(size=n))

                x = load(args.dataset, meta)
                added = noise(len(x))
                derived, calibration = alpha * x + added, noise(8192)
                ident = meta["sample_id"].replace("/", "-") + f"-energy{target}"
                path = out / "inputs" / (ident + ".npz")
                np.savez_compressed(path, iq=derived, quiet=calibration, negative=added[:65536])
                baseline = previous[meta["sample_id"].replace("/", "-")]
                prepared.append(
                    dict(
                        id=ident,
                        split=split,
                        target_energy_ratio=target,
                        attenuation=alpha,
                        injected_complex_noise_variance=noise_power,
                        seed=seed,
                        original_cfo_sro_noise="unknown",
                        injected_cfo_hz=0,
                        injected_sro_ppm=0,
                        timing_truth=None,
                        true_snr_db=None,
                        input_path=str(path),
                        input_sha256=sha(path),
                        source=meta,
                        calibration_sources=quiet_meta,
                        development_calibration=dict(
                            quiet_q999=quiet_quantile, source_q99=burst_quantile
                        ),
                        kind=baseline["kind"],
                        original_complete_positions=baseline["successful_sync_positions"],
                        original_candidate_positions=[
                            f["sync_start_raw"] for f in baseline["frames"]
                        ],
                    )
                )
    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=number(original["sync_word"]),
        data_crc_seed=number(original["crc_seed"]),
        pilot_interval=0,
    )
    plan = dict(
        profile=asdict(profile),
        original_plan_sha256=sha(args.original / "frozen-plan.json"),
        previous_results_sha256=sha(args.previous / "results.jsonl"),
        replayed_from=None if not args.inputs else str(args.inputs),
        inputs=prepared,
        parameters_frozen_before_generation=True,
        capture=dict(legacy=vars(PSKCaptureConfig.legacy()), selected=vars(PSKCaptureConfig())),
        source_sha256={str(p): sha(p) for p in Path("src").rglob("*.py")},
        driver_sha256=sha(Path(__file__)),
        note="Targets are development energy recipes, not measured SNR/compliance limits; original IQ untouched",
    )
    write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta in prepared:
        row = dict(meta, arms={})
        arrays = {}
        assert sha(Path(meta["input_path"])) == meta["input_sha256"]
        with np.load(meta["input_path"]) as data:
            for name, config in [
                ("legacy", PSKCaptureConfig.legacy()),
                ("selected", PSKCaptureConfig()),
            ]:
                rx = PSKStreamReceiver(profile, data["quiet"], capture_config=config)
                events = rx.feed(data["iq"]) + rx.finish()
                frames = score(events, 255, kind=meta["kind"])
                for frame in frames:
                    frame["matched_original_candidate"] = any(
                        abs(frame["sync_start_raw"] - t) <= 4
                        for t in meta["original_candidate_positions"]
                    )
                negative = PSKStreamReceiver(profile, data["quiet"], capture_config=config)
                false_events = negative.feed(data["negative"]) + negative.finish()
                row["arms"][name] = dict(
                    frames=frames,
                    stats=rx.capture_stats,
                    input_samples=len(data["iq"]),
                    searches_per_Msample=rx.sync_searches * 1e6 / len(data["iq"]),
                    full_searches_per_Msample=rx.capture_stats["full_searches"]
                    * 1e6
                    / len(data["iq"]),
                    candidates=len(events),
                    correct=sum(f["correct"] for f in frames),
                    negative_samples=len(data["negative"]),
                    negative_candidates=len(false_events),
                    false_deliveries=sum(e["frame"]["status"] == "valid" for e in false_events),
                    false_candidates=sum(not f["matched_original_candidate"] for f in frames),
                    timing_backend=rx.timing_backend,
                )
                for i, event in enumerate(events):
                    for k in (
                        "symbols",
                        "symbol_centers_raw",
                        "timing_detector_error",
                        "timing_rate_ppm",
                    ):
                        arrays[f"{name}-{i}-{k}"] = event[k]
        row["lost_legacy_success_positions"] = [
            f["sync_start_raw"]
            for f in row["arms"]["legacy"]["frames"]
            if f["correct"]
            and not any(
                g["correct"] and abs(g["sync_start_raw"] - f["sync_start_raw"]) <= 4
                for g in row["arms"]["selected"]["frames"]
            )
        ]
        np.savez_compressed(out / "evidence" / (meta["id"] + ".npz"), **arrays)
        rows.append(row)
        write(out / "results.json", rows)
    summary = []
    for split in ("development", "held-out"):
        for kind in ("QPSK", "8PSK"):
            for target in (2.5, 5.0):
                selected = [
                    r
                    for r in rows
                    if r["split"] == split
                    and r["kind"] == kind
                    and r["target_energy_ratio"] == target
                ]
                for arm in ("legacy", "selected"):
                    frames = [f for r in selected for f in r["arms"][arm]["frames"]]
                    summary.append(
                        dict(
                            split=split,
                            kind=kind,
                            target_energy_ratio=target,
                            arm=arm,
                            streams=len(selected),
                            original_complete_packets=sum(
                                len(r["original_complete_positions"]) for r in selected
                            ),
                            candidates=sum(len(r["arms"][arm]["frames"]) for r in selected),
                            correct=sum(f["correct"] for f in frames),
                            statuses=dict(Counter(f["status"] for f in frames)),
                            negative_candidates=sum(
                                r["arms"][arm]["negative_candidates"] for r in selected
                            ),
                            false_deliveries=sum(
                                r["arms"][arm]["false_deliveries"] for r in selected
                            ),
                        )
                    )
    write(out / "summary.json", summary)
    write(out / "regressions.json", [r for r in rows if r["lost_legacy_success_positions"]])
    write(
        out / "failures.json",
        [
            r
            for r in rows
            if r["arms"]["selected"]["correct"] != len(r["original_complete_positions"])
        ],
    )
    assert all(sha(Path(p)) == h for p, h in plan["source_sha256"].items())
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(json.dumps(summary, indent=2))
    assert not any(r["lost_legacy_success_positions"] for r in rows), "Retained regression evidence"
    assert not any(
        r["arms"]["selected"]["negative_candidates"] or r["arms"]["selected"]["false_deliveries"]
        for r in rows
    ), "Negative budget failed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("original", "previous", "dataset", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    run(parser.parse_args())
