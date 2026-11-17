"""Veto frozen carrier selection on original SHA-identified228 streams/96 packets."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, score, sha, write
from audit_carrier_loop import carrier_adapter

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


def run(args):
    assert (
        sha(args.input_index) == "b612659b09c3172f9363dc681b7f912435381f933f5d6084e989710e25be9270"
    )
    plan = json.loads(args.input_index.read_text())
    policy = json.loads(Path("tests/data/carrier_loop_policy.json").read_text())
    assert policy["loop_module_sha256"] == sha(Path("src/nearlink_sdr/phy/carrier_loop.py"))
    assert policy["power_module_sha256"] == sha(Path("src/nearlink_sdr/phy/carrier_weights.py"))
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, meta in enumerate(plan["cases"]):
        path = Path(meta["path"])
        assert sha(path) == meta["sha256"]
        p = profile(meta["mcs"])
        with np.load(path) as ar:
            x, quiet = ar["iq"], ar["quiet"]
        arms = {}
        paths = {}
        for arm in ("full", policy["selected_arm"]):
            rx = MacStreamReceiver(
                p,
                quiet,
                quiet_batching=False,
                defer_quiet_fir=False,
                correlation_backend="scaled",
                normalized_coarse=True,
                tone_notch=ToneNotchConfig(radius=31 / 32),
                sync_search_method="bounded-grid",
                sync_candidate_count=1,
                sync_coherent_threshold=0.45,
            )
            with carrier_adapter(arm, p) as (fits, traces):
                a = rx.feed(x)
                b = rx.finish()
            events = a.diagnostics["events"] + b.diagnostics["events"]
            result = score(events, meta)
            result["MAC_delivered_payloads"] = [v.mac_payload.hex() for v in a.packets + b.packets]
            arrays = {
                f"{j}-{key}": event[key]
                for j, event in enumerate(events)
                for key in (
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                )
            }
            for j, trace in enumerate(traces):
                for key, value in trace.items():
                    arrays[f"loop-{j}-{key}"] = value
            np.savez_compressed(out / (meta["id"] + "-" + arm + ".npz"), **arrays)
            write(out / (meta["id"] + "-" + arm + "-fits.json"), fits)
            arms[arm] = result
            paths[arm] = arrays
        old, new = arms.values()
        assert (
            old["captured"] == new["captured"]
            and old["false_candidates"] == new["false_candidates"]
        )
        centers = [k for k in paths["full"] if k.endswith("symbol_centers_raw")]
        assert all(
            np.array_equal(paths["full"][k], paths[policy["selected_arm"]][k]) for k in centers
        )
        assert len(old["frames"]) == len(new["frames"])
        lost = [
            i
            for i, (a, b) in enumerate(zip(old["frames"], new["frames"], strict=True))
            if a["correct"] and not b["correct"]
        ]
        restored = [
            i
            for i, (a, b) in enumerate(zip(old["frames"], new["frames"], strict=True))
            if not a["correct"] and b["correct"]
        ]
        rows.append(
            dict(
                input=meta,
                arms=arms,
                old_success_lost_frame_indices=lost,
                recovered_frame_indices=restored,
                capture_timing_exact=True,
            )
        )
        write(out / "results.json", rows)
        print(
            f"{index + 1}/{len(plan['cases'])} {meta['id']} {old['correct']}=>{new['correct']}",
            flush=True,
        )
    summary = {
        arm: {
            k: sum(r["arms"][arm][k] for r in rows)
            for k in (
                "known_packets",
                "captured",
                "correct",
                "missed",
                "false_candidates",
                "false_deliveries",
            )
        }
        for arm in ("full", policy["selected_arm"])
    }
    losses = [
        dict(
            id=r["input"]["id"],
            sha256=r["input"]["sha256"],
            frames=r["old_success_lost_frame_indices"],
        )
        for r in rows
        if r["old_success_lost_frame_indices"]
    ]
    recoveries = [
        dict(id=r["input"]["id"], sha256=r["input"]["sha256"], frames=r["recovered_frame_indices"])
        for r in rows
        if r["recovered_frame_indices"]
    ]
    write(
        out / "model.json",
        dict(
            summary=summary,
            original_success_losses=losses,
            recoveries=recoveries,
            input_index_sha256=sha(args.input_index),
            inputs=len(rows),
            samples=sum(r["input"]["samples"] for r in rows),
            negative_samples=sum(r["input"]["samples"] for r in rows if not r["input"]["positive"]),
            fixed_regression_only=True,
            parameter_tuning_allowed=False,
            default_changed=False,
            complete16msps=False,
        ),
    )
    write(
        out / "failures.json",
        [
            r
            for r in rows
            if any(
                a["correct"] < a["known_packets"] or a["false_candidates"] or a["false_deliveries"]
                for a in r["arms"].values()
            )
        ],
    )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(dict(summary=summary, losses=losses, recoveries=recoveries), indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
