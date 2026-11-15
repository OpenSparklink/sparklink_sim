"""Hold timing/carrier fixed to diagnose sync readiness bias; never select receiver answers by CRC."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import rx_mac_frozen_replay as frozen
import rx_timing_audit as audit
from profile_rx_stream import number
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.psk import PSKDemodulator, rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.rx_pipeline import decode_payload
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver, _cubic
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    regressions = json.loads((args.stream_baseline / "regressions.json").read_text())
    row = next(r for r in regressions if r["id"] == args.sample_id)
    old = next(
        json.loads(line)
        for line in (args.reference / "results.jsonl").read_text().splitlines()
        if json.loads(line)["id"] == args.sample_id
    )
    plan = json.loads((args.original / "frozen-plan.json").read_text())
    path = args.original / (args.sample_id + ".npz")
    assert audit.identity(path) == old["derivative"]
    x = np.load(path)["iq"]
    quiets = [
        r for r in plan["quiet_inputs"] if r["sample_id"] in old["input"]["quiet_control_ids"]
    ]
    q = np.concatenate([frozen.checked_raw(args.dataset, r) for r in quiets])
    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=number(plan["sync_word"]),
        data_crc_seed=number(plan["crc_seed"]),
        pilot_interval=0,
    )
    selected = next(i for i, r in enumerate(row["frames"]) if r["status"] == "bad_data_crc")
    old_row = next(
        r
        for r in old["arms"]["coherent"]["rows"]
        if abs(r.get("sync_start_raw", -1) - row["frames"][selected]["sync_start_raw"]) < 4
    )
    old_index = old_row["burst_index"]
    stream = np.load(args.stream_baseline / row["evidence"])
    reference = np.load(args.reference / old["arms"]["coherent"]["evidence"])
    end = row["frames"][selected]["frame_end_symbol"]
    field_start = row["frames"][selected]["data_start_symbol"]
    expected = prbs9(2040)
    cfg = TxConfig(mcs_index=12, crc_seed=number(plan["crc_seed"]))
    z = fftconvolve(x, rrc_filter(0.4, 4), mode="same")
    spline = CubicSpline(np.arange(len(z)), z)
    ablation = []
    trajectories = [
        (
            "stream",
            stream[f"whole-{selected}-symbols"][:end],
            stream[f"whole-{selected}-symbol_centers_raw"][:end],
            row["frames"][selected],
            "lagrange4",
        ),
        (
            "offline",
            reference[f"symbols_{old_index}"][:end],
            reference[f"centers_{old_index}"][:end],
            old_row["frame"],
            "spline",
        ),
    ]
    for name, symbols, centers, frame, original_sampler in trajectories:
        raw4 = np.array([_cubic(z, 0, t) for t in centers])
        raw_spline = spline(centers)
        correction = symbols / (raw4 if original_sampler == "lagrange4" else raw_spline)
        for method, raw in [("lagrange4", raw4), ("spline", raw_spline)]:
            data = (raw * correction)[field_start:end]
            carrier = frame["carrier"]
            t = np.arange(len(data))
            data *= np.exp(
                -1j
                * (
                    2 * np.pi * carrier["residual_cycles_per_symbol"] * t
                    + carrier["residual_phase_radians"]
                )
            )
            bits = PSKDemodulator("8PSK", sps=1).demap_symbols(data, field_start)
            payload, ok = decode_payload(bits, cfg, 2040)
            ablation.append(
                dict(
                    trajectory=name,
                    sampler=method,
                    crc=bool(ok),
                    bit_errors=int(np.sum(payload != expected)),
                    error_positions=np.flatnonzero(payload != expected).tolist(),
                    carrier_held_fixed=True,
                )
            )
    stop = round(row["frames"][selected]["sync_start_raw"] + 31 * 4 + 16 + 1)
    window = z[stop - 256 : stop]
    # Load the exact trusted baseline helper, rather than reimplementing its bug.
    code = subprocess.check_output(
        ["git", "show", args.baseline_commit + ":src/nearlink_sdr/phy/psk_sync.py"], text=True
    )
    namespace = {}
    exec(compile(code, "baseline-psk-sync.py", "exec"), namespace)
    peaks = {}
    for name, fn, guard in [
        ("observed_maximum", search_psk_sync, 0),
        ("baseline_guard_mask", namespace["search_psk_sync"], 16),
        ("current_readiness_gate", search_psk_sync, 16),
    ]:
        found = fn(window, 4, profile.sync_symbols(), 8, lookahead=guard)
        peaks[name] = (
            None
            if found is None
            else dict(start=stop - 256 + found["phase"] + 4 * found["offset"], score=found["score"])
        )
    rx = PSKStreamReceiver(profile, q)
    events = rx.feed(x) + rx.finish()
    same_packet = next(
        e for e in events if abs(e["sync_start_raw"] - old_row["sync_start_raw"]) < 4
    )
    report = dict(
        id=args.sample_id,
        input_identity=old["derivative"],
        baseline_commit=args.baseline_commit,
        original_sro="unknown",
        injected_ppm=old["injected_ppm"],
        ablation=ablation,
        peaks=peaks,
        after=dict(
            status=same_packet["frame"]["status"],
            sync_start_raw=same_packet["sync_start_raw"],
            whole_file_correct=sum(
                e["frame"]["payload"] == np.packbits(expected, bitorder="little").tobytes()
                for e in events
            ),
        ),
        expected_payload_used_only_for_diagnostics=True,
        source_sync=audit.identity(
            Path(__file__).resolve().parents[1] / "src/nearlink_sdr/phy/psk_sync.py"
        ),
    )
    audit.write(out / "diagnosis.json", report)
    np.savez_compressed(
        out / "trajectories.npz",
        baseline=trajectories[0][2],
        offline=trajectories[1][2],
        after=same_packet["symbol_centers_raw"],
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    positions = np.arange(-2, 1.01, 0.125) + old_row["sync_start_raw"]
    values = []
    ref = profile.sync_symbols()
    dr = ref[1:] * ref[:-1].conj()
    for position in positions:
        samples = spline(position + np.arange(len(ref)) * 4)
        dy = samples[1:] * samples[:-1].conj()
        values.append(
            abs(np.sum(dy * dr.conj())) / np.sqrt(np.sum(abs(dy) ** 2) * np.sum(abs(dr) ** 2))
        )
    ax[0].plot(positions - old_row["sync_start_raw"], values)
    ax[0].axvline(
        row["frames"][selected]["sync_start_raw"] - old_row["sync_start_raw"],
        color="r",
        linestyle="--",
        label="Old guard edge",
    )
    ax[0].axhline(0.85, color="grey", linestyle=":")
    ax[0].set(
        xlabel="Start relative to offline selected peak (samples)",
        ylabel="Known-sync differential score",
    )
    ax[0].legend()
    old_centers = trajectories[1][2]
    ax[1].plot(trajectories[0][2] - old_centers, label="Before fix")
    ax[1].plot(same_packet["symbol_centers_raw"] - old_centers, label="After fix")
    ax[1].set(xlabel="Symbol from sync", ylabel="Center difference to offline receiver (samples)")
    ax[1].legend()
    fig.suptitle("Readiness bias diagnosis: differences are not physical timing truth")
    fig.savefig(out / "guard-bias.png", dpi=180)
    fig.savefig(out / "guard-bias.svg")
    plt.close(fig)
    audit.write(
        out / "manifest.json", {p.name: audit.identity(p) for p in out.iterdir() if p.is_file()}
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stream-baseline", "reference", "original", "dataset", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--baseline-commit", default="d29d447")
    run(parser.parse_args())
