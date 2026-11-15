"""Frozen low-SNR repetitions and post-selection timing/carrier truth diagnostics."""

import argparse
import hashlib
import json
import resource
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import rx_timing_audit as audit
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve

from nearlink_sdr.common.prbs import prbs9
from nearlink_sdr.phy.psk import PSKDemodulator, rrc_filter
from nearlink_sdr.phy.rx_pipeline import decode_payload
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def receive(x, q, profile):
    begin = time.perf_counter()
    rx = PSKStreamReceiver(profile, q)
    events = rx.feed(x) + rx.finish()
    return events, dict(
        seconds=time.perf_counter() - begin,
        sync_searches=rx.sync_searches,
        peak_history_samples=rx.peak_history_samples,
    )


def trajectory(error, sps=4):
    # Operational diagnostics, not standard compliance limits.
    cycle = np.floor(error / sps + 0.5).astype(int)
    sustained = np.convolve((abs(error) >= sps / 2).astype(int), np.ones(32, dtype=int), "valid")
    over = np.flatnonzero(abs(error) > 0.05)
    settle = int(over[-1] + 1) if len(over) else 0
    return dict(
        rms_samples=float(np.sqrt(np.mean(error**2))),
        max_samples=float(max(abs(error))),
        head_mid_tail_samples=[float(error[k]) for k in (0, len(error) // 2, len(error) - 1)],
        symbol_cycle_change_count=int(np.sum(abs(np.diff(cycle)))),
        starts_wrong_symbol=bool(cycle[0]),
        sustained_32_symbol_unlock=bool(len(sustained) and max(sustained) >= 32),
        settling_0p05_sample_suffix_symbol=settle if settle < len(error) else None,
    )


def score(events, length, truth=None, kind=None):
    expected = np.packbits(prbs9(length * 8), bitorder="little").tobytes()
    rows = []
    for e in events:
        f = e["frame"]
        row = {k: v for k, v in f.items() if k != "payload"}
        row.update(
            sync_start_raw=e["sync_start_raw"],
            sync_score=e["sync_score"],
            correct=f["status"] == "valid"
            and f["payload"] == expected
            and f["header"]["data_length"] == length
            and f["mcs"] == (12 if kind == "8PSK" else 8),
            emitted_after_samples=e["emitted_after_samples"],
        )
        if f.get("header_crc") and f.get("data_symbols"):
            decoded_kind = "8PSK" if f["mcs"] == 12 else "QPSK"
            d = audit.score_symbols(e["symbols"], f, decoded_kind, length)
            if d.get("diagnostic_payload_hex"):
                recovered = np.unpackbits(
                    np.frombuffer(bytes.fromhex(d["diagnostic_payload_hex"]), dtype=np.uint8),
                    bitorder="little",
                )
                row["bit_error_positions"] = np.flatnonzero(recovered != prbs9(length * 8)).tolist()
            row.update(
                bit_errors=d.get("bit_errors"), ber=d.get("ber"), ber_windows=d.get("ber_windows")
            )
        if truth:
            errors = e["symbol_centers_raw"] - (
                truth["sync_start"] + np.arange(len(e["symbols"])) * truth["step"]
            )
            row["timing_truth"] = trajectory(errors)
        rows.append(row)
    return rows


def diagnostic_no_carrier_refinement(y, profile, length):
    # Diagnostic only: real head gate retained; payload sampled with supplied truth carrier.
    head = decode_psk_frame(y[:65], profile)
    row = {k: v for k, v in head.items() if k != "payload"}
    if not head.get("header_crc") or head.get("frame_end_symbol", len(y) + 1) > len(y):
        return row
    start, end = head["data_start_symbol"], head["frame_end_symbol"]
    cfg = replace(profile.config, mcs_index=head["mcs"])
    hard = PSKDemodulator(cfg.mod_str, sps=1).demap_symbols(y[start:end], start)[
        : head["coded_bits"]
    ]
    decoded, ok = decode_payload(hard, cfg, length * 8)
    reference = prbs9(length * 8)
    errors = np.flatnonzero(decoded != reference)
    row.update(
        status="valid" if ok else "bad_data_crc",
        data_crc=bool(ok),
        bit_errors=len(errors),
        bit_error_positions=errors.tolist(),
    )
    return row


def oracle_diagnosis(meta, x, q, profile, out):
    events, _ = receive(x, q, profile)
    assert len(events) == 1, meta["id"]
    e = events[0]
    truth = meta["truth"]
    count = len(e["symbols"])
    t = truth["sync_start"] + np.arange(count) * truth["step"]
    h = rrc_filter(0.4, 4, 10)
    filtered = fftconvolve(x, h, mode="same")
    y = CubicSpline(np.arange(len(x)), filtered)(t)
    ref = profile.sync_symbols()
    slope, phase = np.polyfit(
        np.arange(len(ref)), np.unwrap(np.angle(y[: len(ref)] * ref.conj())), 1
    )
    gain = np.mean(abs(y[: len(ref)]))
    sync_normalized = y * np.exp(-1j * (phase + slope * np.arange(count))) / gain
    blind = decode_psk_frame(sync_normalized, profile)
    blind_score = audit.score_symbols(sync_normalized, blind, meta["kind"], meta["length"])
    # Known injection carrier only; no estimated/expected payload decides any phase.
    original_phase = 0.37 + 2 * np.pi * meta["cfo_hz"] * np.arange(len(x)) / 16e6
    corrected = fftconvolve(x * np.exp(-1j * original_phase), h, mode="same")
    interp = CubicSpline(np.arange(len(x)), corrected)
    truth_carrier_actual_timing = interp(e["symbol_centers_raw"]) / 100
    both = interp(t) / 100
    arms = dict(
        actual=score(events, meta["length"], truth, meta["kind"])[0],
        truth_timing_blind_carrier=dict(
            status=blind["status"],
            header_crc=blind["header_crc"],
            data_crc=blind["data_crc"],
            bit_errors=blind_score.get("bit_errors"),
            ber=blind_score.get("ber"),
        ),
        actual_timing_truth_carrier=diagnostic_no_carrier_refinement(
            truth_carrier_actual_timing, profile, meta["length"]
        ),
        truth_timing_truth_carrier=diagnostic_no_carrier_refinement(both, profile, meta["length"]),
    )
    np.savez_compressed(
        out / "evidence" / (meta["id"] + "-oracle.npz"),
        actual_symbols=e["symbols"],
        actual_centers=e["symbol_centers_raw"],
        truth_centers=t,
        truth_timing_blind_carrier=sync_normalized,
        actual_timing_truth_carrier=truth_carrier_actual_timing,
        truth_timing_truth_carrier=both,
    )
    return dict(
        id=meta["id"],
        input=meta["input_identity"],
        parameters={k: meta[k] for k in ("kind", "snr_db", "ppm", "cfo_hz")},
        arms=arms,
        evidence="evidence/" + meta["id"] + "-oracle.npz",
    )


def cases():
    base = dict(length=255, offset=0.277, ppm=100, cfo_hz=30000, amplitude=1, fade_end=1)
    result = []
    for kind in ("QPSK", "8PSK"):
        for snr in (8, 10, 12):
            for repeat in range(24):
                result.append(
                    dict(
                        base,
                        kind=kind,
                        snr_db=snr,
                        family="combined-boundary",
                        seed=83016 + repeat,
                        split="independent-repeat",
                    )
                )
        for family, updates, repeats in [
            ("clean", dict(snr_db=80, ppm=0, cfo_hz=0), 2),
            ("sro-only", dict(snr_db=80, ppm=100, cfo_hz=0), 2),
            ("cfo-only", dict(snr_db=80, ppm=0, cfo_hz=30000), 2),
            ("noise-only", dict(snr_db=8, ppm=0, cfo_hz=0), 8),
            ("noise-cfo", dict(snr_db=8, ppm=0, cfo_hz=30000), 8),
            ("noise-sro", dict(snr_db=8, ppm=100, cfo_hz=0), 8),
            ("noise-cfo-sro-amplitude", dict(snr_db=10, amplitude=0.5, fade_end=0.6), 8),
        ]:
            for repeat in range(repeats):
                result.append(
                    dict(
                        base,
                        **{
                            "kind": kind,
                            "family": family,
                            "seed": 84016 + repeat,
                            "split": "coupling-control",
                            **updates,
                        },
                    )
                )
    return result


def wilson(k, n):
    z = 1.959963984540054
    mid = (k / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(k * (n - k) / n**3 + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, mid - half)), float(min(1.0, mid + half))]


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "inputs").mkdir()
    (out / "evidence").mkdir()
    root = Path(__file__).resolve().parents[1]
    plan = dict(
        commit=audit.git("rev-parse", "HEAD"),
        sources={str(p.relative_to(root)): sha(p) for p in (root / "src").rglob("*.py")},
        driver_sha256=sha(Path(__file__)),
        cases=cases(),
        threshold=0.85,
        subdivisions=8,
        bandwidth=0.003,
        max_rate_ppm=500,
        energy_window_symbols=8,
        noise_definition=(
            "ADC noise E|n|²=2500*10^(-snr_db/10); TX amplitude100 RRC/2, "
            "nominal active sample power2500; matched Es/N0 approx snr_db+6.02dB before fades"
        ),
        operational_lock=dict(
            half_symbol_error_samples=2, sustained_symbols=32, settling_samples=0.05
        ),
        truth_scope=("post-selection diagnostics only; "
                     "never used by actual receive or success counts"),
    )
    plan["reference_sources"] = {f: sha(root / f) for f in ["scripts/rx_timing_audit.py"]}
    audit.write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=audit.WORD, data_crc_seed=0x555555, pilot_interval=0
    )
    archived = [
        json.loads(line) for line in (args.replay / "results.jsonl").read_text().splitlines()
    ]
    oracle = []
    for meta in archived:
        if meta["id"] not in ("new-noise-00", "new-noise-01", "new-noise-08"):
            continue
        path = args.replay / meta["input_iq"]
        assert audit.identity(path) == meta["input_identity"]
        with np.load(path) as data:
            oracle.append(oracle_diagnosis(meta, data["iq"], data["quiet"], profile, out))
    audit.write(out / "oracle-diagnostics.json", oracle)
    prior_rows = None
    if args.inputs:
        prior = json.loads((args.inputs / "frozen-plan.json").read_text())
        assert prior["cases"] == plan["cases"], "Frozen cohort parameters changed"
        prior_rows = json.loads((args.inputs / "results.json").read_text())
        prior_manifest = json.loads((args.inputs / "manifest.json").read_text())
    rows = []
    for index, case in enumerate(plan["cases"]):
        if prior_rows is None:
            x, q, truth, _ = audit.simulate(case)
        else:
            frozen = prior_rows[index]
            source = args.inputs / frozen["input"]["path"]
            assert sha(source) == frozen["input"]["sha256"]
            with np.load(source) as data:
                x, q = data["iq"], data["quiet"]
            step = 4 * (1 + case["ppm"] * 1e-6)
            truth = dict(sync_start=512 + case["offset"] + 40 * step, step=step)
        ident = f"repeat-{index:03}"
        path = out / "inputs" / (ident + ".npz")
        np.savez_compressed(path, iq=x, quiet=q)
        events, cost = receive(x, q, profile)
        frames = score(events, 255, truth, case["kind"])
        # Independent quiet record at the same power. No packet or boundary truth enters RX.
        rng = np.random.default_rng(93016 + index)
        sigma = 50 * 10 ** (-case["snr_db"] / 20)
        quiet = sigma / np.sqrt(2) * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
        if prior_rows is not None:
            source = args.inputs / prior_rows[index]["evidence"]
            assert audit.identity(source) == prior_manifest[prior_rows[index]["evidence"]]
            with np.load(source) as data:
                np.testing.assert_array_equal(quiet, data["quiet_negative"])
                quiet = data["quiet_negative"]
        negative, ncost = receive(quiet, q, profile)
        np.savez_compressed(
            out / "evidence" / (ident + ".npz"),
            quiet_negative=quiet,
            **{
                f"{j}-{key}": e[key]
                for j, e in enumerate(events)
                for key in (
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                )
            },
        )
        rows.append(
            dict(
                case,
                id=ident,
                frames=frames,
                cost=cost,
                quiet_candidates=len(negative),
                quiet_deliveries=sum(e["frame"]["status"] == "valid" for e in negative),
                quiet_cost=ncost,
                input=dict(path=str(path.relative_to(out)), sha256=sha(path)),
                evidence="evidence/" + ident + ".npz",
                correct=sum(f["correct"] for f in frames),
            )
        )
    groups = defaultdict(list)
    for r in rows:
        groups[(r["kind"], r["family"], r["snr_db"])].append(r)
    summary = []
    for (kind, family, snr), group in groups.items():
        k = sum(bool(r["correct"]) for r in group)
        n = len(group)
        errors = [
            f["bit_errors"] for r in group for f in r["frames"] if f.get("bit_errors") is not None
        ]
        summary.append(
            dict(
                kind=kind,
                family=family,
                snr_db=snr,
                inputs=n,
                correct=k,
                success_wilson95=wilson(k, n),
                misses=sum(not r["frames"] for r in group),
                statuses=dict(Counter(f["status"] for r in group for f in r["frames"])),
                conditional_ber=sum(errors) / (len(errors) * 255 * 8) if errors else None,
                comparable_packets=len(errors),
                quiet_candidates=sum(r["quiet_candidates"] for r in group),
                quiet_deliveries=sum(r["quiet_deliveries"] for r in group),
                true_timing_cycle_changes=sum(
                    f["timing_truth"]["symbol_cycle_change_count"]
                    for r in group
                    for f in r["frames"]
                ),
            )
        )
    audit.write(out / "results.json", rows)
    audit.write(out / "summary.json", summary)
    audit.write(
        out / "failures.json", [r for r in rows if r["correct"] != 1 or r["quiet_candidates"]]
    )
    assert all(sha(root / p) == v for p, v in plan["sources"].items())
    audit.write(
        out / "integrity.json",
        dict(
            sources_unchanged=True,
            inputs=len(rows),
            peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for kind in ("QPSK", "8PSK"):
        g = [r for r in summary if r["kind"] == kind and r["family"] == "combined-boundary"]
        xx = [r["snr_db"] for r in g]
        yy = np.array([r["correct"] / r["inputs"] for r in g])
        bounds = np.array([r["success_wilson95"] for r in g]).T
        axes[0].errorbar(
            xx,
            yy,
            yerr=np.maximum(0, [yy - bounds[0], bounds[1] - yy]),
            label=kind,
            marker="o",
            capsize=4,
        )
        axes[1].semilogy(
            xx,
            [r["conditional_ber"] if r["conditional_ber"] else np.nan for r in g],
            "-o",
            label=kind,
        )
    axes[0].set(
        xlabel="Injected ADC sample SNR (dB)",
        ylabel="Whole packet success (Wilson 95%)",
        ylim=(-0.03, 1.03),
    )
    axes[1].set(
        xlabel="Injected ADC sample SNR (dB)", ylabel="BER conditional on decodable payload"
    )
    for ax in axes:
        ax.legend()
        ax.grid()
    fig.suptitle(
        "Uncoded 255 B; 16 MS/s, 4 Msym/s; +100 ppm, +30 kHz; 24 independent repeats/point"
    )
    fig.tight_layout()
    fig.savefig(out / "low-snr-boundary.png", dpi=160)
    fig.savefig(out / "low-snr-boundary.svg")
    plt.close(fig)
    audit.write(
        out / "manifest.json",
        {str(p.relative_to(out)): audit.identity(p) for p in out.rglob("*") if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    run(parser.parse_args())
