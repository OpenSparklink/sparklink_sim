"""Raw-IQ coded PSK comparison with fixed independent noise splits and actual acquisition."""

import argparse
import hashlib
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_strict_receive import iq_record, transmitted  # noqa: E402

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL  # noqa: E402
from nearlink_sdr.phy.psk import refine_psk_carrier  # noqa: E402
from nearlink_sdr.phy.psk_fields import demap_psk_field, soft_demap_psk_field  # noqa: E402
from nearlink_sdr.phy.rx_pipeline import decode_payload  # noqa: E402
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver  # noqa: E402
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile  # noqa: E402
from nearlink_sdr.phy.tx_pipeline import TxConfig  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def diagnostic_bits(e, profile):
    f = e["frame"]
    if f.get("status") not in ("valid", "bad_data_crc"):
        return None
    cfg = replace(profile.config, mcs_index=f["mcs"])
    start, end = f["data_start_symbol"], f["frame_end_symbol"]
    carrier = f.get("carrier", {})
    methods = {
        "blind de-rotated Mth-power coherent FFT/Newton; no expected bits": "coherent",
        "blind de-rotated Mth-power FFT+phase regression; no expected bits": "phase_regression",
        "sync estimate only; fewer than 32 payload wire symbols": "sync_only",
        "causal bounded phase-detector PI;sync-only initialization;no expected bits": "causal_pi",
        "integer causal carrier PI;actual header preroll;no expected bits": "fixed_pi",
        "parallel raw-angle integer carrier PI;actual header preroll;no expected bits": "parallel_pi",
    }
    method = methods.get(carrier.get("method"))
    if method is None:
        raise ValueError("Missing or unknown actual carrier method; BER replay is unverified")
    y = e["symbols"][start:end]
    if (method == "sync_only") != (len(y) < 32):
        raise ValueError("Carrier method is inconsistent with actual payload symbol count")
    if method != "sync_only":
        info = carrier.get("weights")
        options = (
            {}
            if info is None
            else dict(carrier_weight_kappa=info["kappa"], carrier_weight_fixed=info["fixed"])
        )
        # Replay only the actual selected method. CRC is checked afterwards,
        # never used to retry a different carrier interpretation.
        if method in ("causal_pi", "fixed_pi", "parallel_pi"):
            from nearlink_sdr.phy.carrier_loop import track_psk_carrier

            if carrier.get("symbols") != len(y) or carrier.get("start_symbol_index") != start:
                raise ValueError("Carrier PI metadata does not match actual field")
            warm = carrier.get("header_symbols", 0)
            if type(warm) is not int or not 0 <= warm <= start:
                raise ValueError("Invalid actual header pre-roll metadata")
            if warm and (
                warm != start - len(profile.sync_symbols())
                or carrier.get("header_modulation") != ("BPSK" if cfg.frame_type == 4 else "QPSK")
            ):
                raise ValueError("Carrier pre-roll does not match actual header profile")
            options = (
                {}
                if not warm
                else dict(
                    header_symbols=e["symbols"][start - warm : start],
                    header_mod_type=carrier["header_modulation"],
                )
            )
            if method == "parallel_pi":
                from nearlink_sdr.phy.carrier_parallel import track_parallel_carrier

                y, _, _ = track_parallel_carrier(
                    y, cfg.mod_str, start, fraction_bits=carrier["fraction_bits"], **options
                )
            elif method == "fixed_pi":
                from nearlink_sdr.phy.carrier_fixed import track_fixed_carrier

                y, _, _ = track_fixed_carrier(
                    y, cfg.mod_str, start, fraction_bits=carrier["fraction_bits"], **options
                )
            else:
                y, _, _ = track_psk_carrier(
                    y, cfg.mod_str, start, bandwidth=carrier["bandwidth"], **options
                )
        elif "prefix_symbols" in carrier:
            from nearlink_sdr.phy.carrier_prefix import refine_psk_carrier_prefix

            prefix = carrier["prefix_symbols"]
            if carrier.get("estimation_symbols") != min(prefix, len(y)):
                raise ValueError("Prefix carrier metadata does not match actual estimation count")
            y, _ = refine_psk_carrier_prefix(
                y, cfg.mod_str, start, prefix_symbols=prefix, method=method, **options
            )
        else:
            y, _ = refine_psk_carrier(y, cfg.mod_str, start, method=method, **options)
    hard = demap_psk_field(y, cfg.mod_str, start, cfg.pilot_interval)[: f["coded_bits"]]
    llr = None
    if not cfg.is_uncoded and profile.soft_payload:
        info = f["soft_information"]
        llr = soft_demap_psk_field(
            y, cfg.mod_str, start, info["variance"], cfg.pilot_interval, method=info["method"]
        )[: f["coded_bits"]]
    bits, ok = decode_payload(hard, cfg, f["header"]["data_length"] * 8, llr=llr)
    assert bool(ok) == f["data_crc"]
    return bits


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    plan = dict(
        rust=_HAS_RUST_ACCEL,
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")},
        driver_sha256=sha(Path(__file__)),
        seed_development=101016,
        seed_held_out=102016,
        repeats=8,
        snr_db=[4, 6, 8],
        mcs=[6, 10],
        payload_bytes=255,
        pilot_interval=8,
        injected_sro_ppm=100,
        cfo_hz=30000,
        offset=0.277,
        variance="ADC complex noise 2500*10^(-snr_db/10)",
        payload="fixed random vector seed65016; independent held-out noise, not new content",
        arms=["hard", "maxlog"],
        inputs=[],
    )
    plan["reference_sources"] = {
        f: sha(ROOT / f)
        for f in [
            "tests/test_strict_receive.py",
            "tests/test_psk_field_layout.py",
            "scripts/rx_timing_audit.py",
        ]
    }
    prepared = []
    for mcs in plan["mcs"]:
        cfg = TxConfig(mcs_index=mcs, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
        profile = PSKReceiveProfile.ordinary(cfg)
        wire, expected = transmitted(profile, length=255)
        clean, _, _, _ = iq_record(wire, ppm=100, cfo=30000)
        for split, seed in [("development", 101016), ("held-out", 102016)]:
            for snr in plan["snr_db"]:
                for repeat in range(8):
                    ident = f"{split}-m{mcs}-snr{snr}-r{repeat}"
                    path = out / "evidence" / (ident + "-input.npz")
                    if args.inputs:
                        source = args.inputs / "evidence" / path.name
                        oldplan = json.loads((args.inputs / "frozen-plan.json").read_text())
                        entry = next(v for v in oldplan["inputs"] if v["id"] == ident)
                        assert sha(source) == entry["sha256"]
                        with np.load(source) as data:
                            x, q = data["iq"], data["quiet"]
                            assert data["expected"].tobytes() == expected
                        path = source
                    else:
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, mcs, snr, repeat])
                        )
                        sigma = 50 * 10 ** (-snr / 20)

                        def noise(n):
                            return (
                                sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
                            )

                        x, q = clean + noise(len(clean)), noise(8192)
                        np.savez_compressed(
                            path, iq=x, quiet=q, expected=np.frombuffer(expected, dtype=np.uint8)
                        )
                    meta = dict(
                        id=ident,
                        mcs=mcs,
                        split=split,
                        snr_db=snr,
                        repeat=repeat,
                        file=str(path),
                        sha256=sha(path),
                    )
                    plan["inputs"].append(meta)
                    prepared.append((meta, x, q, expected, profile))
    write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for meta, x, q, expected, profile in prepared:
        row = dict(meta, arms={})
        arrays = {}
        reference = None
        for soft in (False, True):
            method = "maxlog" if soft else "hard"
            selected = replace(profile, soft_payload=soft)
            begin = time.perf_counter()
            rx = PSKStreamReceiver(selected, q)
            events = rx.feed(x) + rx.finish()
            seconds = time.perf_counter() - begin
            if reference is None:
                reference = events
            assert len(reference) == len(events)
            for a, b in zip(reference, events, strict=True):
                for key in (
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                ):
                    np.testing.assert_array_equal(a[key], b[key])
            frames = []
            for j, e in enumerate(events):
                f = e["frame"]
                bits = diagnostic_bits(e, selected)
                target = np.unpackbits(np.frombuffer(expected, dtype=np.uint8))
                frames.append(
                    dict(
                        status=f["status"],
                        header_crc=f["header_crc"],
                        data_crc=f["data_crc"],
                        length_ok=f["length_ok"],
                        mcs=f.get("mcs"),
                        correct=f["payload"] == expected
                        and f["header"]["data_length"] == 255
                        and f["mcs"] == meta["mcs"]
                        if f["status"] == "valid"
                        else False,
                        bit_errors=None if bits is None else int(np.sum(bits != target)),
                    )
                )
                for key in (
                    "symbols",
                    "symbol_centers_raw",
                    "timing_rate_ppm",
                    "timing_detector_error",
                ):
                    arrays[f"{method}-{j}-{key}"] = e[key]
                if bits is not None:
                    arrays[f"{method}-{j}-decoded"] = bits
            row["arms"][method] = dict(frames=frames, seconds=seconds)
        np.savez_compressed(out / "evidence" / (meta["id"] + "-results.npz"), **arrays)
        rows.append(row)
    summary = []
    for split in ("development", "held-out"):
        for mcs in plan["mcs"]:
            for snr in plan["snr_db"]:
                subset = [
                    r
                    for r in rows
                    if r["split"] == split and r["mcs"] == mcs and r["snr_db"] == snr
                ]
                for arm in plan["arms"]:
                    frames = [f for r in subset for f in r["arms"][arm]["frames"]]
                    comparable = [f["bit_errors"] for f in frames if f["bit_errors"] is not None]
                    summary.append(
                        dict(
                            split=split,
                            mcs=mcs,
                            snr_db=snr,
                            arm=arm,
                            inputs=len(subset),
                            correct=sum(f["correct"] for f in frames),
                            misses=sum(not r["arms"][arm]["frames"] for r in subset),
                            header_failures=sum(f["header_crc"] is False for f in frames),
                            conditional_ber=sum(comparable) / (len(comparable) * 2040)
                            if comparable
                            else None,
                            comparable_packets=len(comparable),
                        )
                    )
    write(out / "results.json", rows)
    write(out / "summary.json", summary)
    write(
        out / "failures.json",
        [
            r
            for r in rows
            if any(not any(f["correct"] for f in a["frames"]) for a in r["arms"].values())
        ],
    )
    write(
        out / "regressions.json",
        [
            r
            for r in rows
            if any(f["correct"] for f in r["arms"]["hard"]["frames"])
            and not any(f["correct"] for f in r["arms"]["maxlog"]["frames"])
        ],
    )
    assert all(sha(ROOT / p) == h for p, h in plan["source_sha256"].items())
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    run(parser.parse_args())
