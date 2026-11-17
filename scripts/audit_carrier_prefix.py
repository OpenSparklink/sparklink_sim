"""Reproduce rejected prefix-only carrier choices on frozen new development inputs."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from audit_spline_fixed import independent_packet, receive, summarize
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy import strict_rx
from nearlink_sdr.phy.carrier_prefix import refine_psk_carrier_prefix
from nearlink_sdr.phy.psk import refine_psk_carrier
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.carrier_prefix_cost import carrier_prefix_cost


def development_cases():
    rng = np.random.default_rng(690117)
    configs = [
        (2, 8, 0, 1),
        (3, 6, 0, 31),
        (4, 12, 8, 97),
        (2, 6, 8, 287),
        (3, 0, 0, 859),
        (4, 8, 8, 31),
        (2, 12, 0, 859),
        (3, 8, 0, 287),
    ]
    for j, (ft, mcs, pilot, length) in enumerate(configs):
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            pid=4,
            whitening_seed=0x52,
            crc_seed=0x123456,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
            symbol_rate_mhz=4,
        )
        profile = PSKReceiveProfile.ordinary(cfg)
        values = (
            np.resize([255, 0, 128, 1, 91], length) if j % 3 == 0 else rng.integers(0, 256, length)
        )
        payload = np.asarray(values, np.uint8).tobytes()
        cfo, ppm = (-35000, 100) if j % 2 else (21000, -75)
        clean, _, truth, step = iq_record(
            independent_packet(profile, payload), cfo=cfo, ppm=ppm, offset=0.311
        )
        prefix = 119 + j * 23
        n = len(clean) + prefix + 256
        sigma = 1.4 if j < 4 else 2.5
        quiet = sigma / np.sqrt(2) * (rng.normal(size=2048) + 1j * rng.normal(size=2048))
        x = sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        kind = ["noise", "tone", "wrong_sync", "burst"][j % 4]
        k = np.arange(n)
        if kind == "tone":
            x += 3.3 * np.exp(-2j * np.pi * 0.097713 * k + 0.47j)
        if kind == "wrong_sync":
            other = PSKReceiveProfile.ordinary(
                TxConfig(
                    mcs_index=8, pid=7, whitening_seed=0x52, crc_seed=0x123456, symbol_rate_mhz=4
                )
            )
            wrong, *_ = iq_record(
                independent_packet(other, b"different prefix dev interference payload"),
                cfo=17000,
                offset=0.723,
            )
            x += 0.07 * np.resize(wrong, n)
        if kind == "burst":
            for start in (prefix + 419, n // 2 + 13):
                x[start : start + 9] += 32 * np.exp(
                    1j * rng.uniform(-np.pi, np.pi, min(9, n - start))
                )
        background = x.copy()
        amp = (0.14, 0.32, 0.75)[j % 3]
        x[prefix : prefix + len(clean)] += amp * clean
        meta = dict(
            id=f"dev-{j}",
            positive=True,
            frame_type=ft,
            mcs=mcs,
            pilot=pilot,
            pid=4,
            length=length,
            kind=kind,
            amplitude=amp,
            cfo_hz=cfo,
            ppm=ppm,
            sync_start=prefix + truth,
            step=step,
        )
        yield meta, profile, x, quiet, payload
        if j < 4:
            yield (
                {**meta, "id": meta["id"] + "-negative", "positive": False},
                profile,
                background,
                quiet,
                b"",
            )


def inputs(frozen=None):
    if frozen is None:
        yield from development_cases()
        return
    for path in sorted(frozen.glob("dev-*-metadata.json")):
        meta = json.loads(path.read_text())
        f = frozen / (meta["id"] + "-input.npz")
        assert sha(f) == meta["input_sha256"]
        ft = meta["frame_type"]
        profile = PSKReceiveProfile.ordinary(
            TxConfig(
                frame_type=ft,
                mcs_index=meta["mcs"],
                pilot_interval=meta["pilot"],
                pid=meta["pid"],
                whitening_seed=0x52,
                crc_seed=0x123456,
                ctrl_bits_len=28 if ft == 2 else 27,
                head_llid=None if ft == 2 else 0x234567,
                symbol_rate_mhz=4,
            )
        )
        with np.load(f) as ar:
            yield meta, profile, ar["iq"], ar["quiet"], ar["payload"].tobytes()


def receive_arm(folder, meta, profile, x, quiet, payload, limit, kappa, chunking="whole"):
    fits = []

    def chosen(symbols, mod, start, **kwargs):
        options = {**kwargs}
        if kappa is not None:
            options["carrier_weight_kappa"] = kappa
        if limit:
            result, info = refine_psk_carrier_prefix(
                symbols, mod, start, prefix_symbols=limit, **options
            )
        else:
            result, info = refine_psk_carrier(symbols, mod, start, **options)
        fits.append(dict(info=info, mod=mod, start=start, payload_wire_symbols=len(symbols)))
        return result, info

    with patch.object(strict_rx, "refine_psk_carrier", chosen):
        row, arrays, _ = receive(folder, meta, profile, x, quiet, payload, chunking=chunking)
    frames = json.loads((folder / "frames.json").read_text())
    diagnostics = []
    fit_index = 0
    for index, frame in enumerate(frames):
        if frame["status"] not in ("valid", "bad_data_crc"):
            continue
        # A short sync-only payload never calls the blind estimator.
        if fit_index >= len(fits):
            continue
        fit = fits[fit_index]
        fit_index += 1
        start, n = fit["start"], fit["payload_wire_symbols"]
        centers = arrays[f"{index}-centers"][start : start + n]
        if len(centers) > 1 and meta.get("cfo_hz") is not None:
            cps = frame["cfo_hz"] / 4e6 + fit["info"]["residual_cycles_per_symbol"]
            raw_step = float((centers[-1] - centers[0]) / (n - 1))
            error_hz = cps * 16e6 / raw_step - meta["cfo_hz"]
            drift = (
                2 * np.pi * (meta["cfo_hz"] * (centers - centers[0]) / 16e6 - cps * np.arange(n))
            )
            arrays[f"{index}-carrier-truth-relative-drift"] = drift
            diagnostics.append(
                dict(
                    frame=index,
                    after_mean_cfo_error_hz=error_hz,
                    relative_drift_first_mid_tail_rad=[float(drift[k]) for k in (0, n // 2, n - 1)],
                    initial_phase_truth_available=False,
                    meaning="carrier component from injected CFO and actual sample centers;anchored at first data sample,not total phase error or decision residual",
                )
            )
    np.savez_compressed(folder / "events.npz", **arrays)
    write(folder / "carrier-fits.json", fits)
    write(folder / "carrier-truth.json", diagnostics)
    return row, arrays


def verdict(base, candidate, weighted_base=None):
    references = [base] if weighted_base is None else [base, weighted_base]
    losses = []
    for reference in references:
        by_id = {r["id"]: r for r in reference}
        losses.extend(r["id"] for r in candidate if r["correct"] < by_id[r["id"]]["correct"])
    current = summarize(candidate)
    previous = [summarize(r) for r in references]
    missing = sum(r["ber_bits"] for r in candidate) != sum(r["ber_bits"] for r in base)
    return dict(
        success_losses=sorted(set(losses)),
        bit_errors=current["bit_errors"],
        baseline_bit_errors=[r["bit_errors"] for r in previous],
        ber_unmatched=missing,
        eligible=(
            not losses
            and not missing
            and all(current["bit_errors"] <= r["bit_errors"] for r in previous)
            and all(current["false_deliveries"] <= r["false_deliveries"] for r in previous)
        ),
        reason="requires zero losses,matched nonincreased BER and false deliveries;no CRC driven carrier retry",
    )


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    if args.inputs is not None:
        assert (
            args.manifest_sha256 is not None
            and sha(args.inputs / "manifest.json") == args.manifest_sha256
        )
        manifest = json.loads((args.inputs / "manifest.json").read_text())
        assert all(sha(args.inputs / k) == v for k, v in manifest.items())
    sources = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/audit_rx_soft_capture.py"),
            Path("scripts/audit_spline_fixed.py"),
            Path("tests/test_strict_receive.py"),
            Path("tests/test_psk_field_layout.py"),
            Path("scripts/rx_timing_audit.py"),
            Path("pyproject.toml"),
            Path("uv.lock"),
            *Path("src").rglob("*.py"),
        )
    }
    write(
        output / "protocol.json",
        dict(
            source=sources,
            developer_seed=690117,
            candidates=[128, 256],
            combination="only256 plus already frozen kappa4;paired full kappa4 control",
            adoption="zero original success loss,matched payload BER no worse,false deliveries no increase;all tests postselection",
            receiver="floatRRC,existingGardner/PI,scaledcoarse,bounded-grid1peak9phases,coherentQ.45,causalnotch31/32",
            held_not_opened_if_development_fails=True,
            word_or_fixedFFT_accepted=False,
            frozen_inputs_root=str(args.inputs) if args.inputs else None,
            frozen_manifest_sha256=args.manifest_sha256,
        ),
    )
    arms = {
        "full": (0, None),
        "prefix128": (128, None),
        "prefix256": (256, None),
        "full-weight4": (0, 4),
        "prefix256-weight4": (256, 4),
    }
    rows = {k: [] for k in arms}
    comparisons = []
    for meta, profile, x, quiet, payload in inputs(args.inputs):
        path = output / (meta["id"] + "-input.npz")
        if args.inputs is not None:
            shutil.copyfile(args.inputs / path.name, path)
        else:
            np.savez_compressed(path, iq=x, quiet=quiet, payload=np.frombuffer(payload, np.uint8))
        meta = {**meta, "input_sha256": sha(path)}
        write(
            output / (meta["id"] + "-format.json"),
            dict(
                profile=profile.name,
                frame_type=profile.config.frame_type,
                modulation=profile.config.mod_str,
                code_rate=profile.config.rate_str,
                uncoded=profile.config.is_uncoded,
                pilot_interval=profile.config.pilot_interval,
                sync_bits_sha256=hashlib.sha256(profile.sync_bits().tobytes()).hexdigest(),
                crc_seed=profile.config.crc_seed,
                whitening_seed=profile.config.whitening_seed,
                payload_bitorder=profile.payload_bitorder,
            ),
        )
        write(output / (meta["id"] + "-metadata.json"), meta)
        baseline_centers = None
        for arm, (limit, kappa) in arms.items():
            row, arrays = receive_arm(
                output / (meta["id"] + "-" + arm), meta, profile, x, quiet, payload, limit, kappa
            )
            rows[arm].append(row)
            centers = {k: v for k, v in arrays.items() if k.endswith("-centers")}
            if baseline_centers is None:
                baseline_centers = centers
            else:
                exact = centers.keys() == baseline_centers.keys() and all(
                    np.array_equal(v, baseline_centers[k]) for k, v in centers.items()
                )
                comparisons.append(dict(id=meta["id"], arm=arm, capture_timing_exact=exact))
                assert exact
    summary = {k: summarize(r) for k, r in rows.items()}
    gates = {
        k: verdict(rows["full"], rows[k], rows["full-weight4"] if "weight4" in k else None)
        for k in ("prefix128", "prefix256", "prefix256-weight4")
    }
    eligible = [k for k, r in gates.items() if r["eligible"]]
    write(
        output / "selection.json",
        dict(
            development_gates=gates,
            eligible=eligible,
            selected=None,
            frozen=True,
            held_evaluated=False,
            disposition="rejected;no held exposure"
            if not eligible
            else "additional fixed regression and independent held required;not yet accepted",
        ),
    )
    # Recheck one already-viewed original success, veto only;not new acceptance data.
    fixture = Path("tests/data/sync_carrier_tone_success.npz")
    frozen = json.loads(fixture.with_suffix(".json").read_text())
    assert sha(fixture) == frozen["derived_npz_sha256"]
    with np.load(fixture) as ar:
        x, quiet, payload = ar["iq"], ar["quiet"], ar["expected"].tobytes()
    profile = PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=6, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )
    original_truth = frozen["source_input"]["packets"][0]
    assert hashlib.sha256(x.tobytes()).hexdigest() == frozen["iq_bytes_sha256"]
    old_meta = dict(
        id="old-success",
        positive=True,
        input_sha256=sha(fixture),
        source_input_sha256=frozen["source_input"]["sha256"],
        sync_start=original_truth["sync_start"],
        step=original_truth["step"],
        cfo_hz=original_truth["cfo_hz"],
        ppm=original_truth["ppm"],
    )
    write(output / "old-success-metadata.json", old_meta)
    old = []
    for arm, (limit, kappa) in arms.items():
        for chunking in ("whole", "fixed", "random"):
            row, _ = receive_arm(
                output / ("old-success-" + arm + "-" + chunking),
                old_meta,
                profile,
                x,
                quiet,
                payload,
                limit,
                kappa,
                chunking,
            )
            old.append(dict(arm=arm, chunking=chunking, summary=row))
    old_losses = [
        dict(arm=r["arm"], chunking=r["chunking"])
        for r in old
        if r["arm"] != "full" and r["summary"]["correct"] < 1
    ]
    failures = []
    for arm, values in rows.items():
        for r in values:
            if r["positive"] and not r["correct"]:
                meta = json.loads((output / (r["id"] + "-metadata.json")).read_text())
                failures.append(
                    dict(
                        arm=arm,
                        id=r["id"],
                        input_sha256=meta["input_sha256"],
                        bit_errors=r["bit_errors"],
                        ber_bits=r["ber_bits"],
                        head_crc=r["head_crc"],
                        data_crc=r["data_crc"],
                        frames=r["id"] + "-" + arm + "/frames.json",
                        carrier=r["id"] + "-" + arm + "/carrier-fits.json",
                        reason="captured,header CRC good,payload CRC failed;not a theoretical unrecoverability claim",
                    )
                )
    for r in old:
        if r["summary"]["correct"] < 1:
            failures.append(
                dict(
                    arm=r["arm"],
                    id="old-success",
                    chunking=r["chunking"],
                    input_sha256=sha(fixture),
                    bit_errors=r["summary"]["bit_errors"],
                    ber_bits=r["summary"]["ber_bits"],
                    head_crc=r["summary"]["head_crc"],
                    data_crc=r["summary"]["data_crc"],
                    frames="old-success-" + r["arm"] + "-" + r["chunking"] + "/frames.json",
                    carrier="old-success-" + r["arm"] + "-" + r["chunking"] + "/carrier-fits.json",
                    reason="original successful fixed regression lost;veto only,not a parameter tuning set",
                )
            )
    write(output / "rows.json", rows)
    write(output / "summary.json", summary)
    write(output / "failures.json", failures)
    write(output / "old-success.json", old)
    costs = [
        carrier_prefix_cost(n, complex_bits=bits, clock_hz=clock)
        for n in (128, 256)
        for bits in (36, 64)
        for clock in (100_000_000, 160_000_000, 200_000_000)
    ]
    model = dict(
        summary=summary,
        development_gates=gates,
        rejected=not eligible,
        held_evaluated=False,
        old_fixture=old,
        original_fixture_success_losses=old_losses,
        capture_timing_comparisons=comparisons,
        costs=costs,
        failures=failures,
        complete16msps_verified=False,
        fixedFFT_format_accepted=False,
        default_receiver_changed=False,
        original_rrc2percent_and16KiB_failures_preserved=True,
        next="prefix alone failed;any continuing carrier loop is a separate fresh development experiment",
    )
    write(output / "model.json", model)
    write(
        output / "manifest.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            dict(summary=summary, gates=gates, rejected=not eligible, held_evaluated=False),
            indent=2,
        )
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--manifest-sha256")
    raise SystemExit(run(parser.parse_args()))
