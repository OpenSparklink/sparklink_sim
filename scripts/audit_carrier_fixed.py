"""Fresh IQ/representation study of exact carrier words; whole-system gate stays open."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import audit_carrier_loop as loop_audit
import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_prefix import verdict
from audit_spline_fixed import independent_packet, summarize
from check_frozen_corpora import verify_group
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy.carrier_fixed import track_fixed_carrier
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.carrier_fixed_flow import carrier_word_cost, simulate_carrier_words


def validate_profiles(configs, pid):
    profiles = []
    for ft, mcs, pilot, _ in configs:
        if ft in (3, 4) and not 0 <= pid <= 5:
            raise ValueError("FT3/4 actual m-sequence index must be0..5")
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            pid=pid,
            symbol_rate_mhz=4,
            whitening_seed=0x52,
            crc_seed=0x123456,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
        )
        # This published corpus predates the standard B1 wire-length correction.
        p = PSKReceiveProfile.ordinary(
            cfg, length_contract="standard" if ft == 2 else "legacy-b1-payload-length"
        )
        p.sync_symbols()
        profiles.append(p)
    return profiles


def cases(held=False):
    recipe = json.loads(Path("tests/data/carrier_fixed_validation.json").read_text())
    rng = np.random.default_rng(recipe["seed"] if held else 730117)
    configs = [
        (3, 8, 8, 5),
        (4, 12, 0, 59),
        (2, 6, 0, 193),
        (3, 12, 8, 487),
        (4, 0, 0, 937),
        (2, 8, 8, 59),
        (3, 6, 8, 937),
        (4, 12, 8, 193),
    ]
    if held:
        configs = recipe["configs"]
    profiles = validate_profiles(configs, recipe["pid"] if held else 1)
    for j, (ft, mcs, pilot, length) in enumerate(configs):
        p = profiles[j]
        cfg = p.config
        values = (
            (np.arange(length) * (47 if held else 109) + j * 31) % 256
            if j % 2
            else rng.integers(0, 256, length) ^ ((j * 17 + 29) & 255)
        )
        payload = np.asarray(values, np.uint8).tobytes()
        cfo, ppm = (
            ((21000, -75) if j % 2 else (-51000, 25))
            if held
            else ((29000, 100) if j % 2 else (-39000, -50))
        )
        x, _, truth, step = iq_record(
            independent_packet(p, payload), cfo=cfo, ppm=ppm, offset=0.731 if held else 0.257
        )
        prefix = 239 + j * 41 if held else 137 + j * 37
        n = len(x) + prefix + 320
        sigma = 1.7 if j < 4 else 2.7
        q = sigma / np.sqrt(2) * (rng.normal(size=2048) + 1j * rng.normal(size=2048))
        background = sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        kind = ["noise", "two_tone", "wrong_sync", "burst_step"][j % 4]
        if kind == "two_tone":
            k = np.arange(n)
            background += 3.1 * np.exp(2j * np.pi * 0.089113 * k + 0.39j) + 1.3 * np.exp(
                -2j * np.pi * 0.137517 * k
            )
        if kind == "wrong_sync":
            other = PSKReceiveProfile.ordinary(
                TxConfig(
                    mcs_index=8, pid=7, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456
                )
            )
            wrong, *_ = iq_record(
                independent_packet(other, b"word-format distinct interferer content"),
                cfo=11000,
                offset=0.563,
            )
            background += 0.085 * np.resize(wrong, n)
        if kind == "burst_step":
            for start in (prefix + 389, n // 2 + 57):
                background[start : start + 5] += 31 * np.exp(
                    1j * rng.uniform(-np.pi, np.pi, min(5, n - start))
                )
        iq = background.copy()
        amp = ((0.21, 0.43, 0.93) if held else (0.17, 0.39, 0.87))[j % 3]
        iq[prefix : prefix + len(x)] += amp * x
        if kind == "burst_step":
            change = 0.8 if held else 1.25
            iq[n // 2 :] *= change
            background[n // 2 :] *= change
        meta = dict(
            id=f"{'validation' if held else 'development'}-{j}",
            positive=True,
            kind=kind,
            frame_type=ft,
            mcs=mcs,
            pilot=pilot,
            length=length,
            pid=cfg.pid,
            cfo_hz=cfo,
            ppm=ppm,
            sync_start=prefix + truth,
            step=step,
            amplitude=amp,
        )
        yield meta, p, iq, q, payload
        if j < 4:
            yield {**meta, "id": meta["id"] + "-negative", "positive": False}, p, background, q, b""


def receive_arm(folder, meta, p, iq, q, payload, arm):
    original = loop_audit.track_psk_carrier

    def fixed(y, mod, start, **kwargs):
        kwargs.pop("bandwidth")
        return track_fixed_carrier(y, mod, start, fraction_bits=int(arm[1:]), **kwargs)

    with patch.object(loop_audit, "track_psk_carrier", original if arm == "float" else fixed):
        return loop_audit.run_arm(folder, meta, p, iq, q, payload, "medium-head")


def numerical(reference, candidate):
    rows = []
    for k, a in reference.items():
        if not k.endswith("-corrected_symbols"):
            continue
        b = candidate[k]
        scale = max(float(np.sqrt(np.mean(abs(a) ** 2))), 1e-30)
        rms = float(np.sqrt(np.mean(abs(a - b) ** 2))) / scale
        peak = float(np.max(abs(a - b))) / max(float(np.max(abs(a))), 1e-30)
        prefix = k.rsplit("-", 1)[0]
        phase = np.unwrap(candidate[prefix + "-phase_word"] * 2 * np.pi / 2**32) - np.unwrap(
            reference[prefix + "-phase"]
        )
        freq = (
            (
                candidate[prefix + "-frequency_word"] * 2 * np.pi / 2**32
                - reference[prefix + "-frequency"]
            )
            * 4e6
            / (2 * np.pi)
        )
        rows.append(
            dict(
                trace=k,
                normalized_rms=rms,
                normalized_peak=peak,
                phase_rms=float(np.sqrt(np.mean(phase**2))),
                phase_peak=float(np.max(abs(phase))),
                mean_frequency_delta_hz=float(np.mean(freq)),
            )
        )
    return rows


def flow_audit(out):
    rng = np.random.default_rng(730121)
    real = rng.integers(-2000, 2001, 512)
    imag = rng.integers(-2000, 2001, 512)
    scenarios = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        for stalls in ((), ((0, 1000),)):
            report = simulate_carrier_words(
                real,
                imag,
                clock_hz=clock,
                stalls=stalls,
                modulation="8PSK",
                start=3,
                header_symbols=64,
            )
            name = f"flow-{clock}-{'blocked' if stalls else 'ready'}.json"
            write(out / name, report)
            scenarios.append(
                dict(
                    clock_hz=clock,
                    stalls=stalls,
                    resource_packet_invalid=report["resource_packet_invalid"],
                    dropped=len(report["dropped"]),
                    emitted=len(report["emitted"]),
                    peak_input=report["peak_input"],
                    peak_output=report["peak_output"],
                    max_latency=report["max_latency"],
                    file=name,
                    sha256=sha(out / name),
                )
            )
    for depth in (4, 8):
        for batch in (4, 8):
            report = simulate_carrier_words(
                real,
                imag,
                clock_hz=200_000_000,
                source_batch=batch,
                input_depth=depth,
                modulation="8PSK",
                start=3,
                header_symbols=64,
            )
            name = f"flow-batch{batch}-depth{depth}.json"
            write(out / name, report)
            scenarios.append(
                dict(
                    clock_hz=200_000_000,
                    stalls=(),
                    source_batch=batch,
                    input_depth=depth,
                    resource_packet_invalid=report["resource_packet_invalid"],
                    dropped=len(report["dropped"]),
                    emitted=len(report["emitted"]),
                    peak_input=report["peak_input"],
                    peak_output=report["peak_output"],
                    max_latency=report["max_latency"],
                    file=name,
                    sha256=sha(out / name),
                )
            )
    return dict(
        cost=carrier_word_cost(),
        depth8_cost=carrier_word_cost(8),
        scenarios=scenarios,
        integer_cycle_word_oracle="tests/test_carrier_fixed_flow.py,exact values+state,not clocks alone",
        full_rx16msps_verified=False,
        primitive_timing_synthesized=False,
    )


def run(args):
    held = args.split == "validation"
    selection = Path("tests/data/carrier_fixed_format.json")
    selected = json.loads(selection.read_text())
    if held:
        assert all(sha(Path(p)) == v for p, v in selected["source_sha256"].items())
        assert selected["fraction_bits"] == 10
        recipe = json.loads(Path("tests/data/carrier_fixed_validation.json").read_text())
        assert recipe["selected_format_sha256"] == sha(selection)
        assert recipe["budget_sha256"] == sha(Path("tests/data/carrier_fixed_budget.json"))
        verify_group("fixed-validation")
    arms = ("float", "q10") if held else ("float", "q10", "q12")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    policy = Path("tests/data/carrier_fixed_budget.json")
    budget = json.loads(policy.read_text())
    rows = {k: [] for k in arms}
    numbers = {k: [] for k in arms if k != "float"}
    saturations = {k: 0 for k in numbers}
    write(
        out / "protocol.json",
        dict(
            seed=json.loads(Path("tests/data/carrier_fixed_validation.json").read_text())["seed"]
            if held
            else 730117,
            validation_recipe_sha256=sha(Path("tests/data/carrier_fixed_validation.json")),
            split=args.split,
            selection_sha256=sha(selection),
            budget_sha256=sha(policy),
            source={
                str(p): sha(p)
                for p in (
                    Path(__file__),
                    Path("scripts/audit_carrier_loop.py"),
                    Path("scripts/audit_rx_soft_capture.py"),
                    Path("scripts/audit_spline_fixed.py"),
                    Path("tests/test_strict_receive.py"),
                    Path("tests/test_psk_field_layout.py"),
                    Path("scripts/rx_timing_audit.py"),
                    Path("pyproject.toml"),
                    Path("uv.lock"),
                    *Path("src").rglob("*.py"),
                )
            },
            held_generated=held,
            default_changed=False,
            truth_used_for_runtime_selection=False,
        ),
    )
    for meta, p, iq, q, payload in cases(held):
        path = out / (meta["id"] + "-input.npz")
        np.savez_compressed(path, iq=iq, quiet=q, payload=np.frombuffer(payload, np.uint8))
        meta = {**meta, "input_sha256": sha(path)}
        write(out / (meta["id"] + "-metadata.json"), meta)
        write(
            out / (meta["id"] + "-format.json"),
            dict(
                profile=p.name,
                modulation=p.config.mod_str,
                code_rate=p.config.rate_str,
                uncoded=p.config.is_uncoded,
                pilot_interval=p.config.pilot_interval,
                crc_seed=p.config.crc_seed,
                whitening_seed=p.config.whitening_seed,
                sync_bits_sha256=hashlib.sha256(p.sync_bits().tobytes()).hexdigest(),
                bitorder=p.payload_bitorder,
            ),
        )
        reference = None
        for arm in rows:
            row, arrays = receive_arm(out / (meta["id"] + "-" + arm), meta, p, iq, q, payload, arm)
            rows[arm].append(row)
            if arm == "float":
                reference = arrays
            else:
                assert all(
                    np.array_equal(v, reference[k])
                    for k, v in arrays.items()
                    if k.endswith("-centers")
                )
                for n in numerical(reference, arrays):
                    numbers[arm].append(dict(id=meta["id"], **n))
                fits = json.loads((out / (meta["id"] + "-" + arm) / "loop-fits.json").read_text())
                saturations[arm] += sum(
                    f["stats"]["input_saturations"]
                    + f["stats"]["output_saturations"]
                    + f["stats"]["cordic_overflows"]
                    for f in fits
                )
    summary = {k: summarize(r) for k, r in rows.items()}
    gates = {k: verdict(rows["float"], rows[k]) for k in numbers}
    for arm, gate in gates.items():
        numeric = all(
            n["normalized_rms"] <= budget["normalized_output_rms_delta"]
            and n["normalized_peak"] <= budget["normalized_output_peak_delta"]
            and n["phase_rms"] <= budget["phase_trajectory_rms_delta_rad"]
            and n["phase_peak"] <= budget["phase_trajectory_peak_delta_rad"]
            and abs(n["mean_frequency_delta_hz"]) <= budget["mean_frequency_delta_hz"]
            for n in numbers[arm]
        )
        gate.update(numerical_passed=numeric, saturations=saturations[arm])
        gate["eligible"] = gate["eligible"] and numeric and saturations[arm] == 0
    failures = [
        dict(
            arm=arm,
            **row,
            frames=row["id"] + "-" + arm + "/frames.json",
            state=row["id"] + "-" + arm + "/loop-fits.json",
        )
        for arm, values in rows.items()
        for row in values
        if row["positive"] and not row["correct"]
    ]
    write(out / "rows.json", rows)
    write(out / "numerical.json", numbers)
    write(out / "failures.json", failures)
    model = dict(
        flow=flow_audit(out),
        summary=summary,
        gates=gates,
        numerical=numbers,
        failures=failures,
        held_generated=held,
        default_changed=False,
        complete16msps=False,
        normalized_gain_bridge_fixed=False,
        integer_core=True,
        feedback_cycle_verified=False,
        all_original_failures_preserved=True,
    )
    write(out / "model.json", model)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(dict(summary=summary, gates=gates), indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    raise SystemExit(run(parser.parse_args()))
