"""Single-factor raw-angle carrier development; rejected/incomplete gates return1."""

import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import audit_carrier_loop as loop_audit
import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import numerical, validate_profiles
from audit_carrier_prefix import verdict
from audit_spline_fixed import independent_packet, summarize
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy.carrier_fixed import track_fixed_carrier
from nearlink_sdr.phy.carrier_parallel import track_parallel_carrier
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.carrier_parallel_flow import parallel_carrier_cost, simulate_parallel_carrier

POLICY = Path("tests/data/carrier_parallel_study.json")


def cases(held=False):
    policy = json.loads(POLICY.read_text())
    recipe = policy["held" if held else "development"]
    rng = np.random.default_rng(recipe["seed"])
    profiles = validate_profiles(recipe["configs"], recipe["pid"])
    for j, ((ft, mcs, pilot, length), profile) in enumerate(
        zip(recipe["configs"], profiles, strict=True)
    ):
        values = (np.arange(length) ** 2 * 13 + np.arange(length) * 73 + j * 43) % 256
        if j % 3 == 1:
            values = rng.integers(0, 256, length) ^ ((j * 19 + 53) & 255)
        elif j % 3 == 2:
            values = np.resize([7, 248, 51, 204, 129, 126, 15], length)
        if held:
            values = (values.astype(int) * 3 + np.arange(length) * 17 + 89) % 256
        payload = np.asarray(values, np.uint8).tobytes()
        cfo, ppm = (
            ((-33000, -100) if j % 2 else (57000, 50))
            if not held
            else ((-61000, 25) if j % 2 else (37000, -75))
        )
        clean, _, truth, step = iq_record(
            independent_packet(profile, payload),
            cfo=cfo,
            ppm=ppm,
            offset=0.319 if not held else 0.817,
        )
        prefix = 163 + j * 43 if not held else 257 + j * 47
        n = len(clean) + prefix + 320
        sigma = 1.9 if j < 4 else 2.9
        q = sigma / np.sqrt(2) * (rng.normal(size=2048) + 1j * rng.normal(size=2048))
        background = sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        k = np.arange(n)
        kind = ["two_tone", "noise", "burst_step", "wrong_sync"][j % 4]
        if kind == "two_tone":
            background += 2.9 * np.exp(2j * np.pi * 0.073117 * k + 0.71j) + 1.7 * np.exp(
                2j * np.pi * 0.076319 * k - 0.29j
            )
        elif kind == "wrong_sync":
            wrong_profile = PSKReceiveProfile.ordinary(
                TxConfig(
                    mcs_index=8, pid=7, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456
                )
            )
            wrong, *_ = iq_record(
                independent_packet(
                    wrong_profile,
                    b"distinct parallel-angle interferer packet;never desired payload",
                ),
                cfo=-19000,
                offset=0.473,
            )
            background += 0.095 * np.resize(wrong, n)
        elif kind == "burst_step":
            for start in (prefix + 443, n // 2 + 93):
                size = min(7, n - start)
                background[start : start + size] += 29 * np.exp(
                    1j * rng.uniform(-np.pi, np.pi, size)
                )
        amp = ((0.18, 0.46, 0.89) if not held else (0.23, 0.49, 0.95))[j % 3]
        iq = background.copy()
        iq[prefix : prefix + len(clean)] += amp * clean
        if kind == "burst_step":
            change = 1.3 if not held else 0.75
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
            pid=recipe["pid"],
            cfo_hz=cfo,
            ppm=ppm,
            sync_start=prefix + truth,
            step=step,
            amplitude=amp,
        )
        yield meta, profile, iq, q, payload
        if j < 4:
            yield (
                {**meta, "id": meta["id"] + "-negative", "positive": False},
                profile,
                background,
                q,
                b"",
            )


def arm(folder, meta, p, iq, q, payload, name):
    original = loop_audit.track_psk_carrier

    def word(y, mod, start, **kwargs):
        kwargs.pop("bandwidth")
        f = track_fixed_carrier if name == "q10" else track_parallel_carrier
        return f(y, mod, start, **kwargs)

    with patch.object(loop_audit, "track_psk_carrier", original if name == "float" else word):
        return loop_audit.run_arm(folder, meta, p, iq, q, payload, "medium-head")


def as_float_trace(arrays):
    converted = dict(arrays)
    for key, value in arrays.items():
        if key.endswith(("-phase_word", "-frequency_word")):
            converted[key[:-5]] = value * (2 * np.pi / 2**32)
    return converted


def numeric_gate(rows, b):
    return all(
        n["normalized_rms"] <= b["normalized_output_rms_delta"]
        and n["normalized_peak"] <= b["normalized_output_peak_delta"]
        and n["phase_rms"] <= b["phase_trajectory_rms_delta_rad"]
        and n["phase_peak"] <= b["phase_trajectory_peak_delta_rad"]
        and abs(n["mean_frequency_delta_hz"]) <= b["mean_frequency_delta_hz"]
        for n in rows
    )


def flow(out):
    rng = np.random.default_rng(780121)
    real = rng.integers(-2000, 2001, 512)
    imag = rng.integers(-2000, 2001, 512)
    rows = []
    for clock, batch, depth, blocked in [
        (c, 1, 4, b) for c in (100000000, 160000000, 200000000) for b in (False, True)
    ] + [(160000000, b, d, False) for b in (4, 8) for d in (4, 8)]:
        name = f"flow-{clock}-{batch}-{depth}-{blocked}.json"
        d = simulate_parallel_carrier(
            real,
            imag,
            clock_hz=clock,
            source_batch=batch,
            input_depth=depth,
            stalls=((0, 1000),) if blocked else (),
            modulation="8PSK",
            start=3,
            header_symbols=64,
        )
        write(out / name, d)
        rows.append(
            dict(
                clock=clock,
                batch=batch,
                depth=depth,
                blocked=blocked,
                emitted=len(d["emitted"]),
                dropped=len(d["dropped"]),
                resource_packet_invalid=d["resource_packet_invalid"],
                max_latency=d["max_latency"],
                file=name,
                sha256=sha(out / name),
            )
        )
    return dict(
        cost=parallel_carrier_cost(),
        depth8_cost=parallel_carrier_cost(8),
        scenarios=rows,
        whole_receiver16msps=False,
    )


def run(args):
    plan = json.loads(POLICY.read_text())
    held = args.split == "validation"
    if held:
        selected = json.loads(Path("tests/data/carrier_parallel_selection.json").read_text())
        assert selected["eligible"] and selected["policy_sha256"] == sha(POLICY)
        assert all(sha(Path(k)) == v for k, v in selected["source"].items())
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/audit_rx_soft_capture.py"),
            Path("scripts/audit_carrier_loop.py"),
            Path("scripts/audit_carrier_fixed.py"),
            Path("scripts/audit_spline_fixed.py"),
            Path("scripts/rx_timing_audit.py"),
            Path("tests/test_strict_receive.py"),
            Path("tests/test_psk_field_layout.py"),
            Path("pyproject.toml"),
            Path("uv.lock"),
            *Path("src").rglob("*.py"),
        )
    }
    write(
        out / "protocol.json",
        dict(
            plan=plan,
            policy_sha256=sha(POLICY),
            source=sources,
            held_generated=held,
            truth_used_for_runtime_selection=False,
            default_changed=False,
        ),
    )
    rows = {k: [] for k in ("float", "q10", "raw18")}
    numbers = {k: [] for k in ("float", "q10")}
    sat = 0
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
        paths = {}
        for name in rows:
            row, arrays = arm(out / (meta["id"] + "-" + name), meta, p, iq, q, payload, name)
            rows[name].append(row)
            paths[name] = arrays
        for ref in numbers:
            a = paths[ref] if ref == "float" else as_float_trace(paths[ref])
            assert all(
                np.array_equal(v, paths["raw18"][k]) for k, v in a.items() if k.endswith("-centers")
            )
            numbers[ref].extend(dict(id=meta["id"], **n) for n in numerical(a, paths["raw18"]))
        fits = json.loads((out / (meta["id"] + "-raw18") / "loop-fits.json").read_text())
        sat += sum(
            f["stats"][k]
            for f in fits
            for k in ("input_saturations", "output_saturations", "cordic_overflows")
        )
    summaries = {k: summarize(v) for k, v in rows.items()}
    gate = verdict(rows["float"], rows["raw18"], rows["q10"])
    numeric = {k: numeric_gate(v, plan["budget"]) for k, v in numbers.items()}
    false_candidates_ok = all(
        summaries["raw18"]["negative_candidates"] <= summaries[k]["negative_candidates"]
        for k in ("float", "q10")
    )
    gate.update(
        numerical_passed=numeric, saturations=sat, negative_candidates_passed=false_candidates_ok
    )
    gate["eligible"] &= all(numeric.values()) and sat == 0 and false_candidates_ok
    failures = [
        dict(
            arm=k,
            **r,
            frames=r["id"] + "-" + k + "/frames.json",
            state=r["id"] + "-" + k + "/loop-fits.json",
        )
        for k, v in rows.items()
        for r in v
        if r["positive"] and not r["correct"]
    ]
    model = dict(
        summary=summaries,
        gate=gate,
        numerical=numbers,
        failures=failures,
        flow=flow(out),
        held_generated=held,
        default_changed=False,
        whole16msps=False,
        all_original_failures_preserved=True,
    )
    for name, data in [("rows", rows), ("model", model)]:
        write(out / (name + ".json"), data)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(json.dumps(dict(summary=summaries, gate=gate), indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    raise SystemExit(run(parser.parse_args()))
