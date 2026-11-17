"""Fresh development of causal PSK carrier PI; rejected/incomplete gate returns1."""

import argparse
import ast
import hashlib
import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_prefix import verdict
from audit_spline_fixed import independent_packet, receive, summarize
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy import stream_rx, strict_rx
from nearlink_sdr.phy.carrier_loop import track_psk_carrier
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.carrier_loop_cost import carrier_loop_cost


def development(held=False):
    rng = np.random.default_rng(720117 if held else 710117)
    configs = [
        (2, 6, 8, 3),
        (3, 12, 0, 43),
        (4, 8, 8, 181),
        (2, 12, 8, 443),
        (3, 6, 0, 911),
        (4, 0, 8, 43),
        (2, 8, 0, 911),
        (3, 12, 8, 181),
    ]
    if held:
        configs = [
            (4, 12, 0, 7),
            (2, 8, 8, 79),
            (3, 6, 8, 227),
            (4, 12, 8, 557),
            (2, 0, 0, 1021),
            (3, 8, 0, 227),
            (4, 6, 0, 1021),
            (2, 12, 8, 79),
        ]
    for j, (ft, mcs, pilot, length) in enumerate(configs):
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            pid=5 if held else 2,
            whitening_seed=0x52,
            crc_seed=0x123456,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
            symbol_rate_mhz=4,
        )
        p = PSKReceiveProfile.ordinary(cfg)
        values = (
            (np.arange(length) * 113 + j * 29) % 256
            if j % 3 == 0
            else (
                np.resize([254, 1, 127, 0, 239], length)
                if j % 3 == 1
                else rng.integers(0, 256, length)
            )
        )
        if held:
            values = (
                (np.arange(length) * 53 + j * 71) % 256
                if j % 3 == 0
                else np.resize([17, 238, 85, 170, 3, 252], length)
                if j % 3 == 1
                else rng.integers(0, 256, length)
            )
        payload = np.asarray(values, np.uint8).tobytes()
        cfo, ppm = (
            ((23000, -25) if j % 2 else (-47000, 75))
            if held
            else ((-17000, 50) if j % 2 else (43000, -100))
        )
        clean, _, truth, step = iq_record(
            independent_packet(p, payload), cfo=cfo, ppm=ppm, offset=0.683 if held else 0.417
        )
        prefix = 223 + j * 31 if held else 181 + j * 29
        n = len(clean) + prefix + 320
        sigma = (1.8 if j < 4 else 2.8) if held else (2 if j < 4 else 3)
        quiet = sigma / np.sqrt(2) * (rng.normal(size=2048) + 1j * rng.normal(size=2048))
        x = sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        kind = (
            ["tone_burst", "double_tone", "wrong_noise", "burst_step"]
            if held
            else ["noise", "tone", "wrong_sync", "burst"]
        )[j % 4]
        if "tone" in kind:
            x += 3.7 * np.exp(2j * np.pi * 0.071939 * np.arange(n) - 0.37j)
        if kind == "double_tone":
            x += 1.9 * np.exp(-2j * np.pi * 0.119717 * np.arange(n) + 0.53j)
        if "wrong" in kind:
            other = PSKReceiveProfile.ordinary(
                TxConfig(
                    mcs_index=8, pid=7, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456
                )
            )
            wrong, *_ = iq_record(
                independent_packet(other, b"new looping interfering payload;not desired PID"),
                cfo=-13000,
                offset=0.129,
            )
            x += 0.09 * np.resize(wrong, n)
        if "burst" in kind:
            for start in (prefix + 311, n // 2 + 71) if held else (prefix + 537, n // 2 + 31):
                x[start : start + 7] += 29 * np.exp(
                    1j * rng.uniform(-np.pi, np.pi, min(7, n - start))
                )
        background = x.copy()
        amplitude = ((0.16, 0.41, 0.91) if held else (0.19, 0.47, 0.83))[j % 3]
        x[prefix : prefix + len(clean)] += amplitude * clean
        if j == 7:
            change = 0.7 if held else 1.4
            x[n // 2 :] *= change
            background[n // 2 :] *= change
        meta = dict(
            id=f"{'validation' if held else 'dev'}-{j}",
            positive=True,
            frame_type=ft,
            mcs=mcs,
            pilot=pilot,
            pid=5 if held else 2,
            length=length,
            kind=kind,
            amplitude=amplitude,
            cfo_hz=cfo,
            ppm=ppm,
            sync_start=prefix + truth,
            step=step,
        )
        yield meta, p, x, quiet, payload
        if j < 4:
            yield (
                {**meta, "id": meta["id"] + "-negative", "positive": False},
                p,
                background,
                quiet,
                b"",
            )


@contextmanager
def carrier_adapter(arm, profile):
    fits = []
    traces = []
    original = strict_rx.refine_psk_carrier
    decoder = stream_rx.decode_psk_frame
    active = {}

    def decoding(symbols, selected_profile, **kwargs):
        active["symbols"] = symbols
        active["nsync"] = len(selected_profile.sync_symbols())
        return decoder(symbols, selected_profile, **kwargs)

    def chosen(symbols, mod, start, **kwargs):
        if arm == "full":
            return original(symbols, mod, start, **kwargs)
        assert kwargs.get("method", "coherent") == "coherent"
        options = (
            {}
            if not arm.endswith("-head")
            else dict(
                header_symbols=active["symbols"][active["nsync"] : start],
                header_mod_type="BPSK" if profile.config.frame_type == 4 else "QPSK",
            )
        )
        out, info, trace = track_psk_carrier(
            symbols, mod, start, bandwidth=arm.split("-")[0], **options
        )
        fits.append(info)
        traces.append(trace)
        return out, info

    with (
        patch.object(strict_rx, "refine_psk_carrier", chosen),
        patch.object(stream_rx, "decode_psk_frame", decoding),
    ):
        yield fits, traces


def run_arm(folder, meta, profile, x, quiet, payload, arm, chunking="whole"):
    with carrier_adapter(arm, profile) as (fits, traces):
        row, arrays, _ = receive(folder, meta, profile, x, quiet, payload, chunking=chunking)
    frames = json.loads((folder / "frames.json").read_text())
    truth = []
    cursor = 0
    for j, (info, trace) in enumerate(zip(fits, traces, strict=True)):
        for key, value in trace.items():
            arrays[f"loop-{j}-{key}"] = value
        n, start = info["symbols"], info["start_symbol_index"]
        matches = [
            i
            for i in range(cursor, len(frames))
            if frames[i]["status"] in ("valid", "bad_data_crc")
            and len(arrays[f"{i}-centers"]) == start + n
        ]
        assert matches, "Completed field fit must match actual frame length in event order"
        frame_index = matches[0]
        cursor = frame_index + 1
        if meta.get("cfo_hz") is not None:
            n = info["symbols"]
            start = info["start_symbol_index"]
            centers = arrays[f"{frame_index}-centers"][start : start + n]
            if len(centers) == n:
                raw_phase = (
                    2
                    * np.pi
                    * (
                        meta["cfo_hz"] * (centers - centers[0]) / 16e6
                        - frames[frame_index]["cfo_hz"] / 4e6 * np.arange(n)
                    )
                )
                error = raw_phase - (np.unwrap(trace["phase"]) - trace["phase"][0])
                arrays[f"loop-{j}-truth-relative-carrier-drift"] = error
                truth.append(
                    dict(
                        frame=frame_index,
                        relative_drift_first_mid_tail_rad=[
                            float(error[k]) for k in (0, n // 2, n - 1)
                        ],
                        initial_phase_truth_available=False,
                        meaning="injected CFO and actual sample centers minus sync/PI NCO;anchored first data sample;not total phase error or cycle-slip truth",
                    )
                )
    np.savez_compressed(folder / "events.npz", **arrays)
    write(folder / "loop-fits.json", fits)
    write(folder / "carrier-truth.json", truth)
    return row, arrays


def node_validation(folder, policy):
    """Independent analytic carrier phase; truth is postprocessing only."""
    rng = np.random.default_rng(policy["independent_node_budget"]["seed"])
    budget = policy["independent_node_budget"]
    rows = []
    arrays = {}
    n = budget["symbols"]
    h = 64
    k = np.arange(n + h)
    for index, (mod, m) in enumerate((("BPSK", 2), ("QPSK", 4), ("8PSK", 8))):
        points = rng.integers(0, m, n)
        ideal = np.r_[
            np.exp(
                1j * ((2 * rng.integers(0, 4, h) + 1) * np.pi / 4 - ((k[:h] + 3) % 2) * np.pi / 4)
            ),
            np.exp(1j * ((2 * points + 1) * np.pi / m - ((k[h:] + 3) % 2) * np.pi / m)),
        ]
        cps = (-0.00023, 0.00019, 0.00011)[index]
        slope = (1 if index % 2 else -1) * 1e-8
        phase = 0.09 + 2 * np.pi * cps * k + slope * k * k / 2
        amp = 0.85 + 0.12 * np.sin(k / 371)
        y = amp * ideal * np.exp(1j * phase) + 0.14 * (
            rng.normal(size=n + h) + 1j * rng.normal(size=n + h)
        )
        y[1001:1008] *= 40 * np.exp(0.7j)
        y[5019:5026] *= 40 * np.exp(-0.8j)
        y[3333:3337] = 0
        output, info, trace = track_psk_carrier(
            y[h:], mod, 3 + h, bandwidth="medium", header_symbols=y[:h], header_mod_type="QPSK"
        )
        error = phase[h:] - np.unwrap(trace["phase"])
        sector = np.floor((error + np.pi / m) / (2 * np.pi / m)).astype(int)
        rms = float(np.sqrt(np.mean(error[2000:] ** 2)))
        freq_truth = 2 * np.pi * cps + slope * k[h:]
        freq_error_hz = float(
            np.mean(trace["frequency"][2000:] - freq_truth[2000:]) * 4e6 / (2 * np.pi)
        )
        settle = None
        for start in range(2001):
            window = error[start : start + 256]
            if np.sqrt(np.mean(window**2)) <= budget["phase_rms_rad"] and np.all(
                sector[start : start + 256] == 0
            ):
                settle = start
                break
        passes = (
            rms <= budget["phase_rms_rad"]
            and np.all(sector == 0)
            and abs(freq_error_hz) <= budget["mean_frequency_error_hz"]
            and settle is not None
            and settle <= budget["convergence_symbols"]
        )
        rows.append(
            dict(
                modulation=mod,
                phase_rms_rad=rms,
                mean_frequency_error_hz=freq_error_hz,
                sector_boundary_events=int(np.count_nonzero(np.diff(sector))),
                sector_nonzero_samples=int(np.count_nonzero(sector)),
                convergence_start_symbol=settle,
                window_symbols=256,
                passed=bool(passes),
                stats=info["stats"],
            )
        )
        for key, value in dict(
            input=y,
            ideal=ideal,
            true_phase=phase,
            output=output,
            true_phase_error=error,
            sector=sector,
            **trace,
        ).items():
            arrays[mod + "-" + key] = value
    np.savez_compressed(folder / "node-validation.npz", **arrays)
    write(folder / "node-validation.json", rows)
    return dict(
        passed=all(r["passed"] for r in rows),
        rows=rows,
        independent_phase_truth=True,
        fixed_verified=False,
    )


def run(args):
    held = args.split == "validation"
    policy_path = Path("tests/data/carrier_loop_policy.json")
    policy = json.loads(policy_path.read_text())
    if held:
        assert policy["loop_module_sha256"] == sha(Path("src/nearlink_sdr/phy/carrier_loop.py"))
        assert policy["power_module_sha256"] == sha(Path("src/nearlink_sdr/phy/carrier_weights.py"))
        assert policy["selected_arm"] == "medium-head"
        generator = next(
            n
            for n in ast.parse(Path(__file__).read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == "development"
        )
        assert (
            hashlib.sha256(ast.dump(generator, include_attributes=False).encode()).hexdigest()
            == policy["input_generator_ast_sha256"]
        )
    arms = (
        ["full", policy["selected_arm"]]
        if held
        else ["full", "slow", "medium", "slow-head", "medium-head"]
    )
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    sources = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/audit_carrier_prefix.py"),
            Path("scripts/audit_rx_soft_capture.py"),
            Path("scripts/audit_spline_fixed.py"),
            Path("tests/test_strict_receive.py"),
            Path("tests/test_psk_field_layout.py"),
            Path("scripts/rx_timing_audit.py"),
            Path("uv.lock"),
            Path("pyproject.toml"),
            *Path("src").rglob("*.py"),
        )
    }
    write(
        out / "protocol.json",
        dict(
            source=sources,
            seed=720117 if held else 710117,
            split=args.split,
            policy_sha256=sha(policy_path),
            arms=arms,
            candidate_gain_pairs={"slow": [1 / 64, 1 / 8192], "medium": [1 / 32, 1 / 2048]},
            adoption="zero success loss,matched BER nonincrease,negative candidates and deliveries nonincrease;old success veto;freeze before new held",
            waveform="ordinary FT2/3/4;actual sync/header/CRC/coding/pilots;independent layout/pulse/resampling",
            held_generated=held,
            default_changed=False,
        ),
    )
    rows = {k: [] for k in arms}
    comparisons = []
    for meta, p, x, q, payload in development(held):
        f = out / (meta["id"] + "-input.npz")
        np.savez_compressed(f, iq=x, quiet=q, payload=np.frombuffer(payload, np.uint8))
        meta = {**meta, "input_sha256": sha(f)}
        write(out / (meta["id"] + "-metadata.json"), meta)
        write(
            out / (meta["id"] + "-format.json"),
            dict(
                profile=p.name,
                modulation=p.config.mod_str,
                code_rate=p.config.rate_str,
                uncoded=p.config.is_uncoded,
                pilot=p.config.pilot_interval,
                sync_bits_sha256=hashlib.sha256(p.sync_bits().tobytes()).hexdigest(),
                crc_seed=p.config.crc_seed,
                whitening_seed=p.config.whitening_seed,
                payload_bitorder=p.payload_bitorder,
            ),
        )
        reference = None
        for arm in rows:
            row, arrays = run_arm(out / (meta["id"] + "-" + arm), meta, p, x, q, payload, arm)
            rows[arm].append(row)
            centers = {k: v for k, v in arrays.items() if k.endswith("-centers")}
            if reference is None:
                reference = centers
            else:
                exact = reference.keys() == centers.keys() and all(
                    np.array_equal(v, reference[k]) for k, v in centers.items()
                )
                comparisons.append(dict(id=meta["id"], arm=arm, capture_timing_exact=exact))
                assert exact
    summary = {k: summarize(r) for k, r in rows.items()}
    gates = {k: verdict(rows["full"], rows[k]) for k in arms if k != "full"}
    for arm, gate in gates.items():
        neg = summary[arm]["negative_candidates"] <= summary["full"]["negative_candidates"]
        matched = all(
            a["ber_bits"] == b["ber_bits"] for a, b in zip(rows["full"], rows[arm], strict=True)
        )
        gate.update(negative_candidates_nonincreased=neg, per_input_ber_matched=matched)
        gate["eligible"] = gate["eligible"] and neg and matched
    fixture = Path("tests/data/sync_carrier_tone_success.npz")
    frozen = json.loads(fixture.with_suffix(".json").read_text())
    assert sha(fixture) == frozen["derived_npz_sha256"]
    shutil.copyfile(fixture, out / "old-success-input.npz")
    p = PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=6, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )
    with np.load(fixture) as a:
        x, q, payload = a["iq"], a["quiet"], a["expected"].tobytes()
    old = []
    for arm in rows:
        for chunking in ("whole", "fixed", "random"):
            row, _ = run_arm(
                out / f"old-success-{arm}-{chunking}",
                dict(id="old-success", positive=True),
                p,
                x,
                q,
                payload,
                arm,
                chunking,
            )
            old.append(dict(arm=arm, chunking=chunking, summary=row))
    losses = [
        dict(arm=r["arm"], chunking=r["chunking"]) for r in old if r["summary"]["correct"] < 1
    ]
    eligible = [
        k for k, v in gates.items() if v["eligible"] and not any(r["arm"] == k for r in losses)
    ]
    failures = [
        dict(
            arm=k,
            **r,
            input_sha256=json.loads((out / (r["id"] + "-metadata.json")).read_text())[
                "input_sha256"
            ],
            frames=r["id"] + "-" + k + "/frames.json",
            fits=r["id"] + "-" + k + "/loop-fits.json",
        )
        for k, values in rows.items()
        for r in values
        if r["positive"] and not r["correct"]
    ]
    node_report = node_validation(out, policy) if held else None
    float_validation_passed = bool(
        held and policy["selected_arm"] in eligible and node_report["passed"]
    )
    model = dict(
        summary=summary,
        gates=gates,
        old_success=old,
        original_success_losses=losses,
        eligible=eligible,
        selected=policy["selected_arm"] if float_validation_passed else None,
        float_validation_passed=float_validation_passed,
        held_generated=held,
        disposition=(
            "float validation passes;fixed/cycle remain unverified"
            if float_validation_passed
            else "validation rejected;no alternative arm selection"
            if held
            else "development rejected"
            if not eligible
            else "development eligible;freeze plus new held still required"
        ),
        failures=failures,
        capture_timing_comparisons=comparisons,
        cost=carrier_loop_cost(),
        independent_node_validation=node_report,
        default_changed=False,
        complete16msps=False,
        fixed_datapath_verified=False,
        truth_not_used_for_selection=True,
        original2percent16KiB_failures_preserved=True,
    )
    write(out / "rows.json", rows)
    write(out / "model.json", model)
    write(out / "failures.json", failures)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            dict(summary=summary, gates=gates, eligible=eligible, old_losses=losses), indent=2
        )
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], default="development")
    raise SystemExit(run(parser.parse_args()))
