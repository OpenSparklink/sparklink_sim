"""Frozen coded-payload AWGN comparison; oracle alignment is diagnostic, not acquisition."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import resource
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_psk_field_layout import independent_wire  # noqa: E402

from nearlink_sdr.common import polar  # noqa: E402
from nearlink_sdr.common.code_block_seg import segment_without_crc  # noqa: E402
from nearlink_sdr.phy.psk_fields import demap_psk_field, soft_demap_psk_field  # noqa: E402
from nearlink_sdr.phy.rx_pipeline import decode_payload  # noqa: E402
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_payload  # noqa: E402


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, data):
    p.write_text(json.dumps(data, indent=2) + "\n")


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    (out / "evidence").mkdir()
    sources = {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / "src").rglob("*.py")}
    plan = dict(
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
        sources=sources,
        driver_sha256=sha(Path(__file__)),
        rust=polar._HAS_RUST_ACCEL,
        dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
        variance="E[|n|²]=10^(-EsN0_dB/10), unit-energy symbol constellation",
        decoder="existing SSC SC, same frozen set and block geometry for every arm",
        alignment="known symbol boundary, carrier and gain; isolated coded-channel diagnostic",
        methods=["hard", "maxlog", "logmap"],
        llr_clip=64,
        variance_floor=1e-12,
        payload_bytes=255,
        pilot_interval=8,
        start_symbol_index=65,
        development_seed=71016,
        held_out_seed=72016,
        repeats=args.repeats,
        snr_db=[2, 4, 6, 8, 10],
        mcs=[6, 10],
        inputs=[],
    )
    plan["reference_sources"] = {f: sha(ROOT / f) for f in ["tests/test_psk_field_layout.py"]}
    prepared = []
    if args.inputs:
        prior = json.loads((args.inputs / "frozen-plan.json").read_text())
        for item in prior["inputs"]:
            path = args.inputs / item["file"]
            assert sha(path) == item["sha256"], path
            prepared.append((item, path))
        plan["input_plan"] = dict(
            path=str(args.inputs), sha256=sha(args.inputs / "frozen-plan.json")
        )
        plan["inputs"] = prior["inputs"]
    else:
        (out / "inputs").mkdir()
        for split, seed in [("development", 71016), ("held-out", 72016)]:
            for mcs in plan["mcs"]:
                cfg = TxConfig(mcs_index=mcs, whitening_seed=0x52, crc_seed=0x123456)
                for snr in plan["snr_db"]:
                    for repeat in range(args.repeats):
                        ident = f"{split}-m{mcs}-snr{snr}-r{repeat:02}"
                        rng = np.random.default_rng(
                            np.random.SeedSequence([seed, mcs, snr, repeat])
                        )
                        bits = rng.integers(0, 2, 255 * 8, dtype=np.int8)
                        coded = encode_payload(bits, cfg)
                        wire = independent_wire(coded, cfg.mod_str, 65, 8, True)
                        variance = 10 ** (-snr / 10)
                        y = wire + np.sqrt(variance / 2) * (
                            rng.normal(size=len(wire)) + 1j * rng.normal(size=len(wire))
                        )
                        path = out / "inputs" / f"{ident}.npz"
                        np.savez_compressed(path, bits=bits, coded=coded, wire=wire, received=y)
                        item = dict(
                            id=ident,
                            file=str(path.relative_to(out)),
                            sha256=sha(path),
                            split=split,
                            mcs=mcs,
                            snr_db=snr,
                            repeat=repeat,
                            variance=variance,
                            modulation=cfg.mod_str,
                            nominal_rate=cfg.rate_str,
                            geometry=[
                                (n, len(info))
                                for n, info in segment_without_crc(
                                    np.zeros(len(bits) + 24, dtype=int), cfg.rate_str, crc_len=24
                                )
                            ],
                        )
                        plan["inputs"].append(item)
                        prepared.append((item, path))
    write(out / "frozen-plan.json", plan)
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    baseline = None
    if args.legacy_source:
        spec = importlib.util.spec_from_file_location("legacy_payload", args.legacy_source)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        plan["legacy_source"] = dict(path=str(args.legacy_source), sha256=sha(args.legacy_source))
    plan["cost_scope"] = "isolated serial process; warm each shape; cyclic method order by repeat"
    write(out / "frozen-plan.json", plan)
    rows = []
    for item, path in prepared:
        with np.load(path) as d:
            bits, coded, y = d["bits"], d["coded"], d["received"]
        cfg = TxConfig(mcs_index=item["mcs"], whitening_seed=0x52, crc_seed=0x123456)
        hard = demap_psk_field(y, cfg.mod_str, 65, 8)[: len(coded)]
        row = dict(
            item,
            input_path=str(path),
            raw_coded_bit_errors=int(np.sum(hard != coded)),
            raw_coded_bit_count=len(coded),
            arms={},
        )
        arrays = {}
        # Decoder initialization is excluded equally from all three arms.
        decode_payload(coded, cfg, len(bits))
        offset = item["repeat"] % 3
        order = plan["methods"][offset:] + plan["methods"][:offset]
        for method in order:
            begin = time.perf_counter()
            llr = (
                None
                if method == "hard"
                else soft_demap_psk_field(y, cfg.mod_str, 65, item["variance"], 8, method=method)[
                    : len(coded)
                ]
            )
            payload, crc_ok = decode_payload(hard, cfg, len(bits), llr=llr)
            seconds = time.perf_counter() - begin
            errors = int(np.sum(payload != bits)) if len(payload) == len(bits) else None
            row["arms"][method] = dict(
                crc_ok=bool(crc_ok),
                length_ok=len(payload) == len(bits),
                bit_errors=errors,
                bits=len(bits),
                correct=bool(crc_ok and errors == 0),
                seconds=seconds,
            )
            arrays[method + "-decoded"] = payload
            if llr is not None:
                arrays[method + "-llr"] = llr
        if baseline is not None:
            old_bits, old_crc = baseline.decode_payload(hard, cfg, len(bits))
            assert bool(old_crc) == row["arms"]["hard"]["crc_ok"]
            np.testing.assert_array_equal(old_bits, arrays["hard-decoded"])
            row["baseline_hard_equivalent"] = True
        evidence = out / "evidence" / f"{item['id']}.npz"
        np.savez_compressed(evidence, **arrays)
        row["evidence"] = str(evidence.relative_to(out))
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        for method, result in row["arms"].items():
            grouped[(row["split"], row["mcs"], row["snr_db"], method)].append(result)
    summary = []
    for (split, mcs, snr, method), group in grouped.items():
        summary.append(
            dict(
                split=split,
                mcs=mcs,
                snr_db=snr,
                method=method,
                packets=len(group),
                correct=sum(x["correct"] for x in group),
                ber=sum(x["bit_errors"] for x in group) / sum(x["bits"] for x in group),
                seconds=float(sum(x["seconds"] for x in group)),
                decode_latency_p50_p95_ms=(
                    np.percentile([x["seconds"] for x in group], [50, 95]) * 1e3
                ).tolist(),
            )
        )
    write(out / "results.json", rows)
    write(out / "summary.json", summary)
    write(
        out / "failures.json",
        [r for r in rows if any(not a["correct"] for a in r["arms"].values())],
    )
    write(
        out / "comparisons.json",
        dict(
            hard_to_maxlog_recoveries=[
                r["id"]
                for r in rows
                if not r["arms"]["hard"]["correct"] and r["arms"]["maxlog"]["correct"]
            ],
            hard_to_maxlog_regressions=[
                r["id"]
                for r in rows
                if r["arms"]["hard"]["correct"] and not r["arms"]["maxlog"]["correct"]
            ],
            maxlog_logmap_different=[
                r["id"]
                for r in rows
                if r["arms"]["maxlog"]["bit_errors"] != r["arms"]["logmap"]["bit_errors"]
            ],
        ),
    )
    assert all(sha(ROOT / p) == h for p, h in sources.items())
    write(
        out / "integrity.json",
        dict(
            sources_unchanged=True,
            input_count=len(rows),
            max_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, mcs in zip(axes, (6, 10), strict=True):
        for method in plan["methods"]:
            g = [
                x
                for x in summary
                if x["split"] == "held-out" and x["mcs"] == mcs and x["method"] == method
            ]
            ax.plot(
                [x["snr_db"] for x in g],
                [x["correct"] / x["packets"] for x in g],
                "-o",
                label=method,
            )
        ax.set(
            title=f"MCS {mcs}: held-out, {args.repeats} packets/point",
            xlabel="Es/N0 (dB)",
            ylabel="CRC + exact payload success",
        )
        ax.set_ylim(-0.03, 1.03)
        ax.grid()
        ax.legend()
    fig.suptitle("Coded AWGN diagnostic: known symbol alignment/carrier; existing SC decoder")
    fig.tight_layout()
    fig.savefig(out / "coded-awgn.png", dpi=160)
    fig.savefig(out / "coded-awgn.svg")
    plt.close(fig)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--legacy-source", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    run(parser.parse_args())
