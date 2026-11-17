"""Frozen failed-packet truth controls, excluded from actual capture and delivery."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, sha, transmitted, write
from audit_rx_soft_capture import diagnostic_bits
from reproduce_sync_coherent import frozen_snapshot
from scipy.interpolate import CubicSpline
from scipy.signal import lfilter

from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.strict_rx import decode_psk_frame
from nearlink_sdr.phy.tone_notch import CausalToneNotch

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_SHA = "b5580311178fc331c5bccbb07482ba2be9abb69dbfecd593128216c9fa236e8b"


def assess(symbols, p, target, required_symbols, kappa=None):
    """Decode first; expected payload/length/MCS only assess the returned result."""
    f = decode_psk_frame(symbols, p, carrier_weight_kappa=kappa)
    wanted = bytes.fromhex(target["expected_hex"])
    out = {k: v for k, v in f.items() if k != "payload"}
    out.update(
        payload_hex=None if f.get("payload") is None else f["payload"].hex(),
        length_matches_reference=f.get("header", {}).get("data_length") == target["length"],
        mcs_matches_reference=f.get("mcs") == target["mcs"],
        available_symbols=len(symbols), required_symbols=required_symbols,
        complete_span=len(symbols) >= required_symbols,
    )
    out["correct"] = bool(f["status"] == "valid" and f["payload"] == wanted
                          and out["length_matches_reference"] and out["mcs_matches_reference"])
    bits = diagnostic_bits(dict(frame=f, symbols=symbols), p)
    out["bit_errors"] = None
    if bits is not None and len(bits) == len(wanted)*8:
        out["bit_errors"] = int(np.count_nonzero(
            bits != np.unpackbits(np.frombuffer(wanted, np.uint8))))
    return out


def classify(arms, actual):
    """Disjoint evidence classes; neither a production selector nor capacity proof."""
    if any(a["correct"] for a in arms.values()):
        return "diagnostic_recovered_actual_failed"
    weighted = arms.get("actual_symbols_kappa2", {})
    old_errors = (actual or {}).get("bit_errors")
    if (old_errors is not None and weighted.get("bit_errors") is not None
            and 0 < weighted["bit_errors"] < old_errors):
        return "errors_reduced_still_failed"
    truth = arms.get("truth_timing_truth_CFO", {})
    if truth.get("complete_span"):
        return "tested_controls_still_failed"
    return "insufficient_evidence"


def run(args):
    frozen_snapshot(args.evidence)
    reference = args.evidence / "held-coherent"
    manifest_file = reference / "manifest.json"
    assert sha(manifest_file) == _REFERENCE_SHA
    manifest = json.loads(manifest_file.read_text())
    for name, digest in manifest.items():
        assert sha(reference / name) == digest, name
    rows = json.loads((reference / "results.json").read_text())
    formal = json.loads((reference / "summary.json").read_text())
    assert formal["known_packets"] == 96 and formal["correct"] == 61
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "evidence").mkdir()
    sources = {str(p.relative_to(_ROOT)): sha(p) for p in (_ROOT / "src").rglob("*.py")}
    write(args.output / "plan.json", dict(
        diagnostic_only=True, reference_manifest_sha256=sha(manifest_file),
        frozen_selection_sha256=sha(args.evidence / "frozen-selection.json"),
        sources=sources, driver_sha256=sha(Path(__file__)),
        unchanged="profile/CRC/expected payload/carrier FFT and grid/notch31/32/frozen kappa2",
        actual_CFO_control="nominal estimate divided by truth step/4; diagnostic SRO conversion",
        partial_controls="actual trajectories stop at header rejection; not full-packet oracles",
        formal_correct_unchanged="61/96",
    ))
    records = []
    for row in rows:
        meta = row["input"]
        failed = [t for t in meta["packets"] if not any(f["correct"] and
                  abs(f["sync_start"]-t["sync_start"]) <= 4 for f in row["frames"])]
        if not failed:
            continue
        assert sha(Path(meta["path"])) == meta["sha256"]
        with np.load(meta["path"]) as data:
            x = data["iq"]
        p = profile(meta["mcs"])
        h = rrc_filter(.4, p.config.sps)
        z = lfilter(h, [1.], x)
        notch = CausalToneNotch()
        zn = np.concatenate([notch.advance(z[i:i+4096]) for i in range(0, len(z), 4096)])
        splines = [CubicSpline(np.arange(len(z)), v) for v in [z, zn]]

        def sampled(centers, cfo, *, notched=False):
            values = splines[int(notched)](centers+(len(h)-1)//2)
            values *= np.exp(-2j*np.pi*cfo*centers/16e6)
            ref = p.sync_symbols()
            gain = np.vdot(ref, values[:len(ref)])/np.vdot(ref, ref)
            if abs(gain) < 1e-30:
                raise ValueError("Diagnostic sync gain is zero; cannot classify a full control")
            return values/gain

        with np.load(reference / "evidence" / (meta["id"]+".npz")) as saved:
            for target in failed:
                # Length is truth for this diagnostic only; dummy TX bits never choose a result.
                wire, _ = transmitted(p, length=target["length"], payload_seed=1)
                required = len(wire)-40
                centers = target["sync_start"]+np.arange(required)*target["step"]
                matched = [(i, f) for i, f in enumerate(row["frames"])
                           if abs(f["sync_start"]-target["sync_start"]) <= 4]
                assert len(matched) <= 1
                y = sampled(centers, target["cfo_hz"])
                yn = sampled(centers, target["cfo_hz"], notched=True)
                symbol_arms = dict(truth_timing_truth_CFO=y, truth_timing_truth_CFO_kappa2=y,
                                   notch_truth_timing_truth_CFO=yn,
                                   notch_truth_timing_truth_CFO_kappa2=yn)
                if matched:
                    i, f = matched[0]
                    symbol_arms.update(
                        actual_symbols_kappa2=saved[f"{i}-symbols"],
                        actual_timing_truth_CFO=sampled(saved[f"{i}-symbol_centers_raw"],
                                                       target["cfo_hz"]),
                        truth_timing_actual_CFO=sampled(centers,
                            f["carrier_error"]["sync_nominal_hz"]/(target["step"]/4)),
                    )
                arms = {k: assess(v, p, target, required, 2 if k.endswith("kappa2") else None)
                        for k, v in symbol_arms.items()}
                filename = f"{meta['id']}-{len(records)}.npz"
                np.savez_compressed(args.output / "evidence" / filename,
                                    **symbol_arms, truth_centers=centers)
                actual = matched[0][1] if matched else None
                records.append(dict(
                    id=meta["id"], input_sha256=meta["sha256"], target=target, actual=actual,
                    diagnostic_only=True, arms=arms, notch_stats=notch.stats,
                    evidence=filename, evidence_sha256=sha(args.output / "evidence" / filename),
                    evidence_class=classify(arms, actual), theoretical_unrecoverable=None,
                    actual_receiver_recovery=False,
                ))
                write(args.output / "results.json", records)
    assert len(records) == 35
    summary = dict(
        packets=len(records), formal_correct_unchanged="61/96", diagnostic_only=True,
        classes=dict(Counter(r["evidence_class"] for r in records)),
        recovered_by_control={k: [dict(id=r["id"], sync_start=r["target"]["sync_start"],
                                      input_sha256=r["input_sha256"])
                                 for r in records if r["arms"].get(k, {}).get("correct")]
                              for k in sorted({k for r in records for k in r["arms"]})},
        caveat="controls are not a CRC-selected receiver or theoretical capacity proof",
    )
    write(args.output / "summary.json", summary)
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    assert sources == {str(p.relative_to(_ROOT)): sha(p) for p in (_ROOT / "src").rglob("*.py")}
    write(args.output / "manifest.json", {str(p.relative_to(args.output)): sha(p)
                                         for p in args.output.rglob("*") if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 0  # Diagnostic execution only; all 35 remain formal failures.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
