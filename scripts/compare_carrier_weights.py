"""Post-receive paired decisions and burst-region wire BER, never RX control."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, sha, write
from tests.test_psk_field_layout import independent_wire

from nearlink_sdr.phy.psk_fields import demap_psk_field, psk_field_layout
from nearlink_sdr.phy.tx_pipeline import encode_payload


def inspect_arm(directory):
    rows = json.loads((directory / "results.json").read_text())
    details = []
    for row in rows:
        meta = row["input"]
        assert sha(Path(meta["path"])) == meta["sha256"]
        with np.load(directory / "evidence" / (meta["id"] + ".npz")) as data:
            for index, frame in enumerate(row["frames"]):
                if not frame.get("matched") or not frame.get("header_crc"):
                    continue
                target = min(meta["packets"],
                             key=lambda p: abs(p["sync_start"] - frame["sync_start"]))
                if frame.get("status") not in ("valid", "bad_data_crc"):
                    continue
                cfg = replace(profile(meta["mcs"]).config, mcs_index=frame["mcs"])
                y = data[f"{index}-symbols"]
                start, end = frame["data_start_symbol"], frame["frame_end_symbol"]
                wire = y[start:end]
                carrier = frame.get("carrier", {})
                freq = carrier.get("residual_cycles_per_symbol", 0)
                phase = carrier.get("residual_phase_radians", 0)
                corrected = wire * np.exp(-1j * (2 * np.pi * freq * np.arange(len(wire)) + phase))
                wanted = np.unpackbits(np.frombuffer(bytes.fromhex(target["expected_hex"]), np.uint8))
                expected_bits = encode_payload(wanted, cfg)
                nbits = frame["coded_bits"]
                assert len(expected_bits) == nbits
                actual = demap_psk_field(corrected, cfg.mod_str, start,
                                         cfg.pilot_interval)[:nbits]
                assert len(actual) == nbits
                bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3}[cfg.mod_str]
                positions, _ = psk_field_layout(nbits, bps, cfg.pilot_interval)
                centers = data[f"{index}-symbol_centers_raw"][start:end]
                damaged = np.zeros(len(centers), bool)
                after = np.zeros(len(centers), bool)
                for lower, upper in meta["burst_ranges"]:
                    #81-tap RRC center40 plus interpolation future2. These
                    #labels only attribute errors after the actual decision.
                    damaged |= (centers >= lower - 42) & (centers < upper + 42)
                    after |= (centers >= upper + 42) & (centers < upper + 42 + 256)
                after &= ~damaged
                bit_to_symbol = positions[np.arange(nbits) // bps]
                errors = actual != expected_bits
                regions = {}
                for name, mask in dict(all=np.ones(len(centers), bool), damaged=damaged,
                                       post64=after, clean=~damaged).items():
                    chosen = mask[bit_to_symbol]
                    regions[name] = dict(bits=int(chosen.sum()),
                                         errors=int(errors[chosen].sum()))
                truth_wire = independent_wire(expected_bits, cfg.mod_str, start + 40,
                                               cfg.pilot_interval, True)
                assert len(truth_wire) == len(corrected)
                phase_error = np.angle(corrected * truth_wire.conj())
                clean = ~damaged
                details.append(dict(
                    id=meta["id"], event=index, input_sha256=meta["sha256"], kind=meta["kind"],
                    status=frame["status"], correct=frame["correct"], mcs=frame["mcs"],
                    code_rate=frame["code_rate"], regions=regions,
                    info_bit_errors=frame.get("bit_errors"),
                    frequency_error_hz=frame["carrier_error"].get("after_payload_error_hz"),
                    clean_phase_rms=float(np.sqrt(np.mean(phase_error[clean]**2)))
                    if clean.any() else None,
                    clean_phase_bias=float(np.angle(np.sum(np.exp(1j * phase_error[clean]))))
                    if clean.any() else None,
                    weights=carrier.get("weights"),
                ))
    totals = {}
    for row in details:
        group = (row["kind"], "uncoded" if row["code_rate"] == "1/1" else "coded")
        label = "/".join(group)
        total = totals.setdefault(label, {k: dict(bits=0, errors=0)
                                          for k in ["all", "damaged", "post64", "clean"]})
        for region, count in row["regions"].items():
            for key, value in count.items():
                total[region][key] += value
    return dict(details=details, totals=totals)


def compare(before, after, output):
    old = json.loads((before / "results.json").read_text())
    new = json.loads((after / "results.json").read_text())
    assert len(old) == len(new)
    losses, recoveries, false, upstream = [], [], [], []
    for a, b in zip(old, new, strict=True):
        assert a["input"] == b["input"], "Frozen IQ or reference metadata changed"
        ident = a["input"]["id"]
        assert a["capture_config"] == b["capture_config"]
        assert a["timing_config"] == b["timing_config"]
        valid = [f for f in b["frames"] if f["status"] == "valid"]
        assert b["delivered_payloads"] == [f["payload_hex"] for f in valid]
        assert all(f["header_crc"] and f["data_crc"] and f["length_ok"] for f in valid)
        if b["false_candidates"] > a["false_candidates"] or (
                b["false_deliveries"] > a["false_deliveries"]):
            false.append(ident)
        # This experiment changes payload weights only: all upstream events,
        # header outcomes and timing trajectories must remain exactly identical.
        changed = a["receiver_events"] != b["receiver_events"]
        changed |= len(a["frames"]) != len(b["frames"])
        for index, (x, y) in enumerate(zip(a["frames"], b["frames"], strict=True)):
            tag = dict(id=ident, event=index, input_sha256=a["input"]["sha256"])
            changed |= any(x.get(k) != y.get(k) for k in
                           ["header_crc", "header", "mcs", "length_ok", "sync_start"])
            if x["correct"] and not y["correct"]:
                losses.append(tag)
            if y["correct"] and not x["correct"]:
                recoveries.append(tag)
        with np.load(before / "evidence" / (ident + ".npz")) as u, np.load(
                after / "evidence" / (ident + ".npz")) as v:
            assert u.files == v.files
            changed |= any(not np.array_equal(u[k], v[k]) for k in u.files)
        if changed:
            upstream.append(ident)
    report = dict(
        inputs=len(old), actual_samples=sum(r["input"]["samples"] for r in old),
        negative_inputs=sum(not r["input"]["packets"] for r in old),
        negative_samples=sum(r["input"]["samples"] for r in old if not r["input"]["packets"]),
        original_success_losses=losses, recoveries=recoveries,
        false_regressions=false, upstream_changes=upstream,
        functional_constraints_passed=not (losses or false or upstream),
        before=inspect_arm(before), after=inspect_arm(after),
        diagnostic_only="burst/phase/expected-wire labels never enter receive decisions; shared bit encoder remains a conformance limitation",
        whole_system_accepted=False, default_promotion=False,
    )
    write(output, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ["before", "after"]}, indent=2))
    return int(not report["functional_constraints_passed"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ["before", "after", "output"]:
        parser.add_argument("--" + option, type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(compare(args.before, args.after, args.output))
