"""Frozen independent IQ → legacy/strict MAC comparison; failures are retained, not hidden."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.test_strict_receive import iq_record, transmitted  # noqa: E402

from nearlink_sdr.phy.mac_interface import iq_to_mac, iq_to_mac_capture  # noqa: E402
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile  # noqa: E402
from nearlink_sdr.phy.tx_pipeline import TxConfig  # noqa: E402


def identity(data):
    return hashlib.sha256(data).hexdigest()


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    cases = []
    for ft, mcs, pilot in [(2, 6, 8), (2, 8, 0), (2, 12, 8), (3, 8, 4), (4, 0, 4)]:
        cfg = TxConfig(frame_type=ft, mcs_index=mcs, ctrl_bits_len=28 if ft == 2 else 27,
                       head_llid=0x234567 if ft >= 3 else None, whitening_seed=0x52,
                       crc_seed=0x123456, pilot_interval=pilot, symbol_rate_mhz=4)
        for cfo, ppm in [(0, 0), (30000, 100)]:
            cases.append((f"ft{ft}-mcs{mcs}-p{pilot}-cfo{cfo}-sro{ppm}",
                          PSKReceiveProfile.ordinary(cfg), cfo, ppm, {}))
    ordinary = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    for damage in ("bad_head", "bad_data", "empty"):
        cases.append((damage, ordinary, 0, 0, {damage: True}))
    frozen = dict(seed_payload=65016, seed_quiet=65017, offset_samples=.277,
                  timing_subdivisions=8, timing_tracking=True, energy_window_symbols=8,
                  carrier_method="coherent", sync_threshold=.85,
                  python=platform.python_version(), platform=platform.platform(),
                  dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pytest")},
                  commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                  inputs={}, sources={str(p.relative_to(ROOT)): identity(p.read_bytes())
                                     for p in [*sorted((ROOT/"src").rglob("*.py")),
                                               ROOT/"tests/test_strict_receive.py",
                                               ROOT/"tests/test_psk_field_layout.py",
                                               ROOT/"scripts/rx_timing_audit.py", Path(__file__)]})
    prepared = []
    for name, profile, cfo, ppm, damage in cases:
        wire, expected = transmitted(profile, **damage)
        x, q, truth, step = iq_record(wire, cfo=cfo, ppm=ppm)
        path = out/(name + ".npz")
        np.savez_compressed(path, iq=x, quiet=q, expected=np.frombuffer(expected, dtype=np.uint8))
        frozen["inputs"][name] = dict(file=path.name, sha256=identity(path.read_bytes()),
                                      iq_sha256=identity(x.tobytes()), sample_count=len(x),
                                      frame_type=profile.config.frame_type,
                                      mcs=profile.config.mcs_index, pilot_interval=profile.config.pilot_interval,
                                      data_crc_seed=profile.config.crc_seed,
                                      whitening_seed=profile.config.whitening_seed,
                                      head_llid=profile.config.head_llid,
                                      cfo_hz=cfo, injected_sro_ppm=ppm,
                                      expected_length=len(expected), sample_hz=16e6, symbol_hz=4e6)
        prepared.append((name, profile, x, q, expected, truth, step, damage))
    write(out/"frozen-plan.json", frozen)
    rows = []
    for name, profile, x, q, expected, truth, step, damage in prepared:
        t = time.perf_counter()
        old = iq_to_mac(x, profile.config, len(expected))
        old_seconds = time.perf_counter() - t
        t = time.perf_counter()
        new = iq_to_mac_capture(x, q, profile, energy_window_symbols=8)
        seconds = time.perf_counter() - t
        frames = [{k: v for k, v in f.items() if k != "payload"} for f in new.diagnostics["frames"]]
        evidence = {}
        for i, b in enumerate(new.diagnostics["acquisition"]["bursts"]):
            for k in ("symbols", "symbol_centers_raw", "timing_detector_error", "timing_rate_ppm"):
                if k in b:
                    evidence[f"b{i}-{k}"] = b[k]
        np.savez_compressed(out/(name + "-diagnostics.npz"), **evidence)
        row = dict(id=name, frames=frames, legacy_crc=bool(old.crc_ok),
                   legacy_header_crc=bool(old.head_crc_ok), legacy_seconds=old_seconds,
                   packets=len(new.packets), content_ok=[p.mac_payload == expected for p in new.packets],
                   strict_seconds=seconds, strict_input_samples_per_second=len(x)/seconds,
                   truth_sync_start=truth, truth_samples_per_symbol=step,
                   expected_rejection=bool(damage))
        rows.append(row)
    source_ok = all(identity((ROOT/p).read_bytes()) == h for p, h in frozen["sources"].items())
    assert source_ok
    write(out/"results.json", rows)
    write(out/"summary.json", dict(positive_inputs=sum(not r["expected_rejection"] for r in rows),
          correct_positive_inputs=sum(bool(r["packets"] == 1 and all(r["content_ok"]))
                                      for r in rows if not r["expected_rejection"]),
          negative_inputs=sum(r["expected_rejection"] for r in rows),
          false_delivery=sum(r["packets"] for r in rows if r["expected_rejection"]),
          sources_unchanged=source_ok, peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write(out/"manifest.json", {p.name: identity(p.read_bytes()) for p in sorted(out.iterdir()) if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output.resolve())
