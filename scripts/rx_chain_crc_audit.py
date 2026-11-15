"""Reproduce header-CRC acceptance and payload delivery, without selecting RF answers by CRC."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path

import numpy as np

from nearlink_sdr.phy.control_info import ControlInfoA3
from nearlink_sdr.phy.rx_pipeline import decode_head, rx_chain
from nearlink_sdr.phy.sync_sequence import sync_signal_2
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_head, tx_chain


def run():
    rows = []
    for pilot in (0, 8):
        cfg = TxConfig(mcs_index=8, pid=0x879012, ctrl_bits_len=28, pilot_interval=pilot)
        seed = sum(int(b) << i for i, b in enumerate(sync_signal_2(cfg.pid)[:12]))
        good = ControlInfoA3(0, 0, 0, 0, 0, 0, 0, 8).pack(seed)
        data = np.random.default_rng(63016).integers(0, 2, 64)
        for case in ("valid", "bad_crc", "wrong_seed"):
            head = good.copy()
            if case == "bad_crc":
                head[0] ^= 1
            if case == "wrong_seed":
                head = ControlInfoA3(0, 0, 0, 0, 0, 0, 0, 8).pack(seed ^ 1)
            encoded = encode_head(head, cfg)
            iq = tx_chain(head, data, cfg)
            result = rx_chain(iq, cfg, 8)
            rows.append(dict(case=case, pilot_interval=pilot, header_seed=seed,
                             input_sha256=hashlib.sha256(iq.tobytes()).hexdigest(),
                             decode_head_ok=bool(decode_head(encoded, cfg)[1]),
                             head_crc_ok=bool(result.head_crc_ok), data_crc_ok=bool(result.crc_ok),
                             delivered_bits=len(result.data_bits),
                             content_ok=bool(np.array_equal(result.data_bits, data))))
    import nearlink_sdr.phy.rx_pipeline as module
    root = Path(module.__file__).parents[3]
    files = ["src/nearlink_sdr/phy/" + f for f in ("rx_pipeline.py", "tx_pipeline.py",
                                                 "control_info.py", "frame.py")]
    return dict(seed=63016, rows=rows, python=platform.python_version(),
                dependencies={p: importlib.metadata.version(p) for p in ("numpy", "scipy")},
                source_sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                               for f in files})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(run(), indent=2) + "\n")
