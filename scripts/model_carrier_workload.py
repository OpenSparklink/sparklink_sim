"""Expose the existing packet-FFT size and serialized weight-node tail cost."""

import argparse
import json
import math
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.carrier_weight_flow import CarrierWeightFlow
from nearlink_sdr.phy.psk_fields import psk_field_layout
from nearlink_sdr.phy.rx_pipeline import payload_coded_length
from nearlink_sdr.phy.tx_pipeline import TxConfig


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mcs in range(13):
        for length in [1, 255, 2047]:
            cfg = TxConfig(mcs_index=mcs, symbol_rate_mhz=4, pilot_interval=8)
            nbits = payload_coded_length(8 * length, cfg)
            bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3}[cfg.mod_str]
            _, count = psk_field_layout(nbits, bps, cfg.pilot_interval)
            nfft = 0 if count < 32 else 16 * 2**math.ceil(math.log2(count))
            # One hypothetical in-place16+16-bit complex buffer alone. Actual
            # fixed FFT format/architecture is not selected or validated here.
            bits = nfft * 32
            rows.append(dict(
                mcs=mcs, modulation=cfg.mod_str, rate=cfg.rate_str, payload_bytes=length,
                coded_bits=nbits, wire_symbols=count, fft_points=nfft,
                numpy_complex128_fft_vector_bytes=nfft * 16,
                hypothetical_single_complex32_buffer_bits=bits,
                bram36_capacity_lower_tiles=math.ceil(bits / (36 * 1024)),
                fits_entire_chip_bram_capacity=bits <= 140 * 36 * 1024,
                weight_tail_if_serial_cycles=8 * count if nfft else 0,
                weight_tail_if_serial_us_at200mhz=8 * count / 200 if nfft else 0.,
                wire_duration_us=count / 4,
            ))
    report = dict(
        scope="current FFT dimensions + illustrative direct-buffer capacity; not all-transform memory lower bound or new fixed FFT verification",
        format_complex32_validated=False, fft_dataflow_verified=False,
        source_word_node=CarrierWeightFlow().estimate(), cases=rows,
        whole_system_accepted=False,
        excluded="normalization, FFT twiddles/intermediates, Newton/circular-phase buffers, pipeline/ports, decoder/TX/ARM concurrency",
    )
    write(args.output / "workload.json", report)
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "sources.json", {str(p): sha(p) for p in [Path(__file__),
          Path("src/nearlink_sdr/phy/psk.py"), Path("src/nearlink_sdr/phy/psk_fields.py"),
          Path("src/nearlink_sdr/phy/rx_pipeline.py"), Path("uv.lock")]})
    print(json.dumps([r for r in rows if r["mcs"] in [0, 8, 12]
                      and r["payload_bytes"] == 2047], indent=2))
    return 1  # No whole-system acceptance; resource gaps are not execution errors.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
