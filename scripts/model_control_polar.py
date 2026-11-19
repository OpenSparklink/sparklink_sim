"""Registered hard-control SSC bank experiment,not a header/RX acceptance claim."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from nearlink_sdr.common.polar import _HAS_RUST_ACCEL, get_polar_decoder
from nearlink_sdr.phy.control_polar_word import ControlPolarFormat
from nearlink_sdr.sim.control_polar_flow import ControlPolarService, control_polar_cost


def run(output, *, random_cases=64):
    """Same frozen tie semantics and integer values,two actual physical layouts."""
    if type(random_cases) is not int or not 0 <= random_cases <= 512:
        raise ValueError("Bounded diagnostic count must be 0..512")
    output.mkdir(parents=True, exist_ok=False)
    reference_path = Path(__file__).resolve().parents[1] / "tests/data/control_polar_reference.json"
    reference = json.loads(reference_path.read_text())
    rng = random.Random(2026112004)
    rows = []
    for N, K in [(64, 40), (256, 51)]:
        patterns = [[0] * N, [1] * N, [k % 2 for k in range(N)],
                    [int(k < N // 2) for k in range(N)]]
        for position in [0, 1, N // 2 - 1, N // 2, N - 2, N - 1]:
            bits = [0] * N
            bits[position] = 1
            patterns.append(bits)
        patterns += [[rng.randrange(2) for _ in range(N)] for _ in range(random_cases)]
        known = [v for v in reference["vectors"] if v["N"] == N]
        patterns += [v["coded_bits"] for v in known]
        decoder = get_polar_decoder(N, K)
        expected = [decoder.decode(np.asarray([1 - 2 * b for b in bits]) * 10.).tolist()
                    for bits in patterns]
        assert expected[-len(known):] == [v["information_bits"] for v in known]
        for layout in ("dense", "depth-parity"):
            events = []
            service = ControlPolarService(ControlPolarFormat(N, K), layout=layout,
                                          observer=events.append)
            actual, latencies = [], []
            for index, bits in enumerate(patterns):
                if index:
                    service.observer = None
                begin = service.cycle
                assert service.issue(bits)
                for _ in range(6000):
                    if service.ready:
                        break
                    service.step()
                assert service.ready, "Bounded control operation failed to drain"
                latencies.append(service.cycle - begin)
                result = service.take()
                assert result["crc_status"] == "not_checked"
                actual.append(list(result["information_bits"]))
                service.step()
            assert actual == expected
            tag = f"N{N}-K{K}-{layout}"
            archive = output / (tag + ".npz")
            np.savez_compressed(archive, input_bits=np.asarray(patterns, dtype=np.uint8),
                                information_bits=np.asarray(actual, dtype=np.uint8))
            (output / (tag + "-first-trace.json")).write_text(json.dumps(events) + "\n")
            rows.append(dict(N=N, K=K, layout=layout, vectors=len(patterns),
                             data_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                             cycles_min=min(latencies), cycles_max=max(latencies),
                             state=service.stats, physical_bank_peak_ports=service.memory.peak,
                             cost=control_polar_cost(service.fmt, layout=layout),
                             decoded_bit_differences=0, CRC="not_checked"))
    result = dict(stage="H4A control SSC bank experiment", actual_reference_rust=_HAS_RUST_ACCEL,
                  reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  rows=rows, known_standard_matrix_vectors=6, default_receiver_changed=False,
                  reception_gain=None, RF_holdout_evaluation=False, full_RX_gates_passed=False,
                  source={str(p.resolve().relative_to(Path(__file__).resolve().parents[1])):
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                              Path(__file__), Path("src/nearlink_sdr/phy/control_polar_word.py"),
                              Path("src/nearlink_sdr/sim/control_polar_flow.py")]})
    (output / "model.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=64)
    args = parser.parse_args()
    result = run(args.output.resolve(), random_cases=args.random_cases)
    print(json.dumps(dict(rows=[{k: row[k] for k in ["N", "K", "layout", "vectors",
                                                   "cycles_min", "cycles_max"]}
                                for row in result["rows"]], full_RX_gates_passed=False)))


if __name__ == "__main__":
    main()
