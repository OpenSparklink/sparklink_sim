"""Compare bounded RAM copy structures; expected exit1 for incomplete system gate."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from model_sync_candidate_work import estimate

from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions


def replay(model, words, requests, output, *, stalls=()):
    """External direct-array oracle; no reference data enters the scheduler."""
    events = []
    checked = 0

    def sink(event, data):
        nonlocal checked
        if data is not None:
            stop, path = event["after_samples"], event["path"]
            expected = words[stop-model.a.window:stop, path]
            received = np.asarray(data, dtype="<u8")
            assert np.array_equal(received, expected)
            event["window_sha256"] = hashlib.sha256(received.tobytes()).hexdigest()
            checked += 1
        events.append(event)

    result = model.simulate(words, requests, stalls=stalls, sink=sink)
    assert result["input_samples"] == len(words)
    assert result["requests"] == (result["completed"]+result["queue_drops"]
                                  +result["history_drops"]+result["startup_drops"])
    assert checked == result["completed"]
    write(output, events)
    return dict(result=result, checked_windows=checked, events_file=output.name,
                events_sha256=sha(output), estimate=model.estimate())


def rrc_word_control(output):
    """Transport actual19+19+8 outputs; upstream RRC numeric failure remains."""
    folder = Path(__file__).resolve().parents[1]/"tests/data"
    path = folder/"sync_carrier_tone_success.npz"
    meta = json.loads((folder/"sync_carrier_tone_success.json").read_text())
    assert sha(path) == meta["derived_npz_sha256"]
    with np.load(path) as archive:
        iq = archive["iq"][:2048]
    fir = FullMantissaRRC(FixedRRCFormat(12, 18))
    fields = [fir.feed_words(*fir._quantize(iq[i:i+32])) for i in range(0, len(iq), 32)]
    r, i, e = [np.concatenate([row[k] for row in fields]) for k in range(3)]
    words = ((r & (2**19-1)) | ((i & (2**19-1)) << 19) | ((e & 255) << 38)).astype("<u8")
    # Independent signed-field decode, not comparison of shared packers.
    assert np.array_equal(((words & (2**19-1)).astype(np.int64) ^ 2**18)-2**18, r)
    assert np.array_equal((((words >> 19) & (2**19-1)).astype(np.int64) ^ 2**18)-2**18, i)
    assert np.array_equal(((words >> 38).astype(np.int64) ^ 128)-128, e)
    np.save(output/"rrc-output-u46-in-u64.npy", words)
    row = replay(CaptureTransport(CaptureTransportAssumptions(paths=1)), words[:, None],
                 [(j, 0) for j in range(256, len(words)+1, 64)], output/"rrc-events.json")
    return dict(**row, fixture_sha256=sha(path), original_input_sha256=meta["source_input"]["sha256"],
                word_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
                new_quantization_operations=0, additional_word_errors=0,
                upstream_rrc_numeric_gate=False, upstream_rrc_stats=fir.stats,
                scope="actual experimental RRC word packing/transport;not tone-word proof or IQ-to-MAC gain")


def run(output, samples=16384):
    if type(samples) is not int or not 512 <= samples <= 262144:
        raise ValueError("Need512..262144 synthetic continuously arriving words")
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(610117)
    words = rng.integers(0, 2**46, (samples, 2), dtype=np.uint64)
    words[:4] = [[0, 2**46-1], [1, 2**45], [255, 127], [65535, 2**38]]
    np.save(output/"input-u46-in-u64.npy", words)
    # Keep fixed differential one-peak nine-phase search and coherent Q.45.
    # Rejected carrier initialization is absent. These remain optimistic real
    # products: word tiling/spline preparation/normalizers are still unallocated.
    local_products = estimate(1)["direct_mac_product_equivalents"]
    products = local_products+324
    # Dependency: choose differential maximum/CFO, THEN run the32-coefficient
    # confirmation pass (latency4). No reuse of fractional leftover slots
    # across this boundary, no sharing credit without an actual schedule.
    def service(units):
        return math.ceil(local_products/units)+max(36, math.ceil(324/units))

    scenarios = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        required = next(u for u in range(1, 221) if service(u) <= clock/500_000)
        old_units = {100_000_000: 4, 160_000_000: 6, 200_000_000: 7}[clock]
        for mode in ("copied8", "sdp1", "tdp2"):
            for workload in ("spaced", "every32", "old_fine_budget", "blocked"):
                units = old_units if workload == "old_fine_budget" else required
                cycles = service(units)
                interval = 256 if workload == "spaced" else 32
                requests = [(j, (j//interval) % 2) for j in range(256, samples+1, interval)]
                stalls = [(0, min(samples*clock//16_000_000, 50000))] if workload == "blocked" else []
                model = CaptureTransport(CaptureTransportAssumptions(
                    clock_hz=clock, mode=mode, fine_service_cycles=cycles))
                row = replay(model, words, requests, output/f"{clock}-{mode}-{workload}.json",
                             stalls=stalls)
                scenarios.append(dict(clock_hz=clock, mode=mode, workload=workload,
                    optimistic_fine_product_units=units, optimistic_fine_service_cycles=cycles,
                    fine_stages=dict(local_cycles=math.ceil(local_products/units),
                                     confirmation_cycles=max(36, math.ceil(324/units))),
                    transport_budget_bram36=12,
                    transport_memory_within_study_budget=row["estimate"]["transport_bram36"] <= 12,
                    known_capture_lower_with_replacement_controls_bits=146870
                    +row["estimate"]["register_reservation_total_bits"]-3*256,
                    rrc_isolated_input_rate_condition=clock == 200_000_000,
                    required_fine_memory_ports_and_preparation_known=False, **row))
    report = dict(scope="two post-filter histories and three owned windows;FT2/SPS4 only",
        reference="independent direct indexed word oracle;no CRC/payload/truth selection",
        scenarios=scenarios, input_word_sha256=hashlib.sha256(words.tobytes()).hexdigest(),
        input_samples=samples, input_paths=2, fine_products_per_request=products,
        rrc_word_control=rrc_word_control(output),
        original_capture_budget_bits=131072, original_capture_budget_still_failed=True,
        current_known_capture_lower_bits=146870,
        physical_remaining_bram36=67, unapproved_reserve25_incremental_bram36=32,
        system_budget_frozen=False, protected_other_costs_not_zero=[
            "RRC8read and product banks", "coarse/notch fixed nodes and ports",
            "fine spline scratch/read ports and normalization", "packet timing/carrier",
            "Polar/CRC/BCH", "TX preparation/shaping", "realtime MAC/AES/DMA/queues"],
        software_rx_selection_changed=False, new_signal_reception_gain_claim=None,
        drop_unit="fine-window request;not a measured lost packet",
        algorithm_misses_measured=None,
        full_capture_cycle_closed=False, whole_system_accepted=False,
        synthesis_evidence=False)
    write(output/"model.json", report)
    sources = [Path(__file__), Path("pyproject.toml"), Path("uv.lock"),
               Path("scripts/model_sync_candidate_work.py"), *Path("src").rglob("*.py")]
    write(output/"sources.json", {str(p): sha(p) for p in sources})
    (output/"driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(output/"manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(scenarios=len(scenarios),
        verified_windows=sum(r["checked_windows"] for r in scenarios),
        whole_system_accepted=False), indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16384)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.samples))
