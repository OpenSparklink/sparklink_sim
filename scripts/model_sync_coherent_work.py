"""Charge coherent confirmation/NCO additions to the existing partial capture graph."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from model_sync_candidate_work import estimate

from nearlink_sdr.phy import sync_coherent
from nearlink_sdr.phy.capture_capacity import CaptureCapacity, CaptureCapacityAssumptions
from nearlink_sdr.phy.rx_resource_budget import RXResourceBudget, receiver_resource_ledger
from nearlink_sdr.phy.sync_coherent import SyncConfirmationNCO, SyncConfirmationNCOFlow


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    n = 32
    nco = SyncConfirmationNCO.estimate()
    # One generic four-real-product template multiply, one CFO rotate, y energy,
    # coherent magnitude and denominator/threshold products. Reference energy is
    # assumed known at configuration; do not make its storage/ports disappear.
    additions = dict(template_real_products=4*n, rotation_real_products=4*n,
                     signal_energy_squares=2*n, magnitude_squares=2,
                     denominator_product=1, threshold_product=1,
                     diagnostic_divides=1, gate_divides=0, sqrt=0,
                     new_cfo_atan2=0, angle_source="reuse already selected differential phase",
                     reference_energy_precomputed=True,
                     literal_reference_energy_preparation_squares=2*n,
                     product_word_widths_verified=False,
                     normalization_accumulator_format=None)
    products = sum(additions[k] for k in ["template_real_products", "rotation_real_products",
        "signal_energy_squares", "magnitude_squares", "denominator_product", "threshold_product"])
    local = estimate(1)
    extra_state = dict(nco_node=nco["logical_bits"], four_symbol_alignment_reservation=4*46,
                       four_capture_monitor_counters=4*64,
                       template_phase_code_reservation=3*n)
    scenarios = []
    for clock in [100_000_000, 160_000_000, 200_000_000]:
        ledger = receiver_resource_ledger(RXResourceBudget(clock_hz=clock))
        available = ledger["dsp"]["remaining_optimistic_fine_products_per_request"]
        units = math.floor(available*500_000/clock)
        capacity = CaptureCapacity(CaptureCapacityAssumptions(
            clock_hz=clock, fine_multipliers=max(1, units), sample_bits=46))
        # No NCO start until the differential maximum/CFO is selected. The max
        # still optimistically assumes fully overlapped confirmation multiplies
        # and one coefficient per cycle, ignoring unallocated normalizers.
        capacity.service_cycles = math.ceil(local["direct_mac_product_equivalents"]/max(1, units))
        capacity.service_cycles += max(n+nco["latency_cycles"], math.ceil(products/max(1, units)))
        events = []
        queue = capacity.simulate(range(256, 262145, 32), 262144, sink=events.append)
        write(args.output / f"queue-{clock}.json", events)
        capture_state = ledger["storage"]["known_partial_bits"] + 2*3732 + 2*78
        scenarios.append(dict(
            clock_hz=clock, fine_multiplier_units=units,
            local_products=local["direct_mac_product_equivalents"], coherent_products=products,
            available_partial_products_per_request=available,
            direct_graph_within_partial_product_budget=local["direct_mac_product_equivalents"]+products <= available,
            service_cycles_scenario=capacity.service_cycles, queue=queue,
            partial_capture_base_bits=capture_state, additions_bits=extra_state,
            known_capture_partial_total_bits=capture_state+sum(extra_state.values()),
            memory_budget_bits=16*1024*8,
            optional_payload_bounded_carrier_node_bits_not_in_capture=1581,
            unallocated="coherent sum/energy/wide compare registers, template/word format proof, port copies, fine spline preparation, packet processing/TX/MAC",
            full_chain_schedule_verified=False, synthesis_evidence=False))
    flow = SyncConfirmationNCOFlow()
    flows = dict(ready=flow.run(37115, 128), blocked=flow.run(37115, 128, stalls=[(8, 73)]),
                 stopped=flow.run(37115, 128, stalls=[(0, 1000)], max_cycles=100))
    coefficients = np.asarray(sync_coherent._COSINE_QUARTER, dtype="<u4")
    np.save(args.output / "quarter-cosine-u17-in-u32.npy", coefficients)
    report = dict(
        evidence="graph/storage estimate + isolated executable coefficient calendar + optimistic snapshot FCFS;no synthesis",
        scope="FT2/SPS4/one9-phase peak;coherent input/sums still float;fixed NCO only",
        operations=additions, nco=nco, nco_calendars=flows, scenarios=scenarios,
        coefficient_logical_bits=17, coefficient_container_bits=32,
        coefficient_words_sha256=hashlib.sha256(coefficients.tobytes()).hexdigest(),
        system_resource_budget_frozen=False, whole_system_accepted=False,
        no_shared_resource_credit="future payload carrier/TX oscillator or BRAM sharing not scheduled;no implicit TDD credit",
    )
    write(args.output / "model.json", report)
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps([dict(clock_hz=s["clock_hz"],products=s["local_products"]+products,
                          known_bits=s["known_capture_partial_total_bits"],
                          queue_drops=s["queue"]["dropped_requests"])
                      for s in scenarios], indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
