"""Retain the resource rejection of bounded sync initialization experiments."""

import argparse
import json
import math
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.capture_capacity import CaptureCapacity, CaptureCapacityAssumptions
from nearlink_sdr.phy.sync_carrier import sync_carrier_work
from nearlink_sdr.phy.sync_coherent import SyncConfirmationNCOFlow


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    scenarios = []
    for steps in [1, 2]:
        work = sync_carrier_work(steps)
        for clock, fine_units in [(100_000_000, 4), (160_000_000, 6), (200_000_000, 7)]:
            stages = []

            def stage(name, cycles):
                start = stages[-1]["stop"] if stages else 0
                stages.append(dict(name=name, start=start, stop=start+cycles))

            # Explicit serial dependencies and conservative per-symbol II.
            # Divider/atan2 width/implementation are still hypothetical.
            stage("component_range_scan", 32)
            stage("component_divisions", 32*work["literal_data_divisions"])
            stage("template_products", 32*math.ceil(4/fine_units)+4)
            stage("energy_and_validation", math.ceil((4*32+1)/fine_units))
            for index in range(steps):
                stage(f"rotate_moments_{index}", 32*math.ceil(8/fine_units)+4)
                stage(f"gradient_curvature_{index}", math.ceil(6/fine_units))
                stage(f"newton_divide_{index}", 32)
            stage("final_phase_sum", 32*math.ceil(4/fine_units)+4)
            stage("phase_atan2", 32)
            initialization = stages[-1]["stop"]
            baseline = math.ceil(2949/fine_units)+max(36, math.ceil(324/fine_units))
            capacity = CaptureCapacity(CaptureCapacityAssumptions(
                clock_hz=clock, fine_multipliers=fine_units, sample_bits=46))
            capacity.service_cycles = baseline+initialization
            events = []
            queue = capacity.simulate(range(256, 262145, 32), 262144, sink=events.append)
            write(output / f"queue-{steps}-{clock}.json", events)
            scenarios.append(dict(
                steps=steps, clock_hz=clock, work=work,
                fine_multiplier_slots=fine_units, baseline_service_cycles=baseline,
                additional_initializer_service_cycles=initialization,
                service_cycles=capacity.service_cycles, queue=queue,
                initialization_stages=stages,
                retained_partial_bits=146870, dedicated_initializer_nco_bits=17610,
                known_partial_lower_bits=146870+17610, budget_bits=16*1024*8,
                data_accumulator_and_port_storage=None,
                workload="every32-sample fine request also initializes;abstract overload,not legal-load proof",
                sharing_credit="none;NCO/sync RAM sharing must include arbitration and service overlap",
                full_chain_16msps=False, synthesis_evidence=False,
            ))
    report = dict(
        scenarios=scenarios, whole_system_accepted=False, adopted=False,
        coefficient_calendar=SyncConfirmationNCOFlow().run(
            37115, 32, stalls=[(i, i+1) for i in range(5, 100, 2)]),
        evidence="operation/storage lower estimates plus executable finite FCFS only",
        limitations="32-cycle scalar divide/atan2 are schedule assumptions,not a mapped unit;formats/feedback/normalization/wholeRX and legal-load deadlines missing",
    )
    write(output / "model.json", report)
    (output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps([dict(steps=s["steps"], clock=s["clock_hz"],
                          service=s["service_cycles"], drops=s["queue"]["dropped_requests"],
                          lower_bits=s["known_partial_lower_bits"]) for s in scenarios], indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
