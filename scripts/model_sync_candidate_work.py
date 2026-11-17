"""Direct-MAC candidate-search scenarios and bounded FCFS gaps, not synthesis."""

import argparse
import json
import math
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.capture_capacity import CaptureCapacity, CaptureCapacityAssumptions
from nearlink_sdr.phy.rx_resource_budget import RXResourceBudget, receiver_resource_ledger


def estimate(peaks, symbols=32, window=256):
    if peaks not in (1, 2) or symbols not in (32, 128):
        raise ValueError("Need1/2 peaks and32/128-symbol sync")
    trials = 9*peaks
    # Generic Horner MAC allocation. Fractional phases are fixed dyadic values:
    # replacing these products by shift/add is another, unimplemented allocation.
    interpolation = 6*symbols*(trials-peaks)
    dynamic = trials*(5*(symbols-1)+2)
    return dict(
        peaks=peaks, trials=trials, symbols=symbols, window=window,
        spline_horner_real_mac_products=interpolation,
        differential_power_and_magnitude_products=dynamic,
        direct_mac_product_equivalents=interpolation+dynamic,
        interpolation_real_adds=6*symbols*(trials-peaks),
        differential_three_product_adds=5*(symbols-1)*trials,
        energy_sum_adds=(symbols-2)*trials,
        signed_reference_sum_adds_upper=4*(symbols-1)*trials,
        score_divisions=trials, score_sqrts=trials, score_angles=trials,
        fine_comparisons=max(0, trials-1),
        coarse_word_reads_max=131*peaks, q12_score_bits=13,
        additional_peak_register_bits_upper=peaks*(64+13+1),
        spline_float_coefficients=4*(window-1),
        numpy_complex128_coefficient_bytes=4*(window-1)*16,
        example_unvalidated_complex32_coefficient_bits=4*(window-1)*32,
        fine_fixed_format_verified=False,
        excluded="spline preparation/elimination, division/sqrt/angle units, scratch, address/ports, primitive-width tiling, feedback and all packet work",
        evidence="one direct-MAC graph estimate; not an impossibility bound for alternate fixed-phase shift/add or local interpolators",
    )


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    scenarios = []
    for clock in [100_000_000, 160_000_000, 200_000_000]:
        ledger = receiver_resource_ledger(RXResourceBudget(clock_hz=clock))
        # Integer number of dedicated fine multipliers within the optimistic
        # remaining slots. No credit for unproven interleaving/port arbitration.
        available = ledger["dsp"]["remaining_optimistic_fine_products_per_request"]
        units = math.floor(available*500_000/clock)
        for peaks in [1, 2]:
            graph = estimate(peaks)
            capacity = CaptureCapacity(CaptureCapacityAssumptions(
                clock_hz=clock, fine_multipliers=max(1, units), sample_bits=46))
            # Reuse only the existing finite FCFS/snapshot simulator. Its old
            # format-specific capacity estimate is deliberately not reused.
            capacity.service_cycles = math.ceil(graph["direct_mac_product_equivalents"]
                                                 / max(1, units))
            diagnostics = []
            report = capacity.simulate(range(256, 262145, 32), 262144,
                                       sink=diagnostics.append)
            blocked = capacity.simulate(range(256, 262145, 32), 262144,
                                        stalls=[(10000, 20000)])
            scenarios.append(dict(clock_hz=clock, graph=graph,
                fine_units_scenario=units, optimistic_products_available_per_request=available,
                direct_graph_within_partial_product_budget=graph["direct_mac_product_equivalents"] <= available,
                queue=report, downstream_stall=blocked,
                known_state_with_q12_and_peak_registers=ledger["storage"]["known_partial_bits"]
                    + ledger["storage"]["ranked_score_ring_optional_extra_bits"]
                    + 2*graph["additional_peak_register_bits_upper"],
                memory_budget_bits=16*1024*8,
                rrc_at_this_clock_verified=(clock == 200_000_000),
                full_chain_schedule_verified=False, synthesis_evidence=False))
            write(args.output / f"queue-{clock}-{peaks}.json", diagnostics)
    report = dict(scenarios=scenarios, whole_system_accepted=False,
                  scope="isolated direct-MAC service scenarios; ADC detection/front cost never omitted from ledger; queue drops are not algorithmic misses",
                  note="retain16KiB/18DSP module budgets;25% whole-chip reserve still unapproved; LUT-shift allocation and finite spline preparation not implemented")
    write(args.output / "model.json", report)
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps([{k: r[k] for k in ["clock_hz", "fine_units_scenario",
                     "direct_graph_within_partial_product_budget", "queue"]}
                     for r in scenarios], indent=2))
    return 1  # Explicit rejection of full-chain acceptance, with retained diagnostics.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
