"""Execute finite product-owner/port boundaries; numeric acceptance stays separate."""

import argparse
import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads(args.plan.read_text())
    assert frozen["baseline_commit"] == "7a7a50040435671853b173215c45091ebf18e006"
    assumptions = PairRRCFlowAssumptions()
    assert frozen["clock_hz"] == assumptions.clock_hz
    assert frozen["input_hz"] == assumptions.input_hz
    write(
        args.output / "plan.json",
        dict(
            frozen_plan=str(args.plan.resolve()),
            frozen_sha256=sha(args.plan),
            source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
            sources={str(f): sha(f) for f in Path("src").rglob("*.py")},
            driver_sha256=sha(Path(__file__)),
            assumptions=asdict(assumptions),
            samples=4096,
            input="continuous finite-word source tokens, not generated IQ",
            numeric_acceptance=False,
            receiver_promotion=False,
        ),
    )
    cases = []
    for name, changes, stalls in [
        ("base", {}, []),
        ("160MHz", dict(clock_hz=160_000_000), []),
        ("slow-feedback", dict(arithmetic_ii=2), []),
        ("one-coefficient-port", dict(coefficient_ports=1), []),
        ("one-product-read-port", dict(product_read_ports=1), []),
        ("two-product-buffers", dict(product_buffers=2), []),
        ("consumer-II14", dict(consumer_ii=14), []),
        ("blocked", {}, [(100, 4100)]),
    ]:
        model = PairRRCFlow(replace(assumptions, **changes))
        result = model.simulate(4096, downstream_stalls=stalls, trace=True)
        write(args.output / (name + "-trace.json"), result.pop("trace"))
        cases.append(
            dict(
                name=name,
                assumptions=asdict(model.a),
                stalls=stalls,
                result=result,
                resource=model.resource_estimate(),
            )
        )
        write(args.output / "results.json", cases)
    base = cases[0]
    costs = base["resource"]
    budgets = frozen["operator_budgets"]
    accepted = (
        base["result"]["continuous_input_without_loss"]
        and costs["capacity_necessary_condition"]
        and costs["multiplier_units"] <= budgets["real_multipliers"]
        and costs["wide_units_total"] <= budgets["wide_add_compare"]
        and costs["small_units_total"] <= budgets["small_units_8bit"]
        and costs["leading_detectors_32bit"] <= budgets["leading_detectors_32bit"]
        and costs["register_tap_logical_bits"] <= budgets["capture_memory_bits"]
    )
    write(
        args.output / "summary.json",
        dict(
            isolated_dataflow_under_assumptions_passed=accepted,
            cost=costs,
            base=base["result"],
            numeric_acceptance=False,
            receiver_promotion=False,
            whole_rx_16msps=None,
            monitoring_update_schedule_complete=False,
            adc_input_conversion_model_complete=False,
            synthesis=None,
            scope="only finite-word pair/C18 RRC core; token dependency/lifetime/port model",
        ),
    )
    write(
        args.output / "failures.json",
        [c for c in cases if not c["result"]["continuous_input_without_loss"]],
    )
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(
        args.output / "manifest.json",
        {str(f.relative_to(args.output)): sha(f) for f in args.output.rglob("*") if f.is_file()},
    )
    print(
        json.dumps(
            dict(
                isolated_dataflow_under_assumptions_passed=accepted,
                numeric_acceptance=False,
                boundaries=[
                    dict(
                        name=c["name"],
                        dropped=c["result"]["dropped"],
                        invalid=c["result"]["invalid"],
                    )
                    for c in cases
                ],
            ),
            indent=2,
        )
    )
    return int(not accepted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    raise SystemExit(run(parser.parse_args()))
