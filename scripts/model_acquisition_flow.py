"""Freeze cycle/resource scenarios for the selected module; no wall-clock/synthesis claims."""

import argparse
import json
import tarfile
from dataclasses import asdict
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.acquisition_flow import AcquisitionFlow, FlowAssumptions


def run(args):
    selected = json.loads(args.selection.read_text())
    if selected['selected_input_bits'] is None or selected['held_out_seen']:
        raise ValueError('Need a development-only frozen width selection')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'driver.executed.py').write_bytes(Path(__file__).read_bytes())
    sources = {str(p): sha(p) for p in Path('src').rglob('*.py')}
    driver_hash = sha(Path(__file__))
    with tarfile.open(args.output/'source.tar', 'w') as archive:
        for p in [*sources, 'pyproject.toml', 'uv.lock']:
            archive.add(p, arcname=p)
    rows = []
    scenarios = [('parallel2', dict(lanes=2)), ('parallel4', dict(lanes=4)),
                 ('parallel8', dict(lanes=8)), ('clock160MHz', dict(clock_hz=160_000_000)),
                 ('wide_add2cycles', dict(adder_latency=2))]
    for name, options in scenarios:
        assumptions = FlowAssumptions(input_bits=selected['hardware_input_bits'], **options)
        model = AcquisitionFlow(assumptions)
        row = dict(name=name, assumptions=asdict(assumptions), estimate=model.estimate,
                   cycle_result=model.simulate(args.samples))
        rows.append(row)
        write(args.output/'results.json', rows)
        print(json.dumps(dict(name=name, II=model.initiation_interval,
            storage_bits=model.estimate['storage']['total_bits'],
            dropped_tags=row['cycle_result']['dropped_sample_tags'],
            feasible=row['cycle_result']['continuous_rate_feasible'])), flush=True)
    # Counterexample: a PN correlator is not a boxcar rolling sum.
    old = 1-2+3
    naive, actual = old-1+4, 2-3+4
    assert naive != actual
    write(args.output/'rejected-constant-time-update.json', dict(
        coefficients=[1, -1, 1], samples=[1, 2, 3, 4], previous_dot=old,
        naive_subtract_add=naive, actual_dot=actual,
        conclusion='A generic PN correlation needs weighted interior updates; do not use boxcar recurrence'))
    write(args.output/'frozen-plan.json', dict(selection=str(args.selection.resolve()),
        selection_sha256=sha(args.selection), samples=args.samples, scenarios=scenarios,
        driver_sha256=driver_hash,
        sources=sources,
        status='cycle-model results and resource estimates only; no RTL or synthesis'))
    assert all(sha(Path(p)) == h for p, h in sources.items()), 'Source changed during cycle scenarios'
    assert sha(Path(__file__)) == driver_hash
    write(args.output/'manifest.json', {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=1_000_000)
    run(parser.parse_args())
