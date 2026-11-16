"""One-command frozen acquisition signoff; retain failed acceptance without stopping evidence."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_capture_hardware import sha, write


def run(args):
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    selection = json.loads(args.selection.read_text())
    if (selection['selected_input_bits'] != 14 or selection['hardware_input_bits'] != 22 or
            selection['synthetic_input_full_scale'] != 1024 or
            selection['hardware_input_full_scale'] != 262144 or selection['held_out_seen']):
        raise ValueError('This replay validates the existing frozen format; it cannot select new widths')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = {str(p): sha(p) for p in Path('src').rglob('*.py')}
    drivers = {str(p): sha(p) for p in Path('scripts').glob('*.py')}
    write(out/'freeze-before-held-out.json', dict(selection=selection,
        selection_path=str(args.selection.resolve()), selection_sha256=sha(args.selection),
        sources=sources, drivers=drivers,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip(),
        holdout_seen=False, new_holdout_seed=350116,
        budgets=dict(score_error=.005, rank_loss=.005, eligible_reference_score=.55,
            float_squared_score_error=1e-10, center_delta=.25, old_success_loss=0,
            negative_candidates=0, storage_bytes=16384),
        limitations='Only integer differential screen is fixed; no full receiver throughput claim'))
    commands, acceptance = [], {}

    def invoke(name, argv, python=sys.executable):
        command = [str(python), *map(str, argv)]
        with (out/(name+'.log')).open('w') as log:
            status = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        commands.append(dict(name=name, argv=command, exit_code=status.returncode))
        write(out/'commands.json', commands)
        print(name, 'exit', status.returncode, flush=True)
        return status.returncode

    driver = ['scripts/audit_capture_hardware.py']
    invoke('generate-held-out', [*driver, '--generate', 'held-out', '--output', out/'held-inputs'])
    if not (out/'held-inputs/frozen-inputs.json').exists():
        raise RuntimeError('Held-out generation failed; see log')
    for backend, chunks in [('reference', 'whole'), ('structure', 'whole'),
                            ('fixed', 'whole'), ('fixed', 'fixed'), ('fixed', 'random')]:
        name = 'held-'+backend+'-'+chunks
        invoke(name, [*driver, '--inputs', out/'held-inputs', '--output', out/name,
            '--correlation-backend', backend, '--input-bits', 14, '--chunk-pattern', chunks])
        if not (out/name/'summary.json').exists():
            raise RuntimeError(name+' failed before summary; see log')
        if backend != 'reference':
            acceptance[name] = not invoke(name+'-comparison', ['scripts/compare_capture_hardware.py',
                '--before', out/'held-reference-whole', '--after', out/name,
                '--output', out/(name+'-comparison.json')])
    # A single frozen width, never scan or re-select using held-out data.
    from sweep_acquisition_fixed import survey
    acceptance['held-fixed-score'] = survey(out/'held-inputs', out/'held-score-survey', 14)['budget_passed']
    if args.python_reference:
        for backend in ('reference', 'fixed'):
            name = 'held-python-'+backend
            invoke(name, [*driver, '--inputs', out/'held-inputs', '--output', out/name,
                '--correlation-backend', backend, '--input-bits', 14], args.python_reference)
            acceptance[name] = not invoke(name+'-comparison', ['scripts/compare_capture_hardware.py',
                '--before', out/('held-'+backend+'-whole'), '--after', out/name,
                '--output', out/(name+'-comparison.json')])
    for name, groups in [('original', ['original']),
                         ('boundaries', ['uncoded', 'held', 'coded', 'weak']),
                         ('negatives', ['negatives'])]:
        argv = ['scripts/replay_acquisition_regression.py', '--evidence-root', args.evidence_root,
                '--dataset', args.dataset, '--quiet-dataset', args.quiet_dataset,
                '--selection', args.selection, '--output', out/('regression-'+name)]
        for group in groups:
            argv += ['--group', group]
        invoke('regression-'+name, argv)
        verdict = out/('regression-'+name)/'acceptance.json'
        if not verdict.exists():
            raise RuntimeError(name+' replay failed before summary; see log')
        acceptance['regression-'+name] = json.loads(verdict.read_text())
    invoke('cycle-model', ['scripts/model_acquisition_flow.py', '--selection', args.selection,
        '--output', out/'cycle-model', '--samples', args.samples])
    assert sources == {str(p): sha(p) for p in Path('src').rglob('*.py')}, 'Source changed'
    assert drivers == {str(p): sha(p) for p in Path('scripts').glob('*.py')}, 'Drivers changed'
    write(out/'acceptance.json', acceptance)
    write(out/'manifest.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                               if p.is_file()})
    print(json.dumps(acceptance, indent=2), flush=True)
    return any(not (all(v[k] for k in ('functional_budget_passed', 'numerical_budget_passed'))
                    if isinstance(v, dict) else v) for v in acceptance.values())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-root', 'dataset', 'quiet-dataset', 'selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--python-reference', type=Path)
    parser.add_argument('--samples', type=int, default=1000000)
    raise SystemExit(run(parser.parse_args()))
