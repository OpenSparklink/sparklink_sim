"""Replay frozen interference inputs; this accepts function, not hardware capacity."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from compare_capture_hardware import run as compare


def main(args):
    root, out = args.evidence_root, args.output
    out.mkdir(parents=True, exist_ok=False)
    frozen = root/'rx-interference-capture'
    hardware = root/'rx-acquisition-hardware'
    commands = []

    def call(name, command):
        with (out/(name+'.log')).open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        commands.append(dict(name=name, command=command, returncode=result.returncode))
        if result.returncode:
            raise RuntimeError(f'{name} failed; retained {out/(name+".log")}')

    try:
        for split in ('development', 'held'):
            for arm in ('reference', 'fallback'):
                name = split+'-'+arm
                call(name, [sys.executable, 'scripts/audit_interference_capture.py',
                     '--inputs', str(frozen/(split+'-inputs')), '--output', str(out/name),
                     '--radius', 'off' if arm == 'reference' else '0.96875'])
            assert not compare(out/(split+'-reference'), out/(split+'-fallback'),
                               out/(split+'-comparison.json'))
        for chunk in ('fixed', 'random'):
            name = 'held-'+chunk
            call(name, [sys.executable, 'scripts/audit_interference_capture.py',
                 '--inputs', str(frozen/'held-inputs'), '--output', str(out/name),
                 '--radius', '0.96875', '--chunks', chunk])
            assert not compare(out/'held-fallback', out/name, out/(name+'-comparison.json'))
            a, b = [json.loads((out/p/'results.json').read_text()) for p in
                    ('held-fallback', name)]
            assert [r['frames'] for r in a] == [r['frames'] for r in b]
        call('old-regression', [sys.executable, 'scripts/replay_acquisition_regression.py',
             '--evidence-root', str(root), '--dataset', str(args.dataset),
             '--quiet-dataset', str(args.quiet_dataset), '--selection',
             str(hardware/'fixed-development-scan-final/frozen-selection.json'),
             '--backend', 'structure', '--tone-radius', '0.96875',
             '--output', str(out/'old-regression')])
        report = dict(functional_passed=True, hardware_budget_passed=None,
                      scope='opt-in floating structure; fixed notch and full cycle gate pending',
                      python=sys.executable, commands=commands)
    finally:
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
    (out/'acceptance.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-root', 'output', 'dataset', 'quiet-dataset'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
