"""Finite value-carrying spline preparation audit; incomplete system returns1."""

import argparse
import ast
import gzip
import hashlib
import importlib.metadata
import itertools
import json
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from audit_spline_fixed import compare_stream, receive

from nearlink_sdr.phy import spline_fixed
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.phy.spline_fixed import FixedUniformCubicEighth, SplineFixedFormat
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.spline_flow import (
    CycledUniformCubicEighth,
    SplineFlow,
    SplineFlowAssumptions,
    spline_flow_cost,
)

FMT = SplineFixedFormat(18, 14)


def word_cases(held=False):
    """Word-node validation only, not a fresh RF receiver acceptance set."""
    rng = np.random.default_rng(660117 if held else 650117)
    lengths = (5, 23, 127, 255, 377, 631) if held else (4, 17, 64, 256, 376, 632, 640)
    for n in lengths:
        for kind in range(6):
            w = np.column_stack((rng.integers(-2**18, 2**18, (n, 2)),
                                 rng.integers(-128, 128, n)))
            if kind == 0:
                w[:] = 0
            elif kind == 1:
                w[:, 0], w[:, 1], w[:, 2] = -2**18, 2**18-1, 0
                if held:
                    w[::3, :2] *= -1
                    w[:, :2] = np.clip(w[:, :2], -2**18, 2**18-1)
            elif kind == 2:
                w[:, 0] = np.where(np.arange(n) % (3 if held else 2), 2**17, -2**17)
                w[:, 2] = 3 if held else -9
            elif kind == 3:
                w[:n//2, 2], w[n//2:, 2] = (-27, 15) if held else (17, -31)
            elif kind == 4:
                w[:] = 0
                w[n//3 if held else n//2] = (2**18-1, -2**18, -127 if held else 127)
            yield f'{n}-{kind}', tuple(tuple(map(int, row)) for row in w)


class Observer:
    """Independent memory mirror/read-latency checker; never supplies operands."""

    def __init__(self, words, ref, latency, trace=None):
        self.cells = {('raw', j): tuple(w) for j, w in enumerate(words)}
        self.ref, self.latency, self.trace = ref, latency, trace
        self.commands, self.pending = set(), {}
        self.reads = self.writes = self.comparisons = 0

    def __call__(self, e):
        if self.trace is not None:
            self.trace.write(json.dumps(e, sort_keys=True)+'\n')
        kind = e['event']
        if kind in ('read_issue', 'write_issue'):
            command = e['cycle'], e['bank']
            assert command not in self.commands, ('multiple SPO commands', e)
            self.commands.add(command)
        key = (e.get('bank'), e.get('index'))
        if kind == 'write_commit':
            self.cells[key] = tuple(e['word'])
            mark, j = e['mark']
            if mark in ('source', 'second'):
                expected = self.ref.source_words[j] if mark == 'source' else self.ref.second_words[j]
                assert tuple(e['word']) == expected, ('wrong committed value', e)
                self.comparisons += 1
            self.writes += 1
        elif kind == 'read_issue':
            assert key in self.cells, ('uninitialized read request', e)
            tag = e['bank'], tuple(e['tag'])
            assert tag not in self.pending
            self.pending[tag] = e['cycle'], self.cells[key]
        elif kind == 'read_return':
            cycle, value = self.pending.pop((e['bank'], tuple(e['tag'])))
            assert e['cycle']-cycle == self.latency and tuple(e['word']) == value, e
            self.reads += 1


def audit_case(output, ident, words, a, *, trace=False, stalls=(), max_cycles=None):
    ref = FixedUniformCubicEighth.from_rrc_words(*zip(*words, strict=True), FMT)
    with (gzip.open(output/(ident+'.jsonl.gz'), 'wt') if trace else nullcontext()) as handle:
        observer = Observer(words, ref, a.read_latency, handle)
        result = SplineFlow(FMT, a).run(words, sink=observer, stalls=stalls, max_cycles=max_cycles)
    write(output/(ident+'.json'), result)
    if max_cycles is None:
        assert result['released'] and not result['aborted'] and not observer.pending
        assert result['source_words'] == ref.source_words and result['second_words'] == ref.second_words
        assert result['exponent'] == ref.exponent and result['arithmetic'] == ref.stats
        assert observer.comparisons == 2*len(words)
    else:
        assert result['aborted'] and result['dropped_windows'] == 1
        assert not all(result['second_valid'])
    return dict(id=ident, n=len(words), assumptions=vars(a), ready_cycle=result['ready_cycle'],
                cycles=result['cycles'], stats=result['stats'], memory_reads=result['memory_reads'],
                memory_writes=result['memory_writes'], port_deferrals=result['port_deferrals'],
                row_peak_operations=result['row_peak_operations'], observer_reads=observer.reads,
                observer_writes=observer.writes, arithmetic=result['arithmetic'],
                intentionally_aborted=result['aborted'], exact=not result['aborted'])


def regression(output):
    fixture = Path('tests/data/sync_carrier_tone_success.npz')
    frozen = json.loads(fixture.with_suffix('.json').read_text())
    assert sha(fixture) == frozen['derived_npz_sha256']
    with np.load(fixture) as ar:
        x, quiet, payload = ar['iq'], ar['quiet'], ar['expected'].tobytes()
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=6, symbol_rate_mhz=4,
        whitening_seed=0x52, crc_seed=0x123456))
    meta = dict(id='old-success', positive=True, input_sha256=sha(fixture))
    base, arrays, _ = receive(output/'old-reference', meta, profile, x, quiet, payload, fmt=FMT)
    results, preparations = [], []
    def factory(samples, fmt):
        # Full compiled capacity charged; never hide cost using the current length.
        node = CycledUniformCubicEighth(samples, fmt, SplineFlowAssumptions(640))
        preparations.append(dict(n=len(samples), ready_cycle=node.preparation['ready_cycle']))
        return node
    with patch.object(spline_fixed, 'FixedUniformCubicEighth', factory):
        for chunking in ('whole', 'fixed', 'random'):
            row, test_arrays, _ = receive(output/('old-'+chunking), meta, profile, x, quiet,
                payload, fmt=FMT, chunking=chunking)
            comparison = compare_stream(base, row, arrays, test_arrays)
            # A bit-exact structural change requires stronger equality than prior float budgets.
            exact = (arrays.keys() == test_arrays.keys() and all(
                np.array_equal(arrays[k], test_arrays[k]) for k in arrays))
            comparison.update(bitexact_arrays=exact, correct=row['correct'])
            results.append(comparison)
    write(output/'packet-regression.json', dict(input=meta, results=results,
        preparation_windows=preparations, compiled_capacity=640,
        timing_scope='numerical integration only;functional query/score and packet loops;ADC clock not advanced during preparation'))
    assert all(r['passed'] and r['bitexact_arrays'] and r['correct'] == 1 for r in results)
    return results


def resources(output, rows):
    costs = [spline_flow_cost(FMT, SplineFlowAssumptions(n, layout))
             for n, layout in itertools.product((256, 376, 632, 640), ('inplace', 'split'))]
    chosen = costs[0]
    state = sum(chosen[k] for k in (
        'cache_data_tag_control_bits', 'q_response_slot_bits', 'y_pending_tag_flags_bits',
        'cache_protection_tag_flags_bits', 'memory_read_pipeline_bits', 'memory_command_register_bits',
        'raw_command_and_read_pipeline_bits', 'maximum_and_peak_pipeline_bits',
        'alignment_write_pipeline_bits', 'right_previous_and_tail_bits',
        'row_pipeline_reservation_bits', 'control_and_monitor_bits'))
    scenarios = []
    raw = np.random.default_rng(650119).integers(0, 2**46, (4096, 2), dtype=np.uint64)
    np.save(output/'transport-words.npy', raw)
    measured = next(r['cycles'] for r in rows if r['id'] == 'dev-256-0-inplace-r1-a2')
    for clock, interval in itertools.product((100_000_000, 160_000_000, 200_000_000), (32, 256)):
        a = CaptureTransportAssumptions(clock_hz=clock, fine_service_cycles=measured)
        result = CaptureTransport(a).simulate(raw, [(j, (j//interval)%2)
                                                   for j in range(256, 4097, interval)])
        scenarios.append(dict(clock_hz=clock, interval=interval, result=result,
            scope='optimistic preparation-only service;no interpolation/score/packet/decoder work'))
    return dict(costs=costs, serial_ft2_register_reservation_bits=state,
        serial_ft2_coefficients_bits=150, system_base_conditional_bram36=89,
        fourth_raw_owned_slot_bram36=2, four_engine_ft2_bram36_subtotal=91,
        unapproved25percent_reserve_bram36=35, remaining_after_candidate_reserve=14,
        ft2_four_engine_memory_luts=1280, ft2_with_rrc_memory_luts=2344,
        ft2_with_baseline_ideal_slicem=3320, full640_four_engine_memory_luts=5120,
        full640_with_baseline_rrc_ideal_slicem=4280, physical_slicem=4350,
        synthesis_or_placement_proven=False, original16kib_failure_preserved=True,
        original_rrc_2percent_failure_preserved=True, original18dsp_and54wide_gates_unchanged=True,
        scenarios=scenarios, unallocated=['fine scoring/CFO/normalization', 'timing/carrier/LLR/Polar',
            'TX preparation/output', 'realtime MAC/AES/DMA', 'control/mux/bank decode/monitor timing'],
        complete_capture_or16msps_verified=False)


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    policy = json.loads(Path('tests/data/spline_fixed_format.json').read_text())
    tree = ast.parse(Path('src/nearlink_sdr/phy/spline_fixed.py').read_text())
    names = {'_rne_ratio', '_rne_shift', '_tables', '_prefix_tail',
             'SplineFixedFormat', 'FixedUniformCubicEighth'}
    tree.body = [n for n in tree.body if getattr(n, 'name', None) in names]
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == policy['datapath_ast_sha256']
    source = {str(p): sha(p) for p in (Path(__file__), Path('scripts/audit_spline_fixed.py'),
        Path('tests/data/spline_fixed_format.json'), Path('tests/data/sync_carrier_tone_success.npz'),
        Path('pyproject.toml'), Path('uv.lock'), *Path('src').rglob('*.py'))}
    write(output/'protocol.json', dict(source=source, seeds=[650117, 660117],
        acceptance='exact source/second/exponent/numeric-stats;zero uninitialized or undefined SPO;cache3',
        selection='inplace/read1/row2 if exact,FT2<=400memoryLUT and no R1 latency loss vs split',
        validation='new word patterns/lengths only;all RF inputs fixed regression,no new RF rate claim',
        default_rx_changed=False, whole_system_accepted=False))
    inputs, rows = {}, []
    for ident, words in word_cases():
        inputs['dev-'+ident] = np.asarray(words, np.int64)
        for layout, read, row in itertools.product(('inplace', 'split'), (1, 2), (2, 3)):
            name = f'dev-{ident}-{layout}-r{read}-a{row}'
            rows.append(audit_case(output, name, words,
                SplineFlowAssumptions(max(256, len(words)), layout, read, row),
                trace=ident == '256-3'))
    # Real RRC output words, not a fabricated equivalent float port.
    with np.load('tests/data/sync_carrier_tone_success.npz') as ar:
        iq = ar['iq'][800:1056]
    rrc = FullMantissaRRC(FixedRRCFormat(12, 18))
    chunks = [rrc.feed_words(*rrc._quantize(iq[j:j+32])) for j in range(0, len(iq), 32)]
    raw = tuple(tuple(map(int, w)) for w in zip(*(np.concatenate([c[k] for c in chunks])
                                                for k in range(3)), strict=True))
    inputs['actual-rrc'] = np.asarray(raw, np.int64)
    write(output/'actual-rrc-provenance.json', dict(
        input_path='tests/data/sync_carrier_tone_success.npz',
        input_sha256=sha(Path('tests/data/sync_carrier_tone_success.npz')),
        iq_slice=[800,1056], filter_history='zero at start of slice;not original capture filter state',
        input_quantization='FullMantissa12 explicit 32-sample blocks',
        coefficient_bits=18, output_port_bits=[19,19,8]))
    rows.append(audit_case(output, 'actual-rrc', raw, SplineFlowAssumptions(), trace=True))
    write(output/'development.json', rows)
    assert spline_flow_cost(FMT)['memory_luts'] <= 400
    by_id = {r['id']: r for r in rows}
    assert all(r['ready_cycle'] == by_id[r['id'].replace('-inplace-', '-split-')]['ready_cycle']
               for r in rows if r['id'].endswith('-inplace-r1-a2'))
    write(output/'selection.json', dict(frozen=True, format=vars(FMT),
        assumptions=vars(SplineFlowAssumptions()), source=source, changed_signal_parameters=False,
        full640_memory_study_passed=False, whole_rx_accepted=False))
    # First exposure to independent node validation occurs AFTER the selection is written.
    validation = []
    for ident, words in word_cases(held=True):
        inputs['validation-'+ident] = np.asarray(words, np.int64)
        validation.append(audit_case(output, 'validation-'+ident, words,
            SplineFlowAssumptions(max(256, len(words)))))
    write(output/'validation.json', validation)
    np.savez_compressed(output/'inputs.npz', **inputs)
    blocked = audit_case(output, 'result-blocked', raw, SplineFlowAssumptions(),
                         stalls=((1500, 1800),), trace=True)
    aborted = audit_case(output, 'aborted', raw, SplineFlowAssumptions(),
                         max_cycles=1000, trace=True)
    packet = regression(output)
    cost = resources(output, rows)
    model = dict(preparation_node_accepted=True, numerical_pass=True, ft2_memory_study_pass=True,
        full640_memory_study_pass=False, development_cases=len(rows),
        validation_cases=len(validation), selected=dict(layout='inplace', read_latency=1, row_latency=2),
        result_blocked=blocked, aborted=aborted, packet_regressions=packet, resources=cost,
        original_numeric_and_resource_failures_preserved=True,
        full_interpolation_score_and_feedback_cycle_verified=False, main_rx_default_changed=False,
        new_rf_acceptance_gain=None, numerical_packet_regression_gain=0, whole_system_accepted=False)
    write(output/'model.json', model)
    write(output/'dependencies.json', {k: importlib.metadata.version(k) for k in ('numpy', 'scipy')})
    write(output/'manifest.json', {str(p.relative_to(output)): sha(p)
        for p in sorted(output.rglob('*')) if p.is_file()})
    print(json.dumps({k: v for k, v in model.items() if k != 'resources'}, indent=2))
    return 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
