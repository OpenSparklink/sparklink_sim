"""Actual memory values/latency/ownership, not only an address-count calendar."""

import itertools

import numpy as np
import pytest

from nearlink_sdr.phy.spline_fixed import FixedUniformCubicEighth, SplineFixedFormat
from nearlink_sdr.sim.spline_flow import (
    _RAM,
    CycledUniformCubicEighth,
    SplineFlow,
    SplineFlowAssumptions,
    _Run,
    spline_flow_cost,
)

FMT = SplineFixedFormat(18,14)


def expected(words):
    return FixedUniformCubicEighth.from_rrc_words(*zip(*words,strict=True),FMT)


@pytest.mark.parametrize('layout,read,row',itertools.product(('inplace','split'),(1,2),(2,3)))
def test_actual_delayed_words_equal_frozen_integer_reference(layout,read,row):
    rng = np.random.default_rng(650117)
    for n in (4,17,64,256,376,632,640):
        words = [tuple(map(int,v)) for v in zip(rng.integers(-2**18,2**18,n),
            rng.integers(-2**18,2**18,n),rng.integers(-128,128,n),strict=True)]
        ref = expected(words)
        events = []
        flow = SplineFlow(FMT,SplineFlowAssumptions(max(256,n),layout,read,row))
        result = flow.run(words,sink=events.append)
        assert result['released'] and not result['aborted'] and result['dropped_windows'] == 0
        assert result['source_words'] == ref.source_words and result['second_words'] == ref.second_words
        assert result['exponent'] == ref.exponent and result['arithmetic'] == ref.stats
        assert all(result['source_valid']) and all(result['second_valid'])
        assert result['stats']['y_reads'] == 2*n-3
        assert result['stats']['q_reads'] == n-1
        assert result['stats']['peak_cache'] <= 3 and result['stats']['peak_qbuffer'] <= 1
        assert result['stats']['peak_compute_pending'] <= 4
        issues = {}
        for e in events:
            if e['event'] == 'read_issue':
                issues[(e['bank'],tuple(e['tag']))] = e['cycle']
            elif e['event'] == 'read_return':
                assert e['cycle']-issues[(e['bank'],tuple(e['tag']))] == read
            elif e['event'] == 'write_commit':
                assert e['cycle'] <= result['ready_cycle']
        assert not result['unresolved_memory_commands'] and not result['unresolved_memory_responses']


def test_no_functional_reference_preparation_is_called(monkeypatch):
    words = [(1,2,0),(-3,4,1),(5,-6,-1),(7,8,0)]
    ref = expected(words)
    monkeypatch.setattr(FixedUniformCubicEighth,'_prepare',lambda *a:pytest.fail('reference shortcut'))
    result = SplineFlow(FMT).run(words)
    assert result['second_words'] == ref.second_words


def test_explicit_cycle_adapter_preserves_functional_samples_and_all_stats(monkeypatch):
    rng = np.random.default_rng(650120)
    x = rng.normal(size=256)+1j*rng.normal(size=256)
    x[128:] *= 2**-40
    ref = FixedUniformCubicEighth(x,FMT)
    monkeypatch.setattr(FixedUniformCubicEighth,'_prepare',lambda *a:pytest.fail('reference preparation shortcut'))
    cycle = CycledUniformCubicEighth(x,FMT)
    assert cycle.preparation['released']
    assert cycle.source_words == ref.source_words and cycle.second_words == ref.second_words
    assert cycle.stats == ref.stats
    times = np.arange(128)/8
    assert np.array_equal(cycle(times),ref(times))
    assert cycle.stats == ref.stats


def test_aligned_boundary_strong_weak_zero_extremes_and_reset():
    flow = SplineFlow(FMT)
    for values in ([(0,0)]*4,[(2**17-1,-2**17)]*4,
                   [(2**17-1,1),(1,0),(-2**17,1),(1,-1)]):
        # The physical raw19 port can exactly carry these already normalized words.
        ref = expected([(*v,-17) for v in values])
        result = flow.run(ref.source_words,exponent=ref.exponent)
        assert result['released'] and result['second_words'] == ref.second_words
    zero = flow.run([(0,0,127)]*4)
    assert zero['exponent'] == 0 and zero['second_words'] == ((0,0),)*4


def test_stalled_descriptor_preserves_words_and_timeout_records_a_drop():
    words = [(0,1,0),(2,3,1),(4,5,-1),(6,7,0)]
    flow = SplineFlow(FMT)
    base = flow.run(words)
    held = flow.run(words,stalls=[(0,100)])
    assert held['ready_cycle'] == base['ready_cycle'] and held['cycles'] == 101
    assert held['stats']['held_result_cycles'] == 100-base['ready_cycle']
    assert held['source_words'] == base['source_words'] and held['second_words'] == base['second_words']
    assert not held['adc_backpressure']
    cut = flow.run(words,max_cycles=3)
    assert cut['aborted'] and not cut['prepared'] and cut['dropped_windows'] == 1
    assert cut['abort_reason'] == 'diagnostic_cycle_limit'
    assert not any(cut['second_valid']) and all(v is None for v in cut['second_words'])
    partial = flow.run(words,max_cycles=base['ready_cycle']-3)
    assert partial['aborted'] and any(partial['second_valid']) and not all(partial['second_valid'])
    assert all(v is None for v,valid in zip(partial['second_words'],partial['second_valid'],strict=True) if not valid)
    ready_but_blocked = flow.run(words,stalls=[(0,100)],max_cycles=50)
    assert ready_but_blocked['prepared'] and ready_but_blocked['aborted']
    assert all(ready_but_blocked['second_valid']) and not ready_but_blocked['released']


def test_single_spo_command_and_capture_time_value_are_not_hidden_extra_ports():
    events,received = [],[]
    ram = _RAM({'a':(128,(8,))},2,lambda event,**data:events.append((event,data)))
    ram.cells['a'][0] = (7,)
    ram.advance(0,lambda *args:received.append(args))
    assert ram.issue('a',0,'read',('check',0))
    assert not ram.issue('a',1,'read',('check',1))
    assert not ram.issue('a',0,'write',(9,))
    ram.advance(1,lambda *args:received.append(args))
    assert not received
    assert ram.issue('a',0,'write',(9,))
    ram.advance(2,lambda *args:received.append(args))
    assert received == [(('check',0),(7,))] and ram.cells['a'][0] == (9,)
    assert ram.deferrals == 2
    assert ram.issue('a',0,'read',('later',0))
    ram.advance(3,lambda *args:received.append(args))
    ram.advance(4,lambda *args:received.append(args))
    assert received[-1] == (('later',0),(9,))


def test_uninitialized_or_invalid_memory_operations_are_explicit():
    ram = _RAM({'a':(128,(8,))},1,lambda *a,**k:None)
    with pytest.raises(ValueError):
        ram.advance(1,lambda *a:None)
    ram.advance(0,lambda *a:None)
    for args in [('a',128,'read',None),('unknown',0,'read',None),
                 ('a',0,'erase',None),('a',0,'write',(128,)),('a',0,'write',[1])]:
        with pytest.raises(ValueError):
            ram.issue(*args)
    assert ram.issue('a',1,'read',('check',1))
    with pytest.raises(RuntimeError,match='uninitialized'):
        ram.advance(1,lambda *a:None)


def test_finite_reply_cache_and_pipeline_overflow_are_rejected():
    context = _Run(FMT,SplineFlowAssumptions(),[(0,0)]*4,0,None)
    context.cache = {0:(0,0),1:(1,1),2:(2,2)}
    context.order = [0,1,2]
    context.protected = {0,1,2}
    context.y_pending.add(3)
    with pytest.raises(RuntimeError,match='replacement'):
        context.receive(('y',3),(3,3))
    context.qbuffer = (0,(0,0))
    with pytest.raises(RuntimeError,match='slot overflow'):
        context.receive(('q',1),(1,1))
    with pytest.raises(RuntimeError,match='slot overflow'):
        context.receive(('unknown',1),(1,1))
    for _ in range(4):
        context.schedule(3,'row',None)
    with pytest.raises(RuntimeError,match='pipeline'):
        context.schedule(3,'row',None)
    context.committed(('source',0))
    with pytest.raises(RuntimeError,match='Duplicate source'):
        context.committed(('source',0))
    context.committed(('second',0))
    with pytest.raises(RuntimeError,match='Duplicate second'):
        context.committed(('second',0))
    context.ram.advance(0,lambda *a:None)
    context.write(('y0',0),(0,0),('source',0))
    with pytest.raises(RuntimeError,match='collided'):
        context.write(('y0',1),(1,1),('source',2))


@pytest.mark.parametrize('options',[dict(window_capacity=3),dict(window_capacity=True),
    dict(layout='ideal-unlimited'),dict(read_latency=0),dict(read_latency=True),dict(row_latency=1)])
def test_unmodeled_assumptions_are_rejected(options):
    with pytest.raises(ValueError):
        SplineFlowAssumptions(**options)


def test_compiled_capacity_and_port_widths_are_not_silently_expanded():
    flow = SplineFlow(FMT)
    for words in ([(0,0,0)]*3,[(0,0,0)]*257,[(True,0,0)]*4,
                  [(np.bool_(False),0,0)]*4,[(.5,0,0)]*4,[(0,0,128)]*4,[(2**18,0,0)]*4,[(0,0)]*4):
        with pytest.raises(ValueError):
            flow.run(words)
    for exponent in (True,512,1.5):
        with pytest.raises(ValueError):
            flow.run([(0,0)]*4,exponent=exponent)
    with pytest.raises(ValueError):
        SplineFlow(None)
    with pytest.raises(ValueError):
        SplineFlow(FMT,True)
    for kwargs in (dict(stalls=None),dict(stalls=[(0,1)]*17),dict(stalls=[(1,0)]),
                   dict(stalls=[(0,3),(2,4)]),dict(max_cycles=0),dict(max_cycles=True)):
        with pytest.raises(ValueError):
            flow.run([(0,0,0)]*4,**kwargs)


def test_literal_bank_costs_latency_sensitivity_and_no_whole_system_acceptance():
    inplace = spline_flow_cost(FMT)
    split = spline_flow_cost(FMT,SplineFlowAssumptions(layout='split'))
    assert inplace['memory_luts'] == 320 and split['memory_luts'] == 496
    assert inplace['logical_data_bits'] == 20480 and split['logical_data_bits'] == 31744
    assert inplace['allocated_memory_bits'] == 320*64 and inplace['ideal_slicem_sites'] == 80
    assert inplace['node_bram36'] == 0 and not inplace['placement_and_timing_verified']
    assert not inplace['whole_system_accepted']
    assert inplace['front_boundary_work_upper_per_window']['component_abs'] == 512
    assert inplace['front_boundary_work_upper_per_window']['alignment_variable_shifts'] == 512
    for n,mem in ((376,640),(632,1280)):
        assert spline_flow_cost(FMT,SplineFlowAssumptions(n))['memory_luts'] == mem
    for args in ((None,None),(FMT,True)):
        with pytest.raises(ValueError):
            spline_flow_cost(*args)
    words = [(j*37-100,77-j*23,0) for j in range(256)]
    fast = SplineFlow(FMT).run(words)
    slow = SplineFlow(FMT,SplineFlowAssumptions(read_latency=2)).run(words)
    split_slow = SplineFlow(FMT,SplineFlowAssumptions(layout='split',read_latency=2)).run(words)
    assert fast['ready_cycle'] == 1542
    assert slow['ready_cycle'] > split_slow['ready_cycle'] > fast['ready_cycle']
    assert slow['second_words'] == split_slow['second_words'] == fast['second_words']
