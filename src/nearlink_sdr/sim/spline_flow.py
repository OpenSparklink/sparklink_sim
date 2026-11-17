"""Value-carrying finite spline preparation; not a complete capture schedule.

§6.2.3 receiver implementation. Commands drive a bank after edge c; the RAM
captures/commits at edge c+1. Registered reads return at c+read_latency.
One SPO address cannot issue both read and write in the same command cycle.
Arithmetic row latency is an assumption to be tested by later synthesis.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.spline_fixed import (
    FixedUniformCubicEighth,
    SplineFixedFormat,
    _rne_shift,
    fixed_spline_cost,
)

__all__ = ["CycledUniformCubicEighth", "SplineFlow", "SplineFlowAssumptions", "spline_flow_cost"]


@dataclass(frozen=True)
class SplineFlowAssumptions:
    """Compiled capacity and finite memory/recurrence latency scenarios."""

    window_capacity: int = 256
    layout: str = 'inplace'
    read_latency: int = 1
    row_latency: int = 2

    def __post_init__(self):
        if (type(self.window_capacity) is not int or not 4 <= self.window_capacity <= 640
                or self.layout not in ('inplace', 'split')
                or type(self.read_latency) is not int or self.read_latency not in (1, 2)
                or type(self.row_latency) is not int or self.row_latency not in (2, 3)):
            raise ValueError('Need compiled4..640capacity,explicit layout,read1/2,row2/3')


def _physical_depth(count):
    return 128 if count <= 128 else 256*math.ceil(count/256)


def _banks(fmt, a):
    half = _physical_depth((a.window_capacity+1)//2)
    result = {name:(half, (fmt.input_bits,)*2) for name in ('y0','y1')}
    if a.layout == 'inplace':
        result.update({name:(half, (fmt.work_bits,)*2) for name in ('w0','w1')})
    else:
        result.update(fw=(_physical_depth(a.window_capacity), (fmt.work_bits,)*2))
        result.update({name:(half, (fmt.work_bits,)*2) for name in ('m0','m1')})
    return result


class _RAM:
    """Single-address bank images; no oracle data or unbounded response queue."""

    def __init__(self, descriptions, latency, emit):
        self.cells = {name:[None]*depth for name,(depth,_) in descriptions.items()}
        self.formats, self.latency, self.emit = descriptions, latency, emit
        self.commands, self.responses = {}, []
        self.cycle = -1
        self.reads = self.writes = self.deferrals = self.peak_pending = 0

    def advance(self, cycle, receive, committed=None):
        if cycle != self.cycle+1:
            raise ValueError('Need consecutive edges')
        previous, self.commands, self.cycle = self.commands, {}, cycle
        for bank,(kind,index,data,mark) in previous.items():
            if kind == 'write':
                self.cells[bank][index] = data
                self.writes += 1
                self.emit('write_commit', bank=bank, index=index, word=data,mark=mark)
                if committed is not None:
                    committed(mark)
            else:
                word = self.cells[bank][index]
                if word is None:
                    self.emit('uninitialized_read', bank=bank, index=index)
                    raise RuntimeError('Read captured an uninitialized memory word')
                self.responses.append((cycle+self.latency-1,bank,index,data,word))
        if len(self.responses) > len(self.cells)*self.latency:
            raise RuntimeError('Read-register capacity exceeded')
        self.peak_pending = max(self.peak_pending, len(self.responses))
        pending, self.responses = self.responses, []
        for due,bank,index,tag,word in pending:
            if due == cycle:
                self.emit('read_return', bank=bank, index=index, tag=tag, word=word)
                receive(tag, word)
            else:
                self.responses.append((due,bank,index,tag,word))

    def issue(self, bank, index, kind, data, mark=None):
        if bank not in self.cells or type(index) is not int or not 0 <= index < len(self.cells[bank]):
            raise ValueError('Unallocated bank/address')
        if kind not in ('read','write'):
            raise ValueError('Need a read or write command')
        if bank in self.commands:
            self.deferrals += 1
            self.emit('port_defer', bank=bank, index=index, kind=kind)
            return False
        if kind == 'write':
            widths = self.formats[bank][1]
            if (not isinstance(data, tuple) or len(data) != len(widths)
                    or any(type(v) is not int or not -2**(b-1) <= v < 2**(b-1)
                           for v,b in zip(data,widths,strict=True))):
                raise ValueError('Memory write exceeds signed word fields')
        self.commands[bank] = kind,index,data,mark
        self.reads += int(kind == 'read')
        self.emit(kind+'_issue', bank=bank, index=index,
                  **({'tag':data} if kind == 'read' else {'word':data}))
        return True


class _Ops(FixedUniformCubicEighth):
    """Share the frozen RNE/saturation primitive, not its preparation routine."""

    def __init__(self, fmt):
        self.counts = dict(products=0,adds=0,rnes=0,bounds=0,shifts=0)
        self._setup(fmt)

    def _bound(self, v, tag, bits=None):
        self.counts['bounds'] += 1
        return super()._bound(v, tag, bits)

    def _round(self, v, shift):
        self.counts['rnes'] += int(shift > 0)
        return super()._round(v, shift)

    def _add(self, a, b, tag):
        self.counts['adds'] += 2
        return super()._add(a,b,tag)

    def _sub(self, a, b, tag):
        self.counts['adds'] += 2
        return super()._sub(a,b,tag)

    def _shift(self, a, power, tag):
        self.counts['shifts'] += 2
        return super()._shift(a,power,tag)

    def _multiply(self, a, coefficient, tag):
        self.counts['products'] += 2
        return super()._multiply(a,coefficient,tag)


class _Run:
    def __init__(self, fmt, a, words, exponent, sink):
        self.a, self.n, self.cycle, self.sink = a,len(words),-1,sink
        self.ops = _Ops(fmt)
        self.raw_mode = exponent is None
        banks = _banks(fmt,a)
        if self.raw_mode:
            banks['raw'] = (a.window_capacity,(19,19,8))
        self.ram = _RAM(banks,a.read_latency,self.emit)
        if self.raw_mode:
            self.ram.cells['raw'][:len(words)] = words
        self.load = iter(words) if not self.raw_mode else None
        self.exponent = exponent
        self.phase = 'scan' if self.raw_mode else 'load'
        self.index = self.peak_count = self.align_issued = 0
        self.greatest, self.exponent_due = None,None
        self.pipeline, self.busy = [],False
        self.cache, self.order, self.protected, self.y_pending = {},[],set(),set()
        self.qbuffer, self.q_pending = None,None
        self.previous = self.right = self.tail = None
        self.j = 0
        self.ready_cycle = None
        self.stats = dict(y_reads=0,cache_operand_touches=0,peak_cache=0,peak_y_pending=0,
                          q_reads=0,peak_qbuffer=0,peak_compute_pending=0,
                          waiting_operands_cycles=0,held_result_cycles=0)
        self.row_peak = dict.fromkeys(self.ops.counts,0)
        self.source_initialized = self.second_initialized = 0
        self.source_valid, self.second_valid = [False]*self.n,[False]*self.n

    def emit(self, event, **fields):
        if self.sink is not None:
            self.sink(dict(cycle=self.cycle,event=event,**fields))

    @staticmethod
    def address(prefix,j):
        return prefix+str(j%2),j//2

    def q_address(self,j):
        return self.address('w',j) if self.a.layout == 'inplace' else ('fw',j)

    def m_address(self,j):
        return self.address('w' if self.a.layout == 'inplace' else 'm',j)

    def write(self, address, word, mark):
        if not self.ram.issue(*address,'write',word,mark):
            raise RuntimeError('Arithmetic completion collided with another write command')

    def committed(self, mark):
        kind,j = mark
        if kind == 'source':
            if self.source_valid[j]:
                raise RuntimeError('Duplicate source initialization')
            self.source_valid[j] = True
            self.source_initialized += 1
        elif kind == 'second':
            if self.second_valid[j]:
                raise RuntimeError('Duplicate second-derivative commit')
            self.second_valid[j] = True
            self.second_initialized += 1

    def schedule(self, due, kind, data):
        self.pipeline.append((due,kind,data))
        if len(self.pipeline) > 4:
            raise RuntimeError('Finite arithmetic pipeline exceeded')
        self.stats['peak_compute_pending'] = max(self.stats['peak_compute_pending'],len(self.pipeline))

    def receive(self, tag, word):
        kind,j = tag
        if kind == 'peak':
            candidate = None
            for value in word[:2]:
                if value:
                    v = abs(value)
                    trial = (v.bit_length()-1+word[2],v << (19-v.bit_length()),v,word[2])
                    if candidate is None or trial[:2] > candidate[:2]:
                        candidate = trial
            self.schedule(self.cycle+1,'peak',candidate)
        elif kind == 'align':
            r,i,e = word
            fmt = self.ops.format
            converted = tuple(self.ops._bound(self.ops._round(v,self.exponent-e),'input',fmt.input_bits)
                              for v in (r,i))
            self.ops._count('discarded_input_components',sum(v != 0 and w == 0
                for v,w in zip((r,i),converted,strict=True)))
            self.write(self.address('y',j),converted,('source',j))
        elif kind == 'y':
            self.y_pending.remove(j)
            if len(self.cache) == 3:
                victim = next((p for p in self.order if p not in self.protected),None)
                if victim is None:
                    raise RuntimeError('Three-word cache has no legal replacement')
                del self.cache[victim]
                self.order.remove(victim)
            self.cache[j] = word
            self.order.append(j)
            self.stats['peak_cache'] = max(self.stats['peak_cache'],len(self.cache))
        else:
            if kind != 'q' or self.qbuffer is not None:
                raise RuntimeError('Forward-state response slot overflow')
            self.q_pending, self.qbuffer = None,(j,word)
            self.stats['peak_qbuffer'] = 1

    def need_y(self, backward, j):
        if backward:
            return j,j+1
        return (0,1,2) if j == 0 else (self.n-3,self.n-2,self.n-1) if j == self.n-1 else (j-1,j+1)

    def prefetch(self, needed, q=None):
        self.protected = set(needed)
        for j in needed:
            if j not in self.cache and j not in self.y_pending:
                if self.ram.issue(*self.address('y',j),'read',('y',j)):
                    self.y_pending.add(j)
                    self.stats['y_reads'] += 1
                    self.stats['peak_y_pending'] = max(self.stats['peak_y_pending'],len(self.y_pending))
                break  # One source word bus per cycle, not two ideal reads.
        if (q is not None and self.qbuffer is None and self.q_pending is None
                and self.ram.issue(*self.q_address(q),'read',('q',q))):
            self.q_pending = q
            self.stats['q_reads'] += 1

    def operands(self, indices):
        if not all(j in self.cache for j in indices):
            return None
        values = []
        for j in indices:
            values.append(self.cache[j])
            self.order.remove(j)
            self.order.append(j)
        self.stats['cache_operand_touches'] += len(indices)
        return values

    def forward(self,j,values):
        o = self.ops
        if j in (0,self.n-1):
            a,b,c = values
            d = o._sub(b,a,'difference') if j == 0 else o._sub(c,b,'difference')
            other = o._sub(c,a,'difference')
            end = o._shift(o._add(o._shift(d,2,'boundary'),other,'boundary'),-1,
                           'forward' if j == 0 else 'boundary')
            if j == 0:
                return end
            rhs = o._sub(end,o._shift(self.previous,1,'terminal_rhs'),'terminal_rhs')
            coefficient = o._lookup(o._last_prefix,o._last_tail,self.n-4)
            return o._add(o._shift(rhs,1,'terminal'),o._multiply(rhs,coefficient,'terminal'),'terminal')
        d = o._sub(values[1],values[0],'difference')
        rhs = o._sub(o._add(d,o._shift(d,1,'triple'),'triple'),self.previous,'forward_rhs')
        return o._multiply(rhs,o._lookup(o._c_prefix,o._c_tail,j-1),'forward')

    def backward(self,j,values,q):
        o = self.ops
        p = o._shift(self.right,1,'slope') if j == 0 else o._multiply(
            self.right,o._lookup(o._c_prefix,o._c_tail,j-1),'slope')
        slope = o._sub(q,p,'slope')
        d = o._sub(values[1],values[0],'difference')
        triple = o._add(d,o._shift(d,1,'triple'),'triple')
        tail = None
        if j == self.n-2:
            tail = o._shift(o._sub(o._add(slope,o._shift(self.right,1,'second_rhs'),'second_rhs'),
                                   triple,'second_rhs'),1,'second')
        second = o._shift(o._sub(o._sub(triple,o._shift(slope,1,'second_rhs'),'second_rhs'),
                                  self.right,'second_rhs'),1,'second')
        return second,slope,tail

    def complete(self, kind, data):
        if kind == 'peak':
            if data is not None and (self.greatest is None or data[:2] > self.greatest[:2]):
                self.greatest = data
            self.peak_count += 1
        else:
            backward,j,word,slope,tail = data
            self.write(self.m_address(j) if backward else self.q_address(j),word,
                       ('second' if backward else 'forward',j))
            self.emit('row_complete',backward=backward,index=j,word=word)
            self.busy = False
            if backward:
                self.right = slope
                if tail is not None:
                    self.tail = tail
                self.j = j-1
                if j == 0:
                    self.phase = 'tail'
            else:
                self.previous, self.j = word,j+1
                if j == self.n-1:
                    self.phase,self.j,self.right = 'backward',self.n-2,word

    def tick(self):
        self.cycle += 1
        self.ram.advance(self.cycle,self.receive,self.committed)
        previous,self.pipeline = self.pipeline,[]
        for due,kind,data in previous:
            if due == self.cycle:
                self.complete(kind,data)
            else:
                self.pipeline.append((due,kind,data))
        if self.phase == 'scan':
            if self.index < self.n:
                if self.ram.issue('raw',self.index,'read',('peak',self.index)):
                    self.index += 1
            elif self.peak_count == self.n:
                self.phase,self.exponent_due = 'exponent',self.cycle+1
        elif self.phase == 'exponent' and self.cycle == self.exponent_due:
            top = self.greatest
            self.exponent = top[0]-(self.ops.format.input_bits-2) if top else 0
            if top and _rne_shift(top[2],self.exponent-top[3]) > 2**(self.ops.format.input_bits-1)-1:
                self.exponent += 1
                self.ops._count('common_exponent_adjustments')
            self.ops._set_exponent(self.exponent)
            self.phase,self.index = 'align',0
        if self.phase in ('load','align'):
            if self.index < self.n:
                if self.phase == 'load':
                    self.write(self.address('y',self.index),next(self.load),('source',self.index))
                else:
                    self.ram.issue('raw',self.index,'read',('align',self.index))
                self.index += 1
            if self.index == self.n and self.source_initialized == self.n:
                self.phase,self.j = 'forward',0
        if self.phase in ('forward','backward'):
            backward,j = self.phase == 'backward',self.j
            if not self.busy:
                needed = self.need_y(backward,j)
                values = self.operands(needed)
                q_ready = not backward or (self.qbuffer is not None and self.qbuffer[0] == j)
                if values is not None and q_ready:
                    before = self.ops.counts.copy()
                    if backward:
                        second,slope,tail = self.backward(j,values,self.qbuffer[1])
                        self.qbuffer = None
                    else:
                        second,slope,tail = self.forward(j,values),None,None
                    for key in before:
                        self.row_peak[key] = max(self.row_peak[key],self.ops.counts[key]-before[key])
                    self.schedule(self.cycle+self.a.row_latency,'row',(backward,j,second,slope,tail))
                    self.emit('row_launch',backward=backward,index=j)
                    self.busy = True
                else:
                    self.stats['waiting_operands_cycles'] += 1
                    self.prefetch(needed,j if backward else None)
            if self.busy:
                if backward and j > 0:
                    self.prefetch(self.need_y(True,j-1),j-1)
                elif not backward:
                    self.prefetch(self.need_y(False,j+1) if j < self.n-1 else self.need_y(True,self.n-2),
                                  self.n-2 if j == self.n-1 else None)
        elif self.phase == 'tail':
            # Keep one logical write issue/cycle even for different parity banks.
            if not self.ram.commands:
                self.write(self.m_address(self.n-1),self.tail,('second',self.n-1))
                self.phase = 'drain'
        elif (self.phase == 'drain' and self.second_initialized == self.n
              and not self.ram.commands and not self.ram.responses and not self.pipeline):
            self.phase,self.ready_cycle = 'ready',self.cycle
            self.emit('prepared')


class SplineFlow:
    """Run one owned bounded window; explicitly abort on diagnostic cycle limit.

    No expected output, sync word, CRC, payload or true timing enters this model.
    Memory replies carry the actual operands. A run never calls the functional
    reference preparation. Source/history ownership before this node and the
    downstream interpolation/score remain separate integration obligations.
    """

    def __init__(self, fmt, assumptions=None):
        if not isinstance(fmt,SplineFixedFormat):
            raise ValueError('Need an explicit fixed format')
        self.format = fmt
        self.a = SplineFlowAssumptions() if assumptions is None else assumptions
        if not isinstance(self.a,SplineFlowAssumptions):
            raise ValueError('Need SplineFlowAssumptions')

    def run(self, words, *, exponent=None, stalls=(), max_cycles=None, sink=None):
        """Raw triples19+19+8, or aligned component pairs with common exponent.

        Stalls hold an already prepared descriptor; they never stop an ADC.
        All per-window state is reset on each new run; an exhausted diagnostic
        limit explicitly aborts that window, not a successful receiver result.
        """
        if not 4 <= len(words) <= self.a.window_capacity:
            raise ValueError('Input exceeds compiled owned-window capacity')
        raw_mode = exponent is None
        if not raw_mode and (type(exponent) is not int or not -512 <= exponent <= 511):
            raise ValueError('Need signed10 common exponent')
        widths = (19,19,8) if raw_mode else (self.format.input_bits,)*2
        checked = []
        for row in words:
            if len(row) != len(widths) or any(isinstance(v,(bool,np.bool_)) for v in row):
                raise ValueError('Need exact signed integer word fields')
            try:
                values = tuple(operator.index(v) for v in row)
            except TypeError as exc:
                raise ValueError('Need integer word fields') from exc
            if any(not -2**(b-1) <= v < 2**(b-1) for v,b in zip(values,widths,strict=True)):
                raise ValueError('Input word exceeds explicit port format')
            checked.append(values)
        if not isinstance(stalls,(list,tuple)) or len(stalls) > 16:
            raise ValueError('Need finite external result-stall intervals')
        previous = 0
        for lo,hi in stalls:
            if type(lo) is not int or type(hi) is not int or not previous <= lo < hi <= 1_000_000:
                raise ValueError('Need ordered bounded stall intervals')
            previous = hi
        limit = (previous+len(words)*32+256) if max_cycles is None else max_cycles
        if type(limit) is not int or not 1 <= limit <= 2_000_000:
            raise ValueError('Need a finite diagnostic cycle horizon')
        context = _Run(self.format,self.a,checked,exponent,sink)
        released = False
        for _ in range(limit):
            context.tick()
            if context.ready_cycle is not None:
                if not any(lo <= context.cycle < hi for lo,hi in stalls):
                    released = True
                    break
                context.stats['held_result_cycles'] += 1
        def snapshot(prefix, valid):
            return tuple(context.ram.cells[context.address(prefix,j)[0]][j//2] if valid[j] else None
                         for j in range(context.n))
        return dict(released=released,aborted=not released,prepared=context.ready_cycle is not None,
            ready_cycle=context.ready_cycle,cycles=context.cycle+1,exponent=context.exponent,
            source_words=snapshot('y',context.source_valid),
            second_words=snapshot('w' if self.a.layout == 'inplace' else 'm',context.second_valid),
            source_valid=tuple(context.source_valid),second_valid=tuple(context.second_valid),
            stats=context.stats,arithmetic=context.ops.stats,operation_counts=context.ops.counts,
            row_peak_operations=context.row_peak,memory_reads=context.ram.reads,
            memory_writes=context.ram.writes,port_deferrals=context.ram.deferrals,
            peak_memory_read_registers=context.ram.peak_pending,
            unresolved_memory_commands=len(context.ram.commands),
            unresolved_memory_responses=len(context.ram.responses),
            dropped_windows=int(not released),abort_reason=None if released else 'diagnostic_cycle_limit',
            adc_backpressure=False,full_capture_schedule_verified=False)


class CycledUniformCubicEighth(FixedUniformCubicEighth):
    """Explicit numerical integration adapter, not an ADC-timed receiver.

    The existing floating boundary packs the same aligned source words; their
    derivatives come only from the memory/cycle model. Subsequent interpolation
    uses the unchanged functional CSD evaluator. Its timing/score pipeline and
    delayed live-input feedback are NOT modeled by this adapter.
    """

    def __init__(self, samples, fmt, assumptions=None):
        self.flow = SplineFlow(fmt,assumptions)
        self.preparation = None
        super().__init__(samples,fmt)

    def _prepare(self, source):
        result = self.flow.run(source,exponent=self.exponent)
        if not result['released']:
            raise RuntimeError('Cycle preparation aborted;no functional fallback')
        self.preparation = result
        self.source_words,self.second_words = result['source_words'],result['second_words']
        for key,value in result['arithmetic'].items():
            if key == 'peak_signed_bits':
                for node,bits in value.items():
                    self.stats[key][node] = max(self.stats[key].get(node,0),bits)
            elif key == 'saturation_by_node':
                for node,count in value.items():
                    self.stats[key][node] = self.stats[key].get(node,0)+count
            else:
                self._count(key,value)


def spline_flow_cost(fmt, assumptions=None):
    """Concrete single-port LUTRAM layout and explicit state reservations."""
    if not isinstance(fmt,SplineFixedFormat):
        raise ValueError('Need explicit SplineFixedFormat')
    a = SplineFlowAssumptions() if assumptions is None else assumptions
    if not isinstance(a,SplineFlowAssumptions):
        raise ValueError('Need explicit SplineFlowAssumptions')
    banks = _banks(fmt,a)
    descriptions = {name:dict(depth=d,width=sum(w),memory_luts=d*sum(w)//64,
                             primitive='RAM128X1S' if d == 128 else 'RAM256X1S',
                             depth_segments=1 if d == 128 else d//256,
                             shared_read_write_address=True,
                             segment_muxes_and_decode_synthesis_verified=False,
                             extra_segment_outputs_not_used_as_free_ports=True)
                    for name,(d,w) in banks.items()}
    memory_luts = sum(row['memory_luts'] for row in descriptions.values())
    return dict(assumptions=vars(a),banks=descriptions,memory_luts=memory_luts,
        allocated_memory_bits=memory_luts*64,ideal_slicem_sites=math.ceil(memory_luts/4),
        node_bram36=0,raw_owned_bram_already_in_transport=True,
        logical_data_bits=a.window_capacity*(2*fmt.input_bits+(2 if a.layout == 'inplace' else 4)*fmt.work_bits),
        cache_data_tag_control_bits=3*(2*fmt.input_bits+math.ceil(math.log2(a.window_capacity)))+8,
        q_response_slot_bits=2*fmt.work_bits+10+2,
        y_pending_tag_flags_bits=a.read_latency*11,
        cache_protection_tag_flags_bits=3*11,
        cache_tag_compare_equivalents_10bit=9,
        memory_read_pipeline_bits=a.read_latency*sum(sum(widths)+17 for _,widths in banks.values()),
        memory_command_register_bits=sum(sum(widths)+18 for _,widths in banks.values()),
        raw_command_and_read_pipeline_bits=62+63*a.read_latency,
        maximum_and_peak_pipeline_bits=56+74,
        alignment_write_pipeline_bits=2*fmt.input_bits+25,
        right_previous_and_tail_bits=3*2*fmt.work_bits,
        row_pipeline_reservation_bits=(8*2*fmt.work_bits+18)*a.row_latency,
        control_and_monitor_bits=256+(18+9+4+15)*64+16*6+16+10,
        full_monitor_update_logic_and_schedule_verified=False,
        validity_masks_are_host_assertion_shadows=True,
        arithmetic_primitive_counts='recorded per run;row latency assumes an unrolled graph,not aggregate ops divided by free units',
        unrolled_row_peak_scenario=dict(real_constant_product_units=2,
            dsp25x18_product_tiles=2*fixed_spline_cost(fmt,a.window_capacity)['dsp25x18_tiles_per_work_coefficient_product'],
            real_add_sub_operations=14,RNE_results=4,saturated_results=26,
            bound_comparison_equivalents=52,constant_shifted_operands=10,
            note='literal peak operators,not a synthesized minimum;fusion/retiming must preserve RNE/saturation'),
        front_boundary_work_upper_per_window=dict(
            component_abs=2*a.window_capacity,leading_bit_detect=2*a.window_capacity,
            normalized_magnitude_shift=2*a.window_capacity,nonzero_tests=2*a.window_capacity,
            lexicographic_power10_magnitude19_comparisons=2*a.window_capacity,
            alignment_variable_shifts=2*a.window_capacity,
            common_exponent_update_events=1,
            note='conservative operator-use estimate,not included in recurrence operation counts;scan1word/cycle and RNE alignment need separate datapath timing'),
        control_mux_decode_luts=None,placement_and_timing_verified=False,
        memory_ports='one SPO command per physical bank;optional response register stages explicitly reserved',
        snapshot_and_trace_are_host_evidence=True,
        original16kib_gate_unchanged=True,whole_system_accepted=False)
