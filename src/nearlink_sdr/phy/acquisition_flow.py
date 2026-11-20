"""Finite FIFO/cycle model for the integer acquisition bank, without synthesis claims.

Signed25x18 multipliers, one-cycle wide adders/comparators and memory ports are
explicit assumptions. This model does not schedule RRC, fine search, timing,
carrier or decoding. Continuous input always incurs the full per-sample work.
"""

import heapq
import math
import operator
from collections import deque
from dataclasses import dataclass

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, fixed_bank_storage_bits

__all__ = ['AcquisitionFlow', 'FlowAssumptions', 'laned_correlation',
           'multiply_signed_tiles', 'signed_multiplier_tiles']


def _parts(width, payload):
    return max(1, (width+payload-2)//payload)


def signed_multiplier_tiles(a_bits, b_bits):
    """Schoolbook signed limbs:24/17 payload bits fit signed25/18-bit primitives."""
    if type(a_bits) is not int or type(b_bits) is not int or min(a_bits, b_bits) < 2:
        raise ValueError('Need signed operand widths≥2')
    return min(_parts(a_bits, 24)*_parts(b_bits, 17),
               _parts(b_bits, 24)*_parts(a_bits, 17))


def multiply_signed_tiles(a, b, a_bits, b_bits):
    """Executable signed decomposition reference, including negative high limbs."""
    signed_multiplier_tiles(a_bits, b_bits)
    if not (-(1 << (a_bits-1)) <= a < 1 << (a_bits-1) and
            -(1 << (b_bits-1)) <= b < 1 << (b_bits-1)):
        raise ValueError('Operand does not fit declared signed width')
    if _parts(a_bits, 24)*_parts(b_bits, 17) > _parts(b_bits, 24)*_parts(a_bits, 17):
        a, b, a_bits, b_bits = b, a, b_bits, a_bits

    def split(value, width, payload):
        count = _parts(width, payload)
        return [(value >> (i*payload)) if i == count-1 else
                ((value >> (i*payload)) & ((1 << payload)-1)) for i in range(count)]

    left, right = split(a, a_bits, 24), split(b, b_bits, 17)
    return sum(x*y << (24*i+17*j) for i, x in enumerate(left) for j, y in enumerate(right))


def laned_correlation(real, imag, coefficient_real, coefficient_imag, lanes):
    """Four independent sign accumulators per lane; no hidden negative-term multiplier."""
    if type(lanes) is not int or lanes < 1 or not (
            len(real) == len(imag) == len(coefficient_real) == len(coefficient_imag)):
        raise ValueError('Invalid lanes or unequal tap arrays')
    real, imag, coefficient_real, coefficient_imag = [list(map(operator.index, v)) for v in
        (real, imag, coefficient_real, coefficient_imag)]
    if any(v not in (-1, 0, 1) for v in coefficient_real+coefficient_imag):
        raise ValueError('Only sign/swap coefficients are allowed')
    total_real = total_imag = 0
    for lane in range(lanes):
        rr = ii = ir = ri = 0
        for k in range(lane, len(real), lanes):
            rr += int(real[k])*int(coefficient_real[k])
            ii += int(imag[k])*int(coefficient_imag[k])
            ir += int(imag[k])*int(coefficient_real[k])
            ri += int(real[k])*int(coefficient_imag[k])
        total_real += rr+ii
        total_imag += ir-ri
    return total_real, total_imag


def _reduction_cycles(terms, products, adders):
    cycles = 0
    while terms > 1:
        cycles += math.ceil(products*(terms//2)/adders)
        terms = (terms+1)//2
    return cycles


@dataclass(frozen=True)
class FlowAssumptions:
    """All rates/ports/latencies are hypotheses to be checked by future RTL/synthesis."""

    input_bits: int = 22
    raw_detector_bits: int = 16
    taps: int = 31
    sps: int = 4
    window_samples: int = 256
    lanes: int = 4
    clock_hz: int = 200_000_000
    input_hz: int = 16_000_000
    fifo_depth: int = 64
    multiplier_latency: int = 3
    adder_latency: int = 1
    comparator_latency: int = 1
    memory_read_latency: int = 1

    def __post_init__(self):
        for name in ('input_bits', 'raw_detector_bits', 'taps', 'sps', 'window_samples',
                     'lanes', 'clock_hz', 'input_hz', 'fifo_depth', 'multiplier_latency',
                     'adder_latency', 'comparator_latency', 'memory_read_latency'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Flow assumptions must be positive integers')
        if (not 4 <= self.input_bits <= 26 or not 4 <= self.raw_detector_bits <= 26 or
                not 1 <= self.taps <= 127 or not 1 <= self.sps <= 16 or
                self.lanes not in (2, 4, 8) or not 1 <= self.fifo_depth <= 1024 or
                not (self.taps+1)*self.sps <= self.window_samples <= 4096):
            raise ValueError('Flow dimensions exceed declared model bounds')
        if any(v & (v-1) for v in (self.sps, self.window_samples, self.fifo_depth)):
            raise ValueError('Address model requires power-of-two SPS/window/FIFO')


class AcquisitionFlow:
    """Deterministic staged work plus finite queues; overload records dropped sample tags.

    Every stage has separate resources. Stage durations conservatively finish one
    token before accepting the next; primitive latency and partial-product trees
    are included. Delay lines are owned by their consumer stage, so upstream jobs
    cannot overwrite a downstream token's historic tap values.
    """

    def __init__(self, assumptions=None):
        assumptions = FlowAssumptions() if assumptions is None else assumptions
        if not isinstance(assumptions, FlowAssumptions):
            raise ValueError('Need explicit FlowAssumptions')
        self.assumptions = assumptions
        a = assumptions
        self.widths = w = FixedBankFormat(a.input_bits).widths(a.taps, 2*a.taps, .6)
        td = signed_multiplier_tiles(a.input_bits, a.input_bits)
        te = signed_multiplier_tiles(w['difference'], w['difference'])
        tn = signed_multiplier_tiles(w['correlation'], w['correlation'])
        tr = signed_multiplier_tiles(a.raw_detector_bits, a.raw_detector_bits)
        latency = a.multiplier_latency
        al = a.adder_latency
        # Fixed resource allocation. Counts cannot be inferred from real-multiply FLOPs.
        diff = math.ceil(4*td/4)+latency+al*(_reduction_cycles(td, 4, 4)+1)
        energy = math.ceil(2*te/4)+latency+al*(_reduction_cycles(te, 2, 8)+1+2)
        dot = al*(math.ceil(a.taps/a.lanes)+1+math.ceil(math.log2(a.lanes)))+a.memory_read_latency
        score = math.ceil(2*tn/8)+latency+al*(_reduction_cycles(tn, 2, 8)+1)
        terms = w['threshold_integer'].bit_count()
        gate = al*math.ceil(math.log2(terms))+a.comparator_latency
        words = [2*a.input_bits+64, 2*w['difference']+64,
                 2*w['difference']+w['window_energy']+64,
                 2*w['correlation']+w['window_energy']+64,
                 w['numerator']+w['window_energy']+64]
        self.stages = [dict(name=n, cycles=c, input_word_bits=b, multiplier_primitives=m,
                            wide_adders=d) for n, c, b, m, d in zip(
            ('difference', 'energy', 'sign_dot', 'score_squared', 'threshold'),
            (diff, energy, dot, score, gate), words, (4, 4, 0, 8, 0),
            (4, 8, 4*a.lanes, 8, terms-1), strict=True)]
        self.stages.append(dict(name='tag_join_hit_write', cycles=a.memory_read_latency+
                                a.comparator_latency+1, input_word_bits=65,
                                multiplier_primitives=0, wide_adders=0))
        adder_widths = [max(2*a.input_bits+math.ceil(math.log2(td))+1, w['difference']),
            max(2*w['difference']+math.ceil(math.log2(te))+1, w['window_energy']),
            w['correlation'], max(2*w['correlation']+math.ceil(math.log2(tn))+1, w['numerator']),
            w['compare'], 0]
        for stage, bits in zip(self.stages, adder_widths, strict=True):
            stage['wide_adder_bits_upper'] = bits
        raw_cycles = (math.ceil(2*tr/2)+latency+al*(_reduction_cycles(tr, 2, 4)+1+2+2)+
                      a.comparator_latency)
        self.raw_detector = dict(cycles=raw_cycles, multiplier_primitives=2, wide_adders=4,
                                 input_word_bits=2*a.raw_detector_bits+64,
                                 raw_trigger_count_adders=2,
                                 scope='parallel continuous detector arithmetic; threshold format not verified')
        self.initiation_interval = max(raw_cycles, *(s['cycles'] for s in self.stages))
        self.query_cycles = 1+2*al+math.ceil(math.log2(a.window_samples))+1+math.ceil(
            math.log2(a.window_samples))+a.comparator_latency
        self.estimate = dict(
            evidence='cycle/dataflow model under stated assumptions; not synthesis',
            widths=w, signed_multiplier_tile_counts_per_sample=dict(
                difference=4*td, energy=2*te, score=2*tn, raw_detector=2*tr),
            variable_real_multiply_per_sample=10,
            primitive_multiplies_per_sample=4*td+2*te+2*tn+2*tr,
            constant_multiply='binary shift/add tree, no DSP; shifts are wires',
            numeric_add_sub_upper_per_sample=(4*math.ceil(a.taps/a.lanes)*a.lanes+
                2*a.lanes+2*(a.lanes-1)+9+4*(td-1)+2*(te-1)+2*(tn-1)+2*(tr-1)+terms-1),
            raw_trigger_count_add_sub=2,
            numeric_comparisons=3, divide=0, sqrt=0, other_nonlinear=0,
            upstream_quantizer=dict(nearest_even=2, saturation_comparisons=4,
                                   scope='input word-format conversion, outside this core schedule'),
            small_address_add_sub_upper_per_sample=(3+2*a.lanes+
                2*a.lanes*math.ceil(a.taps/a.lanes)+math.ceil(a.taps/a.lanes)),
            small_address_comparisons_upper_per_sample=(2+a.lanes+
                a.lanes*math.ceil(a.taps/a.lanes)+math.ceil(a.taps/a.lanes)),
            source_tag_increment_64bit=1,
            dot_complex_read_ports=a.lanes, coefficient_bits_per_dot_cycle=2*a.lanes,
            energy_power_read_ports=1, energy_power_write_ports=1,
            memory_contract='separate energy/dot histories; read old power before write; consumer-owned delays',
            hit_query=dict(parallel_read_bits=a.window_samples,
                barrel_bit_muxes=a.window_samples*math.ceil(math.log2(a.window_samples)),
                mask_bits=a.window_samples, or_gates=a.window_samples-1,
                cycles=self.query_cycles, request_interval_samples=8*a.sps,
                snapshot='one clock after current hit write, before masked OR'),
            multiplier_primitives=sum(s['multiplier_primitives'] for s in self.stages)+2,
            wide_adders=sum(s['wide_adders'] for s in self.stages)+4,
            address_units=dict(adders=2*a.lanes+4, comparators=a.lanes+3),
            separate_control_units=dict(source_tag_adder_bits=64, query_bound_adders_64bit=2,
                query_bound_comparators_64bit=2, raw_hit_count_adders=dict(
                    count=2, bits=math.ceil(math.log2(a.window_samples))+1),
                raw_join_tag_comparator_64bit=1),
            stages=self.stages, raw_detector=self.raw_detector,
            initiation_interval_cycles=self.initiation_interval,
            nominal_capacity_samples_per_second=a.clock_hz/self.initiation_interval,
            first_score_input_samples=a.taps*a.sps+1,
            core_no_queue_latency_cycles=sum(s['cycles'] for s in self.stages),
            existing_requester_first_query_input_samples=a.window_samples,
            fine_request_rate_upper_per_second=a.input_hz/(8*a.sps),
            downstream_contract='query result consumed when produced; fine-engine/backpressure/window snapshots not modeled',
            input_contract=(
                'post-RRC words/raw detector tags must already align;optional arrival calendar '
                'includes upstream latency;conversion and retiming hardware remain excluded'
            ),
            unresolved='RRC/fine search/carrier/timing/payload and actual resource/timing closure',
        )
        raw_window = 8*a.sps
        raw_p = 2*a.raw_detector_bits+1
        raw_e = raw_p+math.ceil(math.log2(raw_window))
        bank_bits = fixed_bank_storage_bits(a.sps, a.taps, a.window_samples, w)
        raw_bits = raw_window*raw_p+2*raw_e+a.window_samples+64+math.ceil(
            math.log2(raw_window))+2*math.ceil(math.log2(a.window_samples))+1
        fifo_words = [s['input_word_bits'] for s in self.stages]+[
            self.raw_detector['input_word_bits'], 65]  # Raw-flag join.
        fifo_bits = sum(fifo_words)*a.fifo_depth+a.fifo_depth  # Direct tag-store valid bits.
        fifo_control_bits = len(fifo_words)*(2*math.ceil(math.log2(a.fifo_depth))+
                            math.ceil(math.log2(a.fifo_depth+1))+2)+6
        query_bits = 2*(a.window_samples+128+math.ceil(math.log2(a.window_samples)))
        product_registers = (4*(2*a.input_bits+math.ceil(math.log2(td))+1)*td+
            2*(2*w['difference']+math.ceil(math.log2(te))+1)*te+
            2*(2*w['correlation']+math.ceil(math.log2(tn))+1)*tn)
        multiplier_registers = latency*(4*(4*a.input_bits+math.ceil(math.log2(td))+1)+
            4*(4*w['difference']+math.ceil(math.log2(te))+1)+
            8*(4*w['correlation']+math.ceil(math.log2(tn))+1)+8*a.raw_detector_bits)
        dot_registers = 4*a.lanes*w['correlation']+a.lanes*(2*w['difference']+2+16)+128
        constant_registers = (terms-1)*w['compare']+64*math.ceil(math.log2(terms))
        query_registers = a.window_samples*math.ceil(math.log2(a.window_samples))+2*a.window_samples+256
        work_registers = (al*product_registers+multiplier_registers+dot_registers+al*constant_registers+
                          query_registers+2*sum(s['input_word_bits'] for s in self.stages))
        total = bank_bits+raw_bits+fifo_bits+fifo_control_bits+query_bits+work_registers
        self.estimate['storage'] = dict(bank_bits=bank_bits, raw_detector_bits=raw_bits,
            fifo_word_bits=fifo_words, fifo_bits=fifo_bits, fifo_control_bits=fifo_control_bits,
            query_fifo_bits=query_bits,
            work_register_bits=work_registers, total_bits=total,
            scope='conservative explicit arrays/work registers and native primitive registers; excludes RRC/fine/frame',
            declared_16KiB_budget_passed=total <= 16*1024*8)
        self.estimate['ports'] = dict(dot='P independent packed-complex reads/cycle,1 write/token',
            power='1 old-power read and1 write/token, scheduled before overwrite',
            hit='1 bit write/token and a full parallel snapshot/query',
            fifo='each independent FIFO has1 read and1 write port',
            raw_join='direct sample-tag modulo depth;1 tag read/1 tag write;valid register bits up to3 updates/cycle',
            restriction='No single shared dual-port RAM can supply all tap reads; banking/replication/mux timing unproven')

    def simulate(self, samples, *, arrivals=None, sink=None):
        """Reserve bounded work at ideal or actual tagged input arrival clocks.

        Explicit arrivals are an iterator of contiguous(sample,cycle) pairs.
        Missing RRC outputs cannot be relabeled or replaced with zeros. Observer
        events describe future reservations,not early live result handoffs.
        Raw detector flags must already be aligned at this interface;their
        retiming and the upstream word conversion are not modeled here.
        """
        if type(samples) is not int or samples < 1:
            raise ValueError('Need positive integer sample count')
        if sink is not None and not callable(sink):
            raise ValueError('Need an optional reservation observer')
        try:
            source = None if arrivals is None else iter(arrivals)
        except TypeError as error:
            raise ValueError('Need iterable tagged arrival clocks') from error
        exhausted = object()
        previous_arrival = -1
        a = self.assumptions
        if (samples*a.clock_hz+a.input_hz-1)//a.input_hz >= 2**63-2**32:
            raise OverflowError('Cycle timestamp would exceed signed64-bit model bound')
        stages = [*self.stages, self.raw_detector]
        busy, queues, peaks, drops = [0]*len(stages), [deque() for _ in stages], [0]*len(stages), [0]*len(stages)
        output, latency_sum, maximum, minimum = 0, 0, 0, None
        query_busy, query_queue, query_peak, query_drops = 0, deque(), 0, 0
        query_latency_max = 0
        raw_flags, raw_slots, raw_flag_peak, raw_flag_drops = [], {}, 0, 0
        for n in range(samples):
            if source is None:
                offered = (n*a.clock_hz+a.input_hz-1)//a.input_hz
            else:
                item = next(source, None)
                if (not isinstance(item, (tuple, list)) or len(item) != 2
                        or type(item[0]) is not int or item[0] != n):
                    raise ValueError('Missing,duplicate or reordered source sample tag')
                offered = item[1]
                if type(offered) is not int or not previous_arrival < offered < 2**63-2**32:
                    raise ValueError('Need increasing signed64 arrival clocks,one input per edge')
                previous_arrival = offered
            current = offered
            # Raw detection is parallel. It has its own independent input FIFO.
            detector_finish = None
            complete = False
            drop_reason = None
            for j in [len(stages)-1, *range(len(self.stages))]:
                arrival = offered if j == len(stages)-1 else current
                q = queues[j]
                while q and q[0] <= arrival:
                    q.popleft()
                start = max(arrival, busy[j])
                if start > arrival and len(q) >= a.fifo_depth:
                    drops[j] += 1
                    current = arrival
                    drop_reason = 'stage_fifo_full:' + stages[j].get('name', 'raw_detector')
                    break
                if start > arrival:
                    q.append(start)
                    peaks[j] = max(peaks[j], len(q))
                busy[j] = start+stages[j]['cycles']
                if j == len(stages)-1:
                    detector_finish = busy[j]
                else:
                    current = busy[j]
            else:
                complete = True
            if detector_finish is not None:
                while raw_flags and raw_flags[0][0] <= detector_finish:
                    # Heap is simulator event scheduling, not associative hardware storage.
                    _, slot, tag = heapq.heappop(raw_flags)
                    if raw_slots.get(slot) == tag:
                        del raw_slots[slot]
                slot = n % a.fifo_depth
                if slot in raw_slots:
                    raw_flag_drops += 1
                    complete = False
                    drop_reason = 'raw_flag_slot_collision'
                else:
                    raw_slots[slot] = n
                    heapq.heappush(raw_flags, (max(current, detector_finish), slot, n))
                    raw_flag_peak = max(raw_flag_peak, len(raw_slots))
            if sink is not None:
                sink(dict(event='sample_reservation' if complete else 'sample_drop',
                          sample=n, arrived_cycle=offered, core_ready_cycle=current,
                          ready_cycle=max(current, detector_finish) if complete else None,
                          detector_ready_cycle=detector_finish, reason=drop_reason))
            if complete:
                output += 1
                delay = current-offered
                latency_sum += delay
                maximum = max(maximum, delay)
                minimum = delay if minimum is None else min(minimum, delay)
                if n % (8*a.sps) == 8*a.sps-1:
                    arrival = max(current, detector_finish)
                    while query_queue and query_queue[0] <= arrival:
                        query_queue.popleft()
                    start = max(arrival, query_busy)
                    if start > arrival and len(query_queue) >= 2:
                        query_drops += 1
                        if sink is not None:
                            sink(dict(event='query_drop', sample=n, arrived_cycle=offered,
                                      offered_cycle=arrival, reason='query_fifo_full'))
                        continue
                    if start > arrival:
                        query_queue.append(start)
                        query_peak = max(query_peak, len(query_queue))
                    query_busy = start+self.query_cycles
                    query_latency_max = max(query_latency_max, query_busy-offered)
                    if sink is not None:
                        sink(dict(event='query_reservation', sample=n, arrived_cycle=offered,
                                  offered_cycle=arrival, start_cycle=start,
                                  ready_cycle=query_busy))
        if source is not None and next(source, exhausted) is not exhausted:
            raise ValueError('Source calendar exceeds declared sample count')
        return dict(input_samples=samples, completed_sample_tags=output,
            dropped_sample_tags=samples-output, drops_by_stage=drops, peak_waiting_by_stage=peaks,
            peak_raw_flag_join=raw_flag_peak, raw_flag_join_drops=raw_flag_drops,
            fifo_depth=a.fifo_depth, active_token_additional_to_fifo=True,
            query_fifo_depth=2, peak_query_waiting=query_peak, query_drops=query_drops,
            max_sample_latency_cycles=maximum, min_sample_latency_cycles=minimum,
            mean_sample_latency_cycles=latency_sum/output if output else None,
            max_query_latency_from_last_sample_cycles=query_latency_max,
            pipeline_drain_end_cycle=max(*busy, query_busy),
            continuous_rate_feasible=(output == samples and not query_drops and
                                     self.initiation_interval*a.input_hz <= a.clock_hz and
                                     self.estimate['storage']['declared_16KiB_budget_passed']))
