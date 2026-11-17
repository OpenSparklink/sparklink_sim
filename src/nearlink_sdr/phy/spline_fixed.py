"""Experimental integer global cubic preparation (§6.2.3 receiver choice).

One already-owned window shares a binary exponent. This is a representation,
not additional ADC precision, AGC/SIR improvement or a default RX backend.
The float constructor is a reference boundary; from_rrc_words uses integer
19+19+8 operands matching the existing experimental RRC output contract.
"""

import math
import operator
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import numpy as np

from nearlink_sdr.phy.acquisition_flow import signed_multiplier_tiles
from nearlink_sdr.phy.uniform_spline import _DIGITS, preparation_calendar

__all__ = ["FixedUniformCubicEighth", "SplineFixedFormat", "fixed_preparation_calendar", "fixed_spline_cost"]


def _rne_ratio(n, d):
    q, r = divmod(abs(n), d)
    q += int(2*r > d or (2*r == d and q % 2))
    return q if n >= 0 else -q


def _rne_shift(n, shift):
    return n << -shift if shift <= 0 else _rne_ratio(n, 1 << shift)


@lru_cache(maxsize=3)
def _tables(fraction_bits):
    c = [Fraction(2)]
    for _ in range(638):
        c.append(1/(4-c[-1]))
    # C0 is a hardwired left shift, not a signed18bit coefficient containing2.
    fractional = [1/(1-2*c[n-2])-2 for n in range(4, 641)]
    scale = 1 << fraction_bits
    return (tuple(_rne_ratio(x.numerator*scale, x.denominator) for x in c[1:]),
            tuple(_rne_ratio(x.numerator*scale, x.denominator) for x in fractional))


def _prefix_tail(words):
    end = len(words)-1
    while end and words[end-1] == words[-1]:
        end -= 1
    return words[:end], words[-1]


@dataclass(frozen=True)
class SplineFixedFormat:
    """Finite study formats, not an arbitrary user-selectable bitwidth interface."""

    input_bits: int
    coefficient_fraction_bits: int

    def __post_init__(self):
        if (type(self.input_bits) is not int or self.input_bits not in (18, 21, 24)
                or type(self.coefficient_fraction_bits) is not int
                or self.coefficient_fraction_bits not in (14, 16, 17)):
            raise ValueError("Need a declared18/21/24bit source and14/16/17fraction-bit study")

    @property
    def work_bits(self):
        """Four signed guard bits at the recurrence/second-derivative nodes."""
        return self.input_bits+4


class FixedUniformCubicEighth:
    """Finite signed words, RNE, saturation, full products and fixed CSD phases.

    The complete window is scanned before conversion; no future packet samples
    or payload/CRC are consulted. The common signed10 exponent is held through
    both sweeps and all queries. Every arithmetic stored node saturates; callers
    must reject numeric overflow rather than silently trusting a clipped score.
    Diagnostics are finite per-window counters plus peak-bit analysis, not an
    extra copy of the input or the whole capture. State resets for each window.
    """

    def _setup(self, fmt):
        if not isinstance(fmt, SplineFixedFormat):
            raise ValueError("Need explicit SplineFixedFormat")
        self.format = fmt
        self.stats = dict(saturations=0, saturation_by_node={}, peak_signed_bits={},
                          roundings=0, products=0, discarded_input_components=0,
                          common_exponent_adjustments=0)
        c, last = _tables(fmt.coefficient_fraction_bits)
        self._c_prefix, self._c_tail = _prefix_tail(c)
        self._last_prefix, self._last_tail = _prefix_tail(last)

    def _count(self, key, amount=1):
        self.stats[key] += operator.index(amount)
        if self.stats[key] >= 2**64:
            raise OverflowError("Window diagnostic counter exhausted")

    def _bound(self, value, node, bits=None):
        bits = self.format.work_bits if bits is None else bits
        low, high = -(1 << (bits-1)), (1 << (bits-1))-1
        self.stats['peak_signed_bits'][node] = max(
            self.stats['peak_signed_bits'].get(node, 0), abs(value).bit_length()+1)
        if not low <= value <= high:
            self._count('saturations')
            counts = self.stats['saturation_by_node']
            counts[node] = counts.get(node, 0)+1
        return max(low, min(high, value))

    def _round(self, value, shift):
        if shift > 0:
            self._count('roundings')
        return _rne_shift(value, shift)

    def _add(self, a, b, node):
        return tuple(self._bound(x+y, node) for x, y in zip(a, b, strict=True))

    def _sub(self, a, b, node):
        return tuple(self._bound(x-y, node) for x, y in zip(a, b, strict=True))

    def _shift(self, a, power, node):
        return tuple(self._bound(self._round(x, -power), node) for x in a)

    def _multiply(self, a, coefficient, node):
        self._count('products', 2)
        return tuple(self._bound(self._round(x*coefficient,
            self.format.coefficient_fraction_bits), node) for x in a)

    @staticmethod
    def _lookup(prefix, tail, i):
        return prefix[i] if i < len(prefix) else tail

    def _set_exponent(self, exponent):
        if not -512 <= exponent <= 511:
            raise OverflowError("Window exponent exceeds signed10 representation")
        self.exponent = exponent

    def __init__(self, samples, fmt):
        self._setup(fmt)
        y = np.asarray(samples, dtype=complex)
        if y.ndim != 1 or not 4 <= len(y) <= 640 or not np.isfinite(y).all():
            raise ValueError("Need4..640finite complex samples")
        peak = float(max(np.max(abs(y.real)), np.max(abs(y.imag))))
        if peak > 1e70:
            raise OverflowError("Input exceeds existing synchronization range")
        exponent = math.frexp(peak)[1]-(fmt.input_bits-1) if peak else 0
        if peak and round(math.ldexp(peak, -exponent)) > (1 << (fmt.input_bits-1))-1:
            exponent += 1
            self._count('common_exponent_adjustments')
        self._set_exponent(exponent)
        source = []
        for value in y:
            words = tuple(self._bound(round(math.ldexp(float(v), -exponent)), 'input',
                                      fmt.input_bits) for v in (value.real, value.imag))
            self._count('discarded_input_components', sum(v != 0 and w == 0
                for v, w in zip((value.real, value.imag), words, strict=True)))
            source.append(words)
        self._prepare(source)

    @classmethod
    def from_rrc_words(cls, real, imag, exponents, fmt):
        """Integer adapter for signed19 components and signed8 binary exponents.

        Scan normalized magnitudes with19bit comparisons, not arbitrary-width
        reconstitution of every input. Align with RNE only after the maximum is
        known. Very small components can be discarded and are counted.
        """
        obj = cls.__new__(cls)
        obj._setup(fmt)
        if not 4 <= len(real) <= 640 or len(imag) != len(real) or len(exponents) != len(real):
            raise ValueError("Need matching4..640 RRC word vectors")
        source, greatest = [], None
        for row in zip(real, imag, exponents, strict=True):
            if any(isinstance(v, (bool, np.bool_)) for v in row):
                raise ValueError("Need integer RRC words, not booleans")
            try:
                r, i, e = map(operator.index, row)
            except TypeError as exc:
                raise ValueError("Need integer RRC words") from exc
            if not all(-2**18 <= v < 2**18 for v in (r, i)) or not -128 <= e <= 127:
                raise ValueError("RRC word exceeds19+19+8 contract")
            source.append((r, i, e))
            for v in (r, i):
                if v:
                    power = abs(v).bit_length()-1+e
                    normalized = abs(v) << (19-abs(v).bit_length())
                    candidate = (power, normalized, abs(v), e)
                    if greatest is None or candidate[:2] > greatest[:2]:
                        greatest = candidate
        exponent = greatest[0]-(fmt.input_bits-2) if greatest else 0
        if greatest and _rne_shift(greatest[2], exponent-greatest[3]) > 2**(fmt.input_bits-1)-1:
            exponent += 1
            obj._count('common_exponent_adjustments')
        obj._set_exponent(exponent)
        converted = []
        for r, i, e in source:
            words = tuple(obj._bound(obj._round(v, exponent-e), 'input', fmt.input_bits)
                          for v in (r, i))
            obj._count('discarded_input_components', sum(v != 0 and w == 0
                for v, w in zip((r, i), words, strict=True)))
            converted.append(words)
        obj._prepare(converted)
        return obj

    def _prepare(self, source):
        y, n = tuple(source), len(source)
        q = [None]*n
        a = self._shift(self._sub(y[1], y[0], 'difference'), 2, 'boundary')
        q[0] = self._shift(self._add(a, self._sub(y[2], y[0], 'difference'), 'boundary'), -1, 'forward')
        for j in range(1, n-1):
            d = self._sub(y[j+1], y[j-1], 'difference')
            rhs = self._sub(self._add(d, self._shift(d, 1, 'triple'), 'triple'), q[j-1], 'forward_rhs')
            q[j] = self._multiply(rhs, self._lookup(self._c_prefix, self._c_tail, j-1), 'forward')
        end = self._shift(self._add(self._shift(self._sub(y[-1], y[-2], 'difference'),
            2, 'boundary'), self._sub(y[-1], y[-3], 'difference'), 'boundary'), -1, 'boundary')
        rhs = self._sub(end, self._shift(q[-2], 1, 'terminal_rhs'), 'terminal_rhs')
        fractional = self._lookup(self._last_prefix, self._last_tail, n-4)
        q[-1] = self._add(self._shift(rhs, 1, 'terminal'),
                         self._multiply(rhs, fractional, 'terminal'), 'terminal')
        right = q[-1]
        for j in range(n-2, -1, -1):
            product = self._shift(right, 1, 'slope') if j == 0 else self._multiply(
                right, self._lookup(self._c_prefix, self._c_tail, j-1), 'slope')
            slope = self._sub(q[j], product, 'slope')
            d = self._sub(y[j+1], y[j], 'difference')
            triple = self._add(d, self._shift(d, 1, 'triple'), 'triple')
            if j == n-2:
                last_second = self._shift(self._sub(self._add(slope,
                    self._shift(right, 1, 'second_rhs'), 'second_rhs'), triple,
                    'second_rhs'), 1, 'second')
            q[j] = self._shift(self._sub(self._sub(triple, self._shift(slope, 1,
                'second_rhs'), 'second_rhs'), right, 'second_rhs'), 1, 'second')
            right = slope
        q[-1] = last_second
        self.source_words, self.second_words = y, tuple(q)

    def __call__(self, positions):
        raw = np.asarray(positions)
        if raw.dtype.kind not in 'fiu' or raw.ndim > 1 or raw.size > 128:
            raise ValueError("Need scalar or at most128 real eighth-grid positions")
        times = np.atleast_1d(raw.astype(float))
        if (not np.isfinite(times).all() or np.any(times < 0)
                or np.any(times > len(self.source_words)-1)
                or np.any(times*8 != np.rint(times*8))):
            raise ValueError("Positions must be exact in-window eighths")
        output = []
        for tick in (times*8).astype(int):
            j, fraction = divmod(int(tick), 8)
            words = self.source_words[j]
            if fraction:
                inputs = (words, self.source_words[j+1], self.second_words[j], self.second_words[j+1])
                result = []
                for component in (0, 1):
                    total = None
                    for value, digits in zip(inputs, _DIGITS[fraction], strict=True):
                        for sign, power in digits:
                            part = value[component] << power
                            total = part if total is None else total+part if sign == 1 else total-part
                            total = self._bound(total, 'phase_accumulator', self.format.work_bits+11)
                    result.append(self._bound(self._round(total, 10), 'phase'))
                words = result
            output.append(complex(*(math.ldexp(v, self.exponent) for v in words)))
        result = np.asarray(output, complex)
        return complex(result[0]) if raw.ndim == 0 else result


def fixed_spline_cost(fmt, window=256):
    """Word/port and operation estimates; no new synthesis/clock acceptance."""
    if not isinstance(fmt, SplineFixedFormat) or type(window) is not int or not 4 <= window <= 640:
        raise ValueError("Need a study format and bounded window")
    c, last = _tables(fmt.coefficient_fraction_bits)
    p, tail = _prefix_tail(c)
    lp, ltail = _prefix_tail(last)
    cb, wb, ib = fmt.coefficient_fraction_bits+1, fmt.work_bits, fmt.input_bits
    return dict(source_component_bits=ib, work_component_bits=wb, coefficient_signed_bits=cb,
        common_exponent_bits=10, phase_accumulator_bits=wb+11,
        full_product_bits=wb+cb,
        source_and_work_logical_bits=window*(2*ib+2*wb),
        source_tdp36_bram36=math.ceil(2*ib/36)*math.ceil(window/1024),
        work_tdp36_bram36=math.ceil(2*wb/36)*math.ceil(window/1024),
        live_data_control_monitor_reservation_bits=8*2*wb+256+18*64+16+16*6+10,
        monitor_counter_count=18,
        monitor_counter_basis='13possible saturation-node counters plus5global counters;peak-bit fields reserved separately',
        peak_bit_monitor_lzc_units_if_enabled=2,
        monitoring_update_schedule_verified=False,
        source_read_ports=2, work_read_ports=2, work_write_ports=1,
        ports_overlap_rule='conversion/prepare/fine sequential within one owner;no simultaneous two-read+write credit',
        compressed_c_prefix=list(p), compressed_c_tail=tail,
        compressed_last_prefix=list(lp), compressed_last_tail=ltail,
        coefficient_words_before_compression=len(c)+len(last),
        coefficient_words_after_compression=len(p)+len(lp)+2,
        coefficient_bits_after_compression=(len(p)+len(lp)+2)*cb,
        coefficient_lookup='finite prefix+constant tail mux,zero precision loss after quantization;timing/LUT count unverified',
        dsp25x18_tiles_per_work_coefficient_product=signed_multiplier_tiles(wb, cb),
        terminal_whole_coefficient_is_not_operand=True,
        extra_terminal_real_adds=2,
        exponent_max_scan_samples=window, alignment_pass_samples=window,
        scan_and_alignment_assumed_pipeline_latency_each=2,
        serial_preparation_cycles_scenario=6*window+8,
        scan_pipeline_bits_scenario=2*(19+8+19+10),
        maximum_state_bits=19+8+19+10,
        maximum_state_fields='unsigned19 normalized magnitude,signed10 power,unsigned19 original magnitude,signed8 exponent',
        alignment_pipeline_bits_scenario=2*(46+2*ib+10+10+1),
        rrc_integer_input_component_bits=19, rrc_input_exponent_bits=8,
        alignment_barrel_shifters=2, align_rne_guard_sticky_units=2,
        input_lzc_units=2, input_abs_units=2,
        alignment_source_reads=window, alignment_destination_writes=window,
        source_buffers_separate_from_original_transport=True,
        existing_transport_words_not_replaced=True,
        unscaled_score_graph=dict(
            sample_component_bits=wb, difference_component_bits=2*wb+1,
            differential_four_real_product_tiles=4*signed_multiplier_tiles(wb, wb),
            two_difference_power_product_tiles=2*signed_multiplier_tiles(2*wb+1, 2*wb+1),
            correlation_component_bits_32symbol_sign_template=2*wb+7,
            two_correlation_magnitude_product_tiles=2*signed_multiplier_tiles(2*wb+7, 2*wb+7),
            actual_score_and_coherent_path_still_float=True,
            interpretation='hypothetical full-growth direct sign-template graph,not an accepted score format or impossibility bound;normalization/angle/wide compare still unallocated'),
        original_16kib_failure_preserved=True,
        rrc_or_tone_fixed_gate_covered=False, schedule_or_16msps_verified=False,
        synthesis_evidence=False)


def fixed_preparation_calendar(window=256):
    """Address-only scan/align/recurrence calendar with two-cycle input pipelines.

    Events are(phase,raw_read,Y_read_tuple,Y_write,Q_read,Q_write). Original
    46bit words remain in the already charged transport snapshot. Quantized Y
    and Q are separate buffers, not free padding. The exponent is unavailable
    until the maximum scan drains; no recurrence reads incompletely aligned Y.
    Arithmetic/feedback/routing timing remains an explicit scenario.
    """
    if type(window) is not int or not 4 <= window <= 640:
        raise ValueError('Need a finite4..640-sample owned window')
    for i in range(window):
        yield 'range', i, (), None, None, None
    for _ in range(2):
        yield 'range_drain', None, (), None, None, None
    for i in range(window+2):
        yield 'align', i if i < window else None, (), i-2 if i >= 2 else None, None, None
    for phase, source, read, write in preparation_calendar(window):
        if phase != 'range':
            yield phase, None, source, None, read, write
