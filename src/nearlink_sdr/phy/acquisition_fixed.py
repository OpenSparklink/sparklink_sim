"""Integer arithmetic model of the causal acquisition bank, not a quantized receiver.

Input rounding is nearest/even with saturation. Differences, correlations, power
and energy keep their full derived precision; they never silently wrap. Only the
squared threshold constant is quantized. Wide Python integers emulate explicitly
sized datapaths; they are not a claim about one DSP operation or a synthesized RTL.
"""

import math
import operator
import sys
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from nearlink_sdr.phy.acquisition_bank import CorrelationPoint, DifferentialBank, sign_coefficients

__all__ = ['FixedBankFormat', 'FixedDifferentialBank', 'fixed_bank_storage_bits']


def fixed_bank_storage_bits(sps, k, window_samples, widths):
    """Packed bank state, excluding FIFO/work registers; shared with the cycle model."""
    w = widths
    return (sps*2*w['input']+k*sps*(2*w['difference']+w['difference_power'])+
            sps*w['window_energy']+window_samples+2*k+1+3*64+
            math.ceil(math.log2(window_samples))+math.ceil(math.log2(sps))+
            w['threshold_constant']+12)


@dataclass(frozen=True)
class FixedBankFormat:
    """Input width and constant fractional bits; full-scale is a separate static shift."""

    input_bits: int = 16
    threshold_fractional_bits: int = 24

    def __post_init__(self):
        if (type(self.input_bits) is not int or not 4 <= self.input_bits <= 26 or
                type(self.threshold_fractional_bits) is not int or
                not 8 <= self.threshold_fractional_bits <= 32):
            raise ValueError('Need input bits4..26 and threshold fraction8..32')

    def widths(self, k, coefficient_energy, threshold):
        """Conservative full-precision growth, including the normalization compare."""
        d = 2*self.input_bits+1
        c = d+math.ceil(math.log2(2*k))+1
        p = 2*d+1
        e = p+math.ceil(math.log2(k))
        n = 2*c+1
        constant = round(Fraction(str(threshold))**2*coefficient_energy*
                         (1 << self.threshold_fractional_bits))
        bits = max(1, constant.bit_length())
        return dict(input=self.input_bits, difference=d, correlation=c,
                    difference_power=p, window_energy=e, numerator=n,
                    threshold_constant=bits,
                    compare=max(n+self.threshold_fractional_bits, e+bits),
                    threshold_integer=constant)


class FixedDifferentialBank(DifferentialBank):
    """Full-precision signed difference/correlation and unsigned power/energy model.

    np.int64 is used only where derived bounds prove it cannot overflow. Powers,
    energies, numerator and threshold products are bounded Python integers. No
    floating score, sqrt or division enters gate decisions. The inherited bounded
    hit query uses the same integer positions as the floating structure.
    """

    def __init__(self, reference, sps, window_samples, threshold=.6, *, origin=0,
                 input_full_scale=1024., format=None):
        super().__init__(reference, sps, window_samples, threshold,
                         origin=origin, input_full_scale=input_full_scale)
        self.format = FixedBankFormat() if format is None else format
        if not isinstance(self.format, FixedBankFormat) or not math.log2(input_full_scale).is_integer():
            raise ValueError('Need FixedBankFormat and power-of-two full scale')
        self.widths = self.format.widths(self.k, self.coefficient_energy, threshold)
        if self.widths['threshold_integer'] == 0:
            raise ValueError('Threshold rounded to zero; insufficient constant precision')
        if self.widths['correlation'] >= 63:
            raise ValueError('Correlation exceeds the proven int64 arithmetic path')
        self._coefficient_real, self._coefficient_imag = sign_coefficients(reference)
        self._sample_real = np.zeros(sps, np.int64)
        self._sample_imag = np.zeros(sps, np.int64)
        self._difference_real = np.zeros(self.k*sps, np.int64)
        self._difference_imag = np.zeros(self.k*sps, np.int64)
        self._powers = np.full(self.k*sps, 0, object)
        self._energy = np.full(sps, 0, object)
        del self._coefficients, self._sample_delay, self._differences
        self.stats.update(input_saturations=0, intermediate_overflows=0)
        if self.logical_storage_bits > 16*1024*8:
            raise ValueError('Fixed bank exceeds the declared16KiB logical state budget')

    @property
    def logical_storage_bits(self):
        """Packed delay/energy/hit/coefficient/control state; FIFO/pipeline are separate."""
        return fixed_bank_storage_bits(self.sps, self.k, self.window_samples, self.widths)

    @property
    def emulator_state_bytes(self):
        """Conservative Python container/element size, not packed hardware storage."""
        total = 0
        for a in vars(self).values():
            if isinstance(a, np.ndarray):
                total += sys.getsizeof(a)
                if a.dtype == object:
                    total += sum(sys.getsizeof(v) for v in a)
        return total

    def _checked(self, value, width, signed=False):
        limit = 1 << (width-int(signed))
        if not (-limit if signed else 0) <= value < limit:
            self.stats['intermediate_overflows'] += 1
            raise OverflowError('Derived fixed-point accumulator bound violated')
        return value

    def quantize(self, samples):
        """Reference-IQ to input words; normalization is a declared power-of-two shift."""
        x = np.asarray(samples, dtype=np.clongdouble)
        if x.ndim != 1 or not np.isfinite(x).all():
            raise ValueError('Need finite one-dimensional input')
        limit = 1 << (self.format.input_bits-1)
        re = np.rint(x.real/self.input_full_scale*limit)
        im = np.rint(x.imag/self.input_full_scale*limit)
        self.stats['input_saturations'] += int(np.count_nonzero(
            (re < -limit) | (re >= limit)) + np.count_nonzero((im < -limit) | (im >= limit)))
        return np.clip(re, -limit, limit-1).astype(np.int64), np.clip(im, -limit, limit-1).astype(np.int64)

    def advance(self, samples, *, inspect=False):
        """One bounded causal update; each fixed operation follows the declared width."""
        x = np.asarray(samples)
        if x.ndim != 1 or not 1 <= len(x) <= 8*self.sps or not np.isfinite(x).all():
            raise ValueError('Need 1..8*SPS finite filtered samples')
        if self.samples+len(x)+abs(self.origin) >= 2**63:
            raise OverflowError('Signed 64-bit sample counter exhausted')
        r, i = self.quantize(x)
        return self._advance_words(r, i, inspect=inspect)

    def advance_words(self, real, imag, *, inspect=False):
        """Integer-word port for future RTL comparison; reject invalid words, do not re-quantize."""
        if (np.ndim(real) != 1 or np.ndim(imag) != 1 or len(real) != len(imag) or
                not 1 <= len(real) <= 8*self.sps):
            raise ValueError('Need equally sized1..8*SPS integer word arrays')
        try:
            r, i = [list(map(operator.index, values)) for values in (real, imag)]
        except TypeError as error:
            raise ValueError('Integer word port cannot round floating samples') from error
        bound = 1 << (self.format.input_bits-1)
        if any(v < -bound or v >= bound for v in (*r, *i)):
            raise ValueError('Input word exceeds declared signed width')
        if self.samples+len(r)+abs(self.origin) >= 2**63:
            raise OverflowError('Signed 64-bit sample counter exhausted')
        return self._advance_words(np.array(r, np.int64), np.array(i, np.int64), inspect=inspect)

    def _advance_words(self, r, i, *, inspect):
        count = len(r)
        re, im = np.r_[self._sample_real, r], np.r_[self._sample_imag, i]
        # Input≤26 bits: these products/sums have ≤53 bits and cannot overflow int64.
        dr = re[self.sps:]*re[:-self.sps]+im[self.sps:]*im[:-self.sps]
        di = im[self.sps:]*re[:-self.sps]-re[self.sps:]*im[:-self.sps]
        self._sample_real, self._sample_imag = re[-self.sps:].copy(), im[-self.sps:].copy()
        power = np.fromiter((int(a)**2+int(b)**2 for a, b in zip(dr, di, strict=True)), object)
        joined_re = np.r_[self._difference_real, dr]
        joined_im = np.r_[self._difference_imag, di]
        powers = np.r_[self._powers, power]
        corr_re, corr_im = np.empty(count, np.int64), np.empty(count, np.int64)
        energy = np.empty(count, object)
        for phase in range(min(self.sps, count)):
            a, b = joined_re[self.sps+phase::self.sps], joined_im[self.sps+phase::self.sps]
            corr_re[phase::self.sps] = (np.correlate(a, self._coefficient_real, 'valid')+
                                       np.correlate(b, self._coefficient_imag, 'valid'))
            corr_im[phase::self.sps] = (np.correlate(b, self._coefficient_real, 'valid')-
                                       np.correlate(a, self._coefficient_imag, 'valid'))
            lane = (self.samples+phase) % self.sps
            delta = power[phase::self.sps]-powers[phase:count:self.sps]
            values = self._energy[lane]+np.cumsum(delta)
            energy[phase::self.sps] = values
            self._energy[lane] = values[-1]
        points, w = [], self.widths
        for j in range(count):
            self._checked(int(dr[j]), w['difference'], True)
            self._checked(int(di[j]), w['difference'], True)
            self._checked(int(corr_re[j]), w['correlation'], True)
            self._checked(int(corr_im[j]), w['correlation'], True)
            self._checked(power[j], w['difference_power'])
            e = self._checked(int(energy[j]), w['window_energy'])
            n = self._checked(int(corr_re[j])**2+int(corr_im[j])**2, w['numerator'])
            left, right = n << self.format.threshold_fractional_bits, w['threshold_integer']*e
            self._checked(left, w['compare'])
            self._checked(right, w['compare'])
            index = self.samples+j
            ready = index >= self.k*self.sps
            self._hits[index % self.window_samples] = ready and e > 0 and left >= right
            self.stats['zero_energy'] += int(ready and e == 0)
            self.stats['new_correlations'] += int(ready)
            if inspect and ready:
                points.append(CorrelationPoint(self.origin+index-self.k*self.sps,
                                               n, e, self.coefficient_energy))
        self._difference_real = joined_re[-self.k*self.sps:].copy()
        self._difference_imag = joined_im[-self.k*self.sps:].copy()
        self._powers = powers[-self.k*self.sps:].copy()
        self.samples += count
        self.stats['samples'] += count
        return points if inspect else None

    def worst_operations_per_sample(self):
        estimate = super().worst_operations_per_sample()
        estimate['floating_emulator_extra_add_bound'] = 0
        estimate['widths'] = self.widths
        estimate['input_quantizer'] = dict(round_nearest_even=2, saturation_compare=4,
                                           scope='upstream format conversion; not included in core')
        estimate['logical_state_bits'] = self.logical_storage_bits
        estimate['normalization'] = 'integer squared threshold; no sqrt or division'
        return estimate
