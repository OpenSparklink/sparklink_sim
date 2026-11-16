"""Causal integer-grid differential bank, before the floating fine search (§6.5/6.6).

Only new correlation positions are computed. No CRC, payload, fine-phase decision,
or future sample enters this module. The floating structure uses extended precision
to study its arithmetic separately from the NumPy reference's rounding order.
"""

import math
from dataclasses import dataclass

import numpy as np

__all__ = ['CorrelationPoint', 'DifferentialBank', 'normalized_greater',
           'reference_integer_operations', 'sign_coefficients']


@dataclass(frozen=True)
class CorrelationPoint:
    """Squared numerator and input energy; common coefficient scale cancels."""

    start: int
    numerator: object
    energy: object
    coefficient_energy: int

    @property
    def score_squared(self):
        """Diagnostic division only; neither the gate nor ranking needs it."""
        return 0. if self.energy <= 0 else float(
            self.numerator/(self.energy*self.coefficient_energy))


def normalized_greater(first: CorrelationPoint, second: CorrelationPoint):
    """Compare normalized scores by cross products, never raw correlation magnitude.

    Valid nonnegative energies/numerators are guaranteed by the bank. Zero-energy
    points have score zero and cannot win a tie. Fixed point uses Python integers
    here as an emulator of the explicitly sized wide products, not float division.
    """
    if first.energy <= 0:
        return False
    if second.energy <= 0:
        return first.numerator > 0
    return bool(first.numerator*second.energy*second.coefficient_energy >
                second.numerator*first.energy*first.coefficient_energy)


def sign_coefficients(reference):
    """Recognize a common-scale cardinal/diagonal differential reference explicitly."""
    ref = np.asarray(reference)
    if ref.ndim != 1 or not 2 <= len(ref) <= 128 or not np.isfinite(ref).all():
        raise ValueError('Need 2..128 finite sync symbols')
    d = ref[1:]*ref[:-1].conj()
    scale = float(max(np.max(abs(d.real)), np.max(abs(d.imag))))
    if not scale > 0:
        raise ValueError('Need nonzero differential reference')
    real, imag = np.rint(d.real/scale).astype(int), np.rint(d.imag/scale).astype(int)
    norms = real**2+imag**2
    if (not np.allclose(d, scale*(real+1j*imag), atol=scale*1e-12, rtol=0) or
            not np.all(norms == norms[0]) or norms[0] not in (1, 2)):
        raise ValueError('Structure supports only common-scale sign/swap sync coefficients')
    return real, imag


def reference_integer_operations(sps=4, window_samples=256, reference_symbols=32, min_start=1):
    """Source-level worst arithmetic of the existing native integer screen, one call.

    ``hypot`` is kept as a nonlinear primitive rather than assuming its library
    implementation is two multiplies and one sqrt. Address/control arithmetic,
    compiler FMA choices, the raw energy detector and fine search are separate.
    This is an estimate from the current scalar source, not a synthesis result.
    """
    if (type(sps) is not int or sps < 1 or type(reference_symbols) is not int or
            reference_symbols < 2 or type(window_samples) is not int or
            window_samples < reference_symbols*sps or not math.isfinite(min_start)):
        raise ValueError('Invalid reference work dimensions')
    k, differences, positions = reference_symbols-1, 0, 0
    for phase in range(sps):
        n = (window_samples-1-phase)//sps+1
        differences += n-1
        positions += sum(phase+i*sps >= min_start for i in range(n-reference_symbols+1))
    return dict(multiply=5*(k+differences)+positions*(4*k+1),
                add=2*(k+differences)+k+positions*5*k,
                divide=positions, sqrt=positions, hypot=k+differences+positions,
                normalized_positions=positions, new_differences=differences,
                comparison_at_least=2*positions,
                scope='estimated scalar-native arithmetic/call; excludes address, detector/fine')


class DifferentialBank:
    """Finite continuous bank with an input delay, derivative/power rings and hit bits.

    ``advance`` accepts at most eight symbols of new filtered samples. All raw
    samples, including quiet/interference and active frames, must be supplied.
    ``gate`` only enables the unchanged floating fine search. No coarse peak is
    substituted for the receiver's final maximum or its future-sample condition.
    """

    def __init__(self, reference, sps, window_samples, threshold=.6, *, origin=0,
                 input_full_scale=1024.):
        real, imag = sign_coefficients(reference)
        if (type(sps) is not int or not 1 <= sps <= 16 or
                type(window_samples) is not int or
                not len(reference)*sps <= window_samples <= 4096 or
                type(origin) is not int or not -(2**62) <= origin < 2**62 or
                not np.isfinite(threshold) or not 0 < threshold <= 1 or
                not np.isfinite(input_full_scale) or input_full_scale <= 0):
            raise ValueError('Invalid bank SPS/window/origin/threshold/full scale')
        self.sps, self.window_samples, self.origin = sps, window_samples, origin
        self.k, self.coefficient_energy = len(real), int(np.sum(real**2+imag**2))
        self._coefficients = (real+1j*imag).astype(np.clongdouble)
        self._sample_delay = np.zeros(sps, np.clongdouble)
        self._differences = np.zeros(self.k*sps, np.clongdouble)
        self._powers = np.zeros(self.k*sps, np.longdouble)
        self._energy = np.zeros(sps, np.longdouble)
        self._hits = np.zeros(window_samples, bool)
        self.input_full_scale = input_full_scale
        self.threshold_squared = np.longdouble(threshold)**2
        self.samples = 0
        self.stats = dict(samples=0, new_correlations=0, queries=0, queried_positions=0,
                          zero_energy=0, float_energy_rebases=0)

    @property
    def state_array_bytes(self):
        """Actual emulator array capacity, separate from future packed hardware bits."""
        return sum(a.nbytes for a in vars(self).values() if isinstance(a, np.ndarray))

    def _difference(self, samples):
        extended = np.concatenate((self._sample_delay, samples))
        d = extended[self.sps:]*extended[:-self.sps].conj()
        self._sample_delay = extended[-self.sps:].copy()
        return d

    def advance(self, samples, *, inspect=False):
        """Consume a causal bounded block; optional caller-owned diagnostics ≤8*SPS."""
        x = np.asarray(samples)
        if x.ndim != 1 or not 1 <= len(x) <= 8*self.sps or not np.isfinite(x).all():
            raise ValueError('Need 1..8*SPS finite filtered samples')
        if self.samples+len(x)+abs(self.origin) >= 2**63:
            raise OverflowError('Signed 64-bit sample counter exhausted')
        x = x.astype(np.clongdouble)/self.input_full_scale
        new = self._difference(x)
        power = new.real**2+new.imag**2
        joined = np.concatenate((self._differences, new))
        powers = np.concatenate((self._powers, power))
        corr = np.empty(len(new), np.clongdouble)
        energy = np.empty(len(new), np.longdouble)
        for phase in range(min(self.sps, len(new))):
            # Drop one old sample: K-1 historic terms + the new terms, no old dot products.
            seq = joined[self.sps+phase::self.sps]
            corr[phase::self.sps] = np.correlate(seq, self._coefficients, 'valid')
            lane = (self.samples+phase) % self.sps
            delta = power[phase::self.sps]-powers[phase:len(new):self.sps]
            values = self._energy[lane]+np.cumsum(delta)
            lane_power = powers[self.sps+phase::self.sps]
            # Floating cancellation guard only; the fixed integer recurrence is exact.
            small = np.flatnonzero(values <= np.max(powers)*1e-12)
            for i in small:
                values[i] = np.sum(lane_power[i:i+self.k], dtype=np.longdouble)
            self.stats['float_energy_rebases'] += len(small)
            energy[phase::self.sps] = np.maximum(values, 0)
            self._energy[lane] = values[-1]
        numerator = corr.real**2+corr.imag**2
        points = []
        for i in range(len(new)):
            index = self.samples+i
            ready = index >= self.k*self.sps
            accepted = ready and energy[i] > 0 and (
                numerator[i] >= self.threshold_squared*self.coefficient_energy*energy[i])
            self._hits[index % self.window_samples] = accepted
            self.stats['zero_energy'] += int(ready and energy[i] <= 0)
            self.stats['new_correlations'] += int(ready)
            if inspect and ready:
                points.append(CorrelationPoint(self.origin+index-self.k*self.sps,
                                               numerator[i], energy[i], self.coefficient_energy))
        self._differences = joined[-len(self._differences):].copy()
        self._powers = powers[-len(self._powers):].copy()
        self.samples += len(new)
        self.stats['samples'] += len(new)
        return points if inspect else None

    def gate(self, lower_start, upper_start):
        """Any legal integer position meets threshold; hit history is never unbounded."""
        if not np.isfinite([lower_start, upper_start]).all():
            raise ValueError('Need finite query bounds')
        lo = max(self.k*self.sps, math.ceil(lower_start-self.origin)+self.k*self.sps)
        hi = min(self.samples-1, math.floor(upper_start-self.origin)+self.k*self.sps)
        self.stats['queries'] += 1
        if lo > hi:
            return False, 0
        if lo < self.samples-self.window_samples:
            raise ValueError('Query exceeds bounded retained history')
        self.stats['queried_positions'] += hi-lo+1
        indices = np.arange(lo, hi+1) % self.window_samples
        return bool(np.any(self._hits[indices])), hi-lo+1

    def worst_operations_per_sample(self):
        """Real arithmetic graph estimate; includes continuous raw-IQ energy detector.

        Coefficients are sign/swap wiring. Counts are independent of quiet jumping
        and do not include the unchanged RRC, fine search or downstream receiver.
        The floating emulator's cancellation rebases are separately observable.
        """
        return dict(multiply=11, add=4*self.k+7, compare=3, divide=0, sqrt=0,
                    floating_emulator_extra_add_bound=self.k-1,
                    other=dict(coefficient_sign_controls=4*self.k,
                               fixed_power_of_two_shifts=0),
                    data_reads=2*self.k+6, data_writes=9,
                    coefficient_bit_reads=2*self.k,
                    query_hit_reads_max=self.window_samples,
                    query_scope='once per 8*SPS samples; parallel masked OR needs explicit ports',
                    scope='estimate for sign/swap bank plus continuous raw energy; no synthesis')
