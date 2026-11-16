"""Causal bounded persistent-tone detector/notch, on RRC output (§6.2.1.2,6.5/6.6).

This floating structure is an opt-in candidate, not a fully fixed-point frontend.
Two consecutive input lag-SPS coherence blocks enable a frozen narrow complex
notch. No sync reference, CRC, payload, or future packet boundary enters control.
"""

from dataclasses import dataclass

import numpy as np

__all__ = ['CausalToneNotch', 'ToneNotchConfig']


@dataclass(frozen=True)
class ToneNotchConfig:
    """Finite detector blocks and release hangover; coefficient freezes while enabled."""

    radius: float = 31/32
    coherence_squared: float = .6
    block_samples: int = 128
    release_blocks: int = 128

    def __post_init__(self):
        if (not np.isfinite(self.radius) or self.radius not in (15/16, 31/32) or
                not np.isfinite(self.coherence_squared) or not 0 < self.coherence_squared < 1 or
                type(self.block_samples) is not int or self.block_samples not in (64, 128, 256) or
                type(self.release_blocks) is not int or not 1 <= self.release_blocks <= 128):
            raise ValueError('Unsupported bounded tone detector configuration')


class CausalToneNotch:
    """Per-sample state and block detector updates independent of caller chunking.

    H(z)=(1-a*z^-1)/(1-radius*a*z^-1). The complex unit coefficient a is fitted
    from lag1 of the two enabling blocks; no coefficient chasing during a packet.
    Delayed samples and prior energy are retained continuously, including silence.
    """

    def __init__(self, config=None, sps=4):
        self.config = config or ToneNotchConfig()
        if not isinstance(self.config, ToneNotchConfig) or type(sps) is not int or sps != 4:
            raise ValueError('Tone structure currently supports an explicit config and SPS4')
        self.delay = np.zeros(sps, complex)
        self.power_delay = np.zeros(sps)
        self.slot = self.block_count = self.samples = 0
        self.c1 = self.c4 = self.acquire_sum = 0j
        self.e0 = self.e4 = 0.
        self.qualified = self.bad = 0
        self.active = False
        self.coefficient = self.previous_output = 0j
        self.last_coherence_squared = 0.
        self.stats = dict(samples=0, active_samples=0, detector_blocks=0, activations=0,
                          releases=0, nonlinear_coefficient_fits=0)

    @property
    def state_array_bytes(self):
        return self.delay.nbytes+self.power_delay.nbytes

    def advance(self, samples):
        x = np.asarray(samples, dtype=complex)
        if x.ndim != 1 or not 1 <= len(x) <= 4096 or not np.isfinite(x).all():
            raise ValueError('Need 1..4096 finite input samples')
        if np.any(abs(x.real) > 1e70) or np.any(abs(x.imag) > 1e70):
            raise OverflowError('Input exceeds finite fourth-power detector range')
        if self.samples+len(x) >= 2**63:
            raise OverflowError('Signed64 tone sample counter exhausted')
        y = np.empty(len(x), complex)
        cfg = self.config
        for i, value in enumerate(x):
            previous = self.delay[(self.slot-1) % 4]
            out = (value+self.coefficient*(cfg.radius*self.previous_output-previous)
                   if self.active else value)
            if not np.isfinite(out):
                raise OverflowError('Tone filter exceeded floating model range')
            y[i] = out
            self.stats['active_samples'] += int(self.active)
            self.previous_output = out
            self.c1 += value*previous.conjugate()
            self.c4 += value*self.delay[self.slot].conjugate()
            power = float(value.real**2+value.imag**2)
            self.e0 += power
            self.e4 += self.power_delay[self.slot]
            self.delay[self.slot], self.power_delay[self.slot] = value, power
            self.slot = (self.slot+1) % 4
            self.block_count += 1
            if self.block_count == cfg.block_samples:
                # Division below is diagnostic only. The gate uses squared quantities.
                n = float(self.c4.real**2+self.c4.imag**2)
                energy = self.e0*self.e4
                self.last_coherence_squared = n/energy if energy > 0 else 0.
                good = energy > 0 and n >= cfg.coherence_squared*energy
                if good:
                    self.bad = 0
                    if not self.active:
                        self.qualified += 1
                        self.acquire_sum += self.c1
                        if self.qualified == 2:
                            magnitude = abs(self.acquire_sum)
                            if magnitude > 0:
                                self.coefficient = self.acquire_sum/magnitude
                                self.active, self.previous_output = True, 0j
                                self.stats['activations'] += 1
                                self.stats['nonlinear_coefficient_fits'] += 1
                            # Opposite lag1 sums can cancel despite lag4 coherence.
                            # Retry a fresh pair instead of accumulating without bound.
                            self.qualified, self.acquire_sum = 0, 0j
                else:
                    self.bad = min(cfg.release_blocks, self.bad+1)
                    if not self.active:
                        self.qualified, self.acquire_sum = 0, 0j
                    elif self.bad == cfg.release_blocks:
                        self.active = False
                        self.qualified, self.acquire_sum = 0, 0j
                        self.stats['releases'] += 1
                self.c1 = self.c4 = 0j
                self.e0 = self.e4 = 0.
                self.block_count = 0
                self.stats['detector_blocks'] += 1
        self.samples += len(x)
        self.stats['samples'] += len(x)
        return y

    def operation_estimate(self):
        """Unmapped real-operation upper bounds, not DSP or synthesis evidence."""
        return dict(per_sample=dict(multiply=14, add=19, compare=1, divide=0, sqrt=0),
            per_block=dict(multiply=4, add=1, compare=4, divide=0, sqrt=0),
            per_activation=dict(complex_normalization=1),
            scope='includes continuous detector, active notch; diagnostics excluded',
            coefficient_fit='bounded two-block normalization; CORDIC mapping pending cycle model',
            state='four complex input samples, four powers, output/coefficients, finite accumulators/counters')
