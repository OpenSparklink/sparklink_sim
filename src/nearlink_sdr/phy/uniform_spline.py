"""Finite uniform not-a-knot cubic preparation and eighth-grid evaluation.

This retains the global interpolation of the PSK sync reference (§6.2.3).
It is not the rejected four-point replacement. The floating datapath and ROM
constants are reference values; no fixed-point format is accepted here.
"""

import math

import numpy as np

__all__ = ["UniformCubicEighth", "eighth_weights", "preparation_calendar", "spline_work"]


def _rom():
    c = [2.]
    for _ in range(638):
        c.append(1./(4.-c[-1]))
    last = []
    for n in range(4, 641):
        last.append(1./(1.-2*c[n-2]))
    return tuple(c), tuple(last)


_C, _LAST = _rom()  # Offline configuration, never per-window divisions.


def _naf(value):
    """Canonical signed digits, descending powers; configuration-time only."""
    sign, value = (1 if value >= 0 else -1), abs(value)
    digits, power = [], 0
    while value:
        digit = 2-value % 4 if value % 2 else 0
        if digit:
            digits.append((sign*digit, power))
        value = (value-digit)//2
        power += 1
    return tuple(reversed(digits))


def eighth_weights(tick):
    """Integer weights on(y[i],y[i+1],M[i],M[i+1]), denominator1024."""
    if type(tick) is not int or not 0 <= tick <= 7:
        raise ValueError("Need eighth tick0..7")
    m, k = tick, 8-tick
    return 128*k, 128*m, (k**3-64*k)//3, (m**3-64*m)//3


_DIGITS = tuple(tuple(_naf(w) for w in eighth_weights(t)) for t in range(8))


def _shift(value, power):
    return complex(math.ldexp(value.real, power), math.ldexp(value.imag, power))


class UniformCubicEighth:
    """Own one finite window and one in-place forward/second-derivative buffer.

    The first/last derivative equations are s0+2s1=(5d0+d1)/2 and
    2s[n-2]+s[n-1]=(d[n-3]+5d[n-2])/2. Interior equations are
    s[i-1]+4s[i]+s[i+1]=3(y[i+1]-y[i-1]). ROM elimination constants depend
    only on length. The backward pass overwrites forward state with M=y''.
    A last-node result is held until a separate final write (also costed).
    """

    def __init__(self, samples):
        y = np.asarray(samples, dtype=complex)
        if y.ndim != 1 or not 4 <= len(y) <= 640 or not np.isfinite(y).all():
            raise ValueError("Need4..640finite uniformly spaced samples")
        if max(np.max(abs(y.real)), np.max(abs(y.imag))) > 1e70:
            raise OverflowError("Spline reference exceeds the existing sync range")
        self.samples = y.copy()
        y, n = self.samples, len(y)
        q = np.empty(n, complex)
        q[0] = _shift(_shift(y[1]-y[0], 2)+(y[2]-y[0]), -1)
        for i in range(1, n-1):
            d = y[i+1]-y[i-1]
            q[i] = _C[i]*(d+_shift(d, 1)-q[i-1])
        end = _shift(_shift(y[-1]-y[-2], 2)+(y[-1]-y[-3]), -1)
        q[-1] = _LAST[n-4]*(end-_shift(q[-2], 1))
        right = q[-1]
        for i in range(n-2, -1, -1):
            slope = q[i]-(_shift(right, 1) if i == 0 else _C[i]*right)
            d = y[i+1]-y[i]
            triple = d+_shift(d, 1)
            if i == n-2:
                last_second = _shift(slope+_shift(right, 1)-triple, 1)
            q[i] = _shift(triple-_shift(slope, 1)-right, 1)
            right = slope
        q[-1] = last_second
        self.second = q
        self.samples.setflags(write=False)
        self.second.setflags(write=False)

    def __call__(self, positions):
        """At most128eighth-grid points; no extrapolation or hidden sample fetch."""
        raw = np.asarray(positions)
        if raw.dtype.kind not in "fiu" or raw.ndim > 1 or raw.size > 128:
            raise ValueError("Need a scalar or at most128real eighth-grid positions")
        times = np.atleast_1d(raw.astype(float))
        if (not np.isfinite(times).all() or np.any(times < 0)
                or np.any(times > len(self.samples)-1)
                or np.any(times*8 != np.rint(times*8))):
            raise ValueError("Positions must be in-window exact eighths")
        out = np.empty(len(times), complex)
        for row, tick in enumerate((times*8).astype(int)):
            i, fraction = divmod(int(tick), 8)
            if fraction == 0:
                out[row] = self.samples[i]
                continue
            values = (self.samples[i], self.samples[i+1], self.second[i], self.second[i+1])
            total = None
            for value, digits in zip(values, _DIGITS[fraction], strict=True):
                for sign, power in digits:
                    part = _shift(value, power)
                    # First(highest y[i] digit) is positive.
                    total = part if total is None else total+part if sign == 1 else total-part
            out[row] = _shift(total, -10)
        return complex(out[0]) if raw.ndim == 0 else out


def spline_work(window=256, symbols=32, peaks=1):
    """Literal operation graph plus unvalidated storage/latency scenarios.

    No phase/grid/score changes or credit for reusing another consumer's ports.
    Inverse coefficient precision and variable product DSP tiling remain open.
    """
    if (any(type(v) is not int for v in (window, symbols, peaks))
            or not 4 <= window <= 640 or not 2 <= symbols <= 128
            or window <= 4*(symbols-1)+1 or peaks not in (1, 2)):
        raise ValueError("Need an existing bounded search geometry")
    costs = [2*(sum(len(x) for x in _DIGITS[t])-1) for t in (4, 5, 6, 7, 1, 2, 3, 4)]
    return dict(window=window, symbols=symbols, peaks=peaks, trials=9*peaks,
        preparation_real_constant_products=4*window-6,
        preparation_real_adds=16*window-4,
        upstream_owned_window_copy_word_reads=window,
        upstream_owned_window_copy_word_writes=window,
        preparation_source_word_reads=5*window,
        source_read_accounting="includes range scan; boundary duplicate operands held in registers",
        preparation_work_word_reads=2*window-1,
        preparation_work_word_writes=2*window,
        runtime_preparation_divisions=0, runtime_preparation_sqrt=0,
        configuration_rom_reciprocal_evaluations=638+637,
        rom_words=639+637, rom_float_reference_bits=(639+637)*64,
        fractional_real_variable_products=0,
        fractional_real_adds_per_eight_trials=costs,
        fractional_real_adds=sum(costs)*symbols*peaks,
        interpolation_source_word_reads=17*symbols*peaks,
        interpolation_second_word_reads=16*symbols*peaks,
        interpolation_left_shift_max=10, interpolation_final_right_shift=10,
        equal_width_csd_accumulator_growth_bits=11,
        float_owned_sample_and_work_bits=2*window*128,
        held_last_second_word=True,
        added_work_word_bits_scenario=64, work_format_verified=False,
        coefficient_bits_scenario=18, coefficient_format_verified=False,
        source_range_scan_cycles=window,
        forward_cycles_scenario=2*window+2,
        backward_cycles_scenario=2*window+2,
        feedback_latency_cycles_assumption=2,
        whole_rx_accepted=False, synthesis_evidence=False)


def preparation_calendar(window=256):
    """Finite address calendar: two Y reads, one Q read/write, two-cycle recurrence.

    Each event is (phase, source addresses, Q read, Q write). Input is already
    owned; the upstream copy is charged separately. This models ports/lifetime,
    not fixed arithmetic, routing or a proof of the assumed feedback latency.
    Y is immutable. Q is first forward state, then overwritten by second
    derivatives. A terminal M is held until the final write. No score or other
    window may use these ports while preparation owns them.
    """
    if type(window) is not int or not 4 <= window <= 640:
        raise ValueError("Need a finite4..640-sample preparation window")
    n = window
    for i in range(n):
        yield "range", (i,), None, None
    yield "forward_setup", (), None, None
    yield "forward_setup", (), None, None
    for i in range(n):
        y = (0, 1) if i == 0 else (n-3, n-2) if i == n-1 else (i-1, i+1)
        yield "forward_read", y, i-1 if i else None, None
        y = (2,) if i == 0 else (n-1,) if i == n-1 else ()
        yield "forward_write", y, None, i
    yield "backward_setup", (), n-1, None
    yield "backward_setup", (), None, None
    for i in range(n-2, -1, -1):
        yield "backward_read", (i, i+1), i, None
        yield "backward_write", (), None, i
    yield "last_second_held", (), None, None
    yield "last_second_write", (), None, n-1
