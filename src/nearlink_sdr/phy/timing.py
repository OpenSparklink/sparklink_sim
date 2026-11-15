"""Packet-local PSK clock recovery after matched filtering (TXS-10002-2025 §6.2.1.2).

The standard defines the waveform; this NDA Gardner/PI receiver is an implementation
choice, not a mandated algorithm or a statement of clock-compliance limits.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

__all__ = ["GardnerPIState", "PSKTimingConfig", "track_psk_timing"]


@dataclass(frozen=True)
class PSKTimingConfig:
    """Normalized loop bandwidth and experimental integral rate bound.

    Gains assume beta=.4 RRC and power-normalized complex Gardner error. The
    detector gain 1.2 per symbol was measured with the independent TX reference
    for both alternating-rotation QPSK and 8PSK. Bounds are not SLE limits.
    """

    bandwidth: float = .003
    max_rate_ppm: float = 500.0

    def __post_init__(self):
        if (not np.isfinite([self.bandwidth, self.max_rate_ppm]).all()
                or not 0 < self.bandwidth <= .02 or not 0 < self.max_rate_ppm <= 10000):
            raise ValueError("Need finite bandwidth in (0,.02] and rate bound in (0,10000] ppm")


class GardnerPIState:
    """Packet-local power/TED/PI state shared by offline and incremental sampling."""

    def __init__(self, sps: int, power: float, config: PSKTimingConfig):
        self.config, self.sps = config, sps
        self.damping, self.detector_gain = 2 ** -.5, 1.2
        theta = config.bandwidth / (self.damping + .25 / self.damping)
        denominator = 1 + 2 * self.damping * theta + theta * theta
        self.kp = sps * 4 * self.damping * theta / denominator / self.detector_gain
        self.ki = sps * 4 * theta * theta / denominator / self.detector_gain
        self.lower = sps * (1 - config.max_rate_ppm * 1e-6)
        self.upper = sps * (1 + config.max_rate_ppm * 1e-6)
        self.power, self.omega = max(power, 1e-30), float(sps)
        self.rate_limit_events, self.error_clip_events = 0, 0

    def advance(self, current: complex, previous: complex, midpoint: complex | None):
        """Update one symbol; None midpoint holds clock rate during known sync."""
        self.power = max(self.power + (abs(current) ** 2 - self.power) / 64, 1e-30)
        error = 0.0
        if midpoint is not None:
            raw_error = float(((previous - current) * midpoint.conjugate()).real / self.power)
            error = max(-2., min(2., raw_error))
            self.error_clip_events += abs(raw_error) > 2
            proposed = self.omega + self.ki * error
            self.rate_limit_events += proposed < self.lower or proposed > self.upper
            self.omega = max(self.lower, min(self.upper, proposed))
        return self.omega + self.kp * error, error

    def diagnostics(self, hold_symbols, rates):
        return dict(method="NDA complex Gardner + power normalization + second-order PI NCO",
                    bandwidth=self.config.bandwidth, max_rate_ppm=self.config.max_rate_ppm,
                    damping=self.damping, detector_gain=self.detector_gain,
                    kp_samples=self.kp, ki_samples=self.ki, hold_symbols=hold_symbols,
                    rate_limit_events=self.rate_limit_events, error_clip_events=self.error_clip_events,
                    estimated_rate_ppm=float(np.median(rates[max(0, len(rates)-128):])),
                    rate_state_final_ppm=float(rates[-1]), lock_status="not_classified")


def track_psk_timing(filtered: np.ndarray, sps: int, start: float, stop: float, *,
                     hold_symbols: int = 32,
                     config: PSKTimingConfig | None = None) -> dict:
    """Track a single packet without decisions, expected payload, CRC or modulation labels.

    ``start`` is the already selected known-sync center, ``stop`` an exclusive
    original-sample bound. Hold the sync centers fixed, then update the NCO phase
    and rate for every symbol. Error = Re[(previous-current)*conj(midpoint)] / power.
    Positive error advances the next center toward a later sample. Returned rate
    is the PI integrator state; it is not calibrated SRO or an independent truth.

    Cubic coefficients are built once; the feedback loop uses scalar Horner
    evaluation rather than a SciPy call per symbol. Centered matched filtering
    and cubic lookahead remain offline; state resets for every packet.
    """
    if config is None:
        config = PSKTimingConfig()
    z = np.asarray(filtered)
    if (z.ndim != 1 or len(z) < 4 or not np.isfinite(z).all()
            or not isinstance(sps, int) or isinstance(sps, bool) or sps < 2
            or not isinstance(hold_symbols, int) or isinstance(hold_symbols, bool)
            or hold_symbols < 1 or not np.isfinite([start, stop]).all()
            or not 0 <= start < stop <= len(z) - 1):
        raise ValueError("Need finite 1D filtered IQ, integer SPS>=2, positive hold and sample bounds")
    if not isinstance(config, PSKTimingConfig):
        raise ValueError("config must be PSKTimingConfig")
    spline = CubicSpline(np.arange(len(z)), z)
    c0, c1, c2, c3 = spline.c

    def sample(position):
        cell = int(position)
        fraction = position - cell
        return ((c0[cell] * fraction + c1[cell]) * fraction + c2[cell]) * fraction + c3[cell]

    sync_centers = np.arange(start, min(stop, start + hold_symbols * sps), sps)
    power = max(float(np.mean(abs(spline(sync_centers)) ** 2)), 1e-30)
    # A clipped error is in [-2,2], so every update advances by at least this amount.
    loop = GardnerPIState(sps, power, config)
    minimum_step = loop.lower - 2 * loop.kp
    capacity = int(np.ceil((stop - start) / minimum_step)) + 1
    symbols = np.empty(capacity, dtype=complex)
    centers = np.empty(capacity)
    rates = np.empty(capacity)
    errors = np.empty(capacity)
    t, count = float(start), 0
    previous, previous_t = sample(t), t
    while t < stop:
        current = sample(t)
        midpoint = sample((previous_t + t) / 2) if count >= hold_symbols else None
        step, error = loop.advance(current, previous, midpoint)
        symbols[count], centers[count] = current, t
        rates[count], errors[count] = (loop.omega / sps - 1) * 1e6, error
        previous, previous_t = current, t
        t += step
        count += 1
    return dict(symbols=symbols[:count], centers=centers[:count], rate_ppm=rates[:count],
                detector_error=errors[:count], diagnostics=loop.diagnostics(hold_symbols, rates[:count]))
