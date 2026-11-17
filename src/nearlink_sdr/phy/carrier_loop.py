"""Experimental causal PSK carrier PI (§6.2.1.2), not a fixed/RTL receiver.

Input must already use the actual known-sync gain/phase/frequency correction.
No payload, CRC, known packet position or pilot contents enter the detector.
"""

import math
import operator
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights

__all__ = ["CarrierLoopConfig", "PSKCarrierLoop", "track_psk_carrier"]

METHOD = "causal bounded phase-detector PI;sync-only initialization;no expected bits"


@dataclass(frozen=True)
class CarrierLoopConfig:
    """Two predeclared float study bandwidths; not an arbitrary tuning interface."""

    bandwidth: str = "slow"

    def __post_init__(self):
        if self.bandwidth not in ("slow", "medium"):
            raise ValueError("Need explicit slow/medium experimental bandwidth")

    @property
    def gains(self):
        return (1 / 64, 1 / 8192) if self.bandwidth == "slow" else (1 / 32, 1 / 2048)


def _wrap(value):
    return (value + math.pi) % (2 * math.pi) - math.pi


def _samples(symbols):
    y = np.asarray(symbols, dtype=complex)
    if (
        y.ndim != 1
        or not np.isfinite(y).all()
        or max(np.max(abs(y.real), initial=0), np.max(abs(y.imag), initial=0)) > 1e6
    ):
        raise ValueError("Need finite normalized complex symbols in the experimental range")
    return y


class PSKCarrierLoop:
    """Finite per-packet PI state, independent of borrowed caller chunk boundaries.

    Correct symbol k with the previous NCO phase. Only then detect its wrapped
    phase error and commit frequency/phase for k+1. Fixed causal power protection
    suppresses updates on erasures/high-energy outliers; it cannot repair samples.
    Detector clamps, NCO wraps and frequency clamps are observable, not slip truth.
    """

    def __init__(self, modulation, start_symbol_index, *, config=None):
        if modulation not in ("BPSK", "QPSK", "8PSK"):
            raise ValueError("Need alternating BPSK/QPSK/8PSK")
        if isinstance(start_symbol_index, (bool, np.bool_)):
            raise ValueError("Need integer17bit field start")
        try:
            self.start = operator.index(start_symbol_index)
        except TypeError as error:
            raise ValueError("Need integer17bit field start") from error
        if not 0 <= self.start < 2**17:
            raise ValueError("Field start exceeds17bits")
        self.config = CarrierLoopConfig() if config is None else config
        if not isinstance(self.config, CarrierLoopConfig):
            raise ValueError("Need explicit CarrierLoopConfig")
        self.modulation, self.m = modulation, {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation]
        self.kp, self.ki = self.config.gains
        self.phase = self.frequency = 0.0
        self.used, self.closed = 0, False
        self.power = BoundedCarrierWeights(4)
        self.stats = dict(
            low_power_holds=0, high_power_holds=0, error_clamps=0, frequency_clamps=0, phase_wraps=0
        )

    def switch_modulation(self, modulation):
        """Carry phase/power across an actual field boundary; no symbol is consumed."""
        if self.closed:
            raise RuntimeError("Cannot reconfigure a closed carrier packet")
        if modulation not in ("BPSK", "QPSK", "8PSK"):
            raise ValueError("Need actual PSK field modulation")
        self.modulation, self.m = modulation, {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation]

    def feed(self, symbols):
        """Return at most32 corrected symbols and caller-owned diagnostic trajectories."""
        if self.closed:
            raise RuntimeError("Cannot feed a closed carrier packet")
        y = _samples(symbols)
        if not 1 <= len(y) <= 32:
            raise ValueError("Need1..32 symbols per bounded feed")
        if self.used + len(y) > 2**17:
            raise OverflowError("Carrier packet exceeds131072symbols")
        out = np.empty(len(y), complex)
        phases, frequencies, errors, holds = [], [], [], []
        for i, value in enumerate(y):
            phase, frequency = self.phase, self.frequency
            corrected = value * complex(math.cos(phase), -math.sin(phase))
            # Add the known alternating rotation in angle coordinates; no
            # additional complex multiplier or trigonometric evaluation.
            angle = math.atan2(corrected.imag, corrected.real)
            if (self.start + self.used) & 1:
                angle += math.pi / self.m
            power, previous = float(value.real**2 + value.imag**2), self.power.power
            low, high = power < previous / 16, power > 4 * previous
            self.power.advance_power([power])
            error = 0.0
            if not (low or high):
                error = angle % (2 * math.pi / self.m) - math.pi / self.m
                limit = math.pi / (2 * self.m)
                self.stats["error_clamps"] += abs(error) > limit
                error = min(limit, max(-limit, error))
            self.stats["low_power_holds"] += int(low)
            self.stats["high_power_holds"] += int(high)
            proposed = frequency + self.ki * error
            bound = 2 * math.pi * 0.1 / self.m
            self.frequency = min(bound, max(-bound, proposed))
            self.stats["frequency_clamps"] += abs(proposed) > bound
            following = phase + self.frequency + self.kp * error
            self.phase = _wrap(following)
            self.stats["phase_wraps"] += not -math.pi <= following < math.pi
            out[i] = corrected
            phases.append(phase)
            frequencies.append(frequency)
            errors.append(error)
            holds.append(low or high)
            self.used += 1
        return dict(
            symbols=out,
            phase=np.asarray(phases),
            frequency=np.asarray(frequencies),
            detector_error=np.asarray(errors),
            held=np.asarray(holds, dtype=bool),
        )

    def finish(self):
        """Close state without advancing fabricated EOF symbols or repeating estimates."""
        self.closed = True
        return dict(
            status="closed_symbols_only" if self.used else "no_symbols",
            method=METHOD,
            bandwidth=self.config.bandwidth,
            symbols=self.used,
            start_symbol_index=self.start,
            total_processed_symbols=self.used,
            header_symbols=0,
            header_modulation=None,
            final_phase_rad=self.phase,
            final_frequency_rad_per_symbol=self.frequency,
            stats=dict(self.stats),
            power=dict(self.power.stats),
            initial_phase_rad=0.0,
            initial_frequency=0.0,
            phase_wraps_are_not_cycle_slips=True,
            fixed_datapath_verified=False,
            full_receiver_stream_verified=False,
        )


def track_psk_carrier(
    symbols,
    mod_type,
    start_symbol_index,
    *,
    bandwidth="slow",
    header_symbols=None,
    header_mod_type=None,
):
    """Batch adapter; optional actual header pre-roll retains the same causal state.

    Header modulation and length are actual profile context, never desired bits.
    Caller owns the header/data/output/trace vectors. No full-receiver streaming
    or rate claim follows from this adapter or optional pre-roll.
    """
    y = _samples(symbols)
    if not 1 <= len(y) <= 2**17:
        raise ValueError("Need1..131072 synchronized payload wire symbols")
    # Validate the payload context even when a different header is primed.
    payload_node = PSKCarrierLoop(mod_type, start_symbol_index, config=CarrierLoopConfig(bandwidth))
    warm = np.empty(0, complex) if header_symbols is None else _samples(header_symbols)
    if header_symbols is None and header_mod_type is not None:
        raise ValueError("Header modulation requires actual header symbols")
    if len(warm) > 512 or len(y) + len(warm) > 2**17 or len(warm) > payload_node.start:
        raise ValueError("Actual header pre-roll exceeds bounded field context")
    if header_symbols is not None and not len(warm):
        raise ValueError("Need a nonempty actual header pre-roll")
    loop = payload_node
    if len(warm):
        loop = PSKCarrierLoop(
            header_mod_type, payload_node.start - len(warm), config=payload_node.config
        )
        for k in range(0, len(warm), 32):
            loop.feed(warm[k : k + 32])
        loop.switch_modulation(mod_type)
    parts = [loop.feed(y[k : k + 32]) for k in range(0, len(y), 32)]
    trace = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    info = loop.finish()
    info.update(
        symbols=len(y),
        start_symbol_index=payload_node.start,
        total_processed_symbols=len(y) + len(warm),
        header_symbols=len(warm),
        header_modulation=header_mod_type,
    )
    return trace.pop("symbols"), info, trace
