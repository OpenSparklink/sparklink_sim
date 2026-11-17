"""Explicit rejected prefix-only carrier candidate (§6.2.1.2 implementation).

Fit uses only the configured prefix and the unchanged FFT/Newton estimator.
Constant correction still needs a caller-owned output vector. This module is
not a continuous carrier loop, fixed FFT, or a complete streaming receiver.
"""

import operator
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights
from nearlink_sdr.phy.psk import refine_psk_carrier

__all__ = ["CarrierPrefixConfig", "PrefixCarrierFit", "refine_psk_carrier_prefix"]


@dataclass(frozen=True)
class CarrierPrefixConfig:
    """Two predeclared study lengths, not an unlimited search/bitwidth interface."""

    prefix_symbols: int = 256

    def __post_init__(self):
        if type(self.prefix_symbols) is not int or self.prefix_symbols not in (128, 256):
            raise ValueError("Need an explicit128/256-symbol prefix experiment")


def _context(mod_type, start_symbol_index, kappa, fixed):
    if mod_type not in ("BPSK", "QPSK", "8PSK"):
        raise ValueError("Need an explicit alternating PSK modulation")
    if isinstance(start_symbol_index, (bool, np.bool_)):
        raise ValueError("Need a bounded integer field symbol index")
    try:
        start = operator.index(start_symbol_index)
    except TypeError as error:
        raise ValueError("Need a bounded integer field symbol index") from error
    if not 0 <= start <= 65535:
        raise ValueError("Field symbol index exceeds16bits")
    if kappa is not None:
        BoundedCarrierWeights(kappa, fixed=fixed)
    elif type(fixed) is not bool or fixed:
        raise ValueError("Fixed weights require explicit kappa")
    return start


def _samples(symbols):
    y = np.asarray(symbols, dtype=complex)
    if y.ndim != 1 or not np.isfinite(y).all() or np.max(abs(y), initial=0) > 1e70:
        raise ValueError("Need finite complex symbols in the experimental numeric range")
    return y


def refine_psk_carrier_prefix(
    symbols,
    mod_type,
    start_symbol_index,
    *,
    prefix_symbols=256,
    method="coherent",
    carrier_weight_kappa=None,
    carrier_weight_fixed=False,
):
    """Single fit, no retries: crop the estimator, apply its parameters to all symbols.

    Return metadata describes the actual prefix for faithful BER replay. This
    batch adapter has frame-sized input/output temporaries; only the estimator
    working set is bounded independently of payload length.
    """
    cfg = CarrierPrefixConfig(prefix_symbols)
    start = _context(mod_type, start_symbol_index, carrier_weight_kappa, carrier_weight_fixed)
    if method != "coherent":
        raise ValueError("Prefix experiment keeps the coherent FFT/Newton method")
    y = _samples(symbols)
    if not 32 <= len(y) <= 65536:
        raise ValueError("Need32..65536 synchronized payload wire symbols")
    count = min(len(y), cfg.prefix_symbols)
    corrected, fit = refine_psk_carrier(
        y[:count],
        mod_type,
        start,
        carrier_weight_kappa=carrier_weight_kappa,
        carrier_weight_fixed=carrier_weight_fixed,
    )
    if count != len(y):
        t = np.arange(len(y))
        corrected = y * np.exp(
            -1j
            * (2 * np.pi * fit["residual_cycles_per_symbol"] * t + fit["residual_phase_radians"])
        )
    return corrected, {
        **fit,
        "prefix_symbols": cfg.prefix_symbols,
        "estimation_symbols": count,
        "payload_symbols": len(y),
        "estimator_scope": "prefix only;constant whole-payload correction;no ongoing tracking",
    }


class PrefixCarrierFit:
    """Causal finite prefix collector; emits parameters, not a stream of decoded symbols.

    Expected wire length must come from a validated header. Each call borrows at
    most32 symbols and reports exactly how many it consumed; the caller retains
    the remainder. No future tail is inspected by the fit. Missing prefix data
    yields an explicit truncation, never a zero-frequency success fallback.
    """

    def __init__(
        self,
        total_symbols,
        mod_type,
        start_symbol_index,
        *,
        config=None,
        carrier_weight_kappa=None,
        carrier_weight_fixed=False,
    ):
        if type(total_symbols) is not int or not 32 <= total_symbols <= 65536:
            raise ValueError(
                "Need validated32..65536-symbol field length;short fields use sync only"
            )
        self.config = CarrierPrefixConfig() if config is None else config
        if not isinstance(self.config, CarrierPrefixConfig):
            raise ValueError("Need explicit CarrierPrefixConfig")
        self.start = _context(
            mod_type, start_symbol_index, carrier_weight_kappa, carrier_weight_fixed
        )
        self.mod_type, self.total = mod_type, total_symbols
        self.target = min(total_symbols, self.config.prefix_symbols)
        self.buffer = np.empty(self.target, complex)
        self.used, self.estimate, self.closed = 0, None, False
        self.kappa, self.fixed = carrier_weight_kappa, carrier_weight_fixed

    def feed(self, symbols):
        """Return consumed count and optional ready parameters; no CRC/payload truth input."""
        if self.closed:
            raise RuntimeError("Cannot feed a closed prefix fit")
        y = _samples(symbols)
        if not 1 <= len(y) <= 32:
            raise ValueError("Need1..32 symbols per bounded call")
        count = min(len(y), self.target - self.used)
        self.buffer[self.used : self.used + count] = y[:count]
        self.used += count
        if self.used == self.target and self.estimate is None:
            _, self.estimate = refine_psk_carrier_prefix(
                self.buffer,
                self.mod_type,
                self.start,
                prefix_symbols=self.config.prefix_symbols,
                carrier_weight_kappa=self.kappa,
                carrier_weight_fixed=self.fixed,
            )
            self.estimate = {**self.estimate, "payload_symbols": self.total}
        return dict(
            consumed=count,
            ready=self.estimate is not None,
            estimate=self.estimate.copy() if self.estimate is not None else None,
        )

    def finish(self):
        """Close without invented samples or a second estimation pass."""
        self.closed = True
        return dict(
            status="ready" if self.estimate is not None else "truncated_prefix",
            observed=self.used,
            required=self.target,
            estimate=self.estimate.copy() if self.estimate is not None else None,
            prefix_storage_bytes=self.buffer.nbytes,
            full_receiver_stream_verified=False,
        )
