"""Shared PSK field positions; alternating phase counts every transmitted pilot (§6.2.1.2)."""
import numpy as np

from nearlink_sdr.phy.pilot import pilot_symbol
from nearlink_sdr.phy.psk import PSKDemodulator, PSKModulator

__all__ = ["demap_psk_field", "map_psk_field", "psk_field_layout", "soft_demap_psk_field"]


def soft_demap_psk_field(wire: np.ndarray, kind: str, start: int, noise_variance: float,
                         interval: int = 0, *, method: str = "maxlog") -> np.ndarray:
    """Preserve absolute rotation through pilots before dropping their LLRs."""
    demod = PSKDemodulator(kind, sps=1)
    llr = demod.soft_demap_symbols(wire, noise_variance, start, method=method).reshape(
        -1, demod._bits_per_symbol)
    if interval:
        llr = llr[(np.arange(len(wire)) + 1) % (interval + 1) != 0]
    return llr.reshape(-1)


def psk_field_layout(n_bits: int, bits_per_symbol: int, pilot_interval: int,
                     omit_last: bool = True) -> tuple[np.ndarray, int]:
    """Return data-symbol wire positions and total wire length (including bit padding)."""
    if (not isinstance(n_bits, int) or isinstance(n_bits, bool) or n_bits < 0
            or bits_per_symbol not in (1, 2, 3) or pilot_interval not in (0, 1, 4, 8, 16)):
        raise ValueError("Need nonnegative bit count, PSK bit width and legal pilot interval")
    ndata = -(-n_bits // bits_per_symbol)
    positions = np.arange(ndata)
    if not pilot_interval or not ndata:
        return positions, ndata
    positions += positions // pilot_interval
    npilots = (ndata - int(omit_last)) // pilot_interval
    return positions, ndata + npilots


def map_psk_field(bits: np.ndarray, kind: str, start: int, interval: int = 0,
                  omit_last: bool = True) -> np.ndarray:
    """Map with absolute wire indices, preserving phase through field/pilot boundaries."""
    bits = np.asarray(bits)
    if bits.ndim != 1 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("Expected binary field bits")
    mod = PSKModulator(kind, sps=1)
    positions, size = psk_field_layout(len(bits), mod.bits_per_symbol, interval, omit_last)
    values = mod.map_symbols(bits)
    values[1::2] *= np.conj(mod._rotation)
    values[(start + positions) % 2 == 1] *= mod._rotation
    wire = np.empty(size, dtype=complex)
    wire[positions] = values
    pilot_mask = np.ones(size, dtype=bool)
    pilot_mask[positions] = False
    pilots = np.flatnonzero(pilot_mask)
    wire[pilots] = [pilot_symbol(kind, start + int(i)) for i in pilots]
    return wire


def demap_psk_field(wire: np.ndarray, kind: str, start: int,
                    interval: int = 0) -> np.ndarray:
    """Demap wire symbols before discarding pilot positions; return padded data bits."""
    mod = PSKDemodulator(kind, sps=1)
    bits = mod.demap_symbols(wire, start).reshape(-1, mod._bits_per_symbol)
    if interval:
        bits = bits[(np.arange(len(wire)) + 1) % (interval + 1) != 0]
    return bits.reshape(-1)
