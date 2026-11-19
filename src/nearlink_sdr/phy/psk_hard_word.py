"""Exact hard BPSK/QPSK on the actual55bit normalization word (§6.2.1.2).

The common positive binary exponent cancels from unit-constellation ordering.
Mathematical ties choose the smallest numeric Gray label;floating ulp behavior
is intentionally a separately measured comparison,not a hardware requirement.
No carrier,CRC,payload,AGC or soft reliability is inferred here.
"""

from nearlink_sdr.phy.normalization_word import MAX_SYMBOLS, unpack_normalized_word
from nearlink_sdr.phy.sync_fit_word import signed

__all__ = ["hard_psk_word"]


def _validate(word, modulation, symbol_index):
    real, imag, _ = unpack_normalized_word(word)
    if type(modulation) is not str or modulation not in ("BPSK", "QPSK"):
        raise ValueError("Compiled hard header supports only BPSK/QPSK")
    if type(symbol_index) is not int or not 0 <= symbol_index < MAX_SYMBOLS:
        raise ValueError("Need an actual17bit absolute wire-symbol index")
    return real, imag


def _project(real, imag, modulation, odd):
    if modulation == "BPSK":
        return real if odd else imag, 0
    if odd:
        return signed(real - imag, 23), signed(real + imag, 23)
    return real, imag


def _decision(first, second, modulation):
    if modulation == "BPSK":
        label = int(first < 0)
        return label, (label,), first == 0
    label = (int(second < 0) << 1) | int(first < 0)
    return label, (label >> 1, label & 1), first == 0 or second == 0


def hard_psk_word(word, modulation, symbol_index):
    """No rounding,division,square root,multiply or exponent alignment.

    signed22 inputs need signed23 sums/differences. The explicit tie rule also
    applies to zero energy;it does not make an erasure reliable or validate CRC.
    Rotation uses every wire index,including pilots.
    """
    real, imag = _validate(word, modulation, symbol_index)
    first, second = _project(real, imag, modulation, bool(symbol_index & 1))
    label, bits, tie = _decision(first, second, modulation)
    return dict(
        index=symbol_index,
        label=label,
        bits=bits,
        mathematical_tie=tie,
        zero_word=real == imag == 0,
    )
