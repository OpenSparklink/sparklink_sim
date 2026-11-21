"""Finite banked scaled10 words and exact functional reference (§6.2.3).

The clocked owner lives in ``sim.banked_scaled_core``. This module preserves
the existing C/E,gate and Q12 semantics;it does not select a different sync.
"""

import math
import operator
import sys
from dataclasses import dataclass

import numpy as np

from nearlink_sdr.phy.acquisition_bank import CorrelationPoint
from nearlink_sdr.phy.acquisition_maximum import ExponentWindow
from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, ScaledDifferentialBank
from nearlink_sdr.phy.coarse_rank import RankedCoarseBank, normalize_squared

__all__ = [
    "BankedDifferenceHistory",
    "BankedRankedReference",
    "BankedScaledReference",
    "DifferenceWord",
    "aligned_piece",
    "input_word",
    "split_alignment",
]


def _integer(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a typed integer word")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("Need integer words") from error


def input_word(real, imag, exponent):
    """Host format assertion;no quantization,headroom conversion or silent wrap."""
    r, i, e = map(_integer, (real, imag, exponent))
    if not (-512 <= r < 512 and -512 <= i < 512 and -48 <= e <= 32):
        raise ValueError("Need signed10 IQ and exponent[-48,32]")
    return r, i, e


@dataclass(frozen=True)
class DifferenceWord:
    """93 typed data bits;tag/address validity is owned by the history controller.

    Constructor checks describe format assertions,not additional squarers.
    The producer computes power from IQ. No algebraic recomputation is hidden
    in this typed history port.
    """

    real: int
    imag: int
    power: int
    exponent: int

    def __post_init__(self):
        values = [self.real, self.imag, self.power, self.exponent]
        if any(type(v) is not int for v in values):
            raise ValueError("Need explicit integer difference fields")
        if not (
            -(1 << 20) <= self.real < 1 << 20
            and -(1 << 20) <= self.imag < 1 << 20
            and 0 <= self.power < 1 << 43
            and -96 <= self.exponent <= 64
        ):
            raise ValueError("Difference exceeds signed21/unsigned43/exponent8 format")


_ZERO = DifferenceWord(0, 0, 0, 0)


def split_alignment(value, shift):
    """Arithmetic floor and tie/even correction for |word|<2**59.

    Large shifts have fixed zero/sign behavior. No variable-sized mask or
    modulo machine shift is implied. Clocked lanes register this result.
    """
    if type(value) is not int or type(shift) is not int or not 0 <= shift <= 320:
        raise ValueError("Need an integer word and compiled0..320 shift")
    if not -(1 << 59) < value < 1 << 59:
        raise ValueError("Alignment magnitude exceeds59 bits")
    if shift >= 60:
        return (-1, 1) if value < 0 else (0, 0)
    if shift == 0:
        return value, 0
    floor = value >> shift
    remainder = value - (floor << shift)
    half = 1 << (shift - 1)
    return floor, int(remainder > half or (remainder == half and floor & 1))


def aligned_piece(word, common, cr, ci):
    """One term's floors/small corrections;sign coefficients are wiring."""
    if not isinstance(word, DifferenceWord) or type(common) is not int:
        raise ValueError("Need a typed difference and common exponent")
    if not -96 <= common <= 64 or cr not in (-1, 0, 1) or ci not in (-1, 0, 1):
        raise ValueError("Invalid compiled exponent or sign/swap coefficient")
    shift = max(common - word.exponent, 0)
    r, br = split_alignment(word.real, shift)
    i, bi = split_alignment(word.imag, shift)
    p, bp = split_alignment(word.power, 2 * shift)
    return (
        r * cr + i * ci,
        i * cr - r * ci,
        p,
        br * cr + bi * ci,
        bi * cr - br * ci,
        bp,
        int(word.power > 0 and p == 0 and not bp),
    )


class BankedDifferenceHistory:
    """One93-bit read and one write per bank;physical pitch is a power of two.

    Padding never enters the logical k-term window. Unwritten startup addresses
    are bypassed by tag validity,not by a free RAM clear. There are no per-word
    tag registers or second exponent/power histories.
    """

    def __init__(self, terms, lanes):
        if type(terms) is not int or terms not in (31, 61, 125) or lanes not in (8, 16):
            raise ValueError("Need compiled31/61/125 terms and8/16 lanes")
        self.terms, self.lanes = terms, lanes
        self.pitch = 1 << terms.bit_length()
        self.rows = 4 * self.pitch // lanes
        self.memory = [[None] * self.rows for _ in range(lanes)]
        self.next_tag = 0

    def address(self, tag):
        if type(tag) is not int or not 0 <= tag < 2**63:
            raise ValueError("Need a nonnegative63-bit history tag")
        slot = (tag // 4) & (self.pitch - 1)
        return slot % self.lanes, (tag % 4) * (self.pitch // self.lanes) + slot // self.lanes

    def append(self, tag, word):
        if tag != self.next_tag or self.next_tag >= 2**63 - 5:
            raise ValueError("Need contiguous history tags with cache headroom")
        if not isinstance(word, DifferenceWord):
            raise ValueError("Need a typed93-bit difference word")
        bank, row = self.address(tag)
        self.memory[bank][row] = word
        self.next_tag += 1
        return bank, row

    def read_batch(self, tag, first, count):
        if (
            type(tag) is not int
            or not 0 <= tag < self.next_tag
            or type(first) is not int
            or type(count) is not int
            or not 0 <= first < self.terms
            or not 1 <= count <= self.lanes
            or first + count > self.terms
        ):
            raise ValueError("Invalid finite fold request or future history tag")
        latest = self.next_tag - 1 - ((self.next_tag - 1 - tag) % 4)
        earliest = latest - 4 * (self.pitch - 1)
        words, addresses = [], []
        used = set()
        for term in range(first, first + count):
            source = tag - 4 * (self.terms - 1 - term)
            if source < 0:
                words.append(_ZERO)
                addresses.append(None)
                continue
            if source < earliest:
                raise ValueError("Fold history expired before registered read")
            bank, row = self.address(source)
            if bank in used:
                raise RuntimeError("Compiled pitch produced a bank read conflict")
            used.add(bank)
            word = self.memory[bank][row]
            if word is None:
                raise RuntimeError("A valid positive history address was never written")
            words.append(word)
            addresses.append((bank, row))
        return words, addresses

    def estimate(self):
        depth = self.rows
        data_width = 6 if depth <= 32 else 3
        primitive = "RAM32M" if depth <= 32 else "RAM64M"
        count = self.lanes * math.ceil(93 / data_width)
        return dict(
            terms=self.terms,
            lanes=self.lanes,
            phase_pitch=self.pitch,
            bank_depth=depth,
            history_word_bits=93,
            logical_data_bits=4 * self.pitch * 93,
            padding_data_bits=4 * (self.pitch - self.terms) * 93,
            tag_controller_bits=63,
            per_bank_read_ports=1,
            per_bank_write_ports=1,
            read_latency_edges=1,
            runtime_RAM_clear_words=0,
            startup_validity="negative logical tags bypass zero;positive unwritten/expired reject",
            LUTRAM_primitive=primitive,
            LUTRAM_primitive_count=count,
            LUTRAM_LUT_estimate=4 * count,
            SLICEM_slice_estimate=count,
            allocated_LUTRAM_bit_capacity=256 * count,
            unused_LUTRAM_bit_capacity=256 * count - 4 * self.pitch * 93,
            LUTRAM_mapping="UG474 RAM32M32x6SDP or RAM64M64x3SDP,4 LUT per primitive",
            alternate_BRAM36_sites_reserved=2 * self.lanes,
            alternate_BRAM_mapping="twoSDP72-wide blocks/bank;no half-site sharing credit",
            actual_synthesis=False,
            Fmax=None,
        )


class BankedScaledReference(ScaledDifferentialBank):
    """Instantaneous functional reference,not a clock/ADC backpressure adapter.

    This explicitly selected experimental class keeps the existing query/tie
    rules. The cycle model must separately prove FIFO/ports and arrival timing.
    """

    _rank_enabled = False

    def __init__(self, reference, sps, window_samples, *args, lanes=16, **kwargs):
        self.banked_history = None
        fmt = kwargs.get("scaled_format")
        if (
            sps != 4
            or {32: 256, 62: 376, 126: 632}.get(len(reference)) != window_samples
            or kwargs.get("rank_fractional_bits", 12) != 12
            or (
                fmt is not None
                and (not isinstance(fmt, ScaledBankFormat) or fmt.mantissa_bits != 10)
            )
        ):
            raise ValueError("Need compiledSPS4/scaled10/Q12 target geometry")
        super().__init__(reference, sps, window_samples, *args, **kwargs)
        self.banked_history = BankedDifferenceHistory(self.k, lanes)
        self.maximum = ExponentWindow(self.k)
        self.iq_history = [(0, 0, 0)] * 4
        self.outgoing = [None] * 4
        self._banked_failed = False
        for name in (
            "_sample_real",
            "_sample_imag",
            "_sample_exponent",
            "_difference_real",
            "_difference_imag",
            "_difference_exponent",
            "_powers",
        ):
            delattr(self, name)
        if self.logical_storage_bits > 16 * 1024 * 8:
            raise ValueError("Banked functional state exceeds inherited16KiB guard")

    @property
    def logical_storage_bits(self):
        original = super().logical_storage_bits
        if self.banked_history is None:
            return original
        # Original difference93-bit history is replaced,not counted twice.
        return (
            original
            - 4 * self.k * 93
            + 4 * self.banked_history.pitch * 93
            + self.maximum.estimate()["functional_state_bits"]
            + 4 * 73
            + 1
        )

    @property
    def emulator_state_bytes(self):
        """Nested owned Python state;neither packed FPGA bits nor process RSS."""
        seen = set()

        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, dict):
                total += sum(size(k) + size(v) for k, v in value.items())
            elif isinstance(value, (list, tuple)):
                total += sum(size(v) for v in value)
            elif isinstance(value, np.ndarray):
                if value.dtype == object:
                    total += sum(size(v) for v in value.flat)
            elif hasattr(value, "__dict__"):
                total += size(vars(value))
            return total

        return size(self)

    def worst_operations_per_sample(self):
        result = super().worst_operations_per_sample()
        result["compare"] -= self.k - 1
        result["other"] = dict(result["other"], exponent_max_terms=0)
        result["banked_history"] = self.banked_history.estimate()
        result["incremental_maximum"] = self.maximum.estimate()
        result["outgoing_cache_bits"] = 4 * 73
        result["scope"] = "functional exact bank;clocked owner separately bills all pipelines/ports"
        return result

    def advance_words(self, real, imag, exponents, *, inspect=False):
        if self._banked_failed:
            raise ValueError("Need a new banked epoch after an invalid partial update")
        arrays = [np.asarray(a) for a in (real, imag, exponents)]
        if (
            any(a.ndim != 1 for a in arrays)
            or not 1 <= len(arrays[0]) == len(arrays[1]) == len(arrays[2]) <= 32
        ):
            raise ValueError("Need equally sized1..32 word arrays")
        words = [input_word(*values) for values in zip(*arrays, strict=True)]
        if self.samples + len(words) + abs(self.origin) >= 2**63 - 5:
            raise OverflowError("Insufficient63-bit sample/cache headroom")
        points = []
        try:
            for r, i, exponent in words:
                n, phase = self.samples, self.samples % 4
                pr, pi, pe = self.iq_history[phase]
                dr, di = r * pr + i * pi, i * pr - r * pi
                word = DifferenceWord(dr, di, dr * dr + di * di, exponent + pe)
                old = self.outgoing[phase]
                if n >= 4 and (old is None or old[0] != n):
                    raise ValueError("Missing same-phase outgoing cache")
                pair = (0, False) if n < 4 else old[1:]
                self.banked_history.append(n, word)
                common = self.maximum.advance(n, pair, (word.exponent, word.power > 0))
                accum = [0] * 7
                for first in range(0, self.k, self.banked_history.lanes):
                    batch, _ = self.banked_history.read_batch(
                        n, first, min(self.banked_history.lanes, self.k - first)
                    )
                    if first == 0:
                        oldest = batch[0]
                        self.outgoing[phase] = (n + 4, oldest.exponent, oldest.power > 0)
                    for j, value in enumerate(batch, first):
                        piece = aligned_piece(
                            value,
                            common,
                            int(self._coefficient_real[j]),
                            int(self._coefficient_imag[j]),
                        )
                        accum = [a + b for a, b in zip(accum, piece, strict=True)]
                cr, ci, energy = accum[0] + accum[3], accum[1] + accum[4], accum[2] + accum[5]
                self._checked(cr, self.widths["correlation"], True)
                self._checked(ci, self.widths["correlation"], True)
                self._checked(energy, self.widths["window_energy"])
                numerator = self._checked(cr * cr + ci * ci, self.widths["numerator"])
                ready = n >= self.k * 4
                self._hits[n % self.window_samples] = (
                    ready
                    and energy > 0
                    and (numerator << 24 >= energy * self.widths["threshold_integer"])
                )
                self.stats["alignment_discarded_terms"] += accum[6]
                self.stats["zero_energy"] += int(ready and energy == 0)
                self.stats["new_correlations"] += int(ready)
                if ready:
                    if self._rank_enabled:
                        self._rank_scores[n % self.rank_history] = normalize_squared(
                            numerator, energy, self.coefficient_energy
                        )
                        self.stats["normalization_words"] += 1
                        self.stats["normalization_zero_energy"] += int(energy == 0)
                        self.stats["normalization_unity_clamps"] += int(
                            energy > 0 and numerator > energy * self.coefficient_energy
                        )
                    if inspect:
                        points.append(
                            CorrelationPoint(
                                self.origin + n - self.k * 4,
                                numerator,
                                energy,
                                self.coefficient_energy,
                            )
                        )
                self.iq_history[phase] = r, i, exponent
                self.samples += 1
                self.stats["samples"] += 1
        except ValueError, OverflowError:
            self._banked_failed = True
            raise
        return points if inspect else None


class BankedRankedReference(BankedScaledReference, RankedCoarseBank):
    """Exact banked functional core with unchanged Q12 query/tie ownership."""

    _rank_enabled = True
