"""Incremental common exponent for the existing scaled bank (§6.2.3).

The owner supplies the outgoing pair from its existing difference history.
No second sample/exponent delay line, score approximation or search change is
introduced. Counts, bitmaps and every descriptor are separately charged.
"""

import operator
import sys

import numpy as np

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, ScaledDifferentialBank
from nearlink_sdr.phy.coarse_rank import RankedCoarseBank

__all__ = ["ExponentWindow", "RollingMaximumRankedBank", "RollingMaximumScaledBank"]


def _integer(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Boolean is not an integer descriptor")
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError("Need an integer descriptor") from error


class ExponentWindow:
    """Four phase histograms,161 bins,11 groups; no owned IQ history.

    Difference exponents are[-96,64]. Zero-power words never occupy a bin.
    The empty maximum is-96,including the legitimate nonzero bin at-96.
    All host arrays represent the fixed formats returned by ``estimate``.
    """

    fields = ("events", "entered", "left", "empty", "same_bin", "cancelled")

    def __init__(self, terms):
        if type(terms) is not int or terms not in (31, 61, 125):
            raise ValueError("Need compiled31/61/125-term SPS4 geometry")
        self.terms, self.counter_bits = terms, terms.bit_length()
        self.counts = [[0] * 161 for _ in range(4)]
        self.bitmaps, self.groups, self.totals = [0] * 4, [0] * 4, [0] * 4
        self.next_tag = 0
        self.stats = dict.fromkeys(self.fields, 0)
        self._owner = None

    def request(self, tag, outgoing, incoming):
        tag = _integer(tag)
        if tag != self.next_tag or not 0 <= tag < 2**63 - 1:
            raise ValueError("Need contiguous nonnegative sample tags with clock headroom")
        pairs = []
        for pair in (outgoing, incoming):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("Need(exponent,nonzero_power) outgoing/incoming pairs")
            exponent, nonzero = pair
            exponent = _integer(exponent)
            if not -96 <= exponent <= 64 or type(nonzero) is not bool:
                raise ValueError("Need exponent[-96,64] and explicit Boolean power flag")
            pairs.append((exponent + 96, nonzero))
        return tag, tag % 4, *pairs

    def read(self, request):
        tag, phase, outgoing, incoming = request
        if tag != self.next_tag:
            raise ValueError("Histogram context changed before registered read")
        return (
            request,
            self.counts[phase][outgoing[0]],
            self.counts[phase][incoming[0]],
            self.bitmaps[phase],
            self.groups[phase],
            self.totals[phase],
        )

    def prepare(self, read):
        request, old_count, new_count, bitmap, groups, total = read
        tag, _phase, (old, leave), (new, enter) = request
        if tag != self.next_tag:
            raise ValueError("Histogram context changed before update")
        if leave and old_count == 0:
            raise ValueError("Outgoing nonzero exponent is absent from borrowed history")
        changes = {}
        if leave:
            changes[old] = old_count - 1
        if enter:
            changes[new] = changes.get(new, new_count) + 1
        if any(not 0 <= value <= self.terms for value in changes.values()):
            raise OverflowError("Exponent bin exceeds compiled counter range")
        total += int(enter) - int(leave)
        if not 0 <= total <= self.terms:
            raise OverflowError("Exponent membership exceeds the compiled window")
        for key, count in changes.items():
            bitmap = bitmap | (1 << key) if count else bitmap & ~(1 << key)
        for group in {key // 16 for key in changes}:
            present = bool((bitmap >> (16 * group)) & 0xFFFF)
            groups = groups | (1 << group) if present else groups & ~(1 << group)
        increments = (
            1,
            int(enter),
            int(leave),
            int(total == 0),
            int(enter and leave and old == new),
            int(not changes),
        )
        updated = {
            key: self.stats[key] + amount
            for key, amount in zip(self.fields, increments, strict=True)
        }
        if any(not 0 <= value < 2**64 for value in updated.values()):
            raise OverflowError("Unsigned64 exponent statistics exhausted")
        return request, tuple(changes.items()), bitmap, groups, total, updated

    def commit(self, prepared, *, owner=None, write_counts=True):
        if self._owner is not owner:
            raise ValueError("Histogram is owned by another service")
        request, changes, bitmap, groups, total, updated = prepared
        tag, phase, *_ = request
        if tag != self.next_tag:
            raise ValueError("Histogram context changed before commit")
        if write_counts:
            for key, count in changes:
                self.counts[phase][key] = count
        self.bitmaps[phase], self.groups[phase], self.totals[phase] = bitmap, groups, total
        self.stats.update(updated)
        self.next_tag += 1

    def maximum(self, phase):
        if type(phase) is not int or not 0 <= phase < 4:
            raise ValueError("Need phase0..3")
        group = self.groups[phase].bit_length() - 1
        if group < 0:
            return -96
        bits = (self.bitmaps[phase] >> (16 * group)) & 0xFFFF
        return 16 * group + bits.bit_length() - 1 - 96

    def advance(self, tag, outgoing, incoming):
        if self._owner is not None:
            raise ValueError("Clocked service owns this histogram")
        request = self.request(tag, outgoing, incoming)
        prepared = self.prepare(self.read(request))
        self.commit(prepared)
        return self.maximum(request[1])

    def estimate(self):
        bits = self.counter_bits
        counts = 4 * 161 * bits
        bitmap = 4 * (161 + 11)
        state = counts + bitmap + 4 * bits + 63 + 6 * 64
        return dict(
            terms=self.terms,
            phases=4,
            counter_bits=bits,
            bins=161,
            count_state_bits=counts,
            bitmap_state_bits=bitmap,
            functional_state_bits=state,
            extra_IQ_history_bits=0,
            borrowed_history_reads_per_sample=2,
            borrowed_fields_per_history_word=["exponent8", "power_nonzero_from43"],
            history_word_read_ports_minimum=1,
            history_port_schedule_verified=False,
            producer_power_zero_compares43_upper=2,
            register_counter_reads_per_sample_upper=2,
            register_counter_writes_per_sample_upper=2,
            count_add_sub_upper=2,
            membership_add_sub_upper=2,
            count_zero_compare_upper=2,
            count_range_comparisons_upper=6,
            input_guard_widths_upper=[63, 63, 63, 8, 8, 8, 8],
            context_comparisons63_per_event_upper=3,
            statistics_add64_per_event=6,
            tag_add63_per_event=1,
            priority_encoder_widths=[11, 16],
            group_OR16_upper=2,
            bitmap_phase_select_mux2_upper=2 * 3 * (161 + 11 + bits),
            group_slice_mux2_upper=3 * 10 * 16,
            bitmap_onehot_decoders=2,
            real_multiply=0,
            divide=0,
            sqrt=0,
            dynamic_counter_read_mux2_upper=2 * (4 * 161 - 1) * bits,
            write_address_decoders=2,
            write_decode_outputs_upper=2 * 644,
            input_exponent_range=[-96, 64],
            exponent_comparisons_removed_per_sample=self.terms - 1,
            histogram_hardware_mapping=None,
            correlation_and_score_cycles=None,
            full_receiver16MSps=False,
            synthesis_or_Fmax_verified=False,
        )


class RollingMaximumScaledBank(ScaledDifferentialBank):
    """Experimental exact maximum replacement; the remaining score core is unchanged.

    Only scaled10/SPS4 and the current32/62/126-symbol target are compiled.
    No receiver default is replaced. Existing16KiB bank state guard remains;
    that guard does not include the RRC,queues,decoder,TX or whole-system state.
    """

    def __init__(self, reference, sps, window_samples, *args, **kwargs):
        geometry = {32: 256, 62: 376, 126: 632}
        fmt = kwargs.get("scaled_format")
        if (
            np.ndim(reference) != 1
            or sps != 4
            or geometry.get(len(reference)) != window_samples
            or (
                fmt is not None
                and (not isinstance(fmt, ScaledBankFormat) or fmt.mantissa_bits != 10)
            )
        ):
            raise ValueError("Need compiled scaled10/SPS4 target sync and search geometry")
        super().__init__(reference, sps, window_samples, *args, **kwargs)
        self.maximum_state = ExponentWindow(self.k)
        self._maximum_failed = False
        if self.logical_storage_bits > 16 * 1024 * 8:
            raise ValueError("Experimental bank exceeds inherited16KiB bank state guard")

    @property
    def logical_storage_bits(self):
        state = getattr(self, "maximum_state", None)
        return super().logical_storage_bits + (
            state.estimate()["functional_state_bits"] + 1 if state is not None else 0
        )

    def advance_words(self, real, imag, exponents, *, inspect=False):
        if self._maximum_failed or self.maximum_state.next_tag != self.samples:
            raise ValueError("Need a new epoch after an invalid maximum/history context")
        if np.ndim(real) == 1 and any(
            value + len(real) >= 2**64 for value in self.maximum_state.stats.values()
        ):
            raise OverflowError("Maximum statistics lack bounded-batch headroom")
        try:
            return super().advance_words(real, imag, exponents, inspect=inspect)
        except ValueError, OverflowError:
            self._maximum_failed = self.maximum_state.next_tag != self.samples
            raise

    @property
    def emulator_state_bytes(self):
        """Include bounded histogram containers;this is neither process RSS nor FPGA bits."""
        node = getattr(self, "maximum_state", None)
        if node is None:
            return super().emulator_state_bytes
        seen = set()

        def size(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            total = sys.getsizeof(value)
            if isinstance(value, dict):
                total += sum(size(key) + size(item) for key, item in value.items())
            elif isinstance(value, (list, tuple)):
                total += sum(size(item) for item in value)
            return total

        fields = ("counts", "bitmaps", "groups", "totals", "stats", "next_tag", "_owner")
        extra = sys.getsizeof(node) + sys.getsizeof(node.__dict__)
        extra += sum(size(getattr(node, key)) for key in fields)
        return super().emulator_state_bytes + extra

    def _window_maxima(self, exponents, powers):
        result = []
        for j in range(len(exponents)):
            result.append(
                self.maximum_state.advance(
                    self.samples + j,
                    (int(self._difference_exponent[j]), bool(self._powers[j])),
                    (int(exponents[j, -1]), bool(powers[j, -1])),
                )
            )
        return np.asarray(result, np.int16)

    def worst_operations_per_sample(self):
        result = super().worst_operations_per_sample()
        result["compare"] -= self.k - 1
        result["other"] = dict(result["other"], exponent_max_terms=0)
        result["incremental_maximum"] = self.maximum_state.estimate()
        result["scope"] = "exact incremental maximum;fold/score ports and cycles not verified"
        return result


class RollingMaximumRankedBank(RollingMaximumScaledBank, RankedCoarseBank):
    """Same experiment with the unchanged existing Q12 normalized ranking owner."""
