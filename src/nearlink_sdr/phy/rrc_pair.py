"""Pair-local RNE and weighted-product normalization RRC candidate, not promoted.

The input format stays12-bit. Native coefficient products retain exponent tags
until after weighting. This changes quantization placement; it is not an AGC
that improves signal/interference power ratio. New-development budgets apply.
"""

import numpy as np

from nearlink_sdr.phy.acquisition_rounding import split_rne_words
from nearlink_sdr.phy.rrc_word import FixedRRC, FixedRRCFormat

__all__ = ["PairProductRRC"]


def _zero_after_round(floor, increment):
    return ((floor == 0) & ~increment) | ((floor == -1) & increment)


class PairProductRRC(FixedRRC):
    """Keep native paired products, then align to their weighted maximum.

    Paired inputs share the greater of their two nonzero exponents. At least
    one input is unshifted, so RNE corrections are mutually exclusive and fit
    the pre-adder carry input. Product corrections accumulate in small unsigned
    counters, followed by just two wide result corrections. No term saturates.
    """

    def __init__(self, format):
        if not isinstance(format, FixedRRCFormat) or format.input_mantissa_bits != 12:
            raise ValueError("Pair candidate requires explicit12-bit input format")
        super().__init__(format)
        self.product_bits = format.input_mantissa_bits + 1 + format.coefficient_bits
        self.stats["product_components_erased"] = 0

    @property
    def logical_state_bits(self):
        # Persistent functional state only; physical owned-product buffers and
        # pipelines must be charged by the separate cycle structure.
        return super().logical_state_bits + 64

    def _pair_sum(self, left, right, left_shift, right_shift):
        fl, bl = split_rne_words(left, left_shift)
        fr, br = split_rne_words(right, right_shift)
        if np.any(bl & br):
            raise AssertionError("Pair-local max exponent must leave one input unshifted")
        self.stats["aligned_components_erased"] += int(
            np.count_nonzero((left != 0) & _zero_after_round(fl, bl))
            + np.count_nonzero((right != 0) & _zero_after_round(fr, br))
        )
        return fl + fr + (bl | br).astype(np.int64)

    def _product_sum(self, values, shifts):
        # Zero products do not require unbounded shifts. Every nonzero left
        # shift is bounded by normalization into product_bits with headroom.
        s = np.where(values != 0, shifts, 0)
        left = np.maximum(-s, 0)
        if np.any(left >= self.product_bits):
            self.stats["intermediate_overflows"] += 1
            raise OverflowError("Normalized product left shift exceeded proven range")
        floor, increment = split_rne_words(values, np.maximum(s, 0))
        floor = np.where(s < 0, values << left, floor)
        increment = np.where(s < 0, False, increment)
        self.stats["product_components_erased"] += int(
            np.count_nonzero((values != 0) & _zero_after_round(floor, increment))
        )
        # All floors remain full growth; correction bits use a six-bit count.
        return np.sum(floor, axis=1, dtype=np.int64) + np.sum(increment, axis=1, dtype=np.int64)

    def _accumulate(self, wr, wi, we):
        left_e, right_e = we[:, :40], we[:, 80:40:-1]
        left_present = (wr[:, :40] != 0) | (wi[:, :40] != 0)
        right_present = (wr[:, 80:40:-1] != 0) | (wi[:, 80:40:-1] != 0)
        pair_e = np.maximum(
            np.where(left_present, left_e, -48), np.where(right_present, right_e, -48)
        )
        left_shift, right_shift = np.maximum(pair_e - left_e, 0), np.maximum(pair_e - right_e, 0)
        products = []
        for values in (wr, wi):
            pair = self._pair_sum(values[:, :40], values[:, 80:40:-1], left_shift, right_shift)
            products.append(
                np.c_[pair * self.coefficients[:40], values[:, 40] * self.coefficients[40]]
            )
        native_e = np.c_[pair_e, we[:, 40]]
        bits = [
            np.fromiter(
                (abs(int(v)).bit_length() for v in p.ravel()), np.int16, count=p.size
            ).reshape(p.shape)
            for p in products
        ]
        magnitude_bits = np.maximum(*bits)
        common = np.max(
            np.where(magnitude_bits > 0, native_e + magnitude_bits - (self.product_bits - 2), -96),
            axis=1,
        )
        shifts = common[:, None] - native_e
        sums = [self._product_sum(p, shifts) for p in products]
        return sums, common

    def operation_estimate(self):
        result = super().operation_estimate()
        result.update(
            alignment="pair-local inputs, weighted native products; exact RNE correction folding",
            component_alignments=160,
            conditional_input_rne_increment_additions=0,
            pair_preadd_carry_inputs=80,
            pair_exponent_comparisons=40,
            gain_max_comparisons=40,
            product_component_alignments=82,
            guard_sticky_tests=242,
            product_small_rounding_count_additions_upper=80,
            wide_product_rounding_result_corrections=2,
            product_leading_magnitude_extractors=82,
            product_leading_detector_width=self.product_bits,
            product_exponent_component_comparisons=41,
            product_exponent_offset_additions=41,
            product_common_exponent_subtractions=41,
            variable_divisions=0,
            square_roots=0,
            output_leading_zero_detectors=2,
            product_bits=self.product_bits,
            input_rne_carry_in_proof="one member of each exponent-max pair is unshifted",
            scope="operation graph only; new physical buffers, queues, ports and cycle model not yet closed",
        )
        return result
