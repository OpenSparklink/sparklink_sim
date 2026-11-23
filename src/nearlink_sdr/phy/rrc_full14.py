"""Explicit full14/C18 RRC,retaining proven signed31 products (§6.2.1.2).

The floating adapter describes digital representation,not ADC precision or SIR.
The original FullMantissaRRC and its rejected12-bit boundary remain unchanged.
"""

from nearlink_sdr.phy.rrc_input import _quantize_full_rrc_words
from nearlink_sdr.phy.rrc_pair import PairProductRRC
from nearlink_sdr.phy.rrc_word import FixedRRC, FixedRRCFormat

__all__ = ["Full14MantissaRRC", "quantize_full14_rrc_words"]


def quantize_full14_rrc_words(samples, *, stats):
    """Causal signed14 RNE,exponent[-48,32];re-round original input on overflow."""
    return _quantize_full_rrc_words(samples, stats=stats, bits=14)


class Full14MantissaRRC(PairProductRRC):
    """One compiled format;shared pair/product arithmetic,not a width parameter API.

    Fixed coefficients bound both signed15 preadder×C18 products and center
    products below2**30. The nominal33-bit DSP product therefore narrows to31
    by removing sign extension only. Weighted-product normalization keeps the
    original31-bit precision;37-bit accumulation and19-bit output are unchanged.
    """

    def __init__(self):
        # Keep PairProductRRC's public12-only contract. Initialize its shared
        # fixed-word implementation explicitly for this compiled coefficient set.
        FixedRRC.__init__(self, FixedRRCFormat(14, 18))
        pair = 2**14 * max(abs(int(c)) for c in self.coefficients[:40])
        center = 2**13 * abs(int(self.coefficients[40]))
        if max(pair, center) >= 2**30:
            raise OverflowError("Compiled full14 coefficient set exceeds signed31 product bound")
        self.product_bits, self.accumulator_bits = 31, 37
        self.native_product_bounds = dict(pair_abs_upper=pair, center_abs_upper=center,
                                         signed31_limit=2**30, strict_bound_verified=True)
        self.stats["product_components_erased"] = self.stats["input_renormalizations"] = 0

    def _quantize(self, samples):
        return quantize_full14_rrc_words(samples, stats=self.stats)

    @property
    def logical_state_bits(self):
        return super().logical_state_bits + 64

    def operation_estimate(self):
        return super().operation_estimate() | dict(
            compiled_format="full14-c18-p31-a37-o19", input_mantissa_bits=14,
            raw_word_port_bits=36, preadder_signed_bits=15,
            native_multiplier_operand_bits=[15, 18], nominal_native_product_bits=33,
            stored_product_bits=31, lossless_sign_extension_slice=True,
            compiled_product_bounds=self.native_product_bounds,
            floating_adapter="signed14,RNE original input again after positive overflow",
            input_adapter_roundings_upper=4, input_adapter_component_range_comparisons_upper=8,
            input_adapter_peak_comparisons=1, input_adapter_renormalization_exponent_increment=1,
            input_adapter_state_bits_extra=64, input_adapter_physical_structure_verified=False,
            physical_ADC_to_word_cycles=None, default_RX_changed=False,
        )
