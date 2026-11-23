"""Compiled full14 RRC word ports and dependency calendar,not RTL or whole RX."""

from nearlink_sdr.phy.rrc_full14 import Full14MantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow
from nearlink_sdr.sim.rrc_memory import RRCMemory

__all__ = ["PairRRC14Flow", "RRC14Memory"]


class PairRRC14Flow(PairRRCFlow):
    """Widen input-side registers only;reuse proven31/37-bit product/SUM schedule."""

    _input_mantissa_bits = 14

    def __init__(self, assumptions=None):
        self._compiled_bounds = Full14MantissaRRC().native_product_bounds
        super().__init__(assumptions)

    def resource_estimate(self):
        return super().resource_estimate() | dict(
            compiled_format="full14-c18-p31-a37-o19", raw_word_port_bits=36,
            native_multiplier_operand_bits=[15, 18], nominal_native_product_bits=33,
            stored_product_bits=31, lossless_sign_extension_slice=True,
            compiled_product_bounds=self._compiled_bounds,
            physical_ADC_to_word_cycles=None, new_synthesis=False,
            whole_RX_latency_cycles=None, full_RX_16MSps=False,
        )


class RRC14Memory(RRCMemory):
    """Same finite banks/ports/ownership;36-bit raw and unchanged69-bit product words."""

    _raw_word_bits = 36

    def estimate(self):
        return super().estimate() | dict(compiled_raw_word_bits=36, product_word_bits=69,
                                        whole_RX_latency_cycles=None, full_RX_16MSps=False)
