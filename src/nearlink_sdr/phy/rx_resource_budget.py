"""Compositional RX allocation ledger (§6.2.1.2), not a whole-RX schedule.

Keep disjoint state, incompatible formats, unallocated work and port requirements
visible. A passing partial subtotal never promotes the experimental receiver.
DSP bounds apply to the present graphs and stated DSP-only product mapping;
they are not an impossibility proof for other factorizations or LUT products.
"""

import math
from dataclasses import dataclass

from nearlink_sdr.phy.acquisition_flow import AcquisitionFlow, signed_multiplier_tiles
from nearlink_sdr.phy.acquisition_scaled import ScaledDifferentialBank
from nearlink_sdr.phy.capture_capacity import fine_work_floor
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import CausalToneNotch
from nearlink_sdr.phy.tx_pipeline import TxConfig

__all__ = ["RXResourceBudget", "receiver_resource_ledger"]


@dataclass(frozen=True)
class RXResourceBudget:
    """Frozen baseline limits; changing these is a design scenario, not acceptance."""

    clock_hz: int = 200_000_000
    input_hz: int = 16_000_000
    dsp25x18: int = 18
    wide_units: int = 54
    storage_bits: int = 16 * 1024 * 8

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("Need positive integer resource assumptions")
        if self.input_hz > self.clock_hz:
            raise ValueError("This allocation assumes at least one clock per input")


def receiver_resource_ledger(budget=None):
    """Allocate known blocks, bound remaining work and refuse partial acceptance.

    FT2 32-symbol/SPS4/1/8-grid capture is the present workload. FT3/4 longer
    synchronization, tone quantization and payload scheduling remain separate.
    The 46-bit history reservation matches RRC19+19+8 output, but does not prove
    that the floating tone/fine paths meet their budgets in that representation.
    """
    b = RXResourceBudget() if budget is None else budget
    if not isinstance(b, RXResourceBudget):
        raise ValueError("Need RXResourceBudget")
    rrc = PairRRCFlow(PairRRCFlowAssumptions(
        clock_hz=b.clock_hz, input_hz=b.input_hz,
    )).resource_estimate()
    # Use the actual ordinary FT2 template: diagonal vs cardinal differential
    # alphabets have different coefficient energy/threshold widths.
    profile = PSKReceiveProfile.ordinary(TxConfig(symbol_rate_mhz=4))
    bank = ScaledDifferentialBank(profile.sync_symbols(), 4, 256)
    widths = bank.widths
    # Match current four-real-product complex differential graph. Threshold
    # multiplication is a constant shift/add tree, not another variable DSP.
    bank_tiles = (
        4 * signed_multiplier_tiles(10, 10)
        + 2 * signed_multiplier_tiles(widths["difference"], widths["difference"])
        + 2 * signed_multiplier_tiles(widths["correlation"], widths["correlation"])
    )
    # Shared detector counted once, even when both banks are always advanced.
    detector_tiles = 2 * signed_multiplier_tiles(16, 16)
    continuous_tiles = 82 + 2 * bank_tiles + detector_tiles
    fine = fine_work_floor()
    request_rate = b.input_hz / 32
    fine_demand = request_rate * fine["variable_real_products"]
    lower_demand = b.input_hz * continuous_tiles + fine_demand
    # Preserve the concrete eight-unit RRC schedule instead of claiming all
    # unused slots can automatically be reclaimed by an unscheduled consumer.
    reserved = rrc["multiplier_units"]
    remaining_dsp_cycles = max(
        0, (b.dsp25x18 - reserved) * b.clock_hz
        - (2 * bank_tiles + detector_tiles) * b.input_hz,
    )
    storage = dict(
        rrc_owned_products_history_rom_pipeline=rrc["register_tap_logical_bits"],
        two_scaled_banks=2 * bank.logical_storage_bits,
        two_filtered_histories=2 * 384 * 46,
        three_owned_fine_windows=3 * 256 * 46,
        three_fine_descriptors=3 * 256,
    )
    # The old fixed22/one-bank cycle allocation includes its FIFOs/work state;
    # it is an incompatible alternative, not extra state to blindly add here.
    old = AcquisitionFlow().estimate
    excluded = [
        "ADC/input-word conversion and monitoring update schedule",
        "raw-energy detector state/threshold format and pipeline",
        "tone fixed-point state/coefficient fit and schedule",
        "scaled-bank exponent/RNE pipeline, FIFOs and port replication",
        "fine spline coefficients/interpolation/normalization/scratch/pipeline",
        "timing/carrier/demapping/header/payload/Polar buffers and scheduling",
        "delayed selection feedback, shared-port arbitration and sink backpressure",
    ]
    known = sum(storage.values())
    return dict(
        assumptions=vars(b),
        scope="operation/capacity estimates; existing isolated cycle results remain isolated; no synthesis",
        blocks=dict(
            rrc=rrc,
            scaled_bank=dict(
                instances=2, widths=widths, operations=bank.worst_operations_per_sample(),
                signed_primitive_products_per_sample_per_bank=bank_tiles,
                cycle_schedule_verified=False,
            ),
            shared_detector=dict(
                instances=1, input_bits_reservation=16,
                primitive_products_per_sample=detector_tiles,
                format_and_schedule_verified=False,
            ),
            tone=CausalToneNotch().operation_estimate(),
            fine=fine,
            packet_processing=dict(cycle_schedule_verified=False, storage_bits=None),
        ),
        dsp=dict(
            mapping="RRC coefficient products and bank tiled variable products in signed25x18 DSPs; fine one primitive/product is optimistic; tone/packet/spline work excluded",
            continuous_primitive_products_per_sample=continuous_tiles,
            fine_requests_per_second=request_rate,
            fine_products_per_second=fine_demand,
            known_graph_product_lower_demand_per_second=lower_demand,
            available_primitive_products_per_second=b.dsp25x18 * b.clock_hz,
            known_graph_necessary_units=math.ceil(lower_demand / b.clock_hz),
            known_graph_necessary_condition=lower_demand <= b.dsp25x18 * b.clock_hz,
            rrc_reserved_units=reserved,
            remaining_optimistic_fine_products_per_request=remaining_dsp_cycles / request_rate,
            remaining_optimistic_fine_requests_per_second=remaining_dsp_cycles
            / fine["variable_real_products"],
            no_free_reuse="no multiplication credit across different sample values or unscheduled RRC slots",
        ),
        storage=dict(
            components=storage, known_partial_bits=known,
            remaining_before_unallocated_bits=b.storage_bits - known,
            known_partial_within_budget=known <= b.storage_bits,
            width_reservation_verified_for_tone_and_fine=False,
            ranked_score_ring_optional_extra_bits=2 * 3732,
            excluded=excluded,
        ),
        ports=dict(
            rrc=rrc["ports"],
            scaled_bank="four complex correlation reads/lane schedule, exponent/power reads and writes not jointly scheduled",
            fine_snapshot="8 source reads +8 destination writes/clock, plus continuous history writes; bank/replication arbitration not implemented",
            packet="unallocated",
        ),
        wide=dict(
            rrc_reserved=rrc["wide_units_total"],
            remaining_before_other_blocks=b.wide_units - rrc["wide_units_total"],
            others_scheduled=False,
        ),
        incompatible_legacy_cycle_allocation=dict(
            input_bits=22, bank_instances=1, total_bits=old["storage"]["total_bits"],
            compatible_with_selected_scaled10_two_bank_front=False,
            note="do not add or reuse its 15.61KiB/II claim as the new chain schedule",
        ),
        full_chain_cost_complete=False,
        whole_rx_acceptance=False,
        whole_rx_continuous_rate_feasible=None,
        synthesis_evidence=False,
    )
