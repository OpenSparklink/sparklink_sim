"""Conditional prefix-FFT storage/operator estimates; not a numeric FFT cycle model."""

import math

from nearlink_sdr.phy.carrier_prefix import CarrierPrefixConfig
from nearlink_sdr.sim.system_resources import bram36_layout

__all__ = ["carrier_prefix_cost"]


def carrier_prefix_cost(
    prefix_symbols=256, *, frame_symbols=65536, complex_bits=36, clock_hz=200_000_000
):
    """Charge independent memories/ports; no unproved cross-job sharing credit.

    36bit and64bit complex words are sensitivity assumptions, not accepted FFT
     formats. The8-DSP service scenario excludes unimplemented normalization,
     CORDIC/reciprocal and control details. Its FIFO requirement is conditional,
     not a verified worst-case latency or16MS/s claim.
    """
    cfg = CarrierPrefixConfig(prefix_symbols)
    if (
        type(frame_symbols) is not int
        or not 32 <= frame_symbols <= 65536
        or type(complex_bits) is not int
        or complex_bits not in (36, 64)
        or type(clock_hz) is not int
        or clock_hz not in (100_000_000, 160_000_000, 200_000_000)
    ):
        raise ValueError("Need bounded frame,complex36/64 and100/160/200MHz scenario")
    n = min(frame_symbols, cfg.prefix_symbols)
    nfft = 16 * (1 << (n - 1).bit_length())
    stages = nfft.bit_length() - 1
    butterflies = nfft * stages // 2
    band = 1 + 2 * ((nfft - 1) // 10)
    # Both operands use the assumed component width, including twiddles.
    # Rectangular decomposition upper bound includes BOTH DSP25x18 dimensions.
    component = complex_bits // 2
    real_product_tiles = math.ceil(component / 25) * math.ceil(component / 18)
    butterfly_tiles = 4 * real_product_tiles
    butterfly_ii = math.ceil(butterfly_tiles / 8)
    newton_ii = real_product_tiles
    assumed_service = (
        butterflies * butterfly_ii + 7 * stages + 2 * band + 4 + 3 * (n * newton_ii + 128) + 64
    )
    future = math.ceil(assumed_service * 4_000_000 / clock_hz)
    required = min(frame_symbols, n + future + 8)
    fifo = 1 << (required - 1).bit_length()
    bank = bram36_layout(complex_bits, nfft)

    def one_reader_writer(depth):
        tdp = bram36_layout(complex_bits, depth)
        sdp = math.ceil(complex_bits / 72) * math.ceil(depth / 512)
        if sdp < tdp["bram36_tiles"]:
            return dict(
                logical_bits=complex_bits * depth,
                bram36_tiles=sdp,
                mode_width=72,
                mode_depth=512,
                read_ports=1,
                write_ports=1,
                evidence="SDP72x512 allocation estimate,not2read ports",
            )
        return tdp

    raw = one_reader_writer(n)
    pending = one_reader_writer(fifo)
    storage = dict(
        fft_source=bank["bram36_tiles"],
        fft_destination=bank["bram36_tiles"],
        raw_prefix_snapshot=raw["bram36_tiles"],
        weighted_time_prefix=raw["bram36_tiles"],
        correction_fifo=pending["bram36_tiles"],
        fft_twiddle_rom=bram36_layout(complex_bits, nfft // 2)["bram36_tiles"],
        hann_generator_or_coefficients=1,
    )
    total = sum(storage.values())
    return dict(
        prefix_limit=cfg.prefix_symbols,
        fit_symbols=n,
        fft_points=nfft,
        complex_word_bits_assumed=complex_bits,
        fixed_format_accepted=False,
        storage_bram36=storage,
        total_bram36=total,
        proposed_module_study_ceiling=21,
        study_storage_pass=total <= 21,
        same_format_full131072_pingpong_bram36=2
        * bram36_layout(complex_bits, 131072)["bram36_tiles"],
        fft_bank_layout=bank,
        twiddle_layout=bram36_layout(complex_bits, nfft // 2),
        twiddle_entries=nfft // 2,
        twiddle_port_role="independent1R/0W half-circle complex table;no unproved symmetry sharing",
        raw_prefix_layout=raw,
        fifo_layout=pending,
        fft_port_roles="source2R/0W,destination0R/2W;separate TDP buffers swap between stages",
        temporal_port_reuse_is_proposal_not_verified_schedule=True,
        interpolation_or_FFT_phase_memory_shared=False,
        real_product_operand_bits_assumed=[component, component],
        real_product_DSP25x18_tiles_upper=real_product_tiles,
        butterfly_DSP25x18_tiles=butterfly_tiles,
        butterfly_II_scenario=butterfly_ii,
        Newton_real_product_DSP25x18_tiles=real_product_tiles,
        operator_estimate=dict(
            fft_butterflies=butterflies,
            fft_real_products=4 * butterflies,
            fft_real_adds=6 * butterflies,
            mth_power_real_products_upper_8psk=12 * n,
            amplitude_power_real_products=2 * n,
            amplitude_sqrt=n,
            amplitude_reciprocal=n,
            input_complex_hann_real_products=2 * n,
            newton_real_products_upper=3 * (8 * n + 12),
            scalar_newton_divides=3,
            final_atan2=1,
            whole_payload_correction_real_products=4 * frame_symbols,
            whole_payload_correction_real_adds=2 * frame_symbols,
            whole_payload_sincos_pairs=frame_symbols,
            whole_payload_phase_values=frame_symbols,
            band_squared_magnitude_products=2 * band,
            band_comparisons=band - 1,
        ),
        clock_hz_assumed=clock_hz,
        symbol_hz=4_000_000,
        adc_input_hz=16_000_000,
        service_cycles_scenario=assumed_service,
        required_fifo_words_scenario=required,
        allocated_fifo_words_scenario=fifo,
        service_is_verified_bound=False,
        assumptions="8DSP pool,butterflyII1/2 for complex36/64,7cycle stage drain;bandII2;NewtonII1/4,128cycle update,64cycle phase;normalization/control/rounding not closed",
        nonlinear_units_and_control_cost=None,
        guard_and_exponent_and_pipeline_register_cost=None,
        prior_conditional_system_bram36=91,
        partial_system_bram36=91 + total,
        unapproved_reserve_bram36=35,
        partial_with_candidate_reserve=91 + total + 35,
        protects_but_does_not_allocate=[
            "Polar",
            "TX",
            "realtimeMAC/AES",
            "DMA",
            "remaining loops and FIFOs",
        ],
        original16KiB_gate_changed=False,
        synthesis_or_timing_evidence=False,
        complete16msps_verified=False,
    )
