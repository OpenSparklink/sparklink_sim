"""Prospective causal carrier allocation; nonlinear timing/fixed words unverified."""

__all__ = ["carrier_loop_cost"]


def carrier_loop_cost():
    """Count the proposed18-bit components and32-bit phase separately from float.

    No DSP credit for a generic-width multiply: the six proposed18x18 real
    products fit six25x18 tiles. CORDIC, exponent alignment and feedback timing
    are future work; do not substitute Python atan2/sin/cos for hardware proof.
    """
    storage = dict(
        input_fifo=4 * 53,
        output_fifo=4 * 53,
        active_input_and_output=2 * 53,
        phase_frequency_and_error=3 * 32,
        pi_extended_scratch=2 * 34,
        power_state=33,
        power_scratch=3 * 41,
        phase_detector_and_rotation_scratch=6 * 32,
        global_index_and_symbol_count=2 * 18,
        counter_bank=13 * 64,
        flags_and_fifo_control=32,
    )
    return dict(
        evidence="operation and register/FIFO allocation estimate,not synthesis or executable CORDIC schedule",
        floating_per_symbol=dict(
            rotation_real_products=4,
            rotation_real_adds=2,
            power_real_products=2,
            power_real_adds=1,
            atan2=1,
            sincos_pair=1,
            phase_modulo=2,
            PI_constant_shifts=2,
            PI_real_adds=3,
            power_update_shift=1,
            comparisons_upper=40,
            counter_updates_upper=13,
        ),
        proposed_complex_word_bits=36,
        proposed_phase_word_bits=32,
        proposed_real_product_operand_bits=[18, 18],
        proposed_DSP25x18_tiles=6,
        additional_normalization_division=0,
        additional_sqrt=0,
        FFT_points=0,
        runtime_symbol_history_words=0,
        input_fifo_words_proposed=4,
        output_fifo_words_proposed=4,
        storage_bits=storage,
        logical_bits=sum(storage.values()),
        FIFO_ports="each1R/1W;four-entry register proposal,not free BRAM or routing",
        floating_counter_width_bound="packet limit131072 prevents Python counter growth without bound",
        prospective_study_ceiling=dict(DSP25x18=8, BRAM36=2),
        extra_exponent_alignment_and_CORDIC_LUT_FF=None,
        accepted_fixed_formats=False,
        executable_cycle_schedule=False,
        cycles_per_symbol_verified=None,
        clock_scenarios_hz=[100_000_000, 160_000_000, 200_000_000],
        continuous_16msps_verified=False,
        system_resource_gate=False,
        batch_adapter_still_owns_frame_sized_output_and_diagnostics=True,
        existing_rx_frame_and_decode_memory_still_required=True,
        original2percent16KiB_failures_changed=False,
        physical_resource_or_timing_evidence=False,
    )
