"""Finite power/TED overlap (§6.2.1.2 structure);not a default or full-RX claim."""

from nearlink_sdr.phy.timing_word import ted_operands
from nearlink_sdr.sim.hold_timing_flow import PipelinedPacketTimingService
from nearlink_sdr.sim.timing_word_flow import TimingWordService, two_real_products

__all__ = ["ParallelTEDTimingWordService", "PowerTEDPacketTimingService"]


class ParallelTEDTimingWordService(TimingWordService):
    """One bank;TED prep while energy multiplies,power feedback while TED multiplies."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ted_pending_age = self.background_power_age = None
        self.prepared_ted_terms = None
        self.background_power_done = False

    def issue(self, current, midpoint):
        if not super().issue(current, midpoint):
            return False
        self.ted_pending_age = 0 if midpoint is not None else None
        self.background_power_age = None
        self.prepared_ted_terms = None
        self.background_power_done = False
        return True

    def _products(self):
        role = self.role
        super()._products()
        if self.job["midpoint"] is None:
            return
        if role == "power" and self.mode == "power":
            assert self.prepared_ted_terms is not None
            self.background_power_age = 0
            self._start_products(self.prepared_ted_terms, "ted")
            self.prepared_ted_terms = None
        elif role == "ted" and self.mode == "ratio_prepare":
            self.mode = "wait_power"

    def step(self):
        if self.cycle == 2**64 - 1:
            raise OverflowError("Timing service clock exhausted")
        # Snapshot ownership at entry: a newly launched pipeline cannot also
        # traverse its first registered stage at this same edge.
        age = self.ted_pending_age
        power_age = self.background_power_age
        if age is not None:
            if age == 2:
                dr, di, mr, mi, e, lost = ted_operands(
                    self.state.previous_word, self.job["current"], self.job["midpoint"]
                )
                self.job.update(ted_exponent=e, ted_erased=lost)
                self.prepared_ted_terms = two_real_products(dr, mr, di, mi)
                self.ted_pending_age = None
            else:
                self.ted_pending_age = age + 1
        if power_age is not None:
            # Same six registered power edges;temporary dispatch doesn't alter
            # the independently owned DSP products or ratio state.
            saved_mode, saved_age = self.mode, self.age
            self.mode, self.age = "power", power_age
            super()._power()
            done = self.mode != "power"
            self.background_power_age = None if done else self.age
            self.background_power_done |= done
            self.mode, self.age = saved_mode, saved_age
        wait_at_entry = self.mode == "wait_power"
        super().step()
        if wait_at_entry and self.background_power_done:
            self.mode, self.age = "ratio_prepare", 0

    def abort(self):
        super().abort()
        self.ted_pending_age = self.background_power_age = None
        self.prepared_ted_terms = None

    def estimate(self):
        result = super().estimate()
        result["state_register_bits_upper"] += 104
        result.update(
            overlap_increment_register_bits_upper=104,
            overlap_increment_parts=dict(
                TED_operands=94,
                TED_preparation_age_valid=3,
                power_age_valid=4,
                power_done=1,
                operand_valid=1,
                reserved_control=1,
            ),
            overlap_additional_DSP=0,
            overlap_additional_RAM_bits=0,
            overlap_additional_source_ports=0,
            overlap_live_arithmetic=dict(
                DSP25x18=4,
                DSP_product_pair_adders80=2,
                DSP_product_final_adders80=1,
                power_add_or_subtract45=1,
                power_normalizers44=2,
                power_aligners44=2,
                TED_peak_align_difference_macro_edges=3,
                extra_logic_LUT=None,
            ),
            overlap_power_adder_must_not_be_credited_as_shared_with_product_tree=True,
            overlap_shift_normalize_stage_timing_not_synthesized=True,
            overlap_added_FSM_modes=1,
            overlap_existing_mode_width4_suffices=True,
            overlap_ted_operands_before_DSP_ownership=True,
            overlap_ratio_waits_for_registered_power_and_TED=True,
            complete_latency_cycles=None,
            full_receiver16MSps=False,
            synthesized=False,
        )
        return result


class PowerTEDPacketTimingService(PipelinedPacketTimingService):
    """Original hold pipeline,then exactly the compiled overlapping data service."""

    normal_service_type = ParallelTEDTimingWordService

    def estimate(self):
        return super().estimate() | dict(
            prepared_TED_encoding=dict(
                stored_components_signed_bits=[25, 25, 22, 22],
                logical_bits=94,
                expanded_four_DSP_terms_are_wire_fanout=True,
                duplicate_a_and_constant_shifts_are_not_independent_registers=True,
                no_arbitrary_external_prepared_terms_port=True,
                encoding_bijection_checked_on_reachable_node_states=True,
                physical_FF_duplication_or_timing_not_verified=True,
            )
        )
