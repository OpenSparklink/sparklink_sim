"""One bounded head-only prefetch schedule;not another capture or timing algorithm.

Known profile geometry and the existing finite PI step envelope bound a prefix
which every full head needs. No received length,CRC,truth or alternate answer
steers that prefix. Existing1024word1R1W/private and global source ports remain.
"""
from nearlink_sdr.phy.psk_fields import psk_field_layout
from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.phy.timing_word import KP_Q48, LOWER_Q32, rne_shift
from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay

__all__ = ["HeadPrefetchedCaptureReplay"]

MINIMUM_STEP_Q32 = LOWER_Q32 - rne_shift(KP_Q48 * (2 << 32), 48)


def _head_offsets(profile):
    cfg = profile.config
    bits, bps, interval = (64, 2, 16) if cfg.frame_type == 2 else (
        256, 1 if cfg.frame_type == 4 else 2, 4
    )
    positions, _ = psk_field_layout(bits, bps, interval, cfg.pilot_interval == 0)
    end = len(profile.sync_symbols()) + int(positions[-1]) + 1
    offsets = tuple(
        unsigned((((phase << 29) + (end - 1) * MINIMUM_STEP_Q32) >> 32) + 3, 11)
        for phase in range(8)
    )
    return end, offsets


class HeadPrefetchedCaptureReplay(InitializedCapturedFrameReplay):
    """Experimental one-profile schedule,chosen before any CRC;default unchanged.

    At admission a21bit add of a compiled8x11bit offset becomes available after
    one edge. Coherent service must still finish before a copy is admitted.
    Init may then copy only that needed head prefix into its existing private
    ring. Every copy retains the original real source/read/write ownership.
    """
    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self._compiled_head_end, self._head_offsets = _head_offsets(self.profile)
        self.head_floor_end = self.head_floor_available_cycle = 0

    def begin(self, selection, cycle):
        super().begin(selection, cycle)
        if self._word_waiting:
            self.head_floor_end = unsigned(
                selection.start_tick // 8 + self._head_offsets[selection.start_tick % 8], 21
            )
            self.head_floor_available_cycle = unsigned(cycle + 1, 63)

    def _copy_limit(self):
        flow = self.flow
        if flow is None or flow.done or flow.frame is not None:
            return super()._copy_limit()
        if flow.head_owner.end != self._compiled_head_end:
            raise RuntimeError("Compiled head horizon disagrees with the actual head owner")
        if self.cycle < self.head_floor_available_cycle:
            return self.history.end
        return min(self.sample_limit, self.head_floor_end)

    def _terminal(self, event):
        event["bounded_head_prefetch"] = dict(
            needed_head_end_symbol=self._compiled_head_end,
            guaranteed_head_future_floor=self.head_floor_end,
            boundary_available_cycle=self.head_floor_available_cycle,
            compiled_offsets=self._head_offsets,
            minimum_step_q32=MINIMUM_STEP_Q32,
            uses_received_length_or_CRC=False,
        )
        super()._terminal(event)

    def release(self):
        super().release()
        self.head_floor_end = self.head_floor_available_cycle = 0

    def estimate(self):
        base = super().estimate()
        fields = base["new_bridge_control_fields"] | dict(
            head_floor_end=21, head_floor_available_cycle=63
        )
        return base | dict(
            new_bridge_control_fields=fields,
            new_bridge_control_bits=sum(fields.values()),
            known_register_reservation_subtotal=(
                base["known_register_reservation_subtotal"] + 84
            ),
            bounded_head_prefetch=dict(
                additional_register_reservation_bits=84,
                coefficient_bits=88, compiled_profile_count=1,
                offset_lookup_LUT6_upper=11, admission_add21=1,
                admission_latency_edges=1, init_guard_compare63_per_check=1,
                init_bound_compare21_per_check=1,
                additional_DSP=0, additional_BRAM36=0,
                additional_source_read_ports=0, private_capacity_words=1024,
                additional_FF_LUT_control_mapping=None,
                profile_switch_and_multi_profile_mapping=None,
                normalization_divide_sqrt_or_nonlinear_ops=0,
                no_extra_datapath_multiply=True, synthesis=False, whole_RX16MSps=False,
            ),
        )

