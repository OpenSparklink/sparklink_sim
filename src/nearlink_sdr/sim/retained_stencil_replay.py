"""Reuse the existing raw4word staging window across PSK symbols (§6.2.1.2 model).

Addresses are registered NCO results;interpolation,phase,loops and CRC stay
unchanged. Explicit structure only,no whole-RX or synthesized timing claim.
"""

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.sync_fit_word import signed, unsigned
from nearlink_sdr.sim.power_ted_replay import PowerTEDFrameReplay

__all__ = ["RetainedStencilFrameReplay"]


def _overlap_words(previous_start, next_start):
    """Consecutive prefix available from a valid previous four-word window.

Four-word exclusive ends must also fit unsigned64. Backward/gapped windows
miss safely;never wrap addresses or reuse a non-prefix intersection.
"""
    if type(previous_start) is not int or type(next_start) is not int:
        raise ValueError("Need native unsigned64 source-window addresses")
    for value in (previous_start, next_start):
        unsigned(value, 64)
        unsigned(value + 4, 64)
    difference = signed(next_start - previous_start, 65)
    return 4 - difference if 0 <= difference < 4 else 0


class RetainedStencilFrameReplay(PowerTEDFrameReplay):
    """Same4x46bit registers,one packet;no extra source bank,port or query credit."""

    _retain_completed_stencil = True

    def __init__(self, profile, prepared, *, reuse_stencil=False, **options):
        if reuse_stencil is not True:
            raise ValueError("Retained stencil requires explicit ordered within-symbol reuse")
        if (not isinstance(profile, PSKReceiveProfile) or profile.config.sps != 4
                or profile.config.symbol_rate_mhz != 4):
            raise ValueError("Retained stencil is compiled only for4Msym/s SPS4")
        super().__init__(profile, prepared, reuse_stencil=True, **options)
        self.stats["cross_symbol_words_reused"] = 0

    def _reuse_count(self, next_start):
        if (self.mode != "nco_query" or len(self.staged) != 4 or not self.queries
                or self.query_index != len(self.queries) or self.queries[-1]["role"] != 0):
            return 0
        return _overlap_words(self.stage_start, next_start)

    def step(self):
        if not self.done and self.mode == "nco_query" and self.query_service.ready:
            reply = self.query_service.result
            if reply["count"] != self.consumer.word_timing.count:
                raise RuntimeError("Query describes an expired timing state")
            if self.consumer._active["count"] >= len(self.consumer.reference):
                _, midpoint = reply["queries"]
                if midpoint is None:
                    raise RuntimeError("Tracked data requires its registered midpoint query")
                count = self._reuse_count(midpoint["cell"] - 1)
                unsigned(self.stats["cross_symbol_words_reused"] + count, 64)
        return super().step()

    def _plan_samples(self, queries, future_end):
        previous = self.staged  # Existing registers before this edge,not another buffer.
        count = (
            self._reuse_count(queries[-1]["cell"] - 1)
            if self.consumer._active["count"] >= len(self.consumer.reference) else 0
        )
        super()._plan_samples(queries, future_end)
        if count:
            self.staged = previous[4 - count:]
            self._count("cross_symbol_words_reused", count)

    def estimate(self):
        return super().estimate() | dict(
            retained_stencil_additional_controller_bits=64,
            retained_stencil_counter_already_in_common_total=True,
            retained_stencil_existing_raw_registers_bits=4 * 46,
            retained_stencil_existing_start_register_bits=64,
            retained_stencil_additional_DSP=0,
            retained_stencil_additional_raw_RAM_bits=0,
            retained_stencil_additional_source_ports=0,
            retained_stencil_source_read_write_ports=(1, 1),
            retained_stencil_query_credits=1,
            retained_stencil_projection_evaluations_upper=2,
            retained_stencil_window_end_add65_upper=4,
            retained_stencil_window_carry_guards_upper=4,
            retained_stencil_first_address_decrement65_upper=2,
            retained_stencil_difference65_upper=2,
            retained_stencil_range_compare65_upper=4,
            retained_stencil_counter_add65_upper=2,
            retained_stencil_incremental_count_compare64_upper=1,
            retained_stencil_compaction_mux2_upper=368,
            retained_stencil_logic_sharing_credit=False,
            retained_stencil_staging_borrowed_by_other_packets_or_fine=False,
            retained_stencil_steady_raw_reads_upper=8,
            retained_stencil_guard_before_query_take_and_source_retire=True,
            retained_stencil_guard_and_compaction_cycles_assumption=0,
            retained_stencil_assumption="same existing registered NCO edge;wide guards/mux"
                "not synthesized or timed;additional pipeline edges can erase latency saving",
            retained_stencil_LUT_FF_or_Fmax=None,
            full_receiver16MSps=False,
            complete_latency_cycles=None,
        )
