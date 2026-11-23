"""Repeated conservative legal-tail snapshots (§6.3/6.9 implementation trial).

Only private-copy eligibility changes. Actual committed PI words and the real
validated frame end feed one finite serial controller;no candidate/CRC retry.
"""

from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.head_prefetched_capture import MINIMUM_STEP_Q32
from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay
from nearlink_sdr.sim.private_prefetch import ContinuousPrivatePrefetch, FrameTailHorizon

__all__ = ["MovingFrameTailHorizon", "MovingTailCaptureReplay"]


class MovingFrameTailHorizon(FrameTailHorizon):
    """One19-edge request,no queue;keep the old published bound while busy.

    q[m]+(N-1-m)*minimum_step never exceeds the actual final symbol position.
    A delayed snapshot remains conservative. This is a needed prefix,not a
    timing estimate;the actual source/query owner still completes the tail.
    """

    def __init__(self, sample_limit):
        super().__init__(sample_limit)
        self.published_boundary = self.last_issued_count = 0

    def issue(self, position_q32, count, end_symbol):
        if self.mode not in ("idle", "ready"):
            return False
        if (type(position_q32) is not int or not 0 <= position_q32 <= (2**20 + 8) << 32
                or type(count) is not int or type(end_symbol) is not int
                or not 0 <= count < end_symbol < 2**17):
            raise ValueError("Need actual bounded local PI words before a legal final symbol")
        if self.mode == "ready":
            if count < self.last_issued_count:
                raise ValueError("A packet PI count cannot go backwards")
            if count == self.last_issued_count:
                return False
        accepted = unsigned(self.stats["accepted"] + 1, 64)
        # N-1-m is one17bit N+ones_complement(m) adder with discarded carry.
        self.remaining = (end_symbol + (count ^ (2**17 - 1))) & (2**17 - 1)
        self.accumulator, self.shifted = position_q32, MINIMUM_STEP_Q32
        self.steps, self.boundary, self.mode = 17, 0, "multiply"
        self.last_issued_count, self.stats["accepted"] = count, accepted
        return True

    def advance(self, cycle):
        publish = self.mode == "register" and cycle != self.cycle
        super().advance(cycle)
        if publish:
            if self.boundary < self.published_boundary:
                raise RuntimeError("Actual PI words violated the conservative monotone envelope")
            self.published_boundary = self.boundary

    def release(self):
        super().release()
        self.published_boundary = self.last_issued_count = 0

    @staticmethod
    def estimate():
        base = FrameTailHorizon.estimate()
        fields = base["state_fields"] | dict(published_boundary=21, last_issued_count=17)
        return base | dict(
            state_fields=fields, state_bits=sum(fields.values()),
            additional_vs_static_bits=38, local_position_guard_constant_bits=53,
            count_add17_per_request=1, count_complement17_per_request=1,
            count_compare17_per_eligible_edge=1, position_guard_compare53_per_request=1,
            count_legal_compare17_per_request=1, count_direction_compare17_per_request=1,
            source_PI_position_bits=96, source_PI_count_bits=64,
            position_high_zero_OR43_per_request=1, count_high_zero_OR47_per_request=1,
            published_compare21_per_completed_request=1,
            input_accumulator_mux53=1, output_register21=1,
            request_II_edges=19, outstanding_requests=1, request_queue_words=0,
            intermediate_PI_updates_need_no_queue=True,
            source_PI_count_and_end_borrowed_not_copied=True,
            control_mux_enable_reset_mapping=None, full_RX16MSps=False,
        )


class MovingTailCaptureReplay(ContinuousPrivatePrefetch):
    """The M38B copy policy,with a bounded repeat of its legal-tail service.

    Only the actual PI owner supplies q/count. No truth,payload,CRC-selected
    answer or hypothetical rate estimate enters the prefix. Default unchanged.
    """

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.horizon = MovingFrameTailHorizon(self.sample_limit)

    def _copy_limit(self):
        if self.flow is None:
            return InitializedCapturedFrameReplay._copy_limit(self)
        if self.flow.done or self.flow.pending_service is not None:
            return self.history.end
        if self.header is None:
            return self.sample_limit
        return max(InitializedCapturedFrameReplay._copy_limit(self),
                   self.horizon.published_boundary)

    def tick(self, cycle, *, blocked=False):
        if type(blocked) is not bool:
            raise ValueError("Need explicit downstream readiness")
        if self.active and not self.done and cycle != self.cycle + 1:
            raise ValueError("Finite frame bridge requires every actual clock edge")
        self.horizon.advance(cycle)
        InitializedCapturedFrameReplay.tick(self, cycle, blocked=blocked)
        if self.done or self.header is None or self.horizon.mode not in ("idle", "ready"):
            return
        if (self.header.get("header_crc") is not True
                or self.header.get("length_ok") is not True
                or self.flow is None or self.flow.frame is None):
            raise RuntimeError("Only a real validated head supplies the moving legal end")
        frame = self.flow.frame.consumer
        if frame.frame_end_symbol is None:
            raise RuntimeError("A real legal frame end is required")
        pi = frame.pi  # The live committed owner,not a delivered-symbol receipt.
        if (pi.count < frame.frame_end_symbol and (self.horizon.mode == "idle"
                or pi.count != self.horizon.last_issued_count)):
            self.horizon.issue(pi.next_q32, pi.count, frame.frame_end_symbol)

    def _terminal(self, event):
        super()._terminal(event)
        event["bounded_private_prefetch"].update(
            horizon_limit=self.horizon.published_boundary,
            moving_conservative_tail=True, last_issued_PI_count=self.horizon.last_issued_count,
            maximum_outstanding_horizon_requests=1, no_PI_request_queue=True,
            actual_final_symbol_not_extrapolated=True,
        )

    def estimate(self):
        base = super().estimate()
        base["bounded_private_prefetch"].update(
            after_valid_head="max(actual query8,last registered conservative live-PI prefix)",
            actual_moving_PI_snapshot=True, additional_vs_M38B_register_bits=38,
            guard_and_count_ALU=MovingFrameTailHorizon.estimate(),
        )
        return base
