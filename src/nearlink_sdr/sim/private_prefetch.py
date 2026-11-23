"""Bounded private-copy scheduling,with an actual finite frame-horizon service.

Experimental §6.3/6.9 ownership only. One accepted selection,one1024word1R1W
bank and the existing source port;no sync,CFO,PI,CRC or payload answer retry.
"""

from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.head_prefetched_capture import MINIMUM_STEP_Q32
from nearlink_sdr.sim.initialized_captured_frame import InitializedCapturedFrameReplay

__all__ = ["ContinuousPrivatePrefetch", "FrameTailHorizon"]


class FrameTailHorizon:
    """One17-step shift/add,one ceil/clamp and one registered output edge.

    The static minimum PI step gives a conservative needed prefix,not an
    estimate of actual sample timing. Real queries retain ownership of the
    remaining tail. sample_limit is a borrowed compiled configuration wire.
    """

    def __init__(self, sample_limit):
        if type(sample_limit) is not int or not 4 <= sample_limit <= 2**20:
            raise ValueError("Need the existing bounded local source limit")
        self.sample_limit = sample_limit
        self.cycle, self.mode = 0, "idle"
        self.remaining = self.shifted = self.accumulator = self.steps = self.boundary = 0
        self.stats = dict(accepted=0, completed=0, cancelled=0)

    @property
    def ready(self):
        return self.mode == "ready"

    def issue(self, start_tick, end_symbol):
        if self.mode != "idle":
            return False
        if (type(start_tick) is not int or not 0 <= start_tick < 2**23
                or type(end_symbol) is not int or not 1 <= end_symbol < 2**17):
            raise ValueError("Need bounded local eighth-sample start and legal symbol end")
        count = unsigned(self.stats["accepted"] + 1, 64)
        self.remaining, self.shifted = end_symbol - 1, MINIMUM_STEP_Q32
        self.accumulator, self.steps, self.boundary = start_tick << 29, 17, 0
        self.mode, self.stats["accepted"] = "multiply", count
        return True

    def advance(self, cycle):
        if type(cycle) is not int or not self.cycle <= cycle < 2**63:
            raise ValueError("Need monotone finite horizon clock")
        if cycle == self.cycle:
            return
        busy = self.mode in ("multiply", "ceil", "register")
        if busy and cycle != self.cycle + 1:
            raise ValueError("An active horizon needs every real edge")
        if self.mode == "multiply":
            add = self.shifted if self.remaining & 1 else 0
            self.accumulator = unsigned(self.accumulator + add, 53)
            self.remaining >>= 1
            self.shifted = unsigned(self.shifted << 1, 52)
            self.steps -= 1
            if self.steps == 0:
                self.mode = "ceil"
        elif self.mode == "ceil":
            whole = self.accumulator >> 32
            fraction = bool(self.accumulator & ((1 << 32) - 1))
            self.boundary = min(self.sample_limit, unsigned(whole + 3 + fraction, 21))
            self.mode = "register"
        elif self.mode == "register":
            self.stats["completed"] = unsigned(self.stats["completed"] + 1, 64)
            self.mode = "ready"
        self.cycle = cycle

    def close(self, cycle):
        self.advance(cycle)
        if self.mode in ("multiply", "ceil", "register"):
            self.stats["cancelled"] = unsigned(self.stats["cancelled"] + 1, 64)
        self.mode = "closed"

    def release(self):
        if self.mode != "closed":
            raise RuntimeError("Need terminal ownership before horizon release")
        self.remaining = self.shifted = self.accumulator = self.steps = self.boundary = 0
        self.mode = "idle"

    @staticmethod
    def estimate():
        fields = dict(remaining=17, shifted_constant=52, accumulator=53, step_counter=5,
                      boundary=21, mode=3, clock=63, statistics=3 * 64)
        return dict(
            state_fields=fields, state_bits=sum(fields.values()), constant_bits=35,
            latency_edges=19, multiply_edges=17, ceil_clamp_edges=1, output_register_edges=1,
            add53_units=1, add21_compare21_units=1, fraction_nonzero_OR32=1,
            add53_per_request_upper=17, add21_compare21_per_request=1,
            shifts52_and17_per_request=17, division_sqrt_nonlinear=0,
            additional_RAM_DSP_ports=0, per_step_one_clock_assumption=True,
            sample_limit_borrowed_from_existing_config=True,
            physical_LUT_FF_Fmax=None, synthesis=False,
        )


class ContinuousPrivatePrefetch(InitializedCapturedFrameReplay):
    """Fill bounded free space before head;then stop at a finite legal horizon.

    The unverified prefix may include future samples not used by a short or bad
    frame. They remain a bounded unconsumed suffix,not another candidate bank.
    Real correct head CRC/legal length controls the end only,never an answer.
    """

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.horizon = FrameTailHorizon(self.sample_limit)

    def _copy_limit(self):
        if self.flow is None:
            return super()._copy_limit()
        if self.flow.done or self.flow.pending_service is not None:
            return self.history.end
        if self.header is None:
            return self.sample_limit  # Space credit,not this limit,bounds the provisional tail.
        limit = super()._copy_limit()
        return max(limit, self.horizon.boundary) if self.horizon.ready else limit

    def tick(self, cycle, *, blocked=False):
        if type(blocked) is not bool:
            raise ValueError("Need explicit downstream readiness")
        if self.active and not self.done and cycle != self.cycle + 1:
            raise ValueError("Finite frame bridge requires every actual clock edge")
        self.horizon.advance(cycle)
        super().tick(cycle, blocked=blocked)
        if (not self.done and self.header is not None and self.horizon.mode == "idle"):
            if (self.header.get("header_crc") is not True
                    or self.header.get("length_ok") is not True
                    or self.flow is None or self.flow.frame is None
                    or self.flow.frame.consumer.frame_end_symbol is None):
                raise RuntimeError("Only the real validated head may supply a frame horizon")
            self.horizon.issue(self.selection.start_tick,
                               self.flow.frame.consumer.frame_end_symbol)

    def _terminal(self, event):
        self.horizon.close(self.cycle)
        super()._terminal(event)
        event["bounded_private_prefetch"] = dict(
            horizon_statistics=self.horizon.stats.copy(), horizon_limit=self.horizon.boundary,
            horizon_mode=self.horizon.mode,
            provisional_copied_suffix_words=(0 if self.history is None else
                                             event["copied_source_end"]
                                             - event["consumed_source_end"]),
            provisional_words_not_consumed=True, private_capacity_words=1024,
            complete_fixed_or_cycles=False,
        )

    def release(self):
        super().release()
        self.horizon.release()

    def estimate(self):
        base, horizon = super().estimate(), self.horizon.estimate()
        fields = base["new_bridge_control_fields"] | dict(frame_tail_horizon=horizon["state_bits"])
        return base | dict(
            new_bridge_control_fields=fields, new_bridge_control_bits=sum(fields.values()),
            known_register_reservation_subtotal=(base["known_register_reservation_subtotal"]
                                                + horizon["state_bits"]),
            bounded_private_prefetch=dict(
                horizon=horizon, additional_register_reservation_bits=horizon["state_bits"],
                additional_DSP=0, additional_BRAM36=0, additional_source_read_ports=0,
                private_capacity_words=1024,
                before_valid_head="existing FIFO free credit;bounded provisional suffix",
                after_valid_head="max(actual query8,serial minimum legal frame prefix)",
                additional_copy_limit_compare21_mux21=1,
                full_receiver16MSps=False, physical_LUT_FF_Fmax=None,
            ),
        )
