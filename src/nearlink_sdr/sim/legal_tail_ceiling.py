"""One finite legal upper copy fence,not a new timing answer (§6.3/6.9 trial).

The actual clipped PI envelope bounds any remaining legal future. Copying is
still limited by the existing1024wordFIFO and original source port;unused
suffix words remain charged and unconsumed. Default reception is unchanged.
"""

from nearlink_sdr.phy.timing_word import KP_Q48, UPPER_Q32, rne_shift
from nearlink_sdr.sim.private_prefetch import ContinuousPrivatePrefetch, FrameTailHorizon

__all__ = ["LegalTailCeiling", "UpperTailCaptureReplay"]

MAXIMUM_STEP_Q32 = UPPER_Q32 + rne_shift(KP_Q48 * (2 << 32), 48)


class LegalTailCeiling(FrameTailHorizon):
    """Same17-step ALU/ceil/register service,one compiled upper coefficient.

    This is a bound on possible needed input,not a prediction of actual NCO
    centers. The true NCO and frame owner still determine consumption/end.
    """

    def issue(self, start_tick, end_symbol):
        if not super().issue(start_tick, end_symbol):
            return False
        self.shifted = MAXIMUM_STEP_Q32  # Compiled wire,not another runtime datapath.
        return True

    @staticmethod
    def estimate():
        return FrameTailHorizon.estimate() | dict(
            fence_kind="legal maximum PI-step upper envelope",
            compiled_step_Q32=MAXIMUM_STEP_Q32,
            one_request_per_validated_frame=True,
            maximum_accepted_step_from_actual_clipped_PI=True,
            copied_suffix_not_a_consumed_or_delivered_tail=True,
        )


class UpperTailCaptureReplay(ContinuousPrivatePrefetch):
    """M38B copy policy with one legal upper fence instead of its lower fence.

    Actual CRC/length/MCS validation supplies N once. No candidate or answer
    retry,no new sample bank,and no new feedback or phase-search parameter.
    """

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.horizon = LegalTailCeiling(self.sample_limit)

    def _terminal(self, event):
        super()._terminal(event)
        event["bounded_private_prefetch"].update(
            fence_kind="legal upper envelope,not actual consumed end",
            compiled_maximum_step_Q32=MAXIMUM_STEP_Q32,
            actual_NCO_still_owns_frame_end=True, maximum_horizon_requests_per_frame=1,
            unused_suffix_charged_to_same_private_FIFO=True,
        )

    def estimate(self):
        base = super().estimate()
        base["bounded_private_prefetch"].update(
            after_valid_head="max(actual query8,registered legal upper envelope)",
            static_upper_fence=True, extra_state_vs_static_lower_bits=0,
            additional_step_estimation_or_PI_wire_comparison=0,
        )
        return base
