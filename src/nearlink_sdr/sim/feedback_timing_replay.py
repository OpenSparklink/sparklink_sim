"""Fixed error-age sampling closure (§6.2.1.2/6.3/6.9 implementation experiment).

One already selected answer,existing finite source/interpolator and strict frame
gate. PI,normalization,carrier and decoding remain explicit off-clock services;
this is neither an ADC receiver nor a split-owner pipeline implementation.
"""

from nearlink_sdr.phy.interpolant_rx import InterpolantPSKReceiver
from nearlink_sdr.phy.timing import GardnerPIState
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.phy.timing_word_rx import WordTimingPSKReceiver
from nearlink_sdr.sim.feedback_timing_word import FeedbackAgeTimingState
from nearlink_sdr.sim.selected_frame_replay import SelectedFrameReplay

__all__ = ["FeedbackAgeFrameReplay"]


def _validate_age(age):
    if type(age) is not int or age not in (0, 1, 2, 3):
        raise ValueError("Need explicit compiled error age0,1,2,3")


class _FloatAgeState(GardnerPIState):
    """Unchanged detector and gains;fixed zero-filled causal error FIFO.

    Parent floating arithmetic is shared,not an independent detector oracle.
    Its immediate PI update is undone before applying the aged error. This host
    reference does not assign zero hardware cycles to either calculation.
    """

    def __init__(self, *args, error_age):
        _validate_age(error_age)
        super().__init__(*args)
        self.error_age = error_age
        self.pending_errors = [0.] * error_age
        self.closed = False

    def advance(self, current, previous, midpoint):
        if self.closed:
            raise RuntimeError("Released timing packet cannot resume")
        omega, limits = self.omega, self.rate_limit_events
        step, measured = super().advance(current, previous, midpoint)
        if self.error_age == 0 or midpoint is None:
            return step, measured
        used = self.pending_errors[0]
        self.pending_errors = [*self.pending_errors[1:], measured]
        proposed = omega + self.ki * used
        self.rate_limit_events = limits + (proposed < self.lower or proposed > self.upper)
        self.omega = max(self.lower, min(self.upper, proposed))
        return self.omega + self.kp * used, measured

    def release(self):
        self.pending_errors = [0.] * self.error_age
        self.closed = True


class _FloatAgeReceiver(InterpolantPSKReceiver):
    def __init__(self, profile, *, error_age, **options):
        super().__init__(profile, **options)
        self.feedback = _FloatAgeState(
            self.sps, self._active["loop"].power, self.timing_config, error_age=error_age
        )
        self._active["loop"] = self.feedback

    def _event(self, row, *, end_padding=False):
        event = super()._event(row, end_padding=end_padding)
        self.feedback.release()
        return event


class _WordAgeReceiver(WordTimingPSKReceiver):
    def __init__(self, profile, *, error_age, **options):
        self.error_age = error_age
        super().__init__(profile, **options)

    def _make_timing_state(self, *args):
        self.feedback = FeedbackAgeTimingState(*args, error_age=self.error_age)
        return self.feedback.state

    def claim_timing_state(self):
        raise RuntimeError("Fixed-age functional node cannot transfer to the old cycle service")

    def push_symbol(self, current_word, midpoint_word, *, after_samples):
        self.validate_symbol_words(current_word, midpoint_word, after_samples=after_samples)
        self.feedback.advance_words(current_word, midpoint_word)
        self._store_timed_symbol(current_word, after_samples)

    def _tag(self, event):
        super()._tag(event)
        if "feedback_before_release" not in event:
            event["feedback_before_release"] = self.feedback.snapshot()
            self.feedback.release()
        event.update(
            feedback_after_release=self.feedback.snapshot(),
            timing_node_cycle_model="functional fixed age;no clocked service",
            complete_latency_cycles=None,
        )
        return event

    def reference_memory(self):
        return super().reference_memory() | dict(feedback_age_functional=self.feedback.estimate())


class FeedbackAgeFrameReplay(SelectedFrameReplay):
    """Explicit float/finite same-age comparison,one finite packet context.

    No search,truth,expected bytes or CRC retry input. The upstream selection is
    consumed once. Format and age are compiled experimental choices,not packet
    decisions. A float arm cannot silently ignore a finite-format argument.
    """

    def __init__(self, *args, error_age, finite, word_timing_format=None, **options):
        _validate_age(error_age)
        if type(finite) is not bool:
            raise ValueError("Need explicit float/finite arm")
        if finite and not isinstance(word_timing_format, TimingWordFormat):
            raise ValueError("Need an explicit finite timing format")
        if not finite and word_timing_format is not None:
            raise ValueError("Float arm cannot silently ignore a finite format")
        self.error_age, self.finite = error_age, finite
        super().__init__(*args, word_timing_format=word_timing_format, **options)

    def _make_consumer(self, profile, initial_fit, word_timing_format, selection):
        options = dict(initial_fit=initial_fit, error_age=self.error_age, **selection)
        if self.finite:
            return _WordAgeReceiver(profile, word_timing_format=word_timing_format, **options)
        return _FloatAgeReceiver(profile, **options)
