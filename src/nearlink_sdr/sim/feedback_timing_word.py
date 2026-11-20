"""Finite error-age functional reference for Gardner/PI (§6.2.1.2 implementation).

The receiver and existing cycle services do not select this model. Detector,
quantization and gains stay canonical;only the error applied by PI has a fixed
age. No complete scheduling,resource or receiver-performance claim.
"""

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, nearest_ratio, signed, unsigned
from nearlink_sdr.phy.timing_word import (
    KI_Q48,
    KP_Q48,
    WordTimingState,
    division_job,
    power_update,
    ted_operands,
)

__all__ = ["FeedbackAgeTimingState"]


class FeedbackAgeTimingState:
    """One packet,zero-filled fixed age0/1/2/3 FIFO,atomic functional updates.

    Composition deliberately prevents admission as a WordTimingState to the old
    single-owner cycle service. A future split detector/PI scheduler needs its
    own causal ownership and value-carrying validation;host arithmetic is not a
    zero-cycle hardware implementation.
    """

    def __init__(self, power, start_tick, hold_symbols, fmt=None, config=None, *, error_age):
        if type(error_age) is not int or error_age not in (0, 1, 2, 3):
            raise ValueError("Need explicit compiled error age0,1,2,3")
        self.state = WordTimingState(power, start_tick, hold_symbols, fmt, config)
        self._error_age = error_age
        self.pending_errors = [0] * error_age
        self.last_applied_error_word = 0
        self.closed = False

    @property
    def error_age(self):
        return self._error_age

    def advance_words(self, current, midpoint):
        """Measured error is current;PI uses the oldest fixed-age error.

        Known sync updates power without consuming the FIFO. Initial tracked
        symbols use its declared zero fill. A failed update leaves the FIFO,
        observer word and original finite state untouched.
        """
        if self.closed:
            raise RuntimeError("Released timing packet cannot resume")
        s, f = self.state, self.state.fmt
        if len(self.pending_errors) != self.error_age:
            raise RuntimeError("Fixed error-age FIFO identity changed")
        for word in self.pending_errors:
            if type(word) is not int:
                raise ValueError("Need native finite error words")
            signed(word, f.error_bits + 3)
        s.validate_words(current, midpoint)
        r, i, e = unpack_interpolated_word(current, INPUT_FORMAT)
        energy = unsigned(r * r + i * i, 44)
        power, erased, floored = power_update(s.power_word, energy, 2 * e, f)
        error = ted_erased = 0
        clipped = False
        if midpoint is not None:
            dr, di, mr, mi, exponent, ted_erased = ted_operands(
                s.previous_word, current, midpoint
            )
            job = division_job(signed(dr * mr + di * mi, 48), exponent, power, f)
            clipped = job["clipped"]
            error = job.get("bypass")
            if error is None:
                error = nearest_ratio(job["numerator"], job["denominator"])
                if job["negative"]:
                    error = -error
        used = (
            error if self.error_age == 0 else self.pending_errors[0]
        ) if midpoint is not None else 0
        signed(used, f.error_bits + 3)
        shift = 48 + f.error_bits - 32
        ki, kp = rne_shift(KI_Q48 * used, shift), rne_shift(KP_Q48 * used, shift)
        # Canonical commit preflights every state/counter/NCO bound before write.
        s.commit(current, power, error, clipped, erased, ted_erased, floored, ki, kp)
        if midpoint is not None and self.error_age:
            self.pending_errors = [*self.pending_errors[1:], error]
        self.last_applied_error_word = used
        return self.snapshot()

    def release(self):
        """Discard unapplied errors at EOF/end;never advance PI after the packet."""
        self.pending_errors = [0] * self.error_age
        self.closed = True

    def snapshot(self):
        return dict(
            timing=self.state.snapshot(),
            error_age=self.error_age,
            pending_errors=self.pending_errors.copy(),
            last_applied_error_word=self.last_applied_error_word,
            closed=self.closed,
        )

    def estimate(self):
        f = self.state.fmt
        width = f.error_bits + 3
        # Same canonical persistent words/counters;not DSP temporaries or pipeline contexts.
        base = 192 + 52 + f.power_bits + 10 + 35 + width + 6 * 64
        extra = (self.error_age + 1) * width + 1
        return dict(
            error_age=self.error_age,
            error_word_bits=width,
            FIFO_register_bits=self.error_age * width,
            applied_error_observer_bits=width,
            closed_flag_bits=1,
            additional_functional_persistent_bits=extra,
            canonical_functional_persistent_bits=base,
            functional_persistent_bits_lower=base + extra,
            FIFO_initialization="zero;known sync holds FIFO;packet end discards without PI flush",
            FIFO_reads_per_tracked_symbol=int(self.error_age > 0),
            FIFO_writes_per_tracked_symbol=int(self.error_age > 0),
            FIFO_shift_wires_bits=max(0, self.error_age - 1) * width,
            FIFO_only_BRAM=0,
            rounding="canonical power/TED,RNE ratio and coefficient products;rate/error clamps",
            pipeline_context_bits=None,
            pipeline_additional_DSP=None,
            pipeline_RAM_ports=None,
            measured_II=None,
            complete_latency_cycles=None,
            executable_cycle_service=False,
            full_receiver16MSps=False,
            synthesis=False,
        )
