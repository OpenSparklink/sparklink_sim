"""Bounded weight-node calendar, not a replacement for packet FFT scheduling."""

from collections import deque

from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights

__all__ = ["CarrierWeightFlow"]


class CarrierWeightFlow:
    """One42-bit ALU and one saturating64-bit counter ALU, eight-clock feedback.

    Fixed-word input with two range-event flags reserved,4-entry1R1W FIFOs.
    ADC/timing/normalization-to-power conversion and packet FFT are outside.
    Source cannot be paused in this stress model; any lost word invalidates the
    whole packet. Real receiver feedback/admission is not integrated here.
    """

    def __init__(self, kappa=2, *, clock_hz=200_000_000, symbol_hz=4_000_000):
        BoundedCarrierWeights(kappa, fixed=True)
        if (any(type(v) is not int or v < 1 for v in [clock_hz, symbol_hz])
                or symbol_hz > clock_hz):
            raise ValueError("Need positive clock and at most one source word/clock")
        self.kappa, self.clock_hz, self.symbol_hz = kappa, clock_hz, symbol_hz

    def estimate(self):
        storage = dict(power_state=33, input_fifo=4 * 106, output_fifo=4 * 100,
                       active_word_and_tag=106, scratch=148, counters=7 * 64,
                       phase_config_valid_and_overflow=8, fifo_control=14)
        return dict(
            evidence="operation/storage allocation + executable bounded queue calendar; not synthesis",
            clock_hz=self.clock_hz, symbol_hz=self.symbol_hz, ii=8, isolated_latency=8,
            input_fifo_words=4, output_fifo_words=4, multiplier_units_extra=0,
            wide_alu42=1, counter_alu64=1, guard_sticky_and_select="explicit combinational logic, LUT count unestimated",
            state_components=storage, logical_bits=sum(storage.values()),
            data_phases=["weight cap compare", "update cap compare", "signed delta subtract",
                         "arithmetic shift/guard-sticky/RNE increment", "state add",
                         "lower clamp compare", "upper clamp compare", "select/commit/output"],
            counter_phases={0: "input saturation flag", 1: "input erased flag",
                            3: "weight clipping", 4: "update clipping", 5: "lower clamp",
                            6: "upper clamp", 7: "sample count"},
            ports="input/output FIFO each1read1write; power register1read1write; counter bank1update/clock",
            counter_saturation="carry detection and saturating select assumed1cycle; sample exhaustion rejects packet",
            continuous_node_necessary_condition=8 * self.symbol_hz <= self.clock_hz,
            extra_nonlinear=0, extra_division=0, extra_sqrt=0,
            excluded=["symbol normalization/power conversion", "FFT/Newton/circular phase",
                      "frame RAM/decoder scheduling", "whole-RX state feedback", "LUT/FF/physical RAM mapping"],
            whole_system_accepted=False,
        )

    def simulate(self, power_words, *, stalls=()):
        """Caller-owned full trace; runtime queues/active state stay finite."""
        if (not power_words or any(type(v) is not int or not 0 <= v < 2**41
                                   for v in power_words)):
            raise ValueError("Need nonempty unsigned41-bit input powers")
        previous = -1
        for start, stop in stalls:
            if (type(start) is not int or type(stop) is not int
                    or not 0 <= start < stop or start < previous):
                raise ValueError("Need ordered nonoverlapping sink intervals")
            previous = stop
        node = BoundedCarrierWeights(self.kappa, fixed=True)
        queue, output = deque(), deque()
        active, next_start, index, cycle = None, 0, 0, 0
        emitted, dropped = [], []
        peak_input = peak_output = 0
        total = len(power_words)
        last_arrival = (total - 1) * self.clock_hz // self.symbol_hz
        stop_bound = max(last_arrival, previous) + 8 * (total + 10)
        if stop_bound >= 2**63:
            raise OverflowError("Cycle counter exhausted")
        while index < total or queue or active is not None or output:
            if cycle > stop_bound:
                raise AssertionError("Bounded weight calendar failed to drain")
            blocked = any(start <= cycle < stop for start, stop in stalls)
            if output and not blocked:
                tag, weight, arrival = output.popleft()
                emitted.append(dict(tag=tag, weight=weight, cycle=cycle, latency=cycle - arrival))
            if active is not None and cycle >= active["done"] and len(output) < 4:
                tag, word, arrival = active["tag"], active["word"], active["arrival"]
                weight = int(node.advance_words([word])[0])
                output.append((tag, weight, arrival))
                active = None
            if index < total and cycle == index * self.clock_hz // self.symbol_hz:
                if len(queue) == 4:
                    dropped.append(dict(tag=index, cycle=cycle))
                else:
                    queue.append((index, power_words[index], cycle))
                index += 1
            if active is None and queue and cycle >= next_start:
                tag, word, arrival = queue.popleft()
                active = dict(tag=tag, word=word, arrival=arrival, done=cycle + 7)
                next_start = cycle + 8
            peak_input, peak_output = max(peak_input, len(queue)), max(peak_output, len(output))
            cycle += 1
        return dict(emitted=emitted, dropped=dropped, cycles=cycle,
                    packet_valid=not dropped, peak_input=peak_input, peak_output=peak_output,
                    max_latency=max(r["latency"] for r in emitted),
                    state_words_bound=4 + 4 + 1, stats=dict(node.stats),
                    scope="weight-word calendar only; drops invalidate entire packet, no CRC-driven retry")
