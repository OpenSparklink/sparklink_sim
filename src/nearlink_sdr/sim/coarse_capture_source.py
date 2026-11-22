"""Causal RRC/flag event ports→bounded coarse owner→original capture geometry.

Future port descriptors are testbench stimuli,held at one per port. No event
history or expected query map selects a result. Existing RRC and raw-detector
work remains separately charged;this controller does not synthesize them.
"""

from nearlink_sdr.sim.capture_fine import unpack_filter_word
from nearlink_sdr.sim.capture_transport import ReleasedCaptureRequest
from nearlink_sdr.sim.coarse_ingress import CoarseIngress

__all__ = ["CoarseCaptureSource"]


class CoarseCaptureSource:
    """One release/flag per edge,one held candidate;explicit source epoch faults.

    RRC stimuli are(cycle,original_tag,word_or_None);flags are(cycle,tag,bool).
    The limit/horizon bound source arrivals,not a packet or an expected start.
    Existing initialization and downstream work may drain after that horizon.
    All post-fault pulses still reach ingress accounting. No automatic rearm.
    """

    def __init__(
        self, reference, releases, flags, *, sample_limit, source_horizon,
        lanes=16, normalizer="serial2", sink=None,
    ):
        if (type(sample_limit) is not int or not 1 <= sample_limit < 2**60
                or type(source_horizon) is not int or not 1 <= source_horizon < 2**60):
            raise ValueError("Need finite source sample and clock bounds")
        self.sample_limit, self.source_horizon = sample_limit, source_horizon
        self.flow = CoarseIngress(reference, lanes=lanes, normalizer=normalizer,
                                  preserve_source_end=True, sink=sink)
        self.rrc, self.flags = iter(releases), iter(flags)
        self.next_rrc = self._read(self.rrc, False, -1)
        self.next_flag = self._read(self.flags, True, -1)
        self.cycle, self.closed = 0, False
        self.fault_cycle = None
        self.stats = dict.fromkeys(("rrc_pulses", "flag_pulses", "coarse_returns",
                                   "enabled_queries", "disabled_queries", "rejected_queries",
                                   "geometry_rejects", "cancelled_descriptors"), 0)

    def _read(self, stream, flag, after):
        event = next(stream, None)
        if event is None:
            return None
        if (not isinstance(event, tuple) or len(event) != 3
                or type(event[0]) is not int or not after < event[0] <= self.source_horizon
                or type(event[1]) is not int or not 0 <= event[1] < self.sample_limit):
            raise ValueError("Need ordered one-per-edge finite source port descriptors")
        if flag:
            if type(event[2]) is not bool:
                raise ValueError("Raw detector port needs a Boolean flag")
        elif event[2] is not None:
            _, _, exponent = unpack_filter_word(event[2])
            if not -96 <= exponent <= 48:
                raise ValueError("RRC source exponent exceeds compiled range")
        return event

    def _count(self, name):
        if self.stats[name] == 2**64 - 1:
            raise OverflowError("Source controller counter exhausted")
        self.stats[name] += 1

    @property
    def failed(self):
        return bool(self.flow.fault_reason)

    @property
    def done(self):
        return self.closed

    def advance(self, cycle):
        """Return(this edge's RRC pulse,new released request),then advance coarse.

        Candidate consumption precedes new source writes and coarse.step,so a
        newly completed query is unavailable until the following edge.
        """
        if type(cycle) is not int or cycle != self.cycle or cycle >= 2**63 - 1:
            raise ValueError("Source controller exclusively owns consecutive clock edges")
        if self.closed:
            self.cycle += 1
            return None, None
        if self.flow.cycle != cycle:
            raise RuntimeError("Coarse and capture clocks diverged")
        request = None
        candidate = self.flow.take_candidate()
        if candidate is not None:
            self._count("coarse_returns")
            if candidate["reason"] is not None:
                self._count("rejected_queries")
            elif not candidate["enabled"]:
                self._count("disabled_queries")
            else:
                end = candidate["source_end"]
                peak = candidate["chosen"] - (end - self.flow.core.window - 40)
                if not 0 <= peak < self.flow.core.window:
                    self._count("geometry_rejects")
                else:
                    request = ReleasedCaptureRequest(
                        after_samples=end, path=0, peaks=(peak,),
                        minimum_tick=8, lookahead_ticks=128, available_cycle=cycle)
                    self._count("enabled_queries")
        pulse = None
        if self.next_flag is not None and self.next_flag[0] == cycle:
            _, tag, flag = self.next_flag
            self.flow.push_flag(tag, flag)
            self._count("flag_pulses")
            self.next_flag = self._read(self.flags, True, cycle)
        if self.next_rrc is not None and self.next_rrc[0] == cycle:
            _, tag, word = self.next_rrc
            self.flow.push_rrc(tag, word)
            pulse = tag, word
            self._count("rrc_pulses")
            self.next_rrc = self._read(self.rrc, False, cycle)
        self.flow.step()
        if self.next_flag is self.next_rrc is None and self.flow.drained:
            self.flow.finish()
            self.closed = True
        if self.failed and request is not None:
            self._count("cancelled_descriptors")
            request = None
        if self.failed and self.fault_cycle is None:
            self.fault_cycle = cycle
        self.cycle += 1
        return pulse, request

    def estimate(self):
        own = dict(pending_rrc_release=174, pending_raw_flag=128, clock=63,
                   source_limit=63, source_horizon=63, closed=1, fault_cycle_and_valid=64,
                   statistics=8 * 64)
        return dict(
            register_fields=own, own_register_bits_upper=sum(own.values()),
            compiled_source_endpoint_bits=4 * 64,
            coarse_ingress=self.flow.estimate(),
            new_DSP=0, new_IQ_RAM=0, new_sample_ports=0,
            held_future_stimulus_descriptors_each_port=1,
            source_and_flag_ports=1, candidate_ports=1,
            source_end_is_original_query_not_latest_snapshot=True,
            guard_compare63_upper=12, endpoint_subtract63=2,
            endpoint_update_latency="registered query context wires;existing query edges unchanged",
            RRC_arithmetic_cycles="actual external producer,not zero",
            raw_detector_arithmetic_cycles=None,
            carrier_and_frame_decoder_cycles=None,
            LUT_FF_Fmax=None, synthesis=False, complete_RX_16MSps=False,
        )
