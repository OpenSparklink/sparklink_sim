"""Finite one-clock RRC/raw-flag/coarse admission (§6.2.1.2/6.2.3).

RRC releases and detector flags are real external ports. This controller does
not invent their arithmetic, stall the ADC, select a CRC answer or time a fine
search. A lost source identity aborts this epoch; later pulses remain counted.
"""

from types import MappingProxyType

from nearlink_sdr.sim.banked_scaled_core import BankedScaledCore
from nearlink_sdr.sim.capture_fine import unpack_filter_word
from nearlink_sdr.sim.raw_gate_join import RawGateJoin
from nearlink_sdr.sim.rrc_bridge_flow import RRCScaledBridgeService

__all__ = ["CoarseIngress"]


class CoarseIngress:
    """One pending query tag, existing flag/word/coarse queues, one owner.

    Push each source port at most once per edge, then call step. Candidate
    consumption is explicit; no unlimited output collector belongs to hardware.
    An epoch never rearms itself or retries a synchronization alternative.
    """

    reasons = MappingProxyType(
        dict(rrc_gap=1, rrc_invalid=2, rrc_credit=3, join_fault=4, core_fault=5, abort=6)
    )

    def __init__(
        self, reference, *, lanes=16, normalizer="pipeline", sink=None,
        preserve_source_end=False,
    ):
        self.core = BankedScaledCore(
            reference, lanes=lanes, origin=-40, normalizer=normalizer, sink=sink,
            preserve_source_end=preserve_source_end,
        )
        self.join = RawGateJoin(RRCScaledBridgeService(), window_samples=self.core.window)
        self.pending = None
        self.next_rrc_tag, self.last_rrc_cycle = 0, None
        self.fault_reason = self.fault_tag = 0
        self.closed = False
        self.stats = dict.fromkeys(
            (
                "rrc_events",
                "rrc_invalid",
                "rrc_missing_tags",
                "rrc_credit_rejected",
                "rrc_discarded",
                "query_requests",
                "pending_query_dropped",
                "pending_query_canceled",
                "candidate_canceled",
                "query_submitted",
                "core_query_dropped",
                "epoch_faults",
            ),
            0,
        )

    @property
    def cycle(self):
        return self.core.cycle

    @property
    def maximum_tag(self):
        # The query coordinate port is signed62, narrower than the sample ID.
        return min(self.core.maximum_input_tag, 2**62 - 1)

    def _updates(self, **changes):
        values = {key: self.stats[key] + amount for key, amount in changes.items()}
        if any(not 0 <= v < 2**64 for v in values.values()):
            raise OverflowError("Unsigned64 ingress statistic exhausted")
        self.stats.update(values)

    def _fault(self, reason, tag):
        if self.fault_reason:
            return
        self._updates(
            epoch_faults=1,
            pending_query_canceled=int(self.pending is not None),
            candidate_canceled=int(bool(self.core.candidates)),
        )
        self.fault_reason, self.fault_tag = self.reasons[reason], tag
        self.pending = None
        if self.core.candidates:
            self.core.take_candidate()
        self.join.abort()
        if not self.core.failure:
            self.core.input_overrun(self.core.input_count)

    def _synchronize(self):
        if self.core.cycle != self.join.cycle:
            raise RuntimeError("Ingress exclusively owns the shared clock")
        if not self.fault_reason:
            if self.join.fault_reason:
                self._fault("join_fault", self.join.fault_tag)
            elif self.core.failure:
                self._fault("core_fault", self.core.first_lost_tag)

    def push_flag(self, tag, flag):
        """The unchanged detector owns the Boolean value and its nonzero service."""
        if self.closed:
            raise RuntimeError("Cannot write after EOF")
        if type(tag) is not int or not 0 <= tag <= self.maximum_tag:
            raise ValueError("Need a compiled sample tag with coordinate headroom")
        accepted = self.join.push_flag(tag, flag)
        self._synchronize()
        return accepted

    def push_rrc(self, tag, word):
        """One nonstallable RRC release;None explicitly denotes an invalid result.

        Missing tags and credit rejection invalidate the epoch. They never
        renumber samples, manufacture a filter word or silently skip a pulse.
        """
        if self.closed:
            raise RuntimeError("Cannot write after EOF")
        if (
            type(tag) is not int
            or not self.next_rrc_tag <= tag <= self.maximum_tag
            or self.last_rrc_cycle == self.cycle
        ):
            raise ValueError("Need monotone bounded RRC tags and one pulse per edge")
        if word is not None:
            _, _, exponent = unpack_filter_word(word)
            if not -96 <= exponent <= 48:
                raise ValueError("Need the compiled RRC exponent[-96,48]")
        gap = tag - self.next_rrc_tag
        self._updates(rrc_events=1, rrc_missing_tags=gap)
        self.last_rrc_cycle, self.next_rrc_tag = self.cycle, tag + 1
        if gap:
            self._fault("rrc_gap", tag)
        if word is None:
            self._updates(rrc_invalid=1, rrc_discarded=1)
            self._fault("rrc_invalid", tag)
            return False
        if self.fault_reason:
            self._updates(rrc_discarded=1)
            return False
        if not self.join.bridge.issue(word, tag):
            self._updates(rrc_credit_rejected=1, rrc_discarded=1)
            self._fault("rrc_credit", tag)
            return False
        return True

    def step(self):
        """Issue old pending query, transfer old joined word, clock both nodes.

        A query created by this transfer cannot issue on this edge. Its bounds
        remain attached to the retired sample tag, even when points arrive later.
        """
        if self.closed:
            raise RuntimeError("Cannot clock after EOF")
        self._synchronize()
        if not self.fault_reason:
            if self.pending is not None and self.core.committed_count > self.pending:
                tag = self.pending
                lower = tag + 2 - 40 - self.core.window
                upper = tag - 40 - self.core.k * 4
                request_id = self.core.next_query_id
                accepted = (
                    self.core.request_query(lower, upper, source_end=tag + 1)
                    if self.core.preserve_source_end else self.core.request_query(lower, upper)
                )
                self._updates(query_submitted=1, core_query_dropped=int(not accepted))
                self.pending = None
                if self.core.sink is not None:
                    self.core.sink(
                        dict(
                            kind="ingress_query_request",
                            request_id=request_id,
                            cycle=self.cycle,
                            sample_tag=tag,
                            lower=lower,
                            upper=upper,
                            snapshot=self.core.committed_count,
                            accepted=accepted,
                        )
                    )
            if self.join.ready and self.core.can_accept:
                item = self.join.bridge.result
                if not self.core.offer(item["tag"], *item["word"]):
                    raise RuntimeError("Prechecked coarse input credit disappeared")
                value = self.join.take()
                if (
                    value["tag"] + 1 >= self.core.window
                    and (value["tag"] + 1) % 32 == 0
                    and value["window_energy_present"]
                ):
                    self._updates(
                        query_requests=1, pending_query_dropped=int(self.pending is not None)
                    )
                    if self.pending is None:
                        self.pending = value["tag"]
        self.core.step()
        self.join.step()
        self._synchronize()

    def take_candidate(self):
        """Consume only the existing coarse output credit;no CRC retry path."""
        return None if self.fault_reason else self.core.take_candidate()

    @property
    def lossless(self):
        """Source/query admission only,never an RF packet or full-RX verdict."""
        return not self.fault_reason and not any(
            (
                self.stats["pending_query_dropped"],
                self.stats["core_query_dropped"],
                self.core.stats["query_expired"],
                self.core.stats["query_invalid"],
            )
        )

    @property
    def drained(self):
        if self.fault_reason:
            return True
        return (
            self.pending is None
            and self.core.data_idle
            and not self.core.query_queue
            and self.core.query_active is None
            and not self.core.candidates
            and self.join.mode == "idle"
            and self.join.bridge.mode in ("idle", "ready")
            and not (self.join.bridge.ready and self.join.count)
        )

    def finish(self):
        """Close after draining;missing tail flags/words remain explicit failures."""
        if self.closed:
            return self.lossless
        self._synchronize()
        if not self.drained:
            raise RuntimeError("Drain finite work and held candidates before EOF")
        missing = max(0, self.join.next_tag - self.next_rrc_tag)
        self._updates(rrc_missing_tags=missing)
        self.join.finish()
        self._synchronize()
        passed = self.core.finish()
        self.closed = True
        return passed and self.lossless

    def abort(self):
        if not self.closed:
            self._fault("abort", self.core.input_count)

    def estimate(self):
        fields = dict(
            pending_tag=63,
            pending_valid=1,
            next_rrc_tag=63,
            last_rrc_cycle=63,
            last_rrc_cycle_valid=1,
            fault_reason=3,
            fault_tag=63,
            closed=1,
        )
        own = sum(fields.values()) + 64 * len(self.stats)
        bridge, join, core = self.join.bridge.estimate(), self.join.estimate(), self.core.estimate()
        return dict(
            own_register_fields=fields,
            own_statistics_bits=64 * len(self.stats),
            own_register_bits_upper=own,
            own_register_research_cap_bits=1536,
            own_register_research_cap_pass=own <= 1536,
            new_DSP=0,
            new_sample_RAM=0,
            pending_query_credits=1,
            query_stride_samples=32,
            raw_flag_FIFO_depth=16,
            output_credit_owned_by_existing_core=True,
            whole_prefix_excluding_RRC_and_detector_logical_bits=(
                own
                + bridge["register_bits_upper"]
                + join["register_bits_upper"]
                + core["logical_state_bits_upper"]
            ),
            bridge=bridge,
            join=join,
            coarse=core,
            independent_statistics_adders64_upper=len(self.stats),
            query_bound_constant_subtractors64=2,
            query_endpoint_increment63=1,
            tag_gap_subtractor63=1,
            EOF_tail_subtractor63=1,
            next_tag_increment63=1,
            tag_and_clock_compare63_upper=5,
            pending_and_fault_MUX2_bits_upper=2 * 63,
            new_coefficient_or_IQ_ports=0,
            encoded_exponent_port_assertion_comparators8=2,
            encoded_exponent_fault_response_cycles=None,
            automatic_rearm=False,
            fault_transition_edges=1,
            raw_detector_arithmetic_cycles=None,
            RRC_cycles="external existing producer,not zero",
            fine_packet_decoder_cycles=None,
            LUT_FF_Fmax=None,
            actual_synthesis=False,
            whole_RX_16MSps=False,
        )
