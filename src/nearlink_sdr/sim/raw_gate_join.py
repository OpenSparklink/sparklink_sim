"""Finite raw-energy flag/RRC word alignment (§6.2.1.2/6.2.3).

Flags refer to causal FIR output indices, before removing the RRC group delay.
The detector arithmetic is external and has no implied zero-cycle service.
"""

from types import MappingProxyType

from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.rrc_bridge_flow import RRCScaledBridgeService

__all__ = ["RawGateJoin"]


class RawGateJoin:
    """Sixteen flags, one registered read and the existing bridge's IQ credit.

    Flags must arrive in order. No second IQ word is stored. A missing word or
    full flag FIFO invalidates this epoch; the ADC can continue while subsequent
    flags/words are explicitly discarded. A new epoch requires an external reset.
    """

    depth = 16
    reasons = MappingProxyType(
        {
            "flag_overflow": 1,
            "word_gap": 2,
            "ownership": 3,
            "EOF_flag_missing": 4,
            "EOF_word_missing": 5,
            "abort": 6,
        }
    )

    def __init__(self, bridge, *, start_tag=0, window_samples=256):
        if (
            not isinstance(bridge, RRCScaledBridgeService)
            or bridge.fmt.mantissa_bits != 10
            or bridge.mode != "idle"
            or bridge.stats["accepted"] != 0
        ):
            raise ValueError("Need a fresh compiled scaled10 bridge owned by this join")
        if type(start_tag) is not int or not 0 <= start_tag < 2**63 - 1:
            raise ValueError("Need a signed64 initial sample index with increment headroom")
        if type(window_samples) is not int or window_samples not in (256, 376, 632):
            raise ValueError("Need the compiled FT2/3/4 SPS4 search window256/376/632")
        self.bridge, self.cycle = bridge, bridge.cycle
        self.window = window_samples
        self.head_tag = self.next_tag = start_tag
        self.flags = self.count = 0
        self.age = self.window
        self.mode, self.closed, self.pending_gate = "idle", False, False
        self.last_push_cycle = -1
        self.head_visible_cycle = self.read_ready_cycle = 0
        self.fault_reason = self.fault_tag = 0
        self.stats = dict.fromkeys(
            (
                "flags_accepted",
                "flags_discarded",
                "words_joined",
                "read_requests",
                "held_edges",
                "faults",
                "orphan_flags",
                "closes",
            ),
            0,
        )

    def _values(self, changes):
        return {key: unsigned(self.stats[key] + value, 64) for key, value in changes.items()}

    def _updates(self, changes):
        self.stats.update(self._values(changes))

    def _fault(self, reason, tag, **extra):
        if self.mode == "failed":
            return
        updates = self._values(dict(faults=1, orphan_flags=self.count, **extra))
        if self.bridge.mode not in ("idle", "aborted"):
            unsigned(self.bridge.stats["aborted"] + 1, 64)
        self.bridge.abort()
        self.stats.update(updates)
        self.fault_reason, self.fault_tag = self.reasons[reason], tag
        self.flags = self.count = 0
        self.age = self.window
        self.pending_gate, self.mode = False, "failed"

    def push_flag(self, tag, flag):
        """One ordered flag write at this edge; it cannot be read until the next."""
        if self.closed:
            raise RuntimeError("Cannot write after EOF")
        if type(tag) is not int or tag != self.next_tag or type(flag) is not bool:
            raise ValueError("Need the next exact sample tag and a Boolean detector flag")
        if tag + 1 >= 2**63:
            raise OverflowError("Signed64 detector tag exhausted")
        if self.last_push_cycle == self.cycle:
            raise ValueError("One flag write port cannot accept two writes at the same edge")
        visible = unsigned(self.cycle + 1, 63)
        accepted = False
        if self.mode == "failed":
            self._updates(dict(flags_discarded=1))
        elif self.count == self.depth:
            self._fault("flag_overflow", tag, flags_discarded=1)
        else:
            self._updates(dict(flags_accepted=1))
            if not self.count:
                self.head_visible_cycle = visible
            self.flags |= int(flag) << self.count
            self.count += 1
            accepted = True
        self.last_push_cycle, self.next_tag = self.cycle, tag + 1
        return accepted

    @property
    def ready(self):
        return self.mode == "ready" and not self.closed

    @property
    def window_energy_present(self):
        """OR of the compiled window of retired, correctly paired raw-energy flags."""
        return self.age < self.window

    def step(self):
        """Advance the bridge and one real alignment edge under exclusive ownership."""
        if self.closed:
            raise RuntimeError("Cannot clock after EOF")
        edge = unsigned(self.cycle + 1, 63)
        if self.mode != "failed":
            if self.bridge.cycle != self.cycle:
                raise RuntimeError("Bridge clock must be owned by the join")
            self.bridge.step()
        self.cycle = edge
        if self.mode == "failed":
            return
        if self.mode in ("reading", "ready"):
            if not self.bridge.ready or self.bridge.result["tag"] != self.head_tag:
                self._fault("ownership", self.head_tag)
            elif self.mode == "reading" and self.cycle >= self.read_ready_cycle:
                self.mode = "ready"
            elif self.ready:
                self._updates(dict(held_edges=1))
        elif self.bridge.ready:
            tag = self.bridge.result["tag"]
            if self.count and tag != self.head_tag:
                self._fault("word_gap", tag)
            elif not self.count or self.head_visible_cycle > self.cycle:
                self._updates(dict(held_edges=1))
            else:
                deadline = unsigned(self.cycle + 1, 63)
                self._updates(dict(read_requests=1))
                self.pending_gate = bool(self.flags & 1)
                self.read_ready_cycle, self.mode = deadline, "reading"

    def take(self):
        """Atomically retire the held IQ credit and its flag; never duplicate a tag."""
        if not self.ready:
            return None
        if not self.bridge.ready or not self.count or self.bridge.result["tag"] != self.head_tag:
            self._fault("ownership", self.head_tag)
            return None
        updates = self._values(dict(words_joined=1))
        unsigned(self.bridge.stats["retired"] + 1, 64)
        unsigned(self.head_tag + 1, 63)
        new_head_is_fresh = (
            self.count > 1
            and self.head_tag + 1 == self.next_tag - 1
            and self.last_push_cycle == self.cycle
        )
        visible = unsigned(self.cycle + 1, 63) if new_head_is_fresh else self.head_visible_cycle
        result = self.bridge.take()
        self.stats.update(updates)
        self.age = 0 if self.pending_gate else min(self.window, self.age + 1)
        result.update(
            energy_trigger=self.pending_gate, window_energy_present=self.window_energy_present
        )
        self.flags >>= 1
        self.count -= 1
        self.head_tag += 1
        self.head_visible_cycle = visible
        self.pending_gate, self.mode = False, "idle"
        return result

    def finish(self):
        """EOF after draining producers; incomplete identities are explicit failures."""
        if self.closed:
            return
        if self.mode in ("reading", "ready") or self.bridge.mode in ("normalize", "round", "clip"):
            raise RuntimeError("Drain the existing registered service before EOF")
        if self.mode == "failed":
            self._updates(dict(closes=1))
        elif self.bridge.ready and not self.count:
            self._fault("EOF_flag_missing", self.bridge.result["tag"], closes=1)
        elif self.count and self.bridge.mode == "idle":
            self._fault("EOF_word_missing", self.head_tag, closes=1)
        elif self.bridge.ready:
            raise RuntimeError("Drain the available word and flag before EOF")
        else:
            self._updates(dict(closes=1))
        self.closed = True

    def abort(self):
        if not self.closed:
            self._fault("abort", self.head_tag)

    def estimate(self):
        fields = dict(
            flag_FIFO=16,
            trigger_age=self.window.bit_length(),
            head_tag=64,
            next_tag=64,
            count=5,
            mode=2,
            pending_gate=1,
            closed=1,
            clock=64,
            last_push_clock=64,
            head_visible_clock=64,
            read_ready_clock=64,
            fault_tag=64,
            fault_reason=3,
        )
        return dict(
            register_fields=fields,
            counter_bits=8 * 64,
            register_bits_upper=sum(fields.values()) + 8 * 64,
            flag_FIFO_depth=16,
            flag_FIFO_ports=dict(writes=1, registered_reads=1),
            flag_read_edges=1,
            IQ_contexts_added=0,
            IQ_word_held_by_existing_bridge=True,
            new_DSP=0,
            new_RAM_blocks=0,
            new_coefficient_ROM=0,
            operations_per_edge_upper=dict(
                flag_insert_shift16=1,
                flag_pop_shift16=1,
                flag_insert_OR16=1,
                trigger_age_add=1,
                trigger_age_saturating_compare=1,
                trigger_age_window_compare=1,
                tag_adders64=2,
                tag_comparators64=2,
                cycle_adders64=2,
                cycle_comparators64=3,
                count_adders5=2,
                count_comparators5=3,
                overflow_tests64=11,
            ),
            independent_counter_adders64_upper=8,
            composed_unstalled_bridge_II=5,
            scope="alignment only;flag generation and actual scaled-bank cycles remain unknown",
            raw_detector_arithmetic_cycles=None,
            scaled_bank_cycles=None,
            synthesis=False,
            LUT_FF_Fmax=None,
            four_RX_gates_passed=False,
        )
