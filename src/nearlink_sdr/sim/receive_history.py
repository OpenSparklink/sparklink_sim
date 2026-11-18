"""One selected-path46bit history bank with real word/port ownership (§6.2.3).

This is a replay-consumer RAM node. Upstream replay may stall;an ADC cannot.
Interpolation,carrier,head/decoder arithmetic is OUTSIDE this node,not zero cycles.
"""

import math
import operator

__all__ = ["ReceiveHistoryPort"]


class ReceiveHistoryPort:
    """Bounded1R1W bank,registered read1,one held reply and explicit retirement."""

    def __init__(self, capacity=512):
        if type(capacity) is not int or not 4 <= capacity <= 4096 or capacity & (capacity - 1):
            raise ValueError("Need power2 history capacity4..4096")
        self.capacity = capacity
        self.words = [0] * capacity
        self.head = self.end = 0
        self.cycle = self.last_write_cycle = self.last_read_cycle = self.last_take_cycle = -1
        self.last_write_index = -1
        self.pending = None
        self.stats = dict.fromkeys(
            (
                "accepted_words",
                "backpressure",
                "read_requests",
                "read_busy",
                "same_address_deferrals",
                "read_returns",
                "retired_words",
                "peak_live",
            ),
            0,
        )

    def _edge(self, cycle):
        if type(cycle) is not int or not self.cycle <= cycle < 2**63 - 1 or cycle < 0:
            raise ValueError("Need monotonic signed64 clock")
        self.cycle = cycle

    def _count(self, key, n=1):
        if not 0 <= self.stats[key] + n < 2**64:
            raise OverflowError("Finite history counter exhausted")
        self.stats[key] += n

    @property
    def live(self):
        return self.end - self.head

    def write(self, index, word, cycle):
        self._edge(cycle)
        if type(index) is not int or index != self.end or index >= 2**63 - 1:
            raise ValueError("Need next absolute word index")
        if isinstance(word, bool):
            raise ValueError("Need unsigned46bit word")
        try:
            word = operator.index(word)
        except TypeError as error:
            raise ValueError("Need unsigned46bit word") from error
        if not 0 <= word < 2**46:
            raise ValueError("Need unsigned46bit word")
        if self.live == self.capacity or cycle == self.last_write_cycle:
            self._count("backpressure")
            return False
        self.words[index % self.capacity] = word
        self.end += 1
        self.last_write_cycle, self.last_write_index = cycle, index
        self._count("accepted_words")
        self.stats["peak_live"] = max(self.stats["peak_live"], self.live)
        return True

    def request(self, index, cycle):
        self._edge(cycle)
        if type(index) is not int or not self.head <= index < self.end:
            raise ValueError("Requested history is future or expired")
        if self.pending is not None or cycle == self.last_read_cycle:
            self._count("read_busy")
            return False
        if cycle == self.last_write_cycle and index == self.last_write_index:
            self._count("same_address_deferrals")
            return False
        self.pending = (index, self.words[index % self.capacity], cycle + 1)
        self.last_read_cycle = cycle
        self._count("read_requests")
        return True

    def take(self, cycle):
        self._edge(cycle)
        if self.pending is None or cycle < self.pending[2] or cycle == self.last_take_cycle:
            return None
        index, word, _ = self.pending
        self.pending = None
        self.last_take_cycle = cycle
        self._count("read_returns")
        return index, word

    def retire_before(self, index, cycle):
        self._edge(cycle)
        if type(index) is not int or not self.head <= index <= self.end:
            raise ValueError("Need observed retirement boundary")
        if self.pending is not None and self.pending[0] < index:
            raise ValueError("Cannot retire an unconsumed registered reply")
        self._count("retired_words", index - self.head)
        self.head = index

    def estimate(self):
        """Explicit macro mapping/finite-state estimate;not synthesis or full RX."""
        return dict(
            logical_bits=self.capacity * 46,
            capacity_words=self.capacity,
            word_bits=46,
            mapping="RAMB36E1 simple-dual-port72x512;one logical R and W",
            bram36=math.ceil(self.capacity / 512),
            registered_read_latency=1,
            reads_per_clock=1,
            writes_per_clock=1,
            # head/end,5 last-edge/index fields,pending index/ready timestamp:9x64.
            address_timestamp_and_reply_bits=9 * 64 + 46 + 1,
            statistic_counter_bits=len(self.stats) * 64,
            additional_combinational_LUTs=None,
            synthesized=False,
            interpolation_loop_head_decoder_cycles=None,
            full_receiver16msps_verified=False,
        )
