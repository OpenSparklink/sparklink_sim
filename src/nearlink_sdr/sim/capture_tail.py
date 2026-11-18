"""Causal finite filtered-word tail and replay ports (§6.2.3 RX structure).

A coarse admission opens a tail before the fine answer exists. The existing
owned snapshot is borrowed, not copied or overwritten. Replay is serialized
with fine work in the composing controller; no packet/CRC recovery is implied.
"""

import math
import operator
from dataclasses import dataclass

__all__ = ["CaptureTailReplay", "TailReplayAssumptions"]


def _integer(value, lo, hi):
    return type(value) is int and lo <= value < hi


@dataclass(frozen=True)
class TailReplayAssumptions:
    """One46-bit1R1W bank per owner and one registered global read credit."""

    capacity: int = 4096
    owned: int = 3
    paths: int = 2
    consumer_ii: int = 4

    def __post_init__(self):
        if (
            not _integer(self.capacity, 8, 4097)
            or self.capacity & (self.capacity - 1)
            or not _integer(self.owned, 1, 17)
            or self.paths not in (1, 2)
            or type(self.paths) is not int
            or not _integer(self.consumer_ii, 1, 17)
        ):
            raise ValueError("Need power2 tail8..4096,owners1..16,paths1/2,consumerII1..16")


class CaptureTailReplay:
    """Finite live banks, no oracle reads or future-dependent admission.

    The diagnostic sink borrows one accepted word at a time. It does not model
    downstream frame/loop/decoder storage. A live handoff names the next input
    index; downstream continuous processing and its context still need proof.
    """

    def __init__(self, assumptions=None, *, sink=None):
        self.a = TailReplayAssumptions() if assumptions is None else assumptions
        if not isinstance(self.a, TailReplayAssumptions):
            raise ValueError("Need TailReplayAssumptions")
        if sink is not None and not callable(sink):
            raise ValueError("Need a diagnostic callback")
        self.sink = sink
        self.banks = [[0] * self.a.capacity for _ in range(self.a.owned)]
        self.leases = [None] * self.a.owned
        self.input_count = 0
        self.cycle = self.last_input_cycle = self.last_tick_cycle = -1
        self.active = self.prefix = self.pending = None
        self.prefix_index = self.next_issue = 0
        self.finished = False
        self.handoff_index = None
        self.stats = dict.fromkeys(
            (
                "input_words",
                "tail_write_words",
                "overflow_leases",
                "discarded_words",
                "output_words",
                "snapshot_reads",
                "tail_reads",
                "same_address_deferrals",
                "completed_leases",
                "cancelled_leases",
                "peak_owned",
                "peak_tail_words",
            ),
            0,
        )

    def _count(self, key):
        if self.stats[key] == 2**64 - 1:
            raise OverflowError("Finite tail counter exhausted")
        self.stats[key] += 1

    def _lease(self, slot):
        if not _integer(slot, 0, self.a.owned) or self.leases[slot] is None:
            raise ValueError("Need an admitted live owner")
        return self.leases[slot]

    def admit(self, slot, ident, after_samples, path):
        """Called only after successful raw-slot admission at the current edge."""
        if (
            not _integer(slot, 0, self.a.owned)
            or self.leases[slot] is not None
            or not _integer(ident, 0, 2**63)
            or type(after_samples) is not int
            or after_samples != self.input_count
            or not _integer(path, 0, self.a.paths)
        ):
            raise ValueError("Need a free slot and current causal admission count/path")
        self.leases[slot] = dict(
            ident=ident,
            start=after_samples,
            path=path,
            head=after_samples,
            end=after_samples,
            failed=False,
            overflow_index=None,
            last_write=None,
        )
        self.stats["peak_owned"] = max(
            self.stats["peak_owned"], sum(x is not None for x in self.leases)
        )

    def input(self, index, row, cycle):
        """Unstoppable live writes; overflow is sticky and never wraps unread data."""
        if (
            type(index) is not int
            or index != self.input_count
            or not _integer(cycle, 0, 2**63 - 17)
            or cycle <= self.last_input_cycle
            or cycle <= self.last_tick_cycle
            or cycle < self.cycle
            or len(row) != self.a.paths
        ):
            raise ValueError("Need the next live row and monotonic cycle")
        values = []
        for value in row:
            if isinstance(value, bool):
                raise ValueError("Need unsigned46-bit words")
            try:
                value = operator.index(value)
            except TypeError as error:
                raise ValueError("Need unsigned46-bit words") from error
            if not 0 <= value < 2**46:
                raise ValueError("Need unsigned46-bit words")
            values.append(value)
        if self.input_count == 2**63 - 1:
            raise OverflowError("Signed64 input horizon exhausted")
        self.input_count += 1
        self.cycle = self.last_input_cycle = cycle
        self._count("input_words")
        for slot, lease in enumerate(self.leases):
            if lease is None or (self.active == slot and self.finished):
                continue
            if lease["failed"]:
                self._count("discarded_words")
                continue
            if lease["end"] - lease["head"] == self.a.capacity:
                lease["failed"], lease["overflow_index"] = True, index
                self._count("overflow_leases")
                self._count("discarded_words")
                continue
            assert lease["end"] == index
            address = index & (self.a.capacity - 1)
            self.banks[slot][address] = values[lease["path"]]
            lease["end"] += 1
            lease["last_write"] = cycle, address
            self._count("tail_write_words")
            self.stats["peak_tail_words"] = max(
                self.stats["peak_tail_words"], lease["end"] - lease["head"]
            )

    def failed(self, slot):
        return self._lease(slot)["failed"]

    def begin(self, slot, prefix, cycle):
        """Borrow the same owned snapshot after fine has released its read port."""
        lease = self._lease(slot)
        if (
            self.active is not None
            or lease["failed"]
            or not isinstance(prefix, (list, tuple))
            or not 1 <= len(prefix) <= 640
            or len(prefix) > lease["start"]
            or not _integer(cycle, 0, 2**63 - 17)
            or cycle < self.cycle
        ):
            raise ValueError("Need one idle replay,valid snapshot and monotonic cycle")
        self.active, self.prefix = slot, prefix
        self.prefix_index, self.next_issue, self.finished = 0, cycle, False

    def tick(self, cycle, *, blocked=False):
        """Commit one reply,handshake it,then issue at most one registered read."""
        if (
            not _integer(cycle, 0, 2**63 - 17)
            or cycle < self.cycle
            or cycle <= self.last_tick_cycle
            or type(blocked) is not bool
        ):
            raise ValueError("Need monotonic cycle and explicit ready state")
        self.cycle = self.last_tick_cycle = cycle
        if self.active is None or self.finished:
            return
        lease = self._lease(self.active)
        if lease["failed"]:
            self.pending = None
            return
        if self.pending is not None and cycle >= self.pending["ready"] and not blocked:
            word = self.pending
            if self.sink is not None:
                self.sink(
                    dict(
                        event="replay_word",
                        cycle=cycle,
                        id=lease["ident"],
                        path=lease["path"],
                        index=word["index"],
                        value=word["value"],
                        region=word["region"],
                    )
                )
            if word["region"] == "snapshot":
                self.prefix_index += 1
            else:
                assert word["index"] == lease["head"]
                lease["head"] += 1
            self._count("output_words")
            self.pending = None
            self.next_issue = cycle + self.a.consumer_ii - 1
        if (
            self.pending is None
            and not blocked
            and self.prefix_index == len(self.prefix)
            and lease["head"] == lease["end"]
        ):
            self.finished = True
            self.handoff_index = self.input_count
            self._count("completed_leases")
            return
        if self.pending is None and not blocked and cycle >= self.next_issue:
            if self.prefix_index < len(self.prefix):
                value = self.prefix[self.prefix_index]
                if type(value) is not int or not 0 <= value < 2**46:
                    raise ValueError("Owned snapshot word exceeds46bit contract")
                index = lease["start"] - len(self.prefix) + self.prefix_index
                region = "snapshot"
                self._count("snapshot_reads")
            elif lease["head"] < lease["end"]:
                index = lease["head"]
                address = index & (self.a.capacity - 1)
                if lease["last_write"] == (cycle, address):
                    self._count("same_address_deferrals")
                    return
                value, region = self.banks[self.active][address], "tail"
                self._count("tail_reads")
            else:
                self.finished = True
                self.handoff_index = self.input_count
                self._count("completed_leases")
                return
            self.pending = dict(ready=cycle + 1, value=value, index=index, region=region)

    def done(self, slot):
        self._lease(slot)
        return self.active == slot and self.finished

    def report(self, slot):
        lease = self._lease(slot)
        return dict(
            tail_overflow=lease["failed"],
            overflow_index=lease["overflow_index"],
            retained_tail_words=lease["end"] - lease["head"],
            replay_done=self.done(slot),
            next_live_index=self.handoff_index if self.done(slot) else None,
        )

    def close(self, slot):
        """Reset descriptors only; old RAM bits cannot become valid in a new lease."""
        self._lease(slot)
        if not self.done(slot):
            self._count("cancelled_leases")
        self.leases[slot] = None
        if self.active == slot:
            self.active = self.prefix = self.pending = None
            self.finished = False

    def estimate(self):
        """Concrete separate TDP banks; no sharing or new frequency guarantee."""
        fields = dict(
            owner_descriptors=self.a.owned * 512,
            statistics=12 * 64,
            input_and_cycle=256,
            read_pipeline=180,
            replay_control=256,
            input_word_registers=self.a.paths * 46,
            transport_join_timestamps_flags=self.a.owned * 65,
            transport_tail_drop_counter=64,
        )
        return dict(
            assumptions=vars(self.a),
            word_bits=46,
            read_latency=1,
            tail_bram36=2 * math.ceil(self.a.capacity / 1024) * self.a.owned,
            logical_tail_bits=self.a.capacity * self.a.owned * 46,
            register_components=fields,
            register_bits=sum(fields.values()),
            max_global_reads_per_cycle=1,
            max_write_ports_per_owner=1,
            max_input_writes_all_owners=self.a.owned,
            primitive_writes_per_stored_word=2,
            raw_snapshot_copies=0,
            extra_DSP=0,
            ADC_backpressure=False,
            fine_replay_overlap_credit=False,
            output_FIFO_words=1,
            downstream_context_and_frame_RAM_included=False,
            whole_receiver16msps=False,
            synthesis_or_timing_evidence=False,
        )
