"""Bounded, sample-exact history/window transport (§6.2.1.2 receiver model).

This is a memory-port calendar, not a fine-search implementation. Fine service
is an explicit reservation; spline/decoder ports and closed-loop feedback remain
unproved. Input words already have their chosen representation: no quantization,
gain, candidate selection or CRC decisions occur here.
"""

import math
import operator
from collections import deque
from dataclasses import dataclass

__all__ = ["CaptureTransport", "CaptureTransportAssumptions"]


@dataclass(frozen=True)
class CaptureTransportAssumptions:
    """Single-clock synchronous RAM, one-cycle read, input-write priority.

    sdp1: one72-bit SDP read/write per path. tdp2: two36-bit TDP slices
    across a46-bit word, up to two reads when no input write needs a port.
    copied8: eight SDP source copies and eight destination write banks.
    Destinations are owned until fine/result release; no write/read sharing
    credit between different owners. All clocks here are design assumptions.
    """

    clock_hz: int = 200_000_000
    input_hz: int = 16_000_000
    mode: str = "tdp2"
    paths: int = 2
    history: int = 384
    window: int = 256
    word_bits: int = 46
    owned: int = 3
    fine_service_cycles: int = 469

    def __post_init__(self):
        if (any(type(v) is not int or v < 1 for k, v in vars(self).items() if k != "mode")
                or self.mode not in ("sdp1", "tdp2", "copied8")
                or self.input_hz > self.clock_hz or self.paths not in (1, 2)
                or not 256 <= self.history <= 4096 or self.window != 256
                or self.owned > 16 or self.word_bits > 72
                or self.fine_service_cycles >= 2**31):
            raise ValueError("Invalid bounded history/window transport assumptions")


class CaptureTransport:
    """FCFS copy and fine reservations, drop-new on full, oldest-first copying.

    `sink(event, words)` borrows the owned array until it returns. Diagnostics
    retained by a caller do not belong to the modeled hardware. Source access
    checks use absolute counters, not a hidden per-word generation RAM. All
    input paths advance continuously even during fine/result stalls.
    """

    def __init__(self, assumptions=None):
        self.a = CaptureTransportAssumptions() if assumptions is None else assumptions
        if not isinstance(self.a, CaptureTransportAssumptions):
            raise ValueError("Need CaptureTransportAssumptions")

    def estimate(self):
        """Concrete uniform primitive layouts, not a synthesized minimum.

        An SDP72x512 is distinct from a TDP36x1024. Widths include parity lanes.
        Destination banking supplies the copy writes only; fine-search accesses
        are deliberately NOT claimed satisfied by these bank arrangements.
        """
        a = self.a
        lanes = 8 if a.mode == "copied8" else 2 if a.mode == "tdp2" else 1
        width, depth = (36, 1024) if a.mode == "tdp2" else (72, 512)
        source_copies = 8 if a.mode == "copied8" else 1
        destination_banks = 8 if a.mode == "copied8" else 1
        source_tiles = a.paths * source_copies * math.ceil(a.word_bits/width)
        source_tiles *= math.ceil(a.history/depth)
        destination_tiles = a.owned * destination_banks * math.ceil(a.word_bits/width)
        destination_tiles *= math.ceil(math.ceil(a.window/destination_banks)/depth)
        # Explicit finite reservation: absolute/cycle counts, descriptors,
        # one-cycle read data+destination index, control and ten64-bit counters.
        # id/count/arrival/copied/finish can coexist: five64-bit timestamps,
        # plus index/path/slot/flags. Do not reuse the old256-bit descriptor
        # allowance after adding the explicit copy/result lifetime calendar.
        descriptor_fields = dict(id_count_and_three_cycles=5*64, next_index=9,
                                 slot=4, path=1, blocked=1, valid_and_stage=3)
        registers = dict(absolute_and_cycle_counters=128, descriptors=a.owned*384,
                         read_pipeline=lanes*(a.word_bits+8+4+1),
                         copy_control=256, statistics_counters_and_extrema=17*64)
        return dict(mode=a.mode, source_bram36=source_tiles,
                    destination_bram36=destination_tiles,
                    transport_bram36=source_tiles+destination_tiles,
                    source_copies=source_copies, destination_write_banks=destination_banks,
                    primitive_width=width, primitive_depth=depth,
                    max_source_reads_per_cycle=lanes,
                    source_input_writes_per_sample=a.paths,
                    input_writes_broadcast_per_sample=a.paths*source_copies,
                    primitive_input_writes_per_sample=a.paths*source_copies
                    * math.ceil(a.word_bits/width),
                    primitive_reads_per_logical_read=math.ceil(a.word_bits/width),
                    primitive_writes_per_logical_destination_write=math.ceil(a.word_bits/width),
                    max_destination_writes_per_cycle=lanes,
                    logical_sample_bits=(a.paths*a.history+a.owned*a.window)*a.word_bits,
                    register_reservation_bits=registers,
                    descriptor_fields_bits=descriptor_fields,
                    descriptor_reserved_bits_each=384,
                    register_reservation_total_bits=sum(registers.values()),
                    extra_dsp=0, lut_ff_synthesis=None,
                    nonlinear_data_operations=0,
                    control_cost="absolute counters/expiry compare, constant-wrap pointers, admission/FCFS control; width/timing synthesis pending",
                    other_source_readers_permitted=False,
                    fine_memory_ports_verified=False, full_chain_verified=False,
                    interpretation="concrete transport allocation; excludes all fine scratch/reads, RRC, banks, decoder, TX, MAC and interface buffers")

    def simulate(self, words, requests, *, stalls=(), sink=None):
        """Stream fixed-width path tuples and sorted(after_sample_count,path).

        At each edge: release old results, commit prior reads, write newly
        arrived input, admit requests, start ready fine service, issue reads.
        A copied window becomes fine-readable only on the following edge.
        Stalls hold completed results, not the unstoppable input. At EOF no
        artificial samples are written; pending windows drain from real history.
        """
        a = self.a
        n = len(words)
        if type(n) is not int or n < 1 or (n*a.clock_hz+a.input_hz-1)//a.input_hz >= 2**63:
            raise ValueError("Need finite nonempty input inside signed64 cycle range")
        if not isinstance(stalls, (tuple, list)) or len(stalls) > 1024:
            raise ValueError("Need bounded external stall intervals")
        previous = 0
        for lo, hi in stalls:
            if (type(lo) is not int or type(hi) is not int or not previous <= lo < hi
                    or hi >= 2**63):
                raise ValueError("Need sorted disjoint nonnegative cycle stalls")
            previous = hi
        # Leave headroom for eight-lane access counters and the finite drain.
        # Very long experiments fail explicitly rather than wrap timestamps.
        tail = (a.owned+1)*(a.window+a.fine_service_cycles+2)
        if max(n*a.clock_hz//a.input_hz, previous)+tail >= 2**60:
            raise OverflowError("Transport timestamp/access-counter horizon exhausted")
        source = [[0]*a.history for _ in range(a.paths)]
        slots = [[0]*a.window for _ in range(a.owned)]
        free, copying, ready = deque(range(a.owned)), deque(), deque()
        active, pipeline = None, []
        stream = iter(requests)
        pending, last_request, ident = next(stream, None), 0, 0
        count = cycle = stall_index = 0
        stats = dict(requests=0, completed=0, queue_drops=0, history_drops=0, startup_drops=0,
                     input_samples=0, source_reads=0, destination_writes=0,
                     input_writes=0, completed_while_blocked=0, peak_owned=0)
        max_latency = max_copy_latency = peak_source_ports = peak_destination_ports = 0
        read_write_deferrals = 0
        minimum_slack = None

        def notify(job, status, data=None):
            if sink is not None:
                sink(dict(id=job["id"], after_samples=job["count"], path=job["path"],
                          offered_cycle=job["arrival"], cycle=cycle, status=status,
                          copied_cycle=job.get("copied")), data)

        while count < n or pending is not None or copying or ready or active or pipeline:
            while stall_index < len(stalls) and cycle >= stalls[stall_index][1]:
                stall_index += 1
            blocked = (stall_index < len(stalls)
                       and stalls[stall_index][0] <= cycle < stalls[stall_index][1])
            if active is not None and cycle >= active["finish"]:
                if not blocked:
                    stats["completed"] += 1
                    max_latency = max(max_latency, cycle-active["arrival"])
                    notify(active, "completed", slots[active["slot"]])
                    free.append(active["slot"])
                    active = None
                elif not active.get("blocked", False):
                    active["blocked"] = True
                    stats["completed_while_blocked"] += 1
            writes = len(pipeline)
            peak_destination_ports = max(peak_destination_ports,
                int(writes > 0) if a.mode == "copied8" else writes)
            for job, index, value in pipeline:
                slots[job["slot"]][index] = value
                stats["destination_writes"] += 1
                if index == a.window-1:
                    job["copied"] = cycle
                    ready.append(job)
                    max_copy_latency = max(max_copy_latency, cycle-job["arrival"])
            pipeline = []
            input_write = False
            if count < n and cycle == (count*a.clock_hz+a.input_hz-1)//a.input_hz:
                row = words[count]
                if len(row) != a.paths:
                    raise ValueError("Every input word must include every continuously advanced path")
                for path, value in enumerate(row):
                    if isinstance(value, bool):
                        raise ValueError("Need unsigned fixed-width words, not booleans")
                    try:
                        value = operator.index(value)
                    except TypeError as e:
                        raise ValueError("Need integer words without conversion") from e
                    if not 0 <= value < 2**a.word_bits:
                        raise ValueError("Input word outside frozen format")
                    source[path][count % a.history] = value
                count += 1
                input_write = True
                stats["input_samples"] += 1
                stats["input_writes"] += a.paths
            while pending is not None:
                requested, path = pending
                if (type(requested) is not int or type(path) is not int
                        or not 1 <= requested <= n or requested < last_request
                        or not 0 <= path < a.paths):
                    raise ValueError("Need sorted in-input request counts and actual path")
                if requested > count:
                    break
                job = dict(id=ident, count=requested, path=path, arrival=cycle, next=0)
                if ident >= 2**63:
                    raise OverflowError("Signed64 request counter exhausted")
                ident += 1
                last_request = requested
                stats["requests"] += 1
                if requested < a.window:
                    stats["startup_drops"] += 1
                    notify(job, "incomplete_history")
                elif not free:
                    stats["queue_drops"] += 1
                    notify(job, "owned_queue_full")
                else:
                    job["slot"] = free.popleft()
                    copying.append(job)
                    stats["peak_owned"] = max(stats["peak_owned"], a.owned-len(free))
                pending = next(stream, None)
            if active is None and ready and ready[0]["copied"] < cycle:
                active = ready.popleft()
                active["finish"] = cycle+a.fine_service_cycles
            if copying:
                job = copying[0]
                lanes = 8 if a.mode == "copied8" else (2-int(input_write)
                                                       if a.mode == "tdp2" else 1)
                for _ in range(lanes):
                    index = job["count"]-a.window+job["next"]
                    if index < count-a.history:
                        stats["history_drops"] += 1
                        notify(job, "source_expired")
                        free.append(job["slot"])
                        copying.popleft()
                        break
                    # Exclude synchronous read/write same-address behavior.
                    if input_write and index % a.history == (count-1) % a.history:
                        read_write_deferrals += 1
                        break
                    slack = a.history-(count-index)
                    minimum_slack = slack if minimum_slack is None else min(minimum_slack, slack)
                    pipeline.append((job, job["next"], source[job["path"]][index % a.history]))
                    stats["source_reads"] += 1
                    job["next"] += 1
                    if job["next"] == a.window:
                        copying.popleft()
                        break
            reads_per_copy = (int(bool(pipeline)) if a.mode == "copied8" else len(pipeline))
            peak_source_ports = max(peak_source_ports, reads_per_copy+int(input_write))
            assert peak_source_ports <= 2 and peak_destination_ports <= (
                2 if a.mode == "tdp2" else 1)
            cycle += 1
        return dict(**stats, max_latency_cycles=max_latency,
                    max_copy_latency_cycles=max_copy_latency,
                    min_read_expiry_slack_samples=minimum_slack,
                    peak_source_operations_per_primitive=peak_source_ports,
                    peak_destination_writes_per_bank=peak_destination_ports,
                    read_write_same_address_deferrals=read_write_deferrals,
                    drain_cycle=cycle, source_state_words=a.paths*a.history,
                    owned_state_words=a.owned*a.window,
                    adc_backpressure=False, pending_descriptors=0,
                    fine_memory_ports_verified=False, closed_loop_capture_verified=False,
                    whole_system_accepted=False)
