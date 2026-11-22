"""Bounded, sample-exact history/window transport (§6.2.1.2 receiver model).

Fine service is an explicit reservation or a bounded measured-microtrace worker.
The immutable raw-slot handoff does not prove downstream replay/decoder ports
or closed-loop acquisition. Input words already have their chosen representation:
no quantization,gain,candidate selection or CRC decisions occur in transport.
"""

import math
import operator
from collections import deque
from dataclasses import dataclass
from itertools import pairwise

from nearlink_sdr.sim.capture_tail import CaptureTailReplay

__all__ = [
    "CaptureRequest",
    "CaptureTransport",
    "CaptureTransportAssumptions",
    "FineService",
    "ReleasedCaptureRequest",
]


@dataclass(frozen=True)
class CaptureRequest:
    """Finite upstream search descriptor; contains no expected bits or CRC."""

    after_samples: int
    path: int
    peaks: tuple[int, ...]
    minimum_tick: int = 0
    lookahead_ticks: int = 128

    def __post_init__(self):
        if (
            type(self.after_samples) is not int
            or not 1 <= self.after_samples < 2**63
            or type(self.path) is not int
            or self.path not in (0, 1)
            or not isinstance(self.peaks, tuple)
            or len(self.peaks) > 2
            or any(type(p) is not int or not 0 <= p < 640 for p in self.peaks)
            or any(abs(a - b) < 4 for a, b in pairwise(self.peaks))
            or type(self.minimum_tick) is not int
            or not 0 <= self.minimum_tick <= 5120
            or type(self.lookahead_ticks) is not int
            or not 0 <= self.lookahead_ticks <= 32768
        ):
            raise ValueError("Need bounded capture request geometry")


@dataclass(frozen=True, kw_only=True)
class ReleasedCaptureRequest(CaptureRequest):
    """Original source endpoint plus real candidate availability on this clock.

    The endpoint never advances with a delayed response. This is an explicit
    experimental transport port,not a new synchronization or retry strategy.
    """

    available_cycle: int

    def __post_init__(self):
        super().__post_init__()
        if type(self.available_cycle) is not int or not 0 <= self.available_cycle < 2**60:
            raise ValueError("Need bounded signed63 candidate availability")


@dataclass(frozen=True)
class FineService:
    """Bounded measured microtrace result, never a CRC-selected packet answer.

    Scalar metadata only: the caller keeps diagnostic traces outside hardware.
    The selected time is in eighth-sample ticks within the owned window.
    geometry_wait has no selected time or evaluated score;it is not a capture.
    """

    cycles: int
    disposition: str
    start_tick: int | None = None

    def __post_init__(self):
        if (
            type(self.cycles) is not int
            or not 1 <= self.cycles <= 2_000_000
            or type(self.disposition) is not str
            or self.disposition
            not in (
                "ready",
                "confirm_pending",
                "needs_future",
                "below_threshold",
                "no_candidate",
                "numerical_reject",
                "geometry_wait",
            )
            or (
                self.start_tick is not None
                and (type(self.start_tick) is not int or not 0 <= self.start_tick < 8192)
            )
            or (
                (self.start_tick is None)
                != (self.disposition in ("no_candidate", "numerical_reject", "geometry_wait"))
            )
        ):
            raise ValueError("Need bounded fine cycles,disposition and selected tick")


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
        if (
            any(type(v) is not int or v < 1 for k, v in vars(self).items() if k != "mode")
            or self.mode not in ("sdp1", "tdp2", "copied8")
            or self.input_hz > self.clock_hz
            or self.paths not in (1, 2)
            or not 256 <= self.history <= 4096
            or self.window not in (256, 376, 632)
            or self.history < self.window
            or self.owned > 16
            or self.word_bits > 72
            or self.fine_service_cycles >= 2**31
        ):
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

    def estimate(self, *, measured_worker=False, released_requests=False):
        """Concrete uniform primitive layouts, not a synthesized minimum.

        An SDP72x512 is distinct from a TDP36x1024. Widths include parity lanes.
        Destination banking supplies the copy writes only; fine-search accesses
        are deliberately NOT claimed satisfied by these bank arrangements.
        """
        if type(measured_worker) is not bool:
            raise ValueError("Need explicit measured-worker cost mode")
        if type(released_requests) is not bool:
            raise ValueError("Need explicit released-request cost mode")
        a = self.a
        lanes = 8 if a.mode == "copied8" else 2 if a.mode == "tdp2" else 1
        width, depth = (36, 1024) if a.mode == "tdp2" else (72, 512)
        source_copies = 8 if a.mode == "copied8" else 1
        destination_banks = 8 if a.mode == "copied8" else 1
        source_tiles = a.paths * source_copies * math.ceil(a.word_bits / width)
        source_tiles *= math.ceil(a.history / depth)
        destination_tiles = a.owned * destination_banks * math.ceil(a.word_bits / width)
        destination_tiles *= math.ceil(math.ceil(a.window / destination_banks) / depth)
        # Explicit finite reservation: absolute/cycle counts, descriptors,
        # one-cycle read data+destination index,control and explicit64-bit counters.
        # id/count/arrival/copied/finish can coexist: five64-bit timestamps,
        # plus index/path/slot/flags. Do not reuse the old256-bit descriptor
        # allowance after adding the explicit copy/result lifetime calendar.
        index_bits = math.ceil(math.log2(a.window))
        descriptor_bits = 448 if measured_worker else 384
        descriptor_fields = dict(
            id_count_and_three_cycles=5 * 64,
            next_index=math.ceil(math.log2(a.window + 1)),
            slot=4,
            path=1,
            blocked=1,
            valid_and_stage=3,
        )
        if measured_worker:
            descriptor_fields.update(fine_cycles=32, selected_tick=13, disposition_and_valid=4)
            descriptor_fields.update(fine_geometry=2 * 10 + 13 + 16 + 2)
        registers = dict(
            absolute_and_cycle_counters=128,
            descriptors=a.owned * descriptor_bits,
            read_pipeline=lanes * (a.word_bits + index_bits + 4 + 1),
            copy_control=256,
            statistics_counters_and_extrema=17 * 64,
        )
        if measured_worker:
            registers.update(worker_join_and_retention_control=160, worker_statistics=11 * 64)
        if released_requests:
            # One pending input descriptor,not another owned sample window.
            # The release timestamp dies at admission;job arrival is already
            # paid by the existing owned descriptor. Keep the original endpoint.
            registers["released_request_port"] = 63 + 63 + 1 + 2 + 20 + 13 + 16 + 1 + 63
        cost = dict(
            mode=a.mode,
            source_bram36=source_tiles,
            destination_bram36=destination_tiles,
            transport_bram36=source_tiles + destination_tiles,
            source_copies=source_copies,
            destination_write_banks=destination_banks,
            primitive_width=width,
            primitive_depth=depth,
            max_source_reads_per_cycle=lanes,
            source_input_writes_per_sample=a.paths,
            input_writes_broadcast_per_sample=a.paths * source_copies,
            primitive_input_writes_per_sample=a.paths
            * source_copies
            * math.ceil(a.word_bits / width),
            primitive_reads_per_logical_read=math.ceil(a.word_bits / width),
            primitive_writes_per_logical_destination_write=math.ceil(a.word_bits / width),
            max_destination_writes_per_cycle=lanes,
            logical_sample_bits=(a.paths * a.history + a.owned * a.window) * a.word_bits,
            register_reservation_bits=registers,
            descriptor_fields_bits=descriptor_fields,
            descriptor_reserved_bits_each=descriptor_bits,
            register_reservation_total_bits=sum(registers.values()),
            extra_dsp=0,
            lut_ff_synthesis=None,
            nonlinear_data_operations=0,
            control_cost=(
                "absolute counters/expiry compare, constant-wrap pointers, "
                "admission/FCFS control; width/timing synthesis pending"
            ),
            other_source_readers_permitted=False,
            fine_memory_ports_verified=False,
            full_chain_verified=False,
            interpretation=(
                "concrete transport allocation; excludes all fine scratch/reads, RRC, "
                "banks, decoder, TX, MAC and interface buffers"
            ),
        )
        if released_requests:
            cost["released_request_port"] = dict(
                credits=1,
                register_bits=registers["released_request_port"],
                research_cap_bits=256,
                source_window_end_bits=63,
                available_cycle_bits=63,
                last_available_cycle_bits=63,
                geometry_path_and_valid_bits=53,
                unsigned63_comparisons=2,
                existing_request_counter_nonzero_test=True,
                finite_input_horizon_and_source_readiness_preflight_only=True,
                same_primitive_ports=True,
                extra_IQ_RAM_or_DSP=0,
                owned_descriptor_timestamp_duplicated=False,
                physical_timing_verified=False,
            )
        return cost

    def simulate(
        self,
        words,
        requests,
        *,
        stalls=(),
        sink=None,
        fine_worker=None,
        tail_replay=None,
        released_requests=False,
        source_owner=None,
    ):
        """Stream fixed-width path tuples and sorted(after_sample_count,path).

        At each edge: release old results, commit prior reads, write newly
        arrived input, admit requests, start ready fine service, issue reads.
        A copied window becomes fine-readable only on the following edge.
        Stalls hold completed results, not the unstoppable input. At EOF no
        artificial samples are written; pending windows drain from real history.
        """
        if fine_worker is not None and not callable(fine_worker):
            raise ValueError("Need a callable bounded fine microtrace provider")
        if type(released_requests) is not bool:
            raise ValueError("Need explicit released-request mode")
        a = self.a
        if source_owner is not None:
            from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource

            if (not isinstance(source_owner, CoarseCaptureSource)
                    or words is not None or requests != () or not released_requests
                    or a.paths != 1 or a.word_bits != 46 or tail_replay is not None
                    or source_owner.cycle != 0 or source_owner.flow.core.window != a.window):
                raise ValueError(
                    "Need fresh matched one-path live coarse source and no static data"
                )
        if tail_replay is not None and (
            not isinstance(tail_replay, CaptureTailReplay)
            or tail_replay.a.owned != a.owned
            or tail_replay.a.paths != a.paths
            or a.word_bits != 46
            or a.mode != "tdp2"
            or fine_worker is None
            or tail_replay.input_count != 0
            or tail_replay.cycle != -1
        ):
            raise ValueError("Need fresh matched46bit TDP tail replay and measured fine worker")
        n = len(words) if source_owner is None else source_owner.sample_limit
        if type(n) is not int or n < 1 or (n * a.clock_hz + a.input_hz - 1) // a.input_hz >= 2**63:
            raise ValueError("Need finite nonempty input inside signed64 cycle range")
        if not isinstance(stalls, (tuple, list)) or len(stalls) > 1024:
            raise ValueError("Need bounded external stall intervals")
        previous = 0
        for lo, hi in stalls:
            if type(lo) is not int or type(hi) is not int or not previous <= lo < hi or hi >= 2**63:
                raise ValueError("Need sorted disjoint nonnegative cycle stalls")
            previous = hi
        # Leave headroom for eight-lane access counters and the finite drain.
        # Very long experiments fail explicitly rather than wrap timestamps.
        service_bound = a.fine_service_cycles if fine_worker is None else 2_000_000
        tail = (a.owned + 1) * (a.window + service_bound + 2)
        if tail_replay is not None:
            tail += (a.owned + 1) * (a.window + tail_replay.a.capacity + 3)
            tail *= tail_replay.a.consumer_ii
        source_horizon = (n * a.clock_hz // a.input_hz if source_owner is None
                          else source_owner.source_horizon)
        if max(source_horizon, previous) + tail >= 2**60:
            raise OverflowError("Transport timestamp/access-counter horizon exhausted")
        source = [[0] * a.history for _ in range(a.paths)]
        slots = [[0] * a.window for _ in range(a.owned)]
        free, copying, ready = deque(range(a.owned)), deque(), deque()
        active, pipeline = None, []
        stream = iter(requests)
        pending, last_request, ident = next(stream, None), 0, 0
        last_available = 0
        count = cycle = stall_index = 0
        stats = dict(
            requests=0,
            completed=0,
            queue_drops=0,
            history_drops=0,
            startup_drops=0,
            input_samples=0,
            source_reads=0,
            destination_writes=0,
            input_writes=0,
            completed_while_blocked=0,
            peak_owned=0,
        )
        source_pulse, source_epoch_aborted = None, False
        if source_owner is not None:
            stats.update(source_epoch_drops=0, source_invalid_pulses=0, source_gap_samples=0)
        if tail_replay is not None:
            stats["tail_drops"] = 0
        max_latency = max_copy_latency = peak_source_ports = peak_destination_ports = 0
        read_write_deferrals = 0
        minimum_slack = None

        worker_stats = dict(
            ready=0,
            confirm_pending=0,
            needs_future=0,
            below_threshold=0,
            no_candidate=0,
            numerical_reject=0,
            geometry_wait=0,
            ready_retention_lost=0,
            ready_retention_necessary_pass=0,
            confirm_pending_retention_lost=0,
            confirm_pending_retention_necessary_pass=0,
        )

        if tail_replay is not None:
            worker_stats = {k: v for k, v in worker_stats.items() if "retention" not in k}

        def notify(job, status, data=None):
            event = dict(
                id=job["id"],
                after_samples=job["count"],
                path=job["path"],
                offered_cycle=job["arrival"],
                cycle=cycle,
                status=status,
                copied_cycle=job.get("copied"),
            )
            if fine_worker is not None:
                # Completion precedes this edge's unstoppable input write.
                # Include that write for the optimistic retention-only check.
                at_edge = (
                    max(count, source_pulse[0] + 1) if source_pulse is not None else count
                ) if source_owner is not None else count + int(
                    count < n and cycle == (count * a.clock_hz + a.input_hz - 1) // a.input_hz
                )
                service = job.get("service")
                gap = (
                    max(0, at_edge - a.history - job["count"])
                    if service is not None and service.disposition in ("ready", "confirm_pending")
                    else None
                )
                event.update(
                    continuous_count_after_edge=at_edge,
                    fine_cycles=None if service is None else service.cycles,
                    fine_started_cycle=None if service is None else job["finish"] - service.cycles,
                    disposition=None if service is None else service.disposition,
                    start_tick=None if service is None else service.start_tick,
                    continuation_gap_samples=gap,
                    retention_necessary_pass=None if gap is None else gap == 0,
                    downstream_replay_ports_verified=False,
                )
                if status == "completed":
                    worker_stats[service.disposition] += 1
                    if gap is not None and tail_replay is None:
                        prefix = service.disposition + "_retention_"
                        worker_stats[prefix + ("lost" if gap else "necessary_pass")] += 1
            if tail_replay is not None:
                event.update(
                    source_ring_only_gap_samples=event.get("continuation_gap_samples"),
                    continuation_gap_samples=None,
                    retention_necessary_pass=None,
                    tail=None if "slot" not in job else tail_replay.report(job["slot"]),
                    replay_started_cycle=job.get("replay_started"),
                    replay_owns_raw_until_release=True,
                    complete_frame_or_MAC_verified=False,
                )
            if sink is not None:
                sink(event, data)

        while ((count < n if source_owner is None else not source_owner.done)
               or pending is not None or copying or ready or active or pipeline):
            if source_owner is not None:
                source_pulse, request = source_owner.advance(cycle)
                if pending is not None:
                    raise RuntimeError("Live source descriptor outlived its single-edge port")
                pending = request
                if source_owner.failed and not source_epoch_aborted:
                    # At most owned slots;include last-copy replies not yet ready.
                    live = {job["id"]: job for job in (*copying, *ready)}
                    if active is not None:
                        live[active["id"]] = active
                    for job, _, _ in pipeline:
                        live[job["id"]] = job
                    for job in live.values():
                        notify(job, "source_epoch_aborted")
                    stats["source_epoch_drops"] += len(live)
                    copying.clear()
                    ready.clear()
                    active, pipeline = None, []
                    free = deque(range(a.owned))
                    source_epoch_aborted = True
            while stall_index < len(stalls) and cycle >= stalls[stall_index][1]:
                stall_index += 1
            blocked = (
                stall_index < len(stalls)
                and stalls[stall_index][0] <= cycle < stalls[stall_index][1]
            )
            if active is not None and cycle >= active["finish"]:
                replay_needed = tail_replay is not None and active["service"].disposition in (
                    "ready",
                    "confirm_pending",
                )
                if replay_needed:
                    if (
                        not blocked
                        and "replay_started" not in active
                        and not tail_replay.failed(active["slot"])
                    ):
                        tail_replay.begin(active["slot"], slots[active["slot"]], cycle)
                        active["replay_started"] = cycle
                elif not blocked:
                    stats["completed"] += 1
                    max_latency = max(max_latency, cycle - active["arrival"])
                    notify(active, "completed", slots[active["slot"]])
                    if tail_replay is not None:
                        tail_replay.close(active["slot"])
                    free.append(active["slot"])
                    active = None
                if active is not None and blocked and not active.get("blocked", False):
                    active["blocked"] = True
                    stats["completed_while_blocked"] += 1
            writes = len(pipeline)
            peak_destination_ports = max(
                peak_destination_ports, int(writes > 0) if a.mode == "copied8" else writes
            )
            for job, index, value in pipeline:
                slots[job["slot"]][index] = value
                stats["destination_writes"] += 1
                if index == a.window - 1:
                    job["copied"] = cycle
                    ready.append(job)
                    max_copy_latency = max(max_copy_latency, cycle - job["arrival"])
            pipeline = []
            input_write = False
            input_arrived = (
                source_pulse is not None if source_owner is not None else
                count < n and cycle == (count * a.clock_hz + a.input_hz - 1) // a.input_hz
            )
            if input_arrived:
                if source_owner is not None:
                    tag, value = source_pulse
                    stats["source_gap_samples"] += tag - count
                    count = tag
                    if value is None:
                        stats["source_invalid_pulses"] += 1
                        count += 1
                        stats["input_samples"] += 1
                        cycle += 1
                        continue  # Fault has cancelled every owner;never write a fake word.
                    row = (value,)
                else:
                    row = words[count]
                if len(row) != a.paths:
                    raise ValueError(
                        "Every input word must include every continuously advanced path"
                    )
                live_values = []
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
                    live_values.append(value)
                if tail_replay is not None:
                    tail_replay.input(count, live_values, cycle)
                count += 1
                input_write = True
                stats["input_samples"] += 1
                stats["input_writes"] += a.paths
            while pending is not None:
                if isinstance(pending, ReleasedCaptureRequest) != released_requests:
                    raise ValueError("Released descriptors require their explicit matching port")
                geometry = pending if isinstance(pending, CaptureRequest) else None
                requested, path = (
                    (geometry.after_samples, geometry.path) if geometry is not None else pending
                )
                if (
                    type(requested) is not int
                    or type(path) is not int
                    or not 1 <= requested <= n
                    or requested < last_request
                    or not 0 <= path < a.paths
                ):
                    raise ValueError("Need sorted in-input request counts and actual path")
                if released_requests and (
                    (ident > 0 and pending.available_cycle <= last_available)
                    or pending.available_cycle
                    < ((requested - 1) * a.clock_hz + a.input_hz - 1) // a.input_hz
                    or pending.available_cycle
                    > (n * a.clock_hz + a.input_hz - 1) // a.input_hz + tail
                ):
                    raise ValueError(
                        "Need ordered release times,one per edge,with source already available "
                        "inside the finite input/drain horizon"
                    )
                if requested > count or (released_requests and cycle < pending.available_cycle):
                    break
                job = dict(id=ident, count=requested, path=path, arrival=cycle, next=0)
                if geometry is not None:
                    if (
                        any(p >= a.window for p in geometry.peaks)
                        or geometry.minimum_tick > 8 * a.window
                    ):
                        raise ValueError("Capture geometry exceeds the actual window")
                    if fine_worker is not None:
                        job["geometry"] = (
                            CaptureRequest(
                                requested,
                                path,
                                geometry.peaks,
                                geometry.minimum_tick,
                                geometry.lookahead_ticks,
                            )
                            if released_requests
                            else geometry
                        )
                if ident >= 2**63:
                    raise OverflowError("Signed64 request counter exhausted")
                ident += 1
                last_request = requested
                if released_requests:
                    last_available = pending.available_cycle
                stats["requests"] += 1
                if requested < a.window:
                    stats["startup_drops"] += 1
                    notify(job, "incomplete_history")
                elif not free:
                    stats["queue_drops"] += 1
                    notify(job, "owned_queue_full")
                else:
                    job["slot"] = free.popleft()
                    if tail_replay is not None:
                        tail_replay.admit(job["slot"], ident - 1, requested, path)
                    copying.append(job)
                    stats["peak_owned"] = max(stats["peak_owned"], a.owned - len(free))
                pending = next(stream, None) if source_owner is None else None
            if active is None and ready and ready[0]["copied"] < cycle:
                active = ready.popleft()
                service_cycles = a.fine_service_cycles
                if fine_worker is not None:
                    service = fine_worker(
                        dict(
                            id=active["id"],
                            after_samples=active["count"],
                            path=active["path"],
                            window_samples=a.window,
                            geometry=active.get("geometry"),
                        ),
                        slots[active["slot"]],
                    )
                    if not isinstance(service, FineService):
                        raise ValueError("Fine worker must return a bounded FineService")
                    if service.start_tick is not None and service.start_tick >= 8 * a.window:
                        raise ValueError("Fine selected tick exceeds its owned window")
                    active["service"] = service
                    service_cycles = service.cycles
                active["finish"] = cycle + service_cycles
            if copying:
                job = copying[0]
                lanes = (
                    8 if a.mode == "copied8" else (2 - int(input_write) if a.mode == "tdp2" else 1)
                )
                for _ in range(lanes):
                    index = job["count"] - a.window + job["next"]
                    if index < count - a.history:
                        stats["history_drops"] += 1
                        notify(job, "source_expired")
                        if tail_replay is not None:
                            tail_replay.close(job["slot"])
                        free.append(job["slot"])
                        copying.popleft()
                        break
                    # Exclude synchronous read/write same-address behavior.
                    if input_write and index % a.history == (count - 1) % a.history:
                        read_write_deferrals += 1
                        break
                    slack = a.history - (count - index)
                    minimum_slack = slack if minimum_slack is None else min(minimum_slack, slack)
                    pipeline.append((job, job["next"], source[job["path"]][index % a.history]))
                    stats["source_reads"] += 1
                    job["next"] += 1
                    if job["next"] == a.window:
                        copying.popleft()
                        break
            reads_per_copy = int(bool(pipeline)) if a.mode == "copied8" else len(pipeline)
            peak_source_ports = max(peak_source_ports, reads_per_copy + int(input_write))
            assert peak_source_ports <= 2 and peak_destination_ports <= (
                2 if a.mode == "tdp2" else 1
            )
            if tail_replay is not None:
                tail_replay.tick(cycle, blocked=blocked)
                if (
                    active is not None
                    and cycle >= active["finish"]
                    and not blocked
                    and active["service"].disposition in ("ready", "confirm_pending")
                ):
                    slot = active["slot"]
                    if tail_replay.failed(slot) or tail_replay.done(slot):
                        status = (
                            "tail_capacity_exceeded" if tail_replay.failed(slot) else "completed"
                        )
                        stats["tail_drops" if tail_replay.failed(slot) else "completed"] += 1
                        max_latency = max(max_latency, cycle - active["arrival"])
                        notify(active, status, slots[slot])
                        tail_replay.close(slot)
                        free.append(slot)
                        active = None
            cycle += 1
        result = dict(
            **stats,
            max_latency_cycles=max_latency,
            max_copy_latency_cycles=max_copy_latency,
            min_read_expiry_slack_samples=minimum_slack,
            peak_source_operations_per_primitive=peak_source_ports,
            peak_destination_writes_per_bank=peak_destination_ports,
            read_write_same_address_deferrals=read_write_deferrals,
            drain_cycle=cycle,
            source_state_words=a.paths * a.history,
            owned_state_words=a.owned * a.window,
            adc_backpressure=False,
            pending_descriptors=0,
            fine_memory_ports_verified=False,
            closed_loop_capture_verified=False,
            whole_system_accepted=False,
        )
        if released_requests:
            result.update(
                released_request_port=True,
                pending_request_capacity=1,
                original_source_window_end_preserved=True,
                RRC_release_arrivals_and_coarse_owner_integrated=False,
            )
        if source_owner is not None:
            result.update(
                live_source_statistics=dict(source_owner.stats),
                live_coarse_statistics=dict(source_owner.flow.stats),
                live_coarse_failure=source_owner.flow.fault_reason,
                source_fault_cycle=source_owner.fault_cycle,
                live_coarse_lossless=source_owner.flow.lossless,
                source_coarse_transport_clock_composed=True,
                RRC_release_arrivals_and_coarse_owner_integrated=True,
                input_periodic_surrogate_used=False,
                source_owner_cost=source_owner.estimate(),
                source_transport_new_register_bits=1 + 3 * 64,
                source_transport_new_register_research_cap=2048,
                no_later_source_window_substitution=True,
                carrier_frame_and_decoder_cycles=None,
            )
        if fine_worker is not None:
            result.update(
                worker_results=worker_stats,
                retention_is_necessary_only=True,
                replay_read_ports_scheduled=False,
                handoff_check_latency_modeled=False,
                fine_memory_handoff=(
                    "owned source immutable until result release;worker microtrace required"
                ),
                resource_feedback_to_capture_algorithm=False,
            )
        if tail_replay is not None:
            assert not any(tail_replay.leases) and tail_replay.active is None
            result.update(
                tail_statistics=dict(tail_replay.stats),
                tail_cost=tail_replay.estimate(),
                replay_read_ports_scheduled=True,
                retention_is_necessary_only=False,
                source_ring_check_excludes_tail=True,
                raw_snapshot_held_through_replay=True,
                fine_and_replay_serialized=True,
                downstream_live_context_verified=False,
            )
        return result
