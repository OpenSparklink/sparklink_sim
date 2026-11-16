"""Necessary work and finite request queues for capture, not a complete receiver schedule.

The optimistic graph allows free integer-phase reuse, sign correlations and
interpolation. It bounds the present per-phase graph under DSP-only variable
products, not every possible reformulation or LUT multiplier architecture.
"""

import math
from collections import deque
from dataclasses import dataclass

__all__ = ["CaptureCapacity", "CaptureCapacityAssumptions", "fine_work_floor"]


def fine_work_floor(
    window=256,
    sps=4,
    reference_symbols=32,
    subdivisions=8,
    *,
    min_start_ticks=8,
    free_integer_phases=True,
):
    """Exact present-grid dimensions; legal start bound is in1/subdiv sample ticks.

    Complex differential products use the optimistic three-real-product form.
    Difference power needs two squares; each legal correlation magnitude needs
    two squares. Constant reference correlations and all other work are free in
    this lower graph. These are real product equivalents, not synthesized DSPs.
    """
    values = (window, sps, reference_symbols, subdivisions, min_start_ticks)
    if (
        any(type(v) is not int for v in values)
        or not 4 <= window <= 4096
        or not 1 <= sps <= 16
        or not 2 <= reference_symbols <= 128
        or not 1 <= subdivisions <= 32
        or min_start_ticks < 0
        or type(free_integer_phases) is not bool
    ):
        raise ValueError("Invalid finite search dimensions")
    phases, differences, positions, all_positions, all_phases = 0, 0, 0, 0, 0
    for tick in range(sps * subdivisions):
        # Source spline samples satisfy phase+j*SPS < window-1.
        count = max(
            0, (subdivisions * (window - 1) - tick + sps * subdivisions - 1) // (sps * subdivisions)
        )
        if count < reference_symbols:
            continue
        legal = sum(
            tick + j * sps * subdivisions >= min_start_ticks
            for j in range(count - reference_symbols + 1)
        )
        all_phases += 1
        all_positions += legal
        if free_integer_phases and tick % subdivisions == 0:
            continue
        phases += 1
        differences += count - 1
        positions += legal
    return dict(
        all_phases=all_phases,
        legal_positions=all_positions,
        charged_phases=phases,
        differential_products=differences,
        charged_legal_positions=positions,
        variable_real_products=5 * differences + 2 * positions,
        free_integer_phases=free_integer_phases,
        scope="optimistic present per-phase graph only; interpolation, preparation, normalization, ports and frontend excluded",
    )


@dataclass(frozen=True)
class CaptureCapacityAssumptions:
    clock_hz: int = 200_000_000
    input_hz: int = 16_000_000
    multiplier_budget: int = 18
    fine_multipliers: int = 4
    queue_depth: int = 2
    history_samples: int = 384
    window_samples: int = 256
    sample_bits: int = 44
    snapshot_reads_per_cycle: int = 8
    memory_budget_bits: int = 16 * 1024 * 8

    def __post_init__(self):
        if (
            any(type(v) is not int or v < 1 for v in vars(self).values())
            or self.input_hz > self.clock_hz
            or self.fine_multipliers > self.multiplier_budget
            or self.queue_depth > 16
            or self.history_samples > 65536
            or self.window_samples != 256
            or self.window_samples > self.history_samples
            or self.sample_bits > 128
            or self.snapshot_reads_per_cycle > 32
        ):
            raise ValueError("Invalid bounded capture capacity assumptions")


class CaptureCapacity:
    """FCFS snapshot/fine service with bounded owned windows and drop-new admission.

    This diagnostic scheduler does not feed delayed decisions into the receiver.
    Source request traces are an optimistic replay, not a closed-loop capture
    result. Downstream stalls hold the active result and its owned snapshot.
    """

    def __init__(self, assumptions=None):
        self.assumptions = a = CaptureCapacityAssumptions() if assumptions is None else assumptions
        if not isinstance(a, CaptureCapacityAssumptions):
            raise ValueError("Need CaptureCapacityAssumptions")
        self.work = fine_work_floor(window=a.window_samples)
        self.service_cycles = math.ceil(self.work["variable_real_products"] / a.fine_multipliers)
        self.copy_cycles = math.ceil(a.window_samples / a.snapshot_reads_per_cycle)

    def necessary_capacity(self):
        a = self.assumptions
        requests = a.input_hz / 32
        demand = requests * self.work["variable_real_products"]
        available = a.clock_hz * a.multiplier_budget
        # Deliberately omit scratch, pipeline, port replication and decoder state:
        # even a passing lower storage bound is never a full capacity acceptance.
        owned = a.queue_depth + 1
        storage = dict(
            two_banks=2 * 12202,
            two_histories=2 * a.history_samples * a.sample_bits,
            owned_windows=owned * a.window_samples * a.sample_bits,
            descriptors=owned * 256,
            rrc_delay=80 * 32,
            rrc_coefficients=41 * 18,
        )
        total = sum(storage.values())
        return dict(
            fine_work=self.work,
            max_quantum_requests_per_second=requests,
            optimistic_product_demand_per_second=demand,
            all_multiplier_capacity_per_second=available,
            fine_graph_all_multipliers_necessary_condition=demand <= available,
            required_multiplier_units_at_clock=math.ceil(demand / a.clock_hz),
            storage_lower_bound_bits=total,
            storage_components=storage,
            storage_lower_bound_within_budget=total <= a.memory_budget_bits,
            scheduled_fine_product_cycles=self.service_cycles,
            snapshot_copy_cycles=self.copy_cycles,
            excluded=[
                "RRC/detector/two-bank compute",
                "tone normalization latency",
                "spline preparation/interpolation",
                "score normalization/ranking",
                "tone state/input FIFOs",
                "scratch/pipeline/port replication",
                "timing/carrier/header/payload/decoder",
            ],
            mapping="one DSP primitive per real variable product is optimistic; LUT multipliers and alternate factorizations are not analyzed",
            snapshot_ports="eight source reads plus eight destination writes/cycle by default; banked/replicated physical implementation unproven",
            status="necessary bounds plus optimistic finite queue only; full16MS/s unverified",
        )

    def simulate(self, requests, input_samples, *, stalls=(), sink=None):
        """Consume ordered request sample counts lazily; diagnostics are caller-owned.

        Stalls are an externally supplied sorted nonoverlapping cycle schedule,
        not signal-derived future information. Requests at EOF count only actually
        arrived samples. No trace/diagnostic list is retained as hardware state.
        """
        if type(input_samples) is not int or input_samples < 1:
            raise ValueError("Need positive actual input sample count")
        if not isinstance(stalls, (list, tuple)) or len(stalls) > 1024:
            raise ValueError("Need at most1024 external stall intervals")
        a = self.assumptions
        if input_samples * a.clock_hz // a.input_hz >= 2**63 - 2**32:
            raise OverflowError("Signed64 cycle timestamp exhausted")
        previous_end = -1
        for start, end in stalls:
            if (
                type(start) is not int
                or type(end) is not int
                or start < 0
                or start >= end
                or start < previous_end
            ):
                raise ValueError("Need sorted disjoint nonnegative stall intervals")
            previous_end = end
        releases = deque()
        copy_busy = fine_busy = last_count = peak = accepted = dropped = gaps = blocked = 0
        snapshot_gaps = 0
        minimum, maximum, latency_sum, stall_index = None, 0, 0, 0
        for ident, count in enumerate(requests):
            if type(count) is not int or not 1 <= count <= input_samples or count < last_count:
                raise ValueError("Request counts must be sorted and inside actual input")
            last_count = count
            arrival = ((count - 1) * a.clock_hz + a.input_hz - 1) // a.input_hz
            while releases and releases[0] <= arrival:
                releases.popleft()
            if len(releases) == a.queue_depth + 1:
                dropped += 1
                if sink is not None:
                    sink(dict(id=ident, after_samples=count, status="request_queue_full"))
                continue
            copied = max(copy_busy, arrival) + self.copy_cycles
            copy_busy = copied
            copied_count = min(input_samples, copied * a.input_hz // a.clock_hz + 1)
            snapshot_gap = (
                count < a.window_samples
                or copied_count - count > a.history_samples - a.window_samples
            )
            snapshot_gaps += int(snapshot_gap)
            start = max(copied, fine_busy)
            complete = start + self.service_cycles
            released = complete
            while stall_index < len(stalls):
                lo, hi = stalls[stall_index]
                if released < lo:
                    break
                if released < hi:
                    released = hi
                else:
                    stall_index += 1
            blocked += int(released != complete)
            fine_busy = released
            releases.append(released)
            peak = max(peak, len(releases))
            accepted += 1
            latest_count = min(input_samples, released * a.input_hz // a.clock_hz + 1)
            age = latest_count - count
            gap = age > a.history_samples
            gaps += int(gap)
            latency = released - arrival
            minimum = latency if minimum is None else min(minimum, latency)
            maximum = max(maximum, latency)
            latency_sum += latency
            if sink is not None:
                sink(
                    dict(
                        id=ident,
                        after_samples=count,
                        offered_cycle=arrival,
                        copied_cycle=copied,
                        started_cycle=start,
                        released_cycle=released,
                        retained_gap_samples=max(0, age - a.history_samples),
                        snapshot_history_gap=snapshot_gap,
                        status=(
                            "snapshot_history_gap"
                            if snapshot_gap
                            else "history_gap"
                            if gap
                            else "history_available_not_packet_acceptance"
                        ),
                    )
                )
        return dict(
            requests=accepted + dropped,
            accepted_requests=accepted,
            dropped_requests=dropped,
            history_gaps=gaps,
            blocked_results=blocked,
            snapshot_history_gaps=snapshot_gaps,
            peak_owned_windows=peak,
            owned_window_capacity=a.queue_depth + 1,
            max_latency_cycles=maximum,
            min_latency_cycles=minimum,
            mean_latency_cycles=latency_sum / accepted if accepted else None,
            drain_cycle=fine_busy,
            source_input_samples=input_samples,
            continuous_input_detection_charged_samples=input_samples,
            closed_loop_capture_verified=False,
        )
