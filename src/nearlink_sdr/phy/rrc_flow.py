"""Finite RRC queues/operand-tag schedule; no synthesis or whole-RX claim.

All input samples write the history, including when a compute request is lost.
Operand versions are checked at their scheduled read cycle. A lost sample output
is a discontinuity for downstream consumers, never silently stitched away.
"""

import math
from collections import deque
from dataclasses import dataclass

from nearlink_sdr.phy.rrc_word import FixedRRCFormat

__all__ = ["RRCFlow", "RRCFlowAssumptions"]


def _merge_add_cycles(lanes):
    """Schedule two signed addition trees on four one-cycle adders."""
    ready, tasks = {}, []
    for component in range(2):
        level = [(component, i) for i in range(lanes)]
        ready.update({key: 0 for key in level})
        while len(level) > 1:
            following = []
            for first in range(0, len(level), 2):
                if first + 1 == len(level):
                    following.append(level[first])
                else:
                    key = ("add", len(tasks))
                    tasks.append((key, level[first], level[first + 1]))
                    following.append(key)
            level = following
    pending, cycle = tasks[:], 0
    while pending:
        issue = [
            v
            for v in pending
            if v[1] in ready and v[2] in ready and ready[v[1]] <= cycle and ready[v[2]] <= cycle
        ][:4]
        for key, left, right in issue:
            ready[key] = cycle + 1
            pending.remove((key, left, right))
        cycle += 1
    return cycle


@dataclass(frozen=True)
class RRCFlowAssumptions:
    clock_hz: int = 200_000_000
    input_hz: int = 16_000_000
    multiplier_lanes: int = 7
    gain_lanes: int = 7
    arithmetic_ii: int = 1
    coefficient_read_ports: int = 7
    history_data_read_ports: int = 14
    history_tag_read_ports: int = 21
    history_samples: int = 128
    fifo_depth: int = 4
    multiplier_budget: int = 18
    wide_operation_budget: int = 54
    memory_budget_bits: int = 16 * 1024 * 8

    def __post_init__(self):
        if (
            any(type(v) is not int or v < 1 for v in vars(self).values())
            or self.input_hz > self.clock_hz
            or self.multiplier_lanes > 18
            or self.gain_lanes > 16
            or self.arithmetic_ii > 4
            or self.fifo_depth > 16
            or not 81 <= self.history_samples <= 1024
        ):
            raise ValueError("Invalid finite RRC schedule assumptions")


class RRCFlow:
    """Gain scan → integer FIR → merge/round, with bounded descriptors and stalls.

    Gain and FIR have two contexts to overlap final max and arithmetic startup.
    The model schedules real-product/operand groups and checks storage lifetimes;
    integer values are specified by FixedRRC, not recalculated at every cycle.
    """

    def __init__(self, format, assumptions=None):
        if not isinstance(format, FixedRRCFormat):
            raise ValueError("Explicit FixedRRCFormat required")
        self.format = format
        self.a = a = assumptions or RRCFlowAssumptions()
        if not isinstance(a, RRCFlowAssumptions):
            raise ValueError("Need RRCFlowAssumptions")
        self.products = min(
            a.multiplier_lanes,
            a.coefficient_read_ports,
            a.history_data_read_ports // 2,
            (a.history_tag_read_ports - a.gain_lanes) // 2,
        )
        if self.products < 1:
            raise ValueError("No FIR product can be serviced alongside the gain scan")
        self.gain_scan = math.ceil(81 / a.gain_lanes) * a.arithmetic_ii
        self.gain_latency = self.gain_scan + math.ceil(math.log2(a.gain_lanes)) * a.arithmetic_ii
        self.fir_cycles = math.ceil(82 / self.products) * a.arithmetic_ii
        # Align/RNE, preadd, product, accumulation: the last issued group
        # needs three further cycles. RNE carry units are charged separately.
        self.fir_latency = self.fir_cycles + 3 * a.arithmetic_ii
        self.merge_add_cycles = _merge_add_cycles(self.products)
        self.output_cycles = (self.merge_add_cycles + 2) * a.arithmetic_ii

    def resource_estimate(self):
        a, f = self.a, self.format
        accumulator = f.input_mantissa_bits + f.coefficient_bits + 7
        # Register-tap read fanout is charged separately as mux inputs. This
        # alternative RAM copy estimate dedicates one read and one write/clone.
        storage = dict(
            data_history=a.history_samples * 2 * f.input_mantissa_bits,
            exponent_and_present_history=a.history_samples * 8,
            coefficient_rom=a.coefficient_read_ports * 41 * f.coefficient_bits,
            gain_partial_and_merge=24 * a.gain_lanes,
            fir_accumulator_contexts=2 * self.products * 2 * accumulator,
            fir_preadd_product_tags=self.products
            * (f.input_mantissa_bits + 1 + f.coefficient_bits + accumulator + 64),
            merge_and_round=2 * self.products * accumulator + 128,
            input_alignment_pipeline=self.products * 2 * (f.input_mantissa_bits + 1),
            fifo_descriptors=3 * a.fifo_depth * 160,
            pending_partial_results=a.fifo_depth * (2 * self.products * accumulator + 8),
            active_descriptors=5 * 256,
            lifetime_guard_pipeline=256,
            telemetry_and_control=16 * 64,
        )
        total = sum(storage.values())
        arithmetic_peak = 4 * self.products + a.gain_lanes + math.ceil(a.gain_lanes / 2) + 4 + 3
        # Two oldest-operand group bounds/subtracts/comparisons plus one shared
        # expiry bound. Version arrays in simulate are an offline oracle; actual
        # hardware can compare group minima with latest_input-history+1.
        lifetime_guard_operations = 5
        wide_peak = arithmetic_peak + lifetime_guard_operations + 2
        ram_copies = (
            a.history_data_read_ports * storage["data_history"]
            + a.history_tag_read_ports * storage["exponent_and_present_history"]
            + total
            - storage["data_history"]
            - storage["exponent_and_present_history"]
        )
        ii = max(self.gain_scan, self.fir_cycles, self.output_cycles)
        return dict(
            initiation_interval_cycles=ii,
            isolated_latency_cycles=self.gain_latency + self.fir_latency + self.output_cycles,
            gain_scan_cycles=self.gain_scan,
            gain_latency_cycles=self.gain_latency,
            fir_issue_cycles=self.fir_cycles,
            fir_latency_cycles=self.fir_latency,
            merge_add_cycles=self.merge_add_cycles,
            scheduled_products_per_cycle=self.products,
            capacity_necessary_condition=ii * a.input_hz <= a.clock_hz,
            multiplier_units=a.multiplier_lanes,
            arithmetic_operations_peak=arithmetic_peak,
            lifetime_guard_operations_peak=lifetime_guard_operations,
            global_counter_increment_units=2,
            wide_operations_peak=wide_peak,
            additional_narrow_counter_units_upper=14,
            history_data_reads_per_input=162,
            history_tag_reads_per_input=243,
            coefficient_reads_per_input=82,
            pending_partial_payload_bits=2 * self.products * accumulator + 8,
            pending_partial_write_ports=2,
            pending_partial_read_ports=1,
            descriptor_fifo_write_ports_max=2,
            descriptor_fifo_read_ports=1,
            pending_result_read_mux_bit_inputs=a.fifo_depth * (2 * self.products * accumulator + 8),
            pending_result_write_mux_bit_inputs=2
            * a.fifo_depth
            * (2 * self.products * accumulator + 8),
            optional_telemetry_increment_units_upper=12,
            optional_telemetry_update_schedule_complete=False,
            operators_within_module_budget=(
                a.multiplier_lanes <= a.multiplier_budget and wide_peak <= a.wide_operation_budget
            ),
            storage_components=storage,
            register_tap_logical_bits=total,
            register_tap_storage_within_budget=total <= a.memory_budget_bits,
            read_mux_bit_inputs=(
                a.history_data_read_ports * a.history_samples * 2 * f.input_mantissa_bits
                + a.history_tag_read_ports * a.history_samples * 8
            ),
            dedicated_1r1w_ram_copy_bits=ram_copies,
            ram_copy_storage_within_budget=ram_copies <= a.memory_budget_bits,
            scope="module datapath/control cycle model and operation/storage estimate; optional telemetry update scheduling, read fanout, timing, synthesis and whole-RX sharing unverified",
        )

    def simulate(self, samples, *, downstream_stalls=(), trace=False):
        if type(samples) is not int or not 0 <= samples <= 1_000_000:
            raise ValueError("Need0..1000000 sample schedule")
        stalls = list(downstream_stalls)
        if any(
            len(s) != 2 or any(type(v) is not int for v in s) or not 0 <= s[0] < s[1] <= 10_000_000
            for s in stalls
        ):
            raise ValueError("Invalid finite downstream stall intervals")
        a = self.a
        incoming, fir_queue, output_queue = deque(), deque(), deque()
        gain_active, fir_active, output_active = [], [], None
        versions = [-1] * a.history_samples
        records = [] if trace else None
        peak = dict(input=0, gain_contexts=0, fir_contexts=0, fir=0, output=0)
        next_input = next_gain = next_fir = 0
        emitted = rejected = invalid = max_latency = latency_sum = 0
        gain_reads = fir_reads = 0
        cycle = 0
        last_input = ((samples - 1) * a.clock_hz + a.input_hz - 1) // a.input_hz if samples else 0
        limit = (
            last_input
            + samples * (self.gain_latency + self.fir_latency + self.output_cycles + 100)
            + max((s[1] for s in stalls), default=0)
            + 100
        )

        def read(job, index, stage):
            if index >= 0 and versions[index % a.history_samples] != index:
                if not job["invalid"] and trace:
                    records.append(
                        dict(
                            sample=job["sample"],
                            cycle=cycle,
                            event="history_gap",
                            stage=stage,
                            operand=index,
                        )
                    )
                job["invalid"] = True

        while (
            next_input < samples
            or incoming
            or gain_active
            or fir_queue
            or fir_active
            or output_queue
            or output_active
        ):
            if cycle > limit:
                raise RuntimeError("Bounded schedule did not drain")
            if (
                next_input < samples
                and cycle == (next_input * a.clock_hz + a.input_hz - 1) // a.input_hz
            ):
                versions[next_input % a.history_samples] = next_input
                if len(incoming) == a.fifo_depth:
                    rejected += 1
                    if trace:
                        records.append(dict(sample=next_input, cycle=cycle, event="drop_new"))
                else:
                    incoming.append(dict(sample=next_input, arrived=cycle, invalid=False))
                next_input += 1
            if (
                output_active is not None
                and cycle >= output_active["done"]
                and not any(lo <= cycle < hi for lo, hi in stalls)
            ):
                job = output_active
                latency = cycle - job["arrived"]
                max_latency, latency_sum = max(max_latency, latency), latency_sum + latency
                emitted += 1
                invalid += int(job["invalid"])
                if trace:
                    records.append(
                        dict(
                            sample=job["sample"],
                            cycle=cycle,
                            event="invalid_output" if job["invalid"] else "output",
                        )
                    )
                output_active = None
            for job in fir_active[:]:
                if cycle >= job["done"] and len(output_queue) < a.fifo_depth:
                    output_queue.append(job)
                    fir_active.remove(job)
            for job in gain_active[:]:
                if cycle >= job["done"] and len(fir_queue) < a.fifo_depth:
                    fir_queue.append(job)
                    gain_active.remove(job)
            if output_active is None and output_queue:
                output_active = output_queue.popleft()
                output_active["done"] = cycle + self.output_cycles
            if len(fir_active) < 2 and fir_queue and cycle >= next_fir:
                job = fir_queue.popleft()
                job.update(start=cycle, done=cycle + self.fir_latency)
                fir_active.append(job)
                next_fir = cycle + self.fir_cycles
                if trace:
                    records.append(dict(sample=job["sample"], cycle=cycle, event="fir_start"))
            if incoming and len(gain_active) < 2 and cycle >= next_gain:
                job = incoming.popleft()
                job.update(start=cycle, done=cycle + self.gain_latency)
                gain_active.append(job)
                next_gain = cycle + self.gain_scan
                if trace:
                    records.append(dict(sample=job["sample"], cycle=cycle, event="gain_start"))
            for job in gain_active:
                age = cycle - job["start"]
                if age < self.gain_scan and age % a.arithmetic_ii == 0:
                    for offset in range(
                        (age // a.arithmetic_ii) * a.gain_lanes,
                        min(81, (age // a.arithmetic_ii + 1) * a.gain_lanes),
                    ):
                        read(job, job["sample"] - offset, "gain")
                        gain_reads += 1
            for job in fir_active:
                age = cycle - job["start"]
                if age < self.fir_cycles and age % a.arithmetic_ii == 0:
                    for product in range(
                        (age // a.arithmetic_ii) * self.products,
                        min(82, (age // a.arithmetic_ii + 1) * self.products),
                    ):
                        offset = product // 2
                        indices = (
                            [job["sample"] - offset, job["sample"] - 80 + offset]
                            if offset < 40
                            else [job["sample"] - 40]
                        )
                        for index in indices:
                            read(job, index, "fir")
                            fir_reads += 1
            for name, size in [
                ("input", len(incoming)),
                ("gain_contexts", len(gain_active)),
                ("fir_contexts", len(fir_active)),
                ("fir", len(fir_queue)),
                ("output", len(output_queue)),
            ]:
                peak[name] = max(peak[name], size)
            cycle += 1
        return dict(
            input_samples=samples,
            outputs=emitted,
            valid_outputs=emitted - invalid,
            drop_new=rejected,
            invalid_history_outputs=invalid,
            maximum_latency_cycles=max_latency,
            mean_latency_cycles=latency_sum / emitted if emitted else 0,
            cycles=cycle,
            peak_queues=peak,
            gain_word_reads=gain_reads,
            fir_word_reads=fir_reads,
            trace=records,
            continuous_input_without_loss=rejected == invalid == 0,
            note="EOF drains existing work; no IQ buffering or silence skipping. Invalid/missing indices require downstream discontinuity handling.",
        )
