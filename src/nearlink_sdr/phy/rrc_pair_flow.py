"""Owned native-product RRC dataflow; neither RTL nor whole-RX acceptance.

Version arrays are offline oracles. Hardware stores bounded products/ownership,
and tests raw-history expiry with source indices. Every source sample advances
history even if its output request is dropped. Invalid outputs keep source IDs.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import ClassVar

__all__ = ["PairRRCFlow", "PairRRCFlowAssumptions"]


@dataclass(frozen=True)
class PairRRCFlowAssumptions:
    clock_hz: int = 200_000_000
    input_hz: int = 16_000_000
    complex_lanes: int = 4
    coefficient_ports: int = 4
    history_read_ports: int = 8
    product_write_ports: int = 4
    product_read_ports: int = 4
    product_buffers: int = 3
    fifo_depth: int = 4
    history_samples: int = 128
    arithmetic_ii: int = 1
    consumer_ii: int = 12

    def __post_init__(self):
        if (
            any(type(v) is not int or v < 1 for v in vars(self).values())
            or self.input_hz > self.clock_hz
            or self.complex_lanes != 4
            or not 2 <= self.product_buffers <= 4
            or self.fifo_depth > 16
            or self.history_samples != 128
            or self.arithmetic_ii > 4
            or not 2 <= self.history_read_ports <= 8
            or self.coefficient_ports > 4
            or self.product_write_ports > 4
            or self.product_read_ports > 4
            or self.consumer_ii > 255
        ):
            raise ValueError("Invalid finite native-product RRC assumptions")


class PairRRCFlow:
    """FIR/weighted max → immutable product bank → SUM → merge/normalize.

    Fixed12/C18/19 format:31-bit products,37-bit accumulators. Two contexts per
    stage; three owned product slots by default.15-cycle merge/normalization
    overlaps contexts at11-cycle launch interval; eight wide merge ALUs cover
    for the explicitly checked stage operation trace. Wide/leading operations
    are assumed one cycle. Raw memory/ROM ports are explicit, not synthesized.
    """

    # Two addition trees, corrections, abs/max, leading code, exponent, clamp,
    # shift, RNE floor/increment, range comparison and final selection register.
    _merge_wide: ClassVar[dict[int, int]] = {1: 4, 2: 2, 3: 2, 4: 2, 5: 1, 13: 2, 14: 4}

    def __init__(self, assumptions=None):
        self.a = a = assumptions or PairRRCFlowAssumptions()
        if not isinstance(a, PairRRCFlowAssumptions):
            raise ValueError("Need PairRRCFlowAssumptions")
        self.fir_lanes = min(
            a.complex_lanes, a.coefficient_ports, a.history_read_ports // 2, a.product_write_ports
        )
        self.sum_lanes = min(a.complex_lanes, a.product_read_ports)
        self.fir_groups = math.ceil(41 / self.fir_lanes)
        self.sum_groups = math.ceil(41 / self.sum_lanes)
        self.fir_latency = (self.fir_groups - 1 + 10) * a.arithmetic_ii
        self.sum_latency = (self.sum_groups - 1 + 3) * a.arithmetic_ii
        self.merge_ii, self.merge_latency = 11 * a.arithmetic_ii, 15 * a.arithmetic_ii

    def resource_estimate(self):
        """Conservative allocated state/operations; mux/routing not synthesized.

        Four banks pad41products to44entries per owned buffer. Context and
        pipeline data are charged separately from metadata.160-bit reserved
        descriptors include63-bit source ID,63-bit arrival timestamp, slot,
        progress/flags and padding. Version-oracle arrays are not hardware tags.
        """
        a, lanes, acc, product = self.a, 4, 37, 31
        product_entry = 2 * product + 7
        storage = dict(
            combined_raw_history=128 * 32,
            coefficient_banks=4 * 11 * 18,
            coefficient_pipeline=2 * 4 * 18,
            owned_padded_products=a.product_buffers * 4 * 11 * product_entry,
            product_owner_descriptors=a.product_buffers * 160,
            input_fifo=a.fifo_depth * 160,
            output_fifo=a.fifo_depth * (46 + 63 + 1),
            fir_context_descriptors=2 * 160,
            fir_context_maxima=2 * 4 * 8,
            fir_alignment=4 * (4 * 12 + 2 + 7),
            fir_preadd=4 * (2 * 13 + 7),
            fir_product_and_magnitude=2 * 4 * product_entry,
            fir_leading_codes=4 * (2 * 5 + 7),
            fir_weighted_tag=4 * (5 + 7 + 8),
            fir_max_tree=3 * 8,
            fir_metadata_pipeline=10 * 160,
            sum_context_descriptors=2 * 160,
            sum_context_partials=2 * lanes * 2 * (acc + 6),
            sum_product_read_pipeline=4 * product_entry,
            sum_alignment_pipeline=4 * (2 * product + 16),
            sum_floor_pipeline=4 * (2 * product + 2 + 8),
            sum_metadata_pipeline=3 * 160,
            merge_context_descriptors=2 * 160,
            # Context input, two levels, corrected sums, abs/peak, output;
            # both contexts retain all data even while completed/blocked.
            merge_context_data=2
            * (8 * (acc + 6) + 4 * (acc + 6) + 2 * (acc + 6) + 4 * acc + acc + 46 + 32),
            output_quantization_pipeline=2 * (2 * acc + 2 + 16),
            queue_and_owner_control=256,
            source_and_clock_counters=2 * 63,
            startup_and_stage_control=128,
            optional_monitoring_registers=16 * 64,
        )
        total = sum(storage.values())
        # Separate units for concurrent FIR/SUM, pooled merge stages.8-bit
        # control operations spilling the64-small-unit allocation use6wide ALUs.
        wide = dict(
            pair_carry_preadd=8,
            product_magnitude=8,
            sum_accumulation=8,
            merge_normalization_pool=8,
            lifetime_guards=3,
            global_counters=2,
            small_control_assigned_to_wide=6,
        )
        small = dict(
            raw_addresses_startup_and_pair_exponents=28,
            product_leading_tags_and_max=15,
            sum_exponents_round_counts=12,
            merge_round_counts_and_exponents=4,
            remaining_control=5,
        )
        lifetime = self.fir_latency + (self.sum_groups - 1) * a.arithmetic_ii + 1
        necessary_ii = max(
            self.fir_groups * a.arithmetic_ii,
            self.sum_groups * a.arithmetic_ii,
            self.merge_ii,
            math.ceil(self.fir_latency / 2),
            math.ceil(self.sum_latency / 2),
            math.ceil(self.merge_latency / 2),
            math.ceil(lifetime / a.product_buffers),
            a.consumer_ii,
        )
        return dict(
            format=dict(
                input_bits=12,
                coefficient_bits=18,
                output_bits=19,
                product_bits=31,
                accumulator_bits=37,
            ),
            multiplier_units=8,
            wide_units=wide,
            wide_units_total=sum(wide.values()),
            small_units=small,
            small_units_total=sum(small.values()),
            leading_detectors_32bit=12,
            leading_note="8product31-bit detectors +4chunks for two37-bit output magnitudes; 1cycle encoder assumption",
            necessary_ii=necessary_ii,
            capacity_necessary_condition=necessary_ii * a.input_hz <= a.clock_hz,
            fir_latency=self.fir_latency,
            sum_latency=self.sum_latency,
            merge_latency=self.merge_latency,
            isolated_latency_before_consumer=self.fir_latency
            + self.sum_latency
            + self.merge_latency,
            product_owner_lifetime_unstalled=lifetime,
            product_bank_entry_bits=product_entry,
            storage_components=storage,
            register_tap_logical_bits=total,
            dedicated_raw_1r1w_ram_copy_bits=total + 7 * storage["combined_raw_history"],
            raw_read_mux_bit_inputs=8 * 128 * 32,
            product_bank_read_mux_bit_inputs=4 * a.product_buffers * 11 * product_entry,
            ports=dict(
                raw_combined_reads=8,
                raw_writes=1,
                coefficient_reads=4,
                product_banks=4,
                reads_per_product_bank=1,
                writes_per_product_bank=1,
                output_writes=1,
                output_reads=1,
            ),
            monitoring_updates_scheduled=False,
            whole_rx_acceptance=False,
            scope="allocated module estimate + executable dependency/lifetime schedule; 1cycle ALU/encoder, mux/fanout/RAM granularity/synthesis and whole-RX sharing unverified",
        )

    def simulate(self, samples, *, downstream_stalls=(), trace=False):
        if type(samples) is not int or not 0 <= samples <= 1_000_000:
            raise ValueError("Need0..1000000 scheduled source samples")
        stalls = list(downstream_stalls)
        if any(
            len(s) != 2 or any(type(v) is not int for v in s) or not 0 <= s[0] < s[1] <= 10_000_000
            for s in stalls
        ):
            raise ValueError("Invalid downstream stalls")
        a, ii = self.a, self.a.arithmetic_ii
        incoming, ready, output = deque(), deque(), deque()
        fir, sums, merges = [], [], []
        owners = [None] * a.product_buffers
        versions = [[-1] * 41 for _ in owners]  # Offline ownership oracle only.
        raw = [-1] * a.history_samples  # Offline raw lifetime oracle.
        records = [] if trace else None
        peak = dict(input=0, ready=0, output=0, fir=0, sums=0, merges=0, owners=0)
        counts = dict(
            emitted=0,
            dropped=0,
            invalid=0,
            history_reads=0,
            source_history_writes=0,
            product_writes=0,
            product_reads=0,
            ownership_errors=0,
            max_latency=0,
            latency_sum=0,
            max_merge_wide_ops=0,
            max_history_reads_per_cycle=0,
            max_product_writes_per_cycle=0,
            max_product_reads_per_cycle=0,
        )
        next_source = next_fir = next_sum = next_merge = 0
        cycle = 0
        end_input = math.ceil(max(0, samples - 1) * a.clock_hz / a.input_hz)
        limit = end_input + samples * (
            self.fir_latency + self.sum_latency + self.merge_latency + a.consumer_ii + 50
        )
        limit += max((s[1] for s in stalls), default=0) + 100

        def record(event, job, **values):
            if records is not None:
                records.append(dict(event=event, sample=job["sample"], cycle=cycle, **values))

        def check_owner(job):
            if owners[job["slot"]] != job["sample"]:
                counts["ownership_errors"] += 1
                raise AssertionError("Product storage overwritten while still owned")

        while next_source < samples or incoming or fir or ready or sums or merges or output:
            raw_reads = product_writes = product_reads = enqueues = 0
            if cycle > limit:
                raise RuntimeError("Finite native-product schedule failed to drain")
            # Registered output FIFO: no same-cycle enqueue fallthrough.
            if (
                output
                and cycle % a.consumer_ii == 0
                and not any(lo <= cycle < hi for lo, hi in stalls)
            ):
                job = output.popleft()
                latency = cycle - job["arrived"]
                counts["emitted"] += 1
                counts["invalid"] += job["invalid"]
                counts["latency_sum"] += latency
                counts["max_latency"] = max(counts["max_latency"], latency)
                record("output", job, invalid=job["invalid"], latency=latency)
            if (
                next_source < samples
                and cycle == (next_source * a.clock_hz + a.input_hz - 1) // a.input_hz
            ):
                raw[next_source % a.history_samples] = next_source
                counts["source_history_writes"] += 1
                job = dict(sample=next_source, arrived=cycle, invalid=False)
                if len(incoming) == a.fifo_depth:
                    counts["dropped"] += 1
                    record("drop_new", job)
                else:
                    incoming.append(job)
                next_source += 1

            # Product writes have a3-stage path; ownership survives all writes.
            for job in fir:
                dt = cycle - job["start"]
                if dt >= 3 * ii and (dt - 3 * ii) % ii == 0:
                    group = (dt - 3 * ii) // ii
                    if group < self.fir_groups:
                        check_owner(job)
                        lo = group * self.fir_lanes
                        for j in range(lo, min(41, lo + self.fir_lanes)):
                            versions[job["slot"]][j] = job["sample"]
                            counts["product_writes"] += 1
                            product_writes += 1
            for job in fir[:]:
                if cycle >= job["start"] + self.fir_latency:
                    ready.append(job)
                    record("products_ready", job, slot=job["slot"])
                    fir.remove(job)

            # Free only after the final synchronous product read is complete.
            for job in sums:
                if not job["released"] and cycle >= job["release"]:
                    check_owner(job)
                    owners[job["slot"]] = None
                    job["released"] = True
                    record("product_release", job, slot=job["slot"])
            # Completed merge contexts hold all output data under backpressure.
            for job in merges[:]:
                if cycle >= job["done"] and len(output) < a.fifo_depth:
                    output.append(job)
                    enqueues += 1
                    merges.remove(job)
            if cycle >= next_merge and len(merges) < 2:
                completed = [j for j in sums if cycle >= j["done"]]
                if completed:
                    job = completed[0]
                    sums.remove(job)
                    assert job["released"]
                    job.update(start=cycle, done=cycle + self.merge_latency)
                    merges.append(job)
                    record("merge_start", job)
                    next_merge = cycle + self.merge_ii
            if ready and cycle >= next_sum and len(sums) < 2:
                job = ready.popleft()
                job.update(
                    start=cycle,
                    released=False,
                    release=cycle + ((self.sum_groups - 1) * ii) + 1,
                    done=cycle + self.sum_latency,
                )
                sums.append(job)
                record("sum_start", job, slot=job["slot"])
                next_sum = cycle + self.sum_groups * ii
            if incoming and cycle >= next_fir and len(fir) < 2 and None in owners:
                job = incoming.popleft()
                slot = owners.index(None)
                owners[slot] = job["sample"]
                versions[slot][:] = [-1] * 41
                job.update(start=cycle, slot=slot)
                fir.append(job)
                record("fir_start", job, slot=slot)
                next_fir = cycle + self.fir_groups * ii

            # Reads occur after source writes, with explicit current-word bypass.
            for job in fir:
                dt = cycle - job["start"]
                if dt % ii == 0 and dt // ii < self.fir_groups:
                    lo = (dt // ii) * self.fir_lanes
                    for j in range(lo, min(41, lo + self.fir_lanes)):
                        for lag in [j, 80 - j] if j < 40 else [40]:
                            index = job["sample"] - lag
                            counts["history_reads"] += 1
                            raw_reads += 1
                            if index >= 0 and raw[index % a.history_samples] != index:
                                if not job["invalid"]:
                                    record("history_gap", job, operand=index)
                                job["invalid"] = True
            for job in sums:
                dt = cycle - job["start"]
                if dt % ii == 0 and dt // ii < self.sum_groups:
                    check_owner(job)
                    lo = (dt // ii) * self.sum_lanes
                    for j in range(lo, min(41, lo + self.sum_lanes)):
                        if versions[job["slot"]][j] != job["sample"]:
                            counts["ownership_errors"] += 1
                            raise AssertionError("Read unprepared or overwritten product")
                        counts["product_reads"] += 1
                        product_reads += 1
            if (
                raw_reads > a.history_read_ports
                or product_writes > a.product_write_ports
                or product_reads > a.product_read_ports
                or enqueues > 1
            ):
                raise AssertionError("Physical port allocation exceeded")
            for name, value in [
                ("history_reads", raw_reads),
                ("product_writes", product_writes),
                ("product_reads", product_reads),
            ]:
                key = "max_" + name + "_per_cycle"
                counts[key] = max(counts[key], value)
            merge_ops = sum(
                self._merge_wide.get((cycle - j["start"]) // ii, 0)
                for j in merges
                if (cycle - j["start"]) % ii == 0
            )
            counts["max_merge_wide_ops"] = max(counts["max_merge_wide_ops"], merge_ops)
            if merge_ops > 8:
                raise AssertionError("Merge ALU allocation exceeded")
            for k, v in dict(
                input=len(incoming),
                ready=len(ready),
                output=len(output),
                fir=len(fir),
                sums=len(sums),
                merges=len(merges),
                owners=sum(v is not None for v in owners),
            ).items():
                peak[k] = max(peak[k], v)
            cycle += 1
        assert counts["emitted"] + counts["dropped"] == samples
        return dict(
            counts,
            cycles=cycle,
            peak=peak,
            mean_latency=counts["latency_sum"] / max(1, counts["emitted"]),
            continuous_input_without_loss=counts["dropped"] == counts["invalid"] == 0,
            trace=records,
        )
