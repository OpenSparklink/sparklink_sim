"""Value-carrying seven-stage Lagrange pipeline (§6.2.1.2 structure model).

DSP25x18 latency3,wide-add/RNE1 and96x7 division ROM read1 are explicit
assumptions. No synthesized resource or timing claim;no receiver-service credit.
"""

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    pack_interpolated,
    rne_shift,
    signed_word,
    validate_query,
)

__all__ = ["LagrangeInterpolationFlow"]

# Dividing one hexadecimal digit with a remainder0..5 needs only96 entries.
_DIV6 = tuple(divmod(k, 6) for k in range(96))


class LagrangeInterpolationFlow:
    """Seven finite contexts,no instruction list or future answer trace.

    Each stage holds one query. A finished blocked stage holds its words and
    stalls admission;input is a pauseable query stream,not an ADC input port.
    The divider consumes one nibble/clock,then performs nearest-even rounding.
    """

    def __init__(self, fmt):
        if not isinstance(fmt, LagrangeWordFormat):
            raise ValueError("Need explicit interpolation word format")
        self.fmt, self.cycle = fmt, 0
        self.stages = [None] * 7
        self.last_issue = self.last_take = -1
        self.stats = dict(accepted=0, backpressure=0, emitted=0, peak_live=0)
        self.latencies = (
            3,
            3,
            *([5 if fmt.phase_bits == 16 else 6] * 3),
            (fmt.state_bits + 3) // 4 + 1,
            3,
        )

    def _count(self, key):
        if self.stats[key] == 2**64 - 1:
            raise OverflowError("Interpolation pipeline counter exhausted")
        self.stats[key] += 1

    def issue(self, words, phase, tag=0):
        triples = validate_query(words, phase, self.fmt)
        if type(tag) is not int or not 0 <= tag < 2**16:
            raise ValueError("Need unsigned16bit caller tag")
        if self.stages[0] is not None or self.last_issue == self.cycle:
            self._count("backpressure")
            return False
        self.stages[0] = dict(
            age=0, done=False, triples=triples, phase=phase, tag=tag, accepted_cycle=self.cycle
        )
        self.last_issue = self.cycle
        self._count("accepted")
        self.stats["peak_live"] = max(self.stats["peak_live"], self.live)
        return True

    @property
    def live(self):
        return sum(x is not None for x in self.stages)

    def take(self):
        job = self.stages[-1]
        if job is None or not job["done"] or self.last_take == self.cycle:
            return None
        self.stages[-1] = None
        self.last_take = self.cycle
        self._count("emitted")
        return dict(
            word=job["word"],
            info=job["info"],
            tag=job["tag"],
            accepted_cycle=job["accepted_cycle"],
            emitted_cycle=self.cycle,
        )

    def _align(self, job, age):
        if age == 0:
            e = [t[2] if t[0] or t[1] else -128 for t in job["triples"]]
            job["max_pairs"] = max(e[:2]), max(e[2:])
            job["zero"] = all(not (t[0] or t[1]) for t in job["triples"])
        elif age == 1:
            pairs = job.pop("max_pairs")
            job["common"] = 0 if job["zero"] else max(pairs)
            job.pop("zero")
        else:
            triples = job.pop("triples")
            values = tuple(
                tuple(rne_shift(t[k], job["common"] - t[2]) << self.fmt.guard_bits for t in triples)
                for k in (0, 1)
            )
            job["erased"] = sum(
                t[k] != 0 and v == 0
                for k, row in enumerate(values)
                for t, v in zip(triples, row, strict=True)
            )
            job["values"] = values

    def _prepare(self, job, age):
        if age == 0:
            job["partial"] = [
                (d - a, b - c, a - b, 3 * b, 3 * c, 2 * a + d) for a, b, c, d in job.pop("values")
            ]
        elif age == 1:
            job["partial"] = [
                (s0, 3 * s1, s2 - s1, b3, 2 * c3 - b3, ad)
                for s0, s1, s2, b3, c3, ad in job["partial"]
            ]
        else:
            job["coefficients"] = [
                tuple(
                    signed_word(v, self.fmt.state_bits)
                    for v in (s0 + s13, 3 * s3, cpart - ad, 2 * b3)
                )
                for s0, s13, s3, b3, cpart, ad in job.pop("partial")
            ]
            job["value"] = [row[0] for row in job["coefficients"]]

    def _horner(self, job, age, stage):
        wide = self.fmt.phase_bits == 20
        if age == 2:
            # Actual signed25x18 products after3clocks;the high tile is separate.
            job["products"] = [
                (v * (job["phase"] & 131071), v * (job["phase"] >> 17)) for v in job["value"]
            ]
        elif wide and age == 3:
            job["products"] = [
                signed_word(lo + (hi << 17), self.fmt.state_bits + 21) for lo, hi in job["products"]
            ]
        elif age == (4 if wide else 3):
            products = job.pop("products")
            if not wide:
                products = [lo for lo, _ in products]
            job["value"] = [rne_shift(p, self.fmt.phase_bits) for p in products]
        elif age == (5 if wide else 4):
            job["value"] = [
                signed_word(v + co[stage - 1], self.fmt.state_bits)
                for v, co in zip(job["value"], job["coefficients"], strict=True)
            ]
            if stage == 4:
                job.pop("coefficients")

    def _divide(self, job, age):
        digits = (self.fmt.state_bits + 3) // 4
        if age == 0:
            job["division"] = [(abs(v), 0, 0, v < 0) for v in job.pop("value")]
        if age < digits:
            shift = 4 * (digits - 1 - age)
            result = []
            for numerator, quotient, remainder, negative in job["division"]:
                digit = (numerator >> shift) & 15
                q, r = _DIV6[16 * remainder + digit]
                result.append((numerator, (quotient << 4) | q, r, negative))
            job["division"] = result
        else:
            job["value"] = []
            for _, quotient, remainder, negative in job.pop("division"):
                q = quotient + int(remainder > 3 or (remainder == 3 and quotient & 1))
                job["value"].append(-q if negative else q)

    def advance(self):
        """One clock;no same-edge bypass through multiple registered stages."""
        if self.cycle >= 2**63 - 1:
            raise OverflowError("Interpolation pipeline clock exhausted")
        for k in range(6, 0, -1):
            previous = self.stages[k - 1]
            if self.stages[k] is None and previous is not None and previous["done"]:
                self.stages[k], self.stages[k - 1] = previous, None
                previous.update(age=0, done=False)
        for stage, job in enumerate(self.stages):
            if job is None or job["done"]:
                continue
            age = job["age"]
            if stage == 0:
                self._align(job, age)
            elif stage == 1:
                self._prepare(job, age)
            elif stage < 5:
                self._horner(job, age, stage)
            elif stage == 5:
                self._divide(job, age)
            elif age == 2:
                word, info = pack_interpolated(*job.pop("value"), job["common"], self.fmt)
                job["word"], job["info"] = (
                    word,
                    dict(
                        common_exponent=job["common"],
                        aligned_components_erased=job.pop("erased"),
                        **info,
                    ),
                )
            job["age"] += 1
            job["done"] = job["age"] == self.latencies[stage]
        self.cycle += 1

    def estimate(self):
        bits = self.fmt.state_bits
        stage_bits = [
            184 + 32,
            12 * bits,
            10 * bits + (4 if self.fmt.phase_bits == 20 else 2) * 43,
            10 * bits + (4 if self.fmt.phase_bits == 20 else 2) * 43,
            10 * bits + (4 if self.fmt.phase_bits == 20 else 2) * 43,
            2 * (2 * bits + 3 + 1),
            max(2 * bits, 2 * self.fmt.output_bits + 8) + 16,
        ]
        # Conservative per-context live fields,including original coefficients.
        metadata = 7 * (16 + self.fmt.phase_bits + 1 + 8 + 4 + 4 + 1 + 1)
        return dict(
            input_word_bits=46,
            output_word_bits=2 * self.fmt.output_bits + 8,
            stage_latencies=list(self.latencies),
            no_stall_latency=sum(self.latencies),
            no_stall_II=max(self.latencies),
            operations_per_query=dict(
                real_native25x18_products=6 if self.fmt.phase_bits == 16 else 12,
                signed_coefficient_and_horner_add_sub=30,
                product_merge_add46=6 if self.fmt.phase_bits == 20 else 0,
                alignment_RNE=8,
                horner_RNE=6,
                division_RNE=2,
                normalization_RNE_worst=4,
                division_ROM_reads=2 * ((bits + 3) // 4),
                source_word_reads=4,
                output_word_writes=1,
                general_division=0,
                sqrt=0,
                peak_abs=2,
                peak_compare=1,
                priority_encode=1,
                exponent_max_comparators=3,
                control_statistics_not_zero=True,
            ),
            contexts=7,
            input_queue=0,
            output_queue=1,
            DSP25x18=6 if self.fmt.phase_bits == 16 else 12,
            max_parallel_coefficient_adders25=12,
            horner_round_and_add_paths=6,
            wide_product_merge_adders46=6 if self.fmt.phase_bits == 20 else 0,
            alignment_RNE_shifters19=8,
            alignment_max_comparators8=3,
            division_ROM_bits=2 * 96 * 7,
            division_ROM_reads_per_clock=2,
            proposed_division_ROM_LUT6=28,
            proposed_division_ROM_MUXF7=14,
            proposed_division_ROM_capacity_bits=1792,
            proposed_ROM_mapping="two128x7 single-read distributedROM banks;not synthesized",
            additional_control_and_datapath_LUTs=None,
            source_history_RAM_and_ports_included=False,
            division_rule="two replicated96x7 digit tables,radix16 remainder0..5;no divider",
            stage_payload_state_bits=stage_bits,
            metadata_state_bits=metadata,
            pipeline_register_bound_bits=sum(stage_bits) + metadata,
            statistics_bits=4 * 64,
            cycle_debug_bits=64,
            timestamp_debug_host_only=True,
            output_packer="shared exponent;mantissa19/22;priority/RNE/saturation macro3",
            packing_macro_gate_and_control_LUTs=None,
            gain_loop_head_carrier_decoder_cycles=None,
            macro_assumptions="DSP3;fixed-width add/RNE1;ROM1;pack3;timing unverified",
            no_fine_or_RX_TX_sharing_credit=True,
            whole_receiver16msps=False,
            synthesis_or_timing=False,
        )
