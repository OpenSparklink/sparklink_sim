"""Causal scaled10 coarse-word dataflow;not an ADC→MAC or RTL implementation.

Every input is processed. Bank reads,clocked RNE,finite accumulator contexts,
histogram feedback,native products and normalized rank writes are explicit.
Fine search,carrier,head/payload and upstream filtering are separate work.
"""

import math
from collections import deque

from nearlink_sdr.phy.acquisition_bank import sign_coefficients
from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat
from nearlink_sdr.phy.banked_words import BankedDifferenceHistory, input_word
from nearlink_sdr.sim.maximum_flow import ExponentMaximumService
from nearlink_sdr.sim.scaled_native import (
    DifferencePipeline,
    DifferenceQueue,
    FoldResult,
    Q12Pipeline,
    ScorePipeline,
)

__all__ = ["BankedScaledCore"]


def _count(stats, key, amount=1):
    value = stats[key] + amount
    if not 0 <= value < 2**64:
        raise OverflowError("Unsigned64 coarse statistic exhausted")
    stats[key] = value


def _alignment(value, shift, *, power=False):
    """Compiled21/43-bit barrel and guard/sticky lane;large shifts bypass."""
    width = 43 if power else 21
    if shift == 0:
        return value, 0
    if (power and shift > width) or (not power and shift >= width):
        return (0, 0) if power or value >= 0 else (-1, 1)
    floor = value >> shift
    remainder = value - (floor << shift)
    half = 1 << (shift - 1)
    return floor, int(remainder > half or (remainder == half and floor & 1))


def _choose(value, negative, coefficient):
    return value if coefficient == 1 else negative if coefficient == -1 else 0


class _FoldPipeline:
    """Bank-local accumulators;rotate4-bit coefficients/mask,not93-bit data.

    Read→alignment→sign selection→pair addition→accumulation are distinct
    edges. The shared reduction reads a completed accumulator bank once,then
    releases it. A blocked C/E sink holds finite state and propagates credit.
    """

    def __init__(self, history, real, imag, widths):
        self.history, self.cr, self.ci, self.w = history, real, imag, widths
        self.lanes = history.lanes
        self.contexts = [None, None]
        self.reading = None
        self.read = self.aligned = self.selected = self.paired = None
        self.reduction = None
        self.last_reads = []  # Combinational-port observer,not a retained IQ log.
        self.stats = dict(
            requests=0,
            reads=0,
            startup_bypasses=0,
            aligned_terms=0,
            accumulated_terms=0,
            reductions=0,
            completed=0,
            output_held=0,
        )

    def _validate_lane(self, row):
        c, e = self.w["correlation"], self.w["window_energy"]
        correction = (2 * self.history.terms).bit_length() + 1
        count = self.history.terms.bit_length()
        for value, bits, signed in zip(
            row,
            [c, c, e, correction, correction, count, count],
            [True, True, False, True, True, False, False],
            strict=True,
        ):
            if not -(1 << (bits - 1)) * signed <= value < 1 << (bits - signed):
                raise OverflowError("Fold accumulator exceeded its compiled exact width")

    def step(self, requests, ce_pending, ce_capacity, save_outgoing):
        self.last_reads = []
        # Reduction uses old completed state;the final accumulation below cannot
        # be consumed on this same edge.
        if self.reduction is not None:
            tag, nodes = self.reduction
            if len(nodes) == 1:
                if len(ce_pending) < ce_capacity:
                    a = nodes[0]
                    ce_pending.append(FoldResult(tag, a[0] + a[3], a[1] + a[4], a[2] + a[5], a[6]))
                    self.reduction = None
                    _count(self.stats, "completed")
                else:
                    _count(self.stats, "output_held")
            else:
                combined = [
                    [a + b for a, b in zip(nodes[i], nodes[i + 1], strict=True)]
                    for i in range(0, len(nodes), 2)
                ]
                for row in combined:
                    self._validate_lane(row)
                self.reduction = tag, combined
                _count(self.stats, "reductions", len(combined))
        elif any(context is not None and context["complete"] for context in self.contexts):
            index = min(
                (
                    i
                    for i, context in enumerate(self.contexts)
                    if context is not None and context["complete"]
                ),
                key=lambda i: self.contexts[i]["tag"],
            )
            context = self.contexts[index]
            rows = context["values"]
            combined = [
                [a + b for a, b in zip(rows[i], rows[i + 1], strict=True)]
                for i in range(0, self.lanes, 2)
            ]
            for row in combined:
                self._validate_lane(row)
            self.reduction = context["tag"], combined
            self.contexts[index] = None
            _count(self.stats, "reductions", len(combined))
        if self.paired is not None:
            index, tag, last, rows = self.paired
            context = self.contexts[index]
            if context is None or context["tag"] != tag or context["complete"]:
                raise ValueError("Fold feedback lost its accumulator owner")
            for lane, piece in rows:
                value = [a + b for a, b in zip(context["values"][lane], piece, strict=True)]
                self._validate_lane(value)
                context["values"][lane] = value
                _count(self.stats, "accumulated_terms")
            if last:
                context["complete"] = True
        next_paired = None
        if self.selected is not None:
            index, tag, last, rows = self.selected
            result = []
            for lane, values in rows:
                a, b, c, d, p, ba, bb, bc, bd, bp, discard = values
                result.append((lane, (a + b, c + d, p, ba + bb, bc + bd, bp, discard)))
            next_paired = index, tag, last, result
        next_selected = None
        if self.aligned is not None:
            index, tag, last, rows = self.aligned
            result = []
            for lane, values in rows:
                r, i, p, br, bi, bp, cr, ci, discard = values
                nr, ni = -r, -i  # Two explicit22-bit negation units per lane.
                result.append(
                    (
                        lane,
                        (
                            _choose(r, nr, cr),
                            _choose(i, ni, ci),
                            _choose(i, ni, cr),
                            _choose(r, nr, -ci),
                            p,
                            _choose(br, -br, cr),
                            _choose(bi, -bi, ci),
                            _choose(bi, -bi, cr),
                            _choose(br, -br, -ci),
                            bp,
                            discard,
                        ),
                    )
                )
            next_selected = index, tag, last, result
        next_aligned = None
        if self.read is not None:
            index, tag, first, common, last, rows = self.read
            result = []
            oldest_lane = ((tag // 4) - (self.history.terms - 1)) % self.lanes
            outgoing_fields = None
            for lane, word, cr, ci in rows:
                shift = max(common - word.exponent, 0)
                r, br = _alignment(word.real, shift)
                i, bi = _alignment(word.imag, shift)
                p, bp = _alignment(word.power, 2 * shift, power=True)
                nonzero = word.power > 0
                if first == 0 and lane == oldest_lane:
                    outgoing_fields = word.exponent, nonzero
                result.append(
                    (lane, (r, i, p, br, bi, bp, cr, ci, int(nonzero and p == 0 and not bp)))
                )
            if first == 0:
                # Reuse the term's nonzero flag in the9-bit cache selector.
                if outgoing_fields is None:
                    raise RuntimeError("Oldest logical term had no physical bank owner")
                save_outgoing(tag + 4, *outgoing_fields)
            next_aligned = index, tag, last, result
            _count(self.stats, "aligned_terms", len(rows))
        self.paired, self.selected, self.aligned = next_paired, next_selected, next_aligned
        self.read = None
        # At most one read batch per bank/edge. A new request can begin on the
        # edge after the preceding final batch,without a fabricated read port.
        if self.reading is None and requests:
            free = next((i for i, context in enumerate(self.contexts) if context is None), None)
            if free is not None:
                tag, common = requests.popleft()
                self.contexts[free] = dict(
                    tag=tag, complete=False, values=[[0] * 7 for _ in range(self.lanes)]
                )
                self.reading = free, tag, common, 0
                _count(self.stats, "requests")
        if self.reading is not None:
            index, tag, common, first = self.reading
            count = min(self.lanes, self.history.terms - first)
            words, addresses = self.history.read_batch(tag, first, count)
            result = []
            for j, (word, address) in enumerate(zip(words, addresses, strict=True), first):
                source = tag - 4 * (self.history.terms - 1 - j)
                lane = (source // 4) & (self.lanes - 1)
                if address is not None:
                    self.last_reads.append(address)
                    _count(self.stats, "reads")
                else:
                    _count(self.stats, "startup_bypasses")
                result.append((lane, word, int(self.cr[j]), int(self.ci[j])))
            last = first + count == self.history.terms
            self.read = index, tag, first, common, last, result
            self.reading = None if last else (index, tag, common, first + count)

    @property
    def idle(self):
        return (
            self.reading is self.read is self.aligned is self.selected is self.paired is None
            and self.reduction is None
            and all(context is None for context in self.contexts)
        )

    def estimate(self):
        c, e, k, lanes = (
            self.w["correlation"],
            self.w["window_energy"],
            self.history.terms,
            self.lanes,
        )
        correction, count = (2 * k).bit_length() + 1, k.bit_length()
        row = 2 * c + e + 2 * correction + 2 * count
        return dict(
            terms=k,
            lanes=lanes,
            accumulator_contexts=2,
            accumulator_lane_bits=row,
            accumulator_register_bits=2 * (lanes * row + 63 + 2),
            read_controller_bits=1 + 63 + 8 + 7 + 1,
            pipeline_register_bits=[
                lanes * 98 + 81,
                lanes * 94 + 65,
                lanes * 142 + 65,
                lanes * 98 + 65,
            ],
            reduction_register_bits=(lanes // 2) * row + 63 + 5 + 1,
            statistics_bits=8 * 64,
            coefficient_ROM_bits=4 * k,
            coefficient_ROM_read_ports=lanes,
            coefficient_ROM_LUT_upper=4 * lanes,
            coefficient_mask_rotate_MUX2_upper=lanes * (lanes.bit_length() - 1) * 5,
            difference_data_rotate_MUX2=0,
            oldest_cache_field_select_MUX2_upper=(lanes - 1) * 9,
            oldest_cache_additional_power_zero_compares=0,
            power_nonzero_compare_bits=43,
            power_nonzero_compares_per_input=k,
            oldest_cache_tag_add_bits=63,
            barrel_shift_widths=[21, 21, 43],
            barrel_shift_units=3 * lanes,
            shift_guard_sticky_lanes=lanes,
            negation_units22=2 * lanes,
            sign_pair_adders23=2 * lanes,
            small_pair_adders3=2 * lanes,
            accumulator_adders_by_width=[c, c, e, correction, correction, count, count] * lanes,
            reduction_adders_by_width=[c, c, e, correction, correction, count, count]
            * (lanes // 2),
            correction_adders_by_width=[c, c, e],
            feedback_latency_edges=1,
            minimum_read_batches=math.ceil(k / lanes),
            actual_synthesis_or_Fmax=False,
        )


class BankedScaledCore:
    """Bounded word input→C/E→squared gate→Q12→bounded coarse query.

    ``offer`` is ready/valid:False is unconsumed. An unpausable source must call
    ``input_overrun`` on its lost tag;it cannot skip a word and continue.
    ``request_query`` is an event port:False explicitly counts a dropped query.
    Sink callbacks are host-only observations;their results never select data.
    """

    INPUT = 16
    DIFFERENCES = 64
    FOLD = 4
    CE = 4
    QUERIES = 2
    _failure_reasons = (None, "input-overrun", "history-expired", "internal-word-or-owner")

    def __init__(self, reference, *, lanes=16, origin=0, sink=None):
        if (
            len(reference) not in (32, 62, 126)
            or type(origin) is not int
            or not -(2**62) <= origin < 2**62
        ):
            raise ValueError("Need compiledSPS4 target reference and signed origin")
        if sink is not None and not callable(sink):
            raise ValueError("Need an observation callback")
        real, imag = sign_coefficients(reference)
        self.cr, self.ci = tuple(map(int, real)), tuple(map(int, imag))
        self.k = len(real)
        self.coefficient_energy = sum(a * a + b * b for a, b in zip(self.cr, self.ci, strict=True))
        self.window = {31: 256, 61: 376, 125: 632}[self.k]
        self.origin, self.sink = origin, sink
        self.maximum_input_tag = 2**63 - 6 - abs(origin)
        self.widths = FixedBankFormat(10).widths(self.k, self.coefficient_energy, 0.6)
        self.history = BankedDifferenceHistory(self.k, lanes)
        self.prefix = DifferencePipeline()
        self.histogram = ExponentMaximumService(self.k, counter_store="serial1r1w")
        self.fold = _FoldPipeline(self.history, self.cr, self.ci, self.widths)
        self.score = ScorePipeline(self.widths, self.coefficient_energy)
        self.normalizer = Q12Pipeline(self.widths, self.k, self.coefficient_energy)
        self.inputs, self.fold_requests, self.ce = (deque() for _ in range(3))
        self.differences = DifferenceQueue()
        self.outgoing = [None] * 4
        # Combined hit/rank word pays for the longer physical ring. Query
        # semantics still retain only the original min(256,window) rank age.
        self.rank_words = [None] * self.window
        self.rank_write_slot = 0
        self.input_count = self.committed_count = self.cycle = 0
        self.failure_code, self.first_lost_tag = 0, None
        self.closed = False
        self.query_queue, self.candidates = deque(), deque()
        self.query_active = self.query_reply = None
        self.next_query_id = 0
        self.stats = dict.fromkeys(
            (
                "input_accepted",
                "input_backpressured",
                "input_resource_aborts",
                "history_conflicts",
                "cache_wait_edges",
                "points",
                "zero_energy",
                "discarded_terms",
                "unity_clamps",
                "query_events",
                "query_dropped",
                "query_expired",
                "query_invalid",
                "query_reads",
                "query_completed",
                "query_output_held",
                "query_read_write_collisions",
            ),
            0,
        )
        self.high_water = dict.fromkeys(("input", "differences", "fold", "ce", "queries"), 0)

    @property
    def failure(self):
        return self._failure_reasons[self.failure_code]

    @property
    def can_accept(self):
        return not self.closed and not self.failure_code and len(self.inputs) < self.INPUT

    def offer(self, tag, real, imag, exponent):
        if self.closed or self.failure_code:
            raise ValueError("Need a new explicit coarse epoch")
        if type(tag) is not int or tag != self.input_count or not 0 <= tag < 2**63 - 5:
            raise ValueError("Need the next contiguous63-bit word tag")
        if tag > self.maximum_input_tag:
            raise OverflowError("Insufficient origin/index/cache coordinate headroom")
        r, i, e = input_word(real, imag, exponent)
        if len(self.inputs) >= self.INPUT:
            _count(self.stats, "input_backpressured")
            return False
        self.inputs.append((tag, r, i, e))
        self.input_count += 1
        _count(self.stats, "input_accepted")
        self._watermarks()
        return True

    def input_overrun(self, tag):
        if type(tag) is not int or tag != self.input_count:
            raise ValueError("Record the actual first unconsumed source tag")
        self._abort(1, tag)

    def _abort(self, code, tag):
        if not self.failure_code:
            self.failure_code, self.first_lost_tag = code, tag
            self.histogram.abort("upstream-resource")
            _count(self.stats, "input_resource_aborts")

    def _observe(self, kind, **fields):
        if self.sink is not None:
            self.sink(dict(kind=kind, cycle=self.cycle + 1, **fields))

    def _watermarks(self):
        for label, queue in (
            ("input", self.inputs),
            ("differences", self.differences),
            ("fold", self.fold_requests),
            ("ce", self.ce),
            ("queries", self.query_queue),
        ):
            self.high_water[label] = max(self.high_water[label], len(queue))

    def _save_outgoing(self, next_tag, exponent, nonzero):
        phase = next_tag % 4
        if self.outgoing[phase] is not None:
            raise ValueError("Oldest-term feedback cache overwritten before histogram consumption")
        self.outgoing[phase] = next_tag, exponent, nonzero

    def _histogram_input(self):
        if not self.histogram.can_issue or not self.differences:
            return
        # A take/issue on one edge is explicitly forbidden by the unchanged
        # histogram owner. No request counter or history write occurs there.
        if len(self.fold_requests) >= self.FOLD:
            return
        tag, word = self.differences[0]
        phase = tag % 4
        old = self.outgoing[phase]
        if tag >= 4 and (old is None or old[0] != tag):
            _count(self.stats, "cache_wait_edges")
            return
        address = self.history.address(tag)
        if address in self.fold.last_reads:
            _count(self.stats, "history_conflicts")
            return
        outgoing = (0, False) if tag < 4 else old[1:]
        if self.history.next_tag != tag:
            raise ValueError("History write lost histogram reservation ownership")
        if not self.histogram.issue(tag, outgoing, (word.exponent, word.power > 0)):
            raise RuntimeError("Histogram ready observation was not a valid issue credit")
        self.history.append(tag, word)
        self.differences.popleft()
        self.outgoing[phase] = None

    def _commit(self, value):
        source, numerator = value["source"], value["numerator"]
        if source.tag != self.committed_count:
            raise ValueError("Normalized results arrived out of input order")
        ready = source.tag >= self.k * 4
        hit, rank = bool(ready and value["hit"]), value["rank"] if ready else 0
        self.rank_words[self.rank_write_slot] = hit, rank
        self.rank_write_slot += 1
        if self.rank_write_slot == self.window:
            self.rank_write_slot = 0
        self.committed_count += 1
        _count(self.stats, "discarded_terms", source.discarded)
        if ready:
            _count(self.stats, "points")
            _count(self.stats, "zero_energy", int(source.energy == 0))
            _count(
                self.stats,
                "unity_clamps",
                int(source.energy > 0 and numerator > value["denominator"]),
            )
        self._observe(
            "point",
            tag=source.tag,
            real=source.real,
            imag=source.imag,
            energy=source.energy,
            numerator=numerator,
            rank=rank,
            hit=hit,
            ready=ready,
            discarded=source.discarded,
        )

    def request_query(self, lower_start, upper_start, *, phase_origin=None):
        if self.failure_code or self.closed:
            raise ValueError("Need an active explicit coarse epoch")
        if phase_origin is None:
            phase_origin = self.origin
        for value in (lower_start, upper_start, phase_origin):
            if type(value) is not int or not -(2**62) <= value < 2**62:
                raise ValueError("Hardware query port requires bounded integer coordinates")
        if self.next_query_id >= 2**63 - 1:
            raise OverflowError("Coarse query ID exhausted")
        request_id = self.next_query_id
        self.next_query_id += 1
        _count(self.stats, "query_events")
        if len(self.query_queue) >= self.QUERIES:
            _count(self.stats, "query_dropped")
            self._observe("query_drop", request_id=request_id, snapshot=self.committed_count)
            return False
        self.query_queue.append(
            dict(
                request_id=request_id,
                lo=lower_start,
                hi=upper_start,
                phase_origin=phase_origin,
                snapshot=self.committed_count,
                accepted_cycle=self.cycle,
                stage=0,
            )
        )
        self._watermarks()
        return True

    def take_candidate(self):
        return self.candidates.popleft() if self.candidates else None

    def _query_emit(self):
        q = self.query_active
        if self.candidates:
            _count(self.stats, "query_output_held")
            return
        enabled = q.get("any_hit", False) and q.get("reason") is None
        chosen = q.get("best_index") if enabled else None
        response = dict(
            request_id=q["request_id"],
            snapshot=q["snapshot"],
            count=q.get("count", 0),
            enabled=bool(enabled),
            chosen=(self.origin + chosen - self.k * 4 if chosen is not None else None),
            score=q.get("best", 0) if enabled else 0,
            reason=q.get("reason"),
            available_cycle=self.cycle + 1,
        )
        self.candidates.append(response)
        self.query_active = None
        self.query_reply = None
        _count(self.stats, "query_completed")
        self._observe("query", **response)

    def _query_fail(self, reason):
        q = self.query_active
        q["reason"], q["stage"] = reason, 4
        self.query_reply = None
        _count(self.stats, "query_expired" if reason == "history-expired" else "query_invalid")

    def _rank_address(self, index):
        delta = self.committed_count - 1 - index
        if not 0 <= delta < min(256, self.window):
            raise ValueError("Query rank history expired")
        latest_slot = self.rank_write_slot - 1 if self.rank_write_slot else self.window - 1
        address = latest_slot - delta
        return address + self.window if address < 0 else address

    def _query_step(self, planned_write):
        q = self.query_active
        if q is not None:
            stage = q["stage"]
            if stage == 0:
                low, high = q["lo"] + self.k * 4 - self.origin, q["hi"] + self.k * 4 - self.origin
                if not (-(2**63) <= low < 2**63 and -(2**63) <= high < 2**63):
                    self._query_fail("query-range")
                else:
                    q["lo"], q["hi"] = low, high
                    q["offset"] = (self.origin - q.pop("phase_origin")) % 4
                    q["stage"] = 1
            elif stage == 1:
                q["lo"] = max(self.k * 4, q["lo"])
                q["hi"] = min(q["snapshot"] - 1, q["hi"])
                q["count"] = max(0, q["hi"] - q["lo"] + 1)
                if q["count"] > 131 or (
                    q["count"] and q["lo"] < q["snapshot"] - min(256, self.window)
                ):
                    self._query_fail("query-range")
                else:
                    q["stage"] = 2
            elif stage == 2:
                q.update(
                    phase=0,
                    index=q["lo"] + (-q["offset"] - q["lo"]) % 4,
                    issued=0,
                    received=0,
                    any_hit=False,
                    best=0,
                    best_index=None,
                )
                q["stage"] = 3 if q["count"] else 4
            elif stage == 3:
                if self.query_reply is not None:
                    index, hit, score = self.query_reply
                    self.query_reply = None
                    q["any_hit"] |= hit
                    if score > q["best"]:
                        q["best"], q["best_index"] = score, index
                    q["received"] += 1
                if q["received"] == q["count"]:
                    q["stage"] = 4
                else:
                    while q["phase"] < 4 and q["index"] > q["hi"]:
                        q["phase"] += 1
                        q["index"] = q["lo"] + (q["phase"] - q["offset"] - q["lo"]) % 4
                    if q["phase"] < 4:
                        index = q["index"]
                        try:
                            address = self._rank_address(index)
                        except ValueError:
                            self._query_fail("history-expired")
                        else:
                            if planned_write is not None and address == planned_write:
                                # No vendor read-during-write behavior is assumed.
                                _count(self.stats, "query_read_write_collisions")
                            else:
                                word = self.rank_words[address]
                                if word is None:
                                    raise RuntimeError("Valid query rank word was never committed")
                                self.query_reply = index, *word
                                q["issued"] += 1
                                q["index"] += 4
                                _count(self.stats, "query_reads")
            else:
                self._query_emit()
        if self.query_active is None and self.query_queue:
            self.query_active = self.query_queue.popleft()

    def step(self):
        if self.cycle >= 2**63 - 1:
            raise OverflowError("Coarse clock exhausted")
        if self.failure_code:
            self.cycle += 1
            return
        try:
            planned = self.rank_write_slot if self.normalizer.cells[-1] is not None else None
            self._query_step(planned)
            point = self.normalizer.step()
            if point is not None:
                self._commit(point)
            self.score.step(self.ce, self.normalizer)
            self.fold.step(self.fold_requests, self.ce, self.CE, self._save_outgoing)
            self._histogram_input()
            self.histogram.step()
            if self.histogram.ready:
                if len(self.fold_requests) >= self.FOLD:
                    raise RuntimeError("Histogram result lacked its reserved fold credit")
                tag, common, _ = self.histogram.take()
                self.fold_requests.append((tag, common))
            self.differences.advance()
            self.prefix.step(self.inputs, self.differences, self.DIFFERENCES)
            self._watermarks()
        except (ValueError, OverflowError, RuntimeError) as error:
            expired = isinstance(error, ValueError) and "expired" in str(error)
            self._abort(2 if expired else 3, self.history.next_tag)
            if not expired:
                raise
        self.cycle += 1

    @property
    def data_idle(self):
        return (
            not self.inputs
            and not len(self.differences)
            and not self.fold_requests
            and not self.ce
            and self.prefix.idle
            and self.histogram.mode == "idle"
            and self.fold.idle
            and self.score.idle
            and self.normalizer.idle
        )

    def finish(self):
        """End word input;drain bounded work,or retain an explicit resource exit."""
        self.closed = True
        if self.failure_code:
            return False
        for _ in range(8192):
            if self.candidates:
                raise ValueError("Drain the held candidate output before finishing")
            if self.data_idle and not self.query_queue and self.query_active is None:
                if self.committed_count != self.input_count:
                    raise RuntimeError("EOF omitted accepted word results")
                return True
            self.step()
            if self.failure_code:
                return False
        raise RuntimeError("Finite coarse drain deadline exceeded")

    def estimate(self):
        """Allocated typed state and explicit operators;not a LUT/timing netlist."""
        history, histogram, pending = (
            self.history.estimate(),
            self.histogram.estimate(),
            self.differences.estimate(),
        )
        prefix, fold, score, normal = (
            self.prefix.estimate(),
            self.fold.estimate(),
            self.score.estimate(),
            self.normalizer.estimate(),
        )
        c, e, lanes = self.widths["correlation"], self.widths["window_energy"], self.history.lanes
        prefix_bits = (
            sum(prefix["fabric_stage_bits_upper"])
            + prefix["valid_bits"]
            + prefix["IQ_lag_register_bits"]
            + prefix["source_tag_counter_bits"]
            + prefix["statistics_bits"]
            + prefix["DSP_internal_pipeline_bit_capacity"]
        )
        fold_bits = (
            fold["accumulator_register_bits"]
            + fold["read_controller_bits"]
            + sum(fold["pipeline_register_bits"])
            + 3  # Remaining global valid bits.
            + fold["reduction_register_bits"]
            + fold["statistics_bits"]
            + fold["coefficient_ROM_bits"]
        )
        score_bits = sum(
            score[name]
            for name in (
                "DSP_internal_pipeline_bit_capacity",
                "native_metadata_register_bits",
                "native_valid_bits",
                "context_register_bits",
                "context_valid_bits",
                "reply_pair_square_register_bits",
                "reply_pair_square_valid_bits",
                "statistics_bits",
            )
        )
        normal_bits = sum(
            normal[name]
            for name in (
                "input_register_bits",
                "pipeline_register_bits",
                "valid_bits",
                "statistics_bits",
            )
        )
        difference_bits = sum(
            pending[name]
            for name in (
                "logical_RAM_bits",
                "registered_front_bits",
                "controller_bits",
                "statistics_bits",
            )
        )
        ce_word_bits = 63 + 2 * c + e + 7
        queue_bits = dict(
            input_register_words=self.INPUT * 90,
            input_controller=14,
            fold_register_words=self.FOLD * 71,
            fold_controller=7,
            CE_register_words=self.CE * ce_word_bits,
            CE_controller=7,
            query_register_words=self.QUERIES * 381,
            query_controller=4,
            candidate_register_word=63 + 63 + 8 + 1 + 64 + 13 + 3 + 63,
            candidate_valid=1,
        )
        query_work_bits = 381 + 63 + 8 + 3 + 8 + 8 + 1 + 13 + 64 + 1 + 3 + 3 + 78 + 1
        controller_bits = (
            3 * 63
            + math.ceil(math.log2(self.window))
            + 1
            + 2
            + 63
            + 1
            + 63
            + 64
            + 63
            + 20
            + len(self.stats) * 64
        )
        ring_bits, cache_bits = 14 * self.window, 4 * 73
        total = (
            history["logical_data_bits"]
            + history["tag_controller_bits"]
            + histogram["total_state_bits_upper"]
            + difference_bits
            + prefix_bits
            + fold_bits
            + score_bits
            + normal_bits
            + sum(queue_bits.values())
            + query_work_bits
            + controller_bits
            + ring_bits
            + cache_bits
        )
        memory_bits = (
            history["logical_data_bits"]
            + pending["logical_RAM_bits"]
            + histogram["count_state_bits"]
            + ring_bits
        )
        dsp_internal = (
            prefix["DSP_internal_pipeline_bit_capacity"]
            + score["DSP_internal_pipeline_bit_capacity"]
        )
        row_bits = math.ceil(math.log2(self.history.rows))
        return dict(
            scope="full compiled scaled-word core;RRC/raw detector,fine/head/payload/MAC outside",
            terms=self.k,
            lanes=lanes,
            coefficient_energy=self.coefficient_energy,
            coefficient_type="diagonal" if self.coefficient_energy == 2 * self.k else "axis",
            no_single_axis_shortcut=True,
            history=history,
            histogram=histogram,
            pending_difference_queue=pending,
            prefix=prefix,
            fold=fold,
            score=score,
            normalization=normal,
            DSP=prefix["DSP"] + score["DSP"],
            DSP_research_cap=18,
            DSP_research_cap_pass=prefix["DSP"] + score["DSP"] <= 18,
            logical_state_bits_upper=total,
            logical_module_research_cap_bits=16 * 1024 * 8,
            logical_module_research_cap_pass=total <= 16 * 1024 * 8,
            DSP_internal_pipeline_bit_capacity=dsp_internal,
            fabric_register_bits_upper=total
            - memory_bits
            - dsp_internal
            - fold["coefficient_ROM_bits"],
            LUTRAM_LUT_estimate=history["LUTRAM_LUT_estimate"] + pending["LUTRAM_LUT_estimate"],
            SLICEM_slice_estimate=history["SLICEM_slice_estimate"]
            + pending["SLICEM_slice_estimate"],
            native_LUTRAM_bit_capacity=history["allocated_LUTRAM_bit_capacity"]
            + pending["allocated_LUTRAM_bit_capacity"],
            BRAM18_estimate=2,
            BRAM36_sites_reserved=2,
            BRAM_site_scope=(
                "one histogram and one combined hit/rank RAM18;no half-site packing credit"
            ),
            BRAM_native_bit_capacity=2 * 18432,
            combined_hit_rank_logical_bits=ring_bits,
            extra_rank_storage_over_existing256=13 * (self.window - min(256, self.window)),
            extra_rank_query_age_credit=0,
            rank_read_write_ports=dict(read=1, write=1),
            ring_read_write_collision="explicit stall/expiry;no unspecified vendor collision data",
            outgoing_cache_bits=cache_bits,
            outgoing_separate_history_reads=0,
            history_write_conflict_row_compare_bits=row_bits,
            history_write_conflict_select_MUX2=(lanes - 1) * (row_bits + 1),
            old_input_window_count_or_ranking_budgets_relaxed=False,
            queue_storage_bits=queue_bits,
            query_work_register_bits=query_work_bits,
            queue_read_MUX2_upper=15 * 90 + 3 * 71 + 3 * ce_word_bits + 381,
            query_setup_edges=3,
            query_reads_upper=131,
            query_score_compare_bits=13,
            query_age_subtract_compare_bits=63,
            query_stride_add_bits=63,
            query_point_index_conversion_adders65=2,
            query_coordinate_overflow_guard=True,
            maximum_input_tag_register_bits=63,
            input_coordinate_limit_compare_bits=63,
            input_exponent_range_compares7=2,
            input_tag_contiguity_compare_bits=63,
            query_input_coordinate_range_compares64=6,
            query_ID_limit_compare_bits=63,
            fold_owner_tag_compare_bits=63,
            input_FIFO_fallthrough_MUX2_upper=90,
            high_water_compare_widths=[5, 7, 3, 3, 2],
            unique_power_nonzero_compares43_per_input=self.k + 1,
            histogram_producer_zero_scope=(
                "standalone2 is not additive;outgoing reuses fold term flag"
            ),
            origin_bound_configuration_setup_edges=2,
            point_origin_add_sub_bits=64,
            ring_addressing=(
                "write increment/wrap;read latest-slot minus bounded delta,one conditional wrap"
            ),
            global_register_reset_setup_edges=1,
            register_reset_scope=(
                "source epoch begins after setup edge;serial644-word clear still paid"
            ),
            source_ADC_to_word_cycles=None,
            raw_detector_cycles=None,
            RRC_cycles=None,
            fine_search_cycles=None,
            head_payload_MAC_cycles=None,
            whole_RX_16MSps=False,
            LUT_as_logic=None,
            FF_placement=None,
            Fmax=None,
            actual_synthesis=False,
        )
