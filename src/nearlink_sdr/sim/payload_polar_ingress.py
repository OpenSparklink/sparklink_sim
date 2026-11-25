"""Replay the standard B1 Polar ingress calendar with bounded actual RAM owners."""

from collections import deque
from dataclasses import dataclass
from itertools import islice
from math import isfinite

from nearlink_sdr.sim.payload_polar_flow import (
    _SHORT_RESULT_BITS,
    _SHORT_TAIL_CAPACITY,
    _Datapath,
    _formats,
    _quantize_llr,
    payload_polar_calendar,
)

__all__ = ["TimedLLR", "replay_payload_polar_ingress"]


@dataclass(frozen=True)
class TimedLLR:
    """One bit available at the specified decoder ingress edge."""

    edge: int
    value: float


def _sample_beats(rows, period):
    row = next(rows, None)
    while row is not None:
        n, _ = row["shape"]
        bps, phase = row["bits_per_symbol"], row["first_symbol_phase"]
        start = 0
        edge = row["first"]
        pilot = row.get("pilot_interval", 0)
        data_index = row.get("data_symbol_index", 0)
        while True:
            count = min(bps - phase, n - start)
            absent_edge = yield edge, row, start, count
            if absent_edge is not None:
                break
            start += count
            if start == n:
                assert edge == row["last"]
                break
            phase = 0
            data_index += 1
            edge += period * (2 if pilot and data_index % pilot == 0 else 1)
        try:
            row = rows.send(absent_edge)
        except StopIteration:
            return


class _Ingress:
    """Each slot has one access per modulo-three bank and one logical owner."""

    def __init__(self, beats, llrs, scale, observer, drain_at_eof, short_tail_slot, frame_llrs):
        self.beats = iter(beats)
        self.pending = next(self.beats, None)
        self.llrs = iter(llrs) if frame_llrs is None else iter(())
        self.frame_llrs = frame_llrs
        self.input_frame = None
        self.scale = scale
        self.observer = observer
        self.drain_at_eof = drain_at_eof
        sizes = (1024, 1024, _SHORT_TAIL_CAPACITY) if short_tail_slot else (1024, 1024)
        self.banks = tuple(tuple([None] * ((size + 2 - lane) // 3) for lane in range(3))
                           for size in sizes)
        self.owners = [None] * len(sizes)
        self.ready = deque()
        self.edge = self.write_edge = -1
        self.written = set()

    def report(self, kind, row, **extra):
        if self.observer is not None:
            self.observer(dict(kind=kind, edge=self.edge, frame=row["frame"],
                               block=row["block"], word=row["word"], **extra))

    def advance(self, edge):
        if edge < self.edge:
            raise RuntimeError("Ingress clock moved backwards")
        while self.pending is not None and self.pending[0] <= edge:
            self.edge, row, start, count = self.pending
            if self.frame_llrs is not None and row["frame"] != self.input_frame:
                self.check_input_end()
                self.input_frame = row["frame"]
                self.llrs = iter(self.frame_llrs())
            values = tuple(self._value(x) for x in islice(self.llrs, count))
            if len(values) != count:
                self.report("truncated", row, partial_reservation=start > 0 and not row["lost"])
                if not self.drain_at_eof:
                    raise ValueError("Truncated input before complete B1 LLR stream")
                if self.frame_llrs is None:
                    self.pending = None
                    break
                if start > 0 and not row["lost"]:
                    self.owners[row["ingress_slot"]] = None
                try:
                    self.pending = self.beats.send(self.edge)
                except StopIteration:
                    self.pending = None
                continue
            if not all(map(isfinite, values)):
                raise ValueError("Need finite B1 input LLRs")
            if row["lost"]:
                if start == 0:
                    self.report(row["status"], row, first_frame_loss=row["first_frame_loss"])
            else:
                self._write(row, start, values)
            self.pending = next(self.beats, None)
        self.edge = edge

    def check_input_end(self):
        end = object()
        if next(self.llrs, end) is not end:
            raise ValueError("Trailing LLRs do not belong to any scheduled B1 frame")

    def _value(self, item):
        if isinstance(item, TimedLLR):
            if item.edge != self.edge:
                raise ValueError("LLR missed its registered ingress edge")
            item = item.value
        return float(item)

    def _write(self, row, start, values):
        slot = row["ingress_slot"]
        owner = self.owners[slot]
        if start == 0:
            if owner is not None and self.edge < owner["ingress_release"]:
                raise RuntimeError("Ingress bank reused before its owner retired")
            self.report("reserve", row, slot=slot, prior_decode=owner and owner["decoded"],
                        reused_during_decode=owner is not None and self.edge < owner["decoded"])
            self.owners[slot] = row
        elif owner is not row:
            raise RuntimeError("Ingress beat has no slot ownership")
        if self.write_edge != self.edge:
            self.written.clear()
            self.write_edge = self.edge
        for index, value in enumerate(values, start):
            lane, address = index % 3, index // 3
            if (slot, lane) in self.written:
                raise RuntimeError("Ingress beat exceeded a single-port bank")
            self.written.add((slot, lane))
            self.banks[slot][lane][address] = _quantize_llr(value, self.scale)
        self.report("write", row, slot=slot, address=start, count=len(values))
        if start + len(values) == row["shape"][0]:
            if len(self.ready) == len(self.banks):
                raise RuntimeError("More complete words than owned ingress slots")
            self.ready.append(row)

    def read(self, row, lane, address):
        slot = row["ingress_slot"]
        if self.owners[slot] is not row or self.edge >= row["ingress_release"]:
            raise RuntimeError("Decoder read after ingress ownership retired")
        if self.edge == self.write_edge and (slot, lane) in self.written:
            raise RuntimeError("Decoder read conflicted with ingress bank write")
        self.report("read", row, slot=slot, lane=lane, address=address)



class _ShortOutput:
    """One detached result, drained in timestamp order with ingress events."""

    def __init__(self, ingress, output_pair):
        self.ingress, self.output_pair = ingress, output_pair
        self.held = self.completed = None
        self.next_edge = self.offset = None
        self.last_output_edge = -1

    def send(self, row, edge, offset, pair):
        if edge <= self.last_output_edge:
            raise RuntimeError("Polar results competed for the ordered output port")
        self.last_output_edge = edge
        if self.output_pair is not None:
            self.output_pair(row, edge, offset, pair)

    def hold(self, result):
        bits = len(result["info_bits"])
        if self.held is not None or bits > _SHORT_RESULT_BITS:
            raise RuntimeError("Polar short-result slot capacity exceeded")
        if self.ingress.edge != result["short_result_ready"]:
            raise RuntimeError("Polar short result missed the copy deadline")
        self.held = result
        self.offset = 0
        self.next_edge = result["output_start"] + 1
        self.ingress.report("result_buffered", result, bits=bits, until=result["taken"])

    def advance(self, edge):
        while self.held is not None and self.next_edge <= edge:
            at = self.next_edge
            receipt = self.held
            self.ingress.advance(at)
            pair = receipt["info_bits"][self.offset:self.offset + 2]
            self.send(receipt, at, self.offset, pair)
            self.offset += len(pair)
            if self.offset == len(receipt["info_bits"]):
                if at != receipt["taken"] or self.completed is not None:
                    raise RuntimeError("Polar short-result retirement mismatch")
                self.ingress.report("result_buffer_released", receipt)
                self.completed, self.held = receipt, None
            else:
                self.next_edge += 1
        self.ingress.advance(edge)

    def take_completed(self):
        receipt, self.completed = self.completed, None
        return receipt

def replay_payload_polar_ingress(
    frames, llrs, *, word_cycles, clock_hz, lanes=2, fraction_bits=0, geometry="indexed",
    stall_until=0, input_load_cycles=None, discard_failed_suffix=False, observer=None,
    output_pair=None, drain_at_eof=False, short_tail_slot=False, short_result_slot=False,
    _frame_llrs=None,
):
    """Execute value and port checks on an ordered stream of input LLRs.

    Frames start after validated header geometry. Missing required words and
    suppressed suffixes are reported to the observer as separate events.
    Returned words remain unchecked. In particular, a later missing word can
    invalidate a frame that has already produced a prefix of unchecked bits.
    With drain_at_eof, complete admitted words retire after a truncated beat.
    The internal frame-local source callback allows the next independent frame
    to start after a truncated beat, without inserting LLRs for the missing tail.
    A short result can be copied into one bounded slot before it is delivered.
    Ingress and result pairs still advance in absolute clock order.
    """
    _formats(lanes, fraction_bits, geometry)
    rows = payload_polar_calendar(
        frames, word_cycles=word_cycles, clock_hz=clock_hz, stall_until=stall_until,
        input_load_cycles=input_load_cycles, discard_failed_suffix=discard_failed_suffix,
        trace_ownership=True, short_tail_slot=short_tail_slot,
        short_result_slot=short_result_slot,
    )
    ingress = _Ingress(_sample_beats(rows, clock_hz // 4_000_000), llrs,
                       1 << fraction_bits, observer, drain_at_eof, short_tail_slot, _frame_llrs)
    ordered = _ShortOutput(ingress, output_pair)
    advance_clock = ordered.advance if short_result_slot else ingress.advance
    while ingress.pending is not None or ingress.ready:
        if not ingress.ready:
            advance_clock(ingress.pending[0])
            if (completed := ordered.take_completed()) is not None:
                yield completed
            continue
        row = ingress.ready.popleft()
        n, k = row["shape"]
        slot = row["ingress_slot"]
        launch = row["launch"]

        def advance(cycle):
            advance_clock(launch + cycle)

        def read_x(lane, address):
            ingress.read(row, lane, address)

        model = _Datapath(n, k, None, lanes, fraction_bits, geometry,
                          banks=ingress.banks[slot], advance=advance, read_x=read_x)
        advance_clock(launch)
        model.decode()
        if model.cycles != row["decoded"] - launch:
            raise ValueError("Decode cycles disagree with actual Polar execution")
        if input_load_cycles is not None and model.input_load_cycles != input_load_cycles[n, k]:
            raise ValueError("LOAD retirement disagrees with actual Polar execution")
        advance_clock(row["decoded"])
        copy = "short_result_ready" in row
        drain_start = row["decoded"] if copy else row["output_start"]
        copied_bits = []

        def emit(offset, pair):
            edge = drain_start + offset
            advance_clock(edge)
            if copy:
                copied_bits.extend(pair)
                if len(copied_bits) > _SHORT_RESULT_BITS:
                    raise RuntimeError("Polar short-result copy exceeded its data width")
                ingress.report("result_copy_pair", row, offset=2 * (offset - 1))
            else:
                ordered.send(row, edge, 2 * (offset - 1), pair)

        bits, output = model.drain(emit if short_result_slot or output_pair is not None else None)
        last = row["short_result_ready"] if copy else row["taken"]
        if output["output_to_last_bit_cycles"] != last - drain_start:
            raise RuntimeError("Output retirement disagrees with actual result ports")
        advance_clock(last)
        receipt = row | dict(info_bits=bits, input_last_read=launch + model.input_last_read_cycle,
                             load_complete=launch + model.input_load_cycles)
        if (completed := ordered.take_completed()) is not None:
            yield completed
        if copy:
            if tuple(copied_bits) != bits:
                raise RuntimeError("Polar short result changed the decoded bits")
            ordered.hold(receipt | dict(info_bits=tuple(copied_bits)))
        else:
            yield receipt
    if ordered.held is not None:
        advance_clock(ordered.held["taken"])
        yield ordered.take_completed()
    ingress.check_input_end()
