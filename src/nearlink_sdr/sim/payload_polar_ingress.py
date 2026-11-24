"""Replay the standard B1 Polar ingress calendar with two actual RAM owners."""

from collections import deque
from itertools import islice
from math import isfinite

from nearlink_sdr.sim.payload_polar_flow import (
    _Datapath,
    _formats,
    _quantize_llr,
    payload_polar_calendar,
)

__all__ = ["replay_payload_polar_ingress"]


def _sample_beats(rows, period):
    for row in rows:
        n, _ = row["shape"]
        bps, phase = row["bits_per_symbol"], row["first_symbol_phase"]
        start = 0
        edge = row["first"]
        while start < n:
            count = min(bps - phase, n - start)
            yield edge, row, start, count
            start += count
            phase = 0
            edge += period
        assert edge - period == row["last"]


class _Ingress:
    """Each slot has one access per modulo-three bank and one logical owner."""

    def __init__(self, beats, llrs, scale, observer):
        self.beats = iter(beats)
        self.pending = next(self.beats, None)
        self.llrs = iter(llrs)
        self.scale = scale
        self.observer = observer
        self.banks = tuple(tuple([None] * ((1026 - lane) // 3) for lane in range(3))
                           for _ in range(2))
        self.owners = [None, None]
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
            values = tuple(float(x) for x in islice(self.llrs, count))
            if len(values) != count:
                self.report("truncated", row)
                raise ValueError("Truncated input before complete B1 LLR stream")
            if not all(map(isfinite, values)):
                raise ValueError("Need finite B1 input LLRs")
            if row["lost"]:
                if start == 0:
                    self.report(row["status"], row, first_frame_loss=row["first_frame_loss"])
            else:
                self._write(row, start, values)
            self.pending = next(self.beats, None)
        self.edge = edge

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
            if len(self.ready) == 2:
                raise RuntimeError("More than two complete words own ingress slots")
            self.ready.append(row)

    def read(self, row, lane, address):
        slot = row["ingress_slot"]
        if self.owners[slot] is not row or self.edge >= row["ingress_release"]:
            raise RuntimeError("Decoder read after ingress ownership retired")
        if self.edge == self.write_edge and (slot, lane) in self.written:
            raise RuntimeError("Decoder read conflicted with ingress bank write")
        self.report("read", row, slot=slot, lane=lane, address=address)


def replay_payload_polar_ingress(
    frames, llrs, *, word_cycles, clock_hz, lanes=2, fraction_bits=0, geometry="indexed",
    stall_until=0, input_load_cycles=None, discard_failed_suffix=False, observer=None,
):
    """Execute value and port checks on an ordered stream of input LLRs.

    Frames start after validated header geometry. Missing required words and
    suppressed suffixes are reported to the observer as separate events.
    Returned words remain unchecked. In particular, a later missing word can
    invalidate a frame that has already produced a prefix of unchecked bits.
    """
    _formats(lanes, fraction_bits, geometry)
    rows = payload_polar_calendar(
        frames, word_cycles=word_cycles, clock_hz=clock_hz, stall_until=stall_until,
        input_load_cycles=input_load_cycles, discard_failed_suffix=discard_failed_suffix,
        trace_ownership=True,
    )
    ingress = _Ingress(_sample_beats(rows, clock_hz // 4_000_000), llrs,
                       1 << fraction_bits, observer)
    while ingress.pending is not None or ingress.ready:
        if not ingress.ready:
            ingress.advance(ingress.pending[0])
            continue
        row = ingress.ready.popleft()
        n, k = row["shape"]
        slot = row["ingress_slot"]
        launch = row["launch"]

        def advance(cycle):
            ingress.advance(launch + cycle)

        def read_x(lane, address):
            ingress.read(row, lane, address)

        model = _Datapath(n, k, None, lanes, fraction_bits, geometry,
                          banks=ingress.banks[slot], advance=advance, read_x=read_x)
        ingress.advance(launch)
        model.decode()
        if model.cycles != row["decoded"] - launch:
            raise ValueError("Decode cycles disagree with actual Polar execution")
        if input_load_cycles is not None and model.input_load_cycles != input_load_cycles[n, k]:
            raise ValueError("LOAD retirement disagrees with actual Polar execution")
        ingress.advance(row["decoded"])
        bits, output = model.drain()
        if output["output_to_last_bit_cycles"] != row["taken"] - row["output_start"]:
            raise RuntimeError("Output retirement disagrees with actual result ports")
        ingress.advance(row["taken"])
        yield row | dict(info_bits=bits, input_last_read=launch + model.input_last_read_cycle,
                         load_complete=launch + model.input_load_cycles)
    end = object()
    if next(ingress.llrs, end) is not end:
        raise ValueError("Trailing LLRs do not belong to any scheduled B1 frame")
