"""Bounded FT3/4 B1 reassembly and CRC at registered Polar-result edges."""

from collections import Counter, deque
from dataclasses import dataclass

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.crc import CRC24A_POLY, CRC24B_POLY, CRC32_POLY
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.payload_polar_flow import B1PayloadArrival
from nearlink_sdr.sim.payload_polar_ingress import replay_payload_polar_ingress

__all__ = ["B1PayloadFrame", "replay_b1_payload_frames"]

_MAX_BYTES = 2044
_RING_FRAME_LIMIT = 3
_FRAME_POLY = {24: CRC24A_POLY, 32: CRC32_POLY}


@dataclass(frozen=True)
class B1PayloadFrame:
    """Post-header context for a standard coded QPSK/8PSK frame."""

    arrival: B1PayloadArrival
    length_bytes: int
    crc_width: int
    crc_seed: int

    def __post_init__(self):
        if (type(self.crc_width) is not int or self.crc_width not in _FRAME_POLY
                or type(self.crc_seed) is not int):
            raise ValueError("Need valid B1 data CRC context")
        if not 0 <= self.crc_seed < 2**32:
            raise ValueError("Need a 32-bit B1 link CRC seed")
        if (not isinstance(self.arrival, B1PayloadArrival)
                or type(self.length_bytes) is not int or self.length_bytes < 0
                or self.length_bytes + self.crc_width // 8 > 2047
                or self.arrival.payload_with_crc_bits != 8 * self.length_bytes + self.crc_width):
            raise ValueError("Invalid standard B1 frame length")

    @classmethod
    def from_config(cls, first_edge: int, length_bytes: int, cfg: TxConfig):
        if (not isinstance(cfg, TxConfig) or cfg.frame_type not in (3, 4)
                or cfg.b1_code_block_mode != "standard" or cfg.is_uncoded
                or cfg.symbol_rate_mhz != 4 or cfg.mod_str not in ("QPSK", "8PSK")):
            raise ValueError("Need 4 Msym/s standard coded FT3/4 QPSK/8PSK")
        bps = 2 if cfg.mod_str == "QPSK" else 3
        arrival = B1PayloadArrival(first_edge, 8 * length_bytes + cfg.crc_len,
                                   cfg.rate_str, bps)
        return cls(arrival, length_bytes, cfg.crc_len, cfg.crc_seed)


def _crc_pair(state, pair, poly, width):
    mask = (1 << width) - 1
    for bit in pair:
        feedback = (state >> (width - 1)) ^ bit
        state = ((state << 1) & mask) ^ (poly if feedback else 0)
    return state


def _ring_address(address):
    return address - _MAX_BYTES if address >= _MAX_BYTES else address


class _Assembly:
    """One active logical block and one running frame CRC."""

    def __init__(self, frame, spec, base):
        self.frame, self.spec, self.base = frame, spec, base
        plan = b1_code_block_plan(spec.arrival.payload_with_crc_bits,
                                  spec.arrival.rate, standard=True)
        self.first_block, self.last_block = plan[0], plan[-1]
        self.block_count = len(plan)
        self.block_index = self.word_index = 0
        self.block_pos = self.frame_pos = 0
        self.byte = self.bytes_written = 0
        self.frame_crc = spec.crc_seed & ((1 << spec.crc_width) - 1)
        self.block_crc = spec.crc_seed & 0xFFFFFF
        self.bad_blocks = self.checked_blocks = 0

    def consume(self, row, offset, pair, write_byte, counts):
        block = self.last_block if self.block_index == self.block_count - 1 else self.first_block
        word = block.codewords[self.word_index]
        if (row["frame"], row["block"], row["word"], row["shape"]) != (
            self.frame, self.block_index, self.word_index, word,
        ):
            raise RuntimeError("Unexpected physical word ownership in B1 frame")
        _, k = word
        if len(pair) != 2 or offset < 0 or offset + 2 > k:
            raise RuntimeError("Expected an aligned B1 result pair")
        old = self.block_pos
        self.block_pos += 2
        padding = sum(w[1] for w in block.codewords) - block.data_bits - block.crc_bits
        if old >= padding:
            if block.crc_bits:
                self.block_crc = _crc_pair(self.block_crc, pair, CRC24B_POLY, 24)
                counts["block_crc_pairs"] += 1
            if old < padding + block.data_bits:
                width = self.spec.crc_width
                self.frame_crc = _crc_pair(self.frame_crc, pair, _FRAME_POLY[width], width)
                counts["frame_crc_pairs"] += 1
                if self.frame_pos < self.spec.length_bytes * 8:
                    self.byte = (self.byte << 2) | (pair[0] << 1) | pair[1]
                    if self.frame_pos % 8 == 6:
                        write_byte(_ring_address(self.base + self.bytes_written), self.byte)
                        self.bytes_written += 1
                        self.byte = 0
                        counts["frame_ram_writes"] += 1
                self.frame_pos += 2
        if offset + 2 == k:
            self.word_index += 1
            if self.word_index == len(block.codewords):
                if block.crc_bits:
                    self.checked_blocks += 1
                    self.bad_blocks += self.block_crc != 0
                self.word_index = self.block_pos = 0
                self.block_index += 1
                self.block_crc = self.spec.crc_seed & 0xFFFFFF
        return self.block_index == self.block_count


@dataclass
class _Owner:
    spec: B1PayloadFrame
    pending: int = 0
    failed: bool = False


@dataclass
class _Delivery:
    frame: int
    length: int
    edge: int
    base: int
    position: int = 0
    pending: int | None = None


class _Consumer:
    def __init__(self, mac_stall_until, observer, owner_limit, mac_ring, mac_frame_capacity):
        self.mac_stall_until = mac_stall_until
        self.owner_limit = owner_limit
        self.observer = observer
        self.memory = bytearray(_MAX_BYTES)
        self.owners = {}
        self.wire_frame = self.offered = None
        self.wire_failed = False
        self.active = None
        self.deliveries = deque()
        self.mac_ring = mac_ring
        self.mac_frame_capacity = mac_frame_capacity
        self.ring_tail = self.ring_bytes = self.ring_peak = self.ring_owners_peak = 0
        self.ring_wraps = self.ring_read_write_edges = 0
        self.edge = self.last_read_edge = self.last_read_address = -1
        self.counts = Counter()
        self.peak_owners = self.peak_buffered_bytes = 0

    def emit(self, kind, frame, edge, **data):
        self.counts[kind] += 1
        if self.observer is not None:
            self.observer(dict(kind=kind, frame=frame, edge=edge, **data))

    def advance(self, edge):
        if edge < self.edge:
            raise RuntimeError("B1 service clock moved backwards")
        while self.deliveries and self.deliveries[0].edge <= edge:
            out = self.deliveries[0]
            if out.pending is not None:
                self.emit("mac_byte", out.frame, out.edge,
                          position=out.position - 1, value=out.pending)
            if out.position == out.length:
                self.emit("mac_complete", out.frame, out.edge, length_bytes=out.length)
                self.deliveries.popleft()
                if self.mac_ring:
                    self.ring_bytes -= out.length
                if self.deliveries:
                    next_frame = self.deliveries[0]
                    next_frame.edge = max(next_frame.edge, out.edge + 1)
            else:
                address = _ring_address(out.base + out.position)
                out.pending = self.memory[address]
                self.last_read_edge, self.last_read_address = out.edge, address
                self.counts["frame_ram_reads"] += 1
                out.position += 1
                out.edge += 1
        self.edge = edge

    def write_byte(self, address, value):
        if self.edge == self.last_read_edge:
            if self.last_read_address == address:
                raise RuntimeError("B1 RAM read and write addressed the same byte")
            self.ring_read_write_edges += 1
        self.memory[address] = value

    def reserve_payload(self, frame, spec, edge):
        base = 0
        if self.mac_ring:
            if len(self.deliveries) >= self.mac_frame_capacity:
                self.fail(frame, edge, "mac_descriptor_full")
                return
            if self.ring_bytes + spec.length_bytes > _MAX_BYTES:
                self.fail(frame, edge, "mac_capacity_full")
                return
            base = self.ring_tail
            end = base + spec.length_bytes
            self.ring_wraps += int(end > _MAX_BYTES)
            self.ring_tail = _ring_address(end)
            self.ring_bytes += spec.length_bytes
            self.ring_peak = max(self.ring_peak, self.ring_bytes)
            self.ring_owners_peak = max(self.ring_owners_peak, len(self.deliveries) + 1)
        elif self.deliveries:
            self.fail(frame, edge, "mac_buffer_busy")
            return
        self.active = _Assembly(frame, spec, base)

    def fail(self, frame, edge, reason):
        if frame == self.wire_frame:
            self.wire_failed = True
        if owner := self.owners.get(frame):
            owner.failed = True
        if self.active is not None and self.active.frame == frame:
            if self.mac_ring:
                self.ring_tail = self.active.base
                self.ring_bytes -= self.active.spec.length_bytes
            self.active = None
        self.emit("frame_failed", frame, edge, reason=reason)

    def event(self, event):
        kind, frame, edge = event["kind"], event["frame"], event["edge"]
        if kind in ("result_buffered", "result_buffer_released"):
            self.advance(edge)
            extra = {"bits": event["bits"], "until": event["until"]} if "bits" in event else {}
            self.emit(kind, frame, edge, **extra)
            return
        if kind not in ("reserve", "resource_drop", "truncated"):
            return
        self.advance(edge)
        if frame != self.wire_frame:
            self.wire_frame, self.wire_failed = frame, False
        if kind == "reserve":
            if frame not in self.owners:
                if len(self.owners) == self.owner_limit:
                    raise RuntimeError("B1 ingress exceeded its outstanding frame owner limit")
                self.owners[frame] = _Owner(self.offered, failed=self.wire_failed)
            self.owners[frame].pending += 1
            if "packed_base" in event:
                self.emit("packed_tail_reserved", frame, edge,
                          slot=event["slot"], base=event["packed_base"])
            self.peak_owners = max(self.peak_owners, len(self.owners))
        elif kind == "resource_drop":
            if event["first_frame_loss"] and not self.wire_failed:
                self.fail(frame, edge, "resource_drop")
        else:
            if not self.wire_failed:
                self.fail(frame, edge, "truncated")
            if event["partial_reservation"]:
                self.retire(frame)

    def retire(self, frame):
        owner = self.owners[frame]
        owner.pending -= 1
        if owner.pending == 0:
            del self.owners[frame]

    def pair(self, row, edge, offset, pair):
        self.advance(edge)
        frame = row["frame"]
        owner = self.owners[frame]
        if not owner.failed and self.active is None:
            self.reserve_payload(frame, owner.spec, edge)
        if not owner.failed:
            state = self.active
            complete = state.consume(row, offset, pair, self.write_byte, self.counts)
            self.peak_buffered_bytes = max(self.peak_buffered_bytes, state.bytes_written)
            if complete:
                self.finish_frame(state, edge)
        if offset + 2 == row["shape"][1]:
            self.retire(frame)

    def finish_frame(self, state, edge):
        length = state.spec.length_bytes
        if (state.frame_pos != length * 8 + state.spec.crc_width
                or state.bytes_written != length):
            raise RuntimeError("Reassembled length disagrees with B1 head")
        frame_ok, block_ok = state.frame_crc == 0, state.bad_blocks == 0
        self.emit("crc_result", state.frame, edge, block_crc_ok=block_ok,
                  frame_crc_ok=frame_ok, checked_blocks=state.checked_blocks,
                  bad_blocks=state.bad_blocks, length_bytes=length)
        if block_ok and frame_ok:
            self.active = None
            self.deliveries.append(
                _Delivery(state.frame, length, max(edge + 1, self.mac_stall_until), state.base)
            )
        else:
            self.fail(state.frame, edge, "crc")

    def finish(self):
        if self.active is not None or self.owners:
            raise RuntimeError("B1 consumer has unfinished owned state")
        while self.deliveries:
            out = self.deliveries[0]
            self.advance(out.edge + out.length - out.position)
        if self.ring_bytes:
            raise RuntimeError("B1 payload reservations leaked")
        result = dict(self.counts, peak_frame_owners=self.peak_owners,
                      peak_buffered_bytes=self.peak_buffered_bytes, frame_ram_bytes=_MAX_BYTES)
        if self.mac_ring:
            result.update(peak_ram_reserved_bytes=self.ring_peak,
                          peak_payload_reservations=self.ring_owners_peak,
                          ram_wrap_reservations=self.ring_wraps,
                          ram_same_edge_read_writes=self.ring_read_write_edges)
            if self.mac_frame_capacity != _RING_FRAME_LIMIT:
                result["mac_frame_capacity"] = self.mac_frame_capacity
        return result


def replay_b1_payload_frames(
    frames, llrs, *, word_cycles, clock_hz, lanes=2, fraction_bits=0, geometry="indexed",
    stall_until=0, input_load_cycles=None, discard_failed_suffix=False,
    mac_stall_until=0, observer=None, short_tail_slot=False, mac_ring=False,
    short_result_slot=False, prefer_short_tail=False, packed_tail_slot=False,
    mac_frame_capacity=_RING_FRAME_LIMIT, _frame_llrs=False,
):
    """Reassemble at actual Polar pair edges and deliver only completed, CRC-valid frames.

    Default ownership is one frame in the 2044-byte RAM. Opt-in mac_ring
    reserves disjoint byte ranges in the same RAM, with three owners by
    default. The explicit fourth descriptor retains the original byte limit.
    CRC failure rolls back only the active, newest reservation.
    """
    if type(mac_stall_until) is not int or mac_stall_until < 0:
        raise ValueError("Need a nonnegative MAC readiness edge")
    if type(short_tail_slot) is not bool:
        raise ValueError("Need a boolean short-tail slot option")
    if type(mac_ring) is not bool:
        raise ValueError("Need a boolean MAC ring option")
    if type(short_result_slot) is not bool:
        raise ValueError("Need a boolean short-result slot option")
    if type(packed_tail_slot) is not bool:
        raise ValueError("Need a boolean packed-tail slot option")
    if (type(mac_frame_capacity) is not int or mac_frame_capacity not in (3, 4)
            or (mac_frame_capacity == 4 and not mac_ring)):
        raise ValueError("Need three MAC descriptors, or four with the MAC ring")
    consumer = _Consumer(
        mac_stall_until, observer, 3 + short_tail_slot + short_result_slot + packed_tail_slot,
        mac_ring, mac_frame_capacity,
    )

    local_llrs = None

    def wire():
        nonlocal local_llrs
        for entry in frames:
            if _frame_llrs:
                spec, local_llrs = entry
            else:
                spec = entry
            if not isinstance(spec, B1PayloadFrame):
                raise ValueError("Expected validated B1 payload frame descriptions")
            consumer.offered = spec
            yield spec.arrival

    for _ in replay_payload_polar_ingress(
        wire(), llrs, word_cycles=word_cycles, clock_hz=clock_hz, lanes=lanes,
        fraction_bits=fraction_bits, geometry=geometry, stall_until=stall_until,
        input_load_cycles=input_load_cycles, discard_failed_suffix=discard_failed_suffix,
        observer=consumer.event, output_pair=consumer.pair, drain_at_eof=True,
        short_tail_slot=short_tail_slot, short_result_slot=short_result_slot,
        prefer_short_tail=prefer_short_tail, packed_tail_slot=packed_tail_slot,
        _frame_llrs=(lambda: local_llrs) if _frame_llrs else None,
    ):
        pass
    return consumer.finish()
