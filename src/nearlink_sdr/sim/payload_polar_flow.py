"""Value-carrying soft Polar service for standard B1 words (§6.9.1.4, Appendix C).

The decoder consumes descrambled, finite LLRs. Scheduling at this boundary does
not charge carrier, demapping, reassembly, or CRC work as zero service.
"""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import islice
from math import ceil, isfinite

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.polar import RATE_TABLE, RELIABILITY_SEQ_1024
from nearlink_sdr.phy.control_polar_word import ssc_geometry

__all__ = [
    "B1PayloadArrival",
    "decode_payload_polar_word",
    "payload_polar_calendar",
    "payload_polar_cost",
]

_SHAPES = frozenset(
    shape for table in RATE_TABLE.values()
    for shape in (*table.items(), (1024, table[64] * 16))
)


def _formats(lanes, fraction_bits, geometry):
    if type(lanes) is not int or lanes not in (1, 2):
        raise ValueError("Need one or two SSC issue lanes")
    if type(fraction_bits) is not int or fraction_bits not in (0, 1):
        raise ValueError("Need Q8.0 or Q9.1 ingress")
    if geometry not in ("indexed", "ordered", "threshold"):
        raise ValueError("Need indexed, ordered or threshold Polar geometry")


@lru_cache(maxsize=1)
def _threshold_rom():
    """Compile two read-only banks shared by all code lengths and rates.

    Each 8-bit word holds the first allowed rank with any information bits and
    the first with all information bits. A heap node absent from all formats
    uses the size of its format family as a sentinel.
    """
    banks = [[], []]
    directories = {}
    for n in sorted({n for n, _ in _SHAPES}):
        dimensions = tuple(sorted(k for size, k in _SHAPES if size == n))
        base = tuple(map(len, banks))
        directories[n] = dimensions, base
        heap = [0] * (2 * n - 1)
        for rank, position in enumerate(i for i in RELIABILITY_SEQ_1024 if i < n):
            first = bisect_left(dimensions, n - rank)
            assert first < 16
            heap[n - 1 + position] = (first << 4) | first
        for node in range(n - 2, -1, -1):
            left, right = heap[2 * node + 1:2 * node + 3]
            some = min(left >> 4, right >> 4)
            full = max(left & 15, right & 15)
            heap[node] = (some << 4) | full
        for parity in range(2):
            banks[parity].extend(heap[parity::2])
    return tuple(tuple(bank) for bank in banks), directories


def _depth_base(n_code, depth, *, beta=False):
    # Full alpha length per depth. For beta, children share a depth's RAM in
    # disjoint halves until the parent's combine finishes. Root has one half.
    return sum(
        (n_code >> d) * (2 if beta and d else 1)
        for d in range(depth & 1, depth, 2)
    )


def _address(n_code, plane, depth, offset):
    if plane in ("L", "B"):
        return (plane, depth & 1, offset & 1), (
            _depth_base(n_code, depth, beta=plane == "B") + offset
        ) // 2
    lane_count = 3 if plane == "X" else 2
    return (plane, offset % lane_count), offset // lane_count


def _depth_shapes(n_code):
    shapes = {}
    for d in range(n_code.bit_length()):
        for plane in ("L", "B"):
            width = (n_code >> d) * (2 if plane == "B" and d else 1)
            for lane in range(min(2, width)):
                bank, address = _address(n_code, plane, d, width - 1 - ((width - 1 - lane) & 1))
                shapes[bank] = address + 1
    return shapes


@dataclass(frozen=True)
class _Op:
    reads: tuple
    writes: tuple


def _evaluate(kind, operands):
    if kind in ("load", "node", "extract"):
        return operands
    if kind == "zero":
        return (0,)
    if kind == "hard":
        bit = int(operands[0] < 0)
        return bit, bit
    if kind in ("xor", "combine"):
        a, b = operands
        return (a ^ b, b) if kind == "combine" else (a ^ b,)
    if kind == "f":
        a, b = operands
        magnitude = min(abs(a), abs(b))
        return (-magnitude if (a < 0) != (b < 0) else magnitude,)
    a, b, beta = operands
    return (b - a if beta else b + a,)


class _Datapath:
    """Registered data and ROM banks; a bounded issue/ALU/writeback pipeline."""

    def __init__(self, n_code, k_info, llr, lanes, fraction_bits, geometry, *,
                 banks=None, advance=None, read_x=None):
        self.n, self.k, self.lanes = n_code, k_info, lanes
        self.ordered = geometry != "indexed"
        self.family = None
        self.packed_count = 0
        self.maximum = (64 << fraction_bits) * n_code
        self.data = {bank: [None] * depth for bank, depth in _depth_shapes(n_code).items()}
        for lane in range(3):
            self.data["X", lane] = llr[lane::3] if banks is None else banks[lane]
        self.advance, self.read_x = advance, read_x
        self.input_last_read_cycle = self.input_load_cycles = None
        if geometry == "threshold":
            banks, directories = _threshold_rom()
            dimensions, self.node_base = directories[n_code]
            self.family = dimensions.index(k_info)
            self.data["T", 0], self.data["T", 1] = banks
        else:
            nodes, info = ssc_geometry(n_code, k_info)
            self.data["T", 0] = nodes
            if not self.ordered:
                for lane in range(2):
                    self.data["I", lane] = info[lane::2]
                    self.data["O", lane] = [None] * ((k_info + 1 - lane) // 2)
        scratch_length = k_info if self.ordered else n_code
        for lane in range(2):
            self.data["U", lane] = [None] * ((scratch_length + 1 - lane) // 2)
        self.limits = {bank: 2 if bank[0] in "LBU" else 1 for bank in self.data}
        self.cycles = 0
        self.peak_llr = 0
        self.peak_ports = Counter()
        self.reads = Counter()
        self.writes = Counter()
        self.ops = Counter()
        self.port_stalls = 0
        self.blocks = 0
        self.splits = 0

    def group(self, kind, count, command):
        """One registered block setup, followed by issue and three drain stages.

        Each bundle occupies one pipeline register at a time. Commit has port
        priority. Extraction reads the I ROM, then uses its registered address
        for the U read. No other opcode competes for U within that block.
        """
        self.blocks += 1
        self.cycles += 1
        pipe = [(), (), ()]
        i = 0
        last = None
        while i < count or any(pipe):
            if self.advance is not None:
                self.advance(self.cycles)
            ports = Counter()
            overwritten = set()
            for writes, values in pipe[2]:
                last = values
                for (bank, address), value in zip(writes, values[:len(writes)], strict=True):
                    assert 0 <= address < len(self.data[bank])
                    if bank[0] == "L":
                        self.peak_llr = max(self.peak_llr, abs(value))
                        if abs(value) > self.maximum:
                            raise OverflowError("SSC accumulator exceeded its proved signed range")
                    self.data[bank][address] = value
                    ports[bank] += 1
                    self.writes[bank] += 1
                    overwritten.add((bank, address))
            advance = []
            for writes, operands in pipe[0]:
                if kind == "extract":
                    addr = _address(self.n, "U", 0, operands[0])
                    operands = self._read((addr,), ports)
                elif kind == "node" and self.family is not None:
                    (cutoffs,) = operands
                    some, full = cutoffs >> 4, cutoffs & 15
                    operands = (0 if self.family < some else 1 if self.family >= full else 2,)
                advance.append((writes, _evaluate(kind, operands)))
            launch = []
            for _ in range(min(self.lanes, count - i)):
                op = command(i)
                demands = Counter(bank for bank, _ in op.reads)
                if (any(ports[bank] + used > self.limits[bank]
                        for bank, used in demands.items())
                        or overwritten.intersection(op.reads)):
                    if launch:
                        self.splits += 1
                    else:
                        self.port_stalls += 1
                    break
                launch.append((op.writes, self._read(op.reads, ports)))
                self.ops[kind] += 1
                i += 1
            for bank, used in ports.items():
                assert used <= self.limits[bank]
                self.peak_ports[bank] = max(self.peak_ports[bank], used)
            pipe = [launch, advance, pipe[1]]
            self.cycles += 1
        return last

    def _read(self, requests, ports):
        values = []
        for bank, address in requests:
            if bank[0] == "X":
                self.input_last_read_cycle = self.cycles
                if self.read_x is not None:
                    self.read_x(bank[1], address)
            value = self.data[bank][address]
            if value is None:
                raise RuntimeError("SSC RAM read before write")
            self.reads[bank] += 1
            ports[bank] += 1
            values.append(value)
        return tuple(values)

    def _node_address(self, node):
        if self.family is None:
            return ("T", 0), node
        parity = node & 1
        return ("T", parity), self.node_base[parity] + node // 2

    def decode(self):
        n = self.n
        def a(plane, depth, i):
            return _address(n, plane, depth, i)
        self.group("load", n, lambda i: _Op((a("X", 0, i),), (a("L", 0, i),)))
        self.input_load_cycles = self.cycles

        def walk(start, length, depth, node, right):
            # A frame has only these bounded fields; the maximum depth is 11.
            (kind,) = self.group("node", 1, lambda _: _Op((self._node_address(node),), ()))
            info_base = self.packed_count if self.ordered else start
            def alpha(i):
                return a("L", depth, i)
            def beta(i):
                return a("B", depth, right * length + i)
            def u(i):
                return a("U", 0, info_base + i)
            if kind == 0:
                self.group("zero", length, lambda i: _Op((), (beta(i),)))
                return
            if kind == 1:
                if self.ordered:
                    # SSC traverses left before right. Each rate-1 node owns
                    # the next consecutive rank range until the result drains.
                    self.packed_count += length
                    assert self.packed_count <= self.k
                self.group("hard", length, lambda i: _Op((alpha(i),), (beta(i), u(i))))
                for shift in range(length.bit_length() - 2, -1, -1):
                    h = 1 << shift

                    def butterfly(i):
                        pos = (i // h) * (2 * h) + i % h
                        return _Op((u(pos), u(pos + h)), (u(pos),))

                    self.group("xor", length // 2, butterfly)
                return
            half = length // 2
            def left(i):
                return a("B", depth + 1, i)
            def next_alpha(i):
                return a("L", depth + 1, i)
            self.group("f", half, lambda i: _Op(
                (alpha(i), alpha(half + i)), (next_alpha(i),),
            ))
            walk(start, half, depth + 1, 2 * node + 1, 0)
            self.group("g", half, lambda i: _Op(
                (alpha(i), alpha(half + i), left(i)), (next_alpha(i),),
            ))
            walk(start + half, half, depth + 1, 2 * node + 2, 1)
            self.group("combine", half, lambda i: _Op(
                (left(i), left(half + i)), (beta(i), beta(half + i)),
            ))

        walk(0, n, 0, 0, 0)
        if self.ordered:
            assert self.packed_count == self.k
        else:
            self.group("extract", self.k, lambda i: _Op((a("I", 0, i),), (a("O", 0, i),)))

    def drain(self, emit=None):
        """Pass at most two bits per edge via registered, single-read ports."""
        plane = "U" if self.ordered else "O"
        result, pending = [], ()
        read_edges = 0
        used = Counter()
        for start in range(0, self.k, 2):
            if pending and emit is not None:
                emit(read_edges, pending)
            result.extend(pending)
            addrs = tuple(_address(self.n, plane, 0, i)
                          for i in range(start, min(start + 2, self.k)))
            ports = Counter()
            pending = self._read(addrs, ports)
            for bank, count in ports.items():
                assert count == 1 and count <= self.limits[bank]
            used.update(ports)
            read_edges += 1
        # The last registered pair is visible one edge after the final read.
        if emit is not None:
            emit(read_edges, pending)
        result.extend(pending)
        return tuple(result), dict(
            output_to_last_bit_cycles=read_edges,
            output_reads={str(bank): count for bank, count in used.items()},
            output_peak_reads_per_bank=1,
        )

    def summary(self):
        return dict(
            cycles=self.cycles, input_last_read_cycle=self.input_last_read_cycle,
            input_load_cycles=self.input_load_cycles, peak_abs_alpha=self.peak_llr,
            port_stalls=self.port_stalls,
            split_issue_edges=self.splits, blocks=self.blocks, operations=dict(self.ops),
            reads={str(bank): count for bank, count in self.reads.items()},
            writes={str(bank): count for bank, count in self.writes.items()},
            peak_ports={str(bank): count for bank, count in self.peak_ports.items()},
        )


def _quantize_llr(value, scale):
    return round(max(-64., min(64., value)) * scale)


def decode_payload_polar_word(
    llr, n_code, k_info, *, lanes=2, fraction_bits=0, geometry="indexed",
):
    """Return ordered decoded bits and a value/port-derived edge count.

    Input rounds nearest-even after symmetric ±64 clipping. Internal LLRs
    are exact signed integers, with hard-decision ties mapping to zero.
    Geometry controls only node ROM storage and ordered output addressing.
    """
    _formats(lanes, fraction_bits, geometry)
    if type(n_code) is not int or type(k_info) is not int or (n_code, k_info) not in _SHAPES:
        raise ValueError("Need a standard B1 physical Polar word shape")
    values = tuple(float(x) for x in islice(llr, n_code + 1))
    if len(values) != n_code or not all(map(isfinite, values)):
        raise ValueError("Need exactly N finite input LLRs")
    scale = 1 << fraction_bits
    q = [_quantize_llr(value, scale) for value in values]
    model = _Datapath(n_code, k_info, q, lanes, fraction_bits, geometry)
    model.decode()
    summary = model.summary()
    bits, output = model.drain()
    return bits, summary | output | dict(
        clipped_inputs=sum(abs(v) > 64 for v in values), quantized_zeros=q.count(0),
        input_lsb=1 / scale, signed_alpha_bits=model.maximum.bit_length() + 1,
    )


_SHORT_TAIL_CAPACITY = 128
_SHORT_RESULT_CODE = 64
_SHORT_RESULT_BITS = max(k for n, k in _SHAPES if n == _SHORT_RESULT_CODE)


def payload_polar_cost(
    *, lanes=2, fraction_bits=0, geometry="indexed", short_tail_slot=False, short_result_slot=False,
):
    """Maximum-width scratch/RAM packing estimate across all 36 B1 formats.

    ROMs are paid for all resident shapes. Each parity bank uses a separate
    physical primitive with simple one-read/one-write or true dual ports.
    """
    _formats(lanes, fraction_bits, geometry)
    if type(short_tail_slot) is not bool or type(short_result_slot) is not bool:
        raise ValueError("Need boolean short-tail and short-result slot options")
    alpha_bits = 18 + fraction_bits
    words = _depth_shapes(1024)
    width = {bank: alpha_bits if bank[0] == "L" else 1 for bank in words}
    ports = dict.fromkeys(words, 2)
    max_k = max(k for _, k in _SHAPES)
    for lane in range(2):
        count = (max_k + 1 - lane) // 2
        bank = "U", lane
        words[bank], width[bank], ports[bank] = 512 if geometry == "indexed" else count, 1, 2
        if geometry == "indexed":
            bank = "I", lane
            words[bank] = sum((k + 1 - lane) // 2 for _, k in _SHAPES)
            width[bank], ports[bank] = 10, 1
            bank = "O", lane
            words[bank], width[bank], ports[bank] = count, 1, 1
    for slot, size in enumerate((1024, 1024, _SHORT_TAIL_CAPACITY) if short_tail_slot
                                else (1024, 1024)):
        for lane in range(3):
            bank = "X", slot, lane
            words[bank], width[bank], ports[bank] = (size + 2 - lane) // 3, 8 + fraction_bits, 1
    if geometry == "threshold":
        threshold_banks, _ = _threshold_rom()
        for lane, values in enumerate(threshold_banks):
            bank = "T", lane
            words[bank], width[bank], ports[bank] = len(values), 8, 1
    else:
        words["T", 0] = sum(2*n - 1 for n, _ in _SHAPES)
        width["T", 0], ports["T", 0] = 2, 1
    # RAMB18 true dual permits x18; 19 bits need x36 on a RAMB36. Cascaded
    # banks use a registered select (the one-edge read is only an assumption).
    depth_native = {1: 16384, 2: 8192, 10: 1024, 18: 1024, 19: 1024, 8: 2048, 9: 2048}
    banks = {
        str(bank): dict(depth=depth, width=width[bank], ports=ports[bank],
                        ramb18=ceil(depth / depth_native[width[bank]])
                        * (2 if width[bank] == 19 else 1))
        for bank, depth in words.items()
    }
    # Per-frame stack, node/block/index control, registered operands/addresses,
    # symmetric clip interface, two ingress ownership tags and instrumentation.
    registers = dict(stack=11 * 64, issue_alu_write=lanes * 3 * 256,
                     control_address_select=384, ingress_ownership=2 * 128,
                     instrumentation=64 * 64, format_directory=len(_SHAPES) * 56,
                     depth_base_table=5 * 11 * 2 * 11)
    if geometry != "indexed":
        # Count goes through 896; the adds use non-conflicting rank ranges.
        registers["ordered_rate1_cursor_and_base"] = 20
    if geometry == "threshold":
        registers["threshold_family_word_and_bases"] = 4 + 8 + 2 * 11
    if short_tail_slot:
        registers.update(short_tail_ingress_descriptor=128, short_tail_MAC_owner=256,
                         short_tail_registered_read=3 * (8 + fraction_bits),
                         short_tail_control_and_mux_select=32)
    if short_result_slot:
        registers.update(short_result_data=_SHORT_RESULT_BITS, short_result_row=128,
                         short_result_edge=64, short_result_cursor_and_valid=7,
                         short_result_MAC_owner=256)
    return dict(
        shapes=len(_SHAPES), lanes=lanes, alpha_bits=alpha_bits, DSP=0,
        geometry=geometry, geometry_threshold_comparisons=2 if geometry == "threshold" else 0,
        ordered_base_and_cursor_adders=(
            dict(address=lanes, cursor=1, width=10) if geometry != "indexed" else None
        ),
        threshold_bank_address_adder_bits=11 if geometry == "threshold" else 0,
        logical_ram_rom_bits=sum(v["depth"] * v["width"] for v in banks.values()),
        absolute_alpha_bound=64 * (1 << fraction_bits) * 1024,
        minsum_lanes=dict(add_subtract=lanes, magnitude_min_compare=lanes,
                          hard_sign_decisions=lanes, beta_xor=lanes),
        per_bank_macro_read_cycles_assumed=1,
        RAMB18_equivalents=sum(v["ramb18"] for v in banks.values()),
        RAMB36_sites_upper=sum(ceil(v["ramb18"] / 2) for v in banks.values()),
        physical_banks=banks, register_bits=registers, LUT=None, FF=None,
        synthesized=False, verified_clock_hz=None,
    )


@dataclass(frozen=True)
class B1PayloadArrival:
    """One already-validated B1 payload, at its first demapped symbol edge."""

    first_edge: int
    payload_with_crc_bits: int
    rate: str
    bits_per_symbol: int
    pilot_interval: int = 0


def _wire_index(data_index, pilot_interval):
    return data_index + (data_index // pilot_interval if pilot_interval else 0)


def payload_polar_calendar(
    frames, *, word_cycles, clock_hz, stall_until=0, input_load_cycles=None,
    discard_failed_suffix=False, trace_ownership=False, short_tail_slot=False,
    short_result_slot=False,
):
    """Schedule two ingress slots and a single ordered held output.

    Completion refers to the unchecked Polar output boundary, not MAC delivery.
    B1 input cannot pause. A word with no free slot on its first beat is lost;
    earlier owned words remain fully accounted for. With measured LOAD edges,
    ingress ownership may retire before decode completes. First loss marks the
    frame incomplete. Optional suffix suppression affects later reservations.
    Ownership tracing exposes the wire geometry for finite RAM replay.
    The optional short slot is only reserved for a healthy frame's final
    physical codeword when neither full input slot is free. The optional short
    result slot detaches one complete N64 result when output is blocked at
    decode completion. Both paths retain issue order and the two-bit output rate.
    """
    if clock_hz not in (100_000_000, 160_000_000, 200_000_000):
        raise ValueError("Use a modeled clock: 100, 160 or 200 MHz")
    if type(stall_until) is not int or stall_until < 0:
        raise ValueError("Downstream release edge must be non-negative")
    if type(short_tail_slot) is not bool:
        raise ValueError("Need a boolean short-tail slot option")
    if type(short_result_slot) is not bool:
        raise ValueError("Need a boolean short-result slot option")
    period = Fraction(clock_hz, 4_000_000)
    detailed = (trace_ownership or input_load_cycles is not None or discard_failed_suffix
                or short_tail_slot or short_result_slot)
    # Expired reservations retain no ownership.
    slots = [0] * (3 if short_tail_slot else 2)
    engine_until = output_until = short_result_until = 0
    next_boundary = 0
    peak = 0
    for frame_no, frame in enumerate(frames):
        bps, pilot = frame.bits_per_symbol, frame.pilot_interval
        if type(pilot) is not int or pilot not in (0, 1, 4, 8, 16):
            raise ValueError("Need legal B1 pilot interval")
        if type(frame.first_edge) is not int or frame.first_edge < next_boundary:
            raise ValueError("Payload symbol intervals must be non-overlapping and non-negative")
        if bps not in (2, 3) or (type(frame.payload_with_crc_bits) is not int
                or not 24 <= frame.payload_with_crc_bits <= 2047 * 8
                or frame.payload_with_crc_bits % 8):
            raise ValueError("Need legal-range B1 length and QPSK/8PSK")
        blocks = b1_code_block_plan(frame.payload_with_crc_bits, frame.rate, standard=True)
        physical_bit = 0
        failed = False
        words = ((block_no, word_no, shape) for block_no, block in enumerate(blocks)
                 for word_no, shape in enumerate(block.codewords))
        for block_no, word_no, (n, k) in words:
            data_index, phase = divmod(physical_bit, bps)
            first = frame.first_edge + ceil(_wire_index(data_index, pilot) * period)
            physical_bit += n
            last_index = _wire_index((physical_bit - 1) // bps, pilot)
            last = frame.first_edge + ceil(last_index * period)
            cost = word_cycles[n, k]
            if type(cost) is not int or cost <= 0:
                raise ValueError("Need positive executed Polar edge counts")
            load = cost if input_load_cycles is None else input_load_cycles[n, k]
            if type(load) is not int or not 0 < load <= cost:
                raise ValueError("Need LOAD retirement within measured decode service")
            row = dict(frame=frame_no, block=block_no, word=word_no, shape=(n,k),
                       first=first, last=last)
            if detailed:
                row.update(bits_per_symbol=bps, first_symbol_phase=phase)
                if pilot:
                    row.update(pilot_interval=pilot, data_symbol_index=data_index)
            slot = None
            if discard_failed_suffix and failed:
                scheduled = row | dict(lost=True, peak_ingress=peak,
                                       status="incomplete_suffix", first_frame_loss=False)
            else:
                slot = next((i for i in range(2) if slots[i] <= first), None)
                if (slot is None and short_tail_slot and not failed and slots[2] <= first
                        and n <= _SHORT_TAIL_CAPACITY and block_no == len(blocks) - 1
                        and word_no == len(blocks[-1].codewords) - 1):
                    slot = 2
                if slot is None:
                    if detailed:
                        row.update(status="resource_drop", first_frame_loss=not failed)
                    failed = True
                    scheduled = row | dict(lost=True, peak_ingress=peak)
                else:
                    prior = slots[slot], engine_until, output_until, short_result_until
                    start = max(last + 1, engine_until)
                    done = start + cost
                    output_start = max(done, stall_until, output_until)
                    # Result RAM banks have registered reads, two bits per edge.
                    pair_count = (k + 1) // 2
                    detached = (short_result_slot and n == _SHORT_RESULT_CODE
                                and short_result_until <= done and done < output_start)
                    if detached:
                        row["short_result_ready"] = done + pair_count
                        output_start = max(output_start, row["short_result_ready"] + 1)
                    release_output = output_start + pair_count
                    engine_until = output_until = release_output + 1
                    if detached:
                        short_result_until = output_until
                        engine_until = row["short_result_ready"] + 1
                    slots[slot] = start + load
                    peak = max(peak, sum(end > first for end in slots))
                    if detailed:
                        row.update(status="accepted", ingress_release=slots[slot],
                                   first_frame_loss=False)
                    scheduled = row | dict(launch=start, decoded=done, output_start=output_start,
                                           taken=release_output, lost=False, ingress_slot=slot,
                                           peak_ingress=peak, queue_wait=start-last-1,
                                           output_wait=output_start-done,
                                           ordered_result_edge=release_output)
            absent_edge = yield scheduled
            if absent_edge is not None:
                # Only the word whose input beat was missing is speculative.
                # All earlier complete words keep their ownership and deadlines.
                if slot is not None:
                    slots[slot], engine_until, output_until, short_result_until = prior
                next_boundary = absent_edge + 1
                break
        else:
            next_boundary = frame.first_edge + ceil((last_index + 1) * period)
