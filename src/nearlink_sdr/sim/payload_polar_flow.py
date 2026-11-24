"""Value-carrying soft Polar service for standard B1 words (§6.9.1.4, Appendix C).

The decoder consumes descrambled, finite LLRs. Scheduling at this boundary does
not charge carrier, demapping, reassembly, or CRC work as zero service.
"""

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from itertools import islice
from math import ceil, isfinite

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.polar import RATE_TABLE
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


def _formats(lanes, fraction_bits):
    if type(lanes) is not int or lanes not in (1, 2):
        raise ValueError("Need one or two SSC issue lanes")
    if type(fraction_bits) is not int or fraction_bits not in (0, 1):
        raise ValueError("Need Q8.0 or Q9.1 ingress")


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

    def __init__(self, n_code, k_info, llr, lanes, fraction_bits):
        self.n, self.k, self.lanes = n_code, k_info, lanes
        self.maximum = (64 << fraction_bits) * n_code
        nodes, info = ssc_geometry(n_code, k_info)
        self.data = {bank: [None] * depth for bank, depth in _depth_shapes(n_code).items()}
        for lane in range(3):
            self.data["X", lane] = llr[lane::3]
        for lane in range(2):
            for plane, values in (("I", info),
                                  ("U", [None] * n_code), ("O", [None] * k_info)):
                self.data[plane, lane] = values[lane::2]
        self.data["T", 0] = nodes
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
            value = self.data[bank][address]
            if value is None:
                raise RuntimeError("SSC RAM read before write")
            self.reads[bank] += 1
            ports[bank] += 1
            values.append(value)
        return tuple(values)

    def decode(self):
        n = self.n
        def a(plane, depth, i):
            return _address(n, plane, depth, i)
        self.group("load", n, lambda i: _Op((a("X", 0, i),), (a("L", 0, i),)))

        def walk(start, length, depth, node, right):
            # A frame has only these bounded fields; the maximum depth is 11.
            (kind,) = self.group("node", 1, lambda _: _Op(((("T", 0), node),), ()))
            def alpha(i):
                return a("L", depth, i)
            def beta(i):
                return a("B", depth, right * length + i)
            def u(i):
                return a("U", 0, start + i)
            if kind == 0:
                self.group("zero", length, lambda i: _Op((), (beta(i),)))
                return
            if kind == 1:
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
        self.group("extract", self.k, lambda i: _Op((a("I", 0, i),), (a("O", 0, i),)))
        return tuple(self.data["O", i & 1][i // 2] for i in range(self.k))

    def summary(self):
        return dict(
            cycles=self.cycles, peak_abs_alpha=self.peak_llr, port_stalls=self.port_stalls,
            split_issue_edges=self.splits, blocks=self.blocks, operations=dict(self.ops),
            reads={str(bank): count for bank, count in self.reads.items()},
            writes={str(bank): count for bank, count in self.writes.items()},
            peak_ports={str(bank): count for bank, count in self.peak_ports.items()},
        )


def decode_payload_polar_word(llr, n_code, k_info, *, lanes=2, fraction_bits=0):
    """Return ordered decoded bits and a value/port-derived edge count.

    Input rounds nearest-even after symmetric ±64 clipping. Internal LLRs
    are exact signed integers, with hard-decision ties mapping to zero.
    """
    _formats(lanes, fraction_bits)
    if type(n_code) is not int or type(k_info) is not int or (n_code, k_info) not in _SHAPES:
        raise ValueError("Need a standard B1 physical Polar word shape")
    values = tuple(float(x) for x in islice(llr, n_code + 1))
    if len(values) != n_code or not all(map(isfinite, values)):
        raise ValueError("Need exactly N finite input LLRs")
    scale = 1 << fraction_bits
    q = [round(max(-64., min(64., value)) * scale) for value in values]
    model = _Datapath(n_code, k_info, q, lanes, fraction_bits)
    bits = model.decode()
    return bits, model.summary() | dict(
        clipped_inputs=sum(abs(v) > 64 for v in values), quantized_zeros=q.count(0),
        input_lsb=1 / scale, signed_alpha_bits=model.maximum.bit_length() + 1,
    )


def payload_polar_cost(*, lanes=2, fraction_bits=0):
    """Maximum-width scratch/RAM packing estimate across all 36 B1 formats.

    ROMs are paid for all resident shapes. Each parity bank uses a separate
    physical primitive with simple one-read/one-write or true dual ports.
    """
    _formats(lanes, fraction_bits)
    alpha_bits = 18 + fraction_bits
    words = _depth_shapes(1024)
    width = {bank: alpha_bits if bank[0] == "L" else 1 for bank in words}
    ports = dict.fromkeys(words, 2)
    for lane in range(2):
        for plane, depth, bits, native_ports in (
            ("U", 512, 1, 2),
            ("I", sum((k + 1 - lane) // 2 for _, k in _SHAPES), 10, 1),
            ("O", ceil(max(k for _, k in _SHAPES) / 2), 1, 1),
        ):
            bank = plane, lane
            words[bank], width[bank], ports[bank] = depth, bits, native_ports
    for slot in range(2):
        for lane in range(3):
            bank = "X", slot, lane
            words[bank], width[bank], ports[bank] = (1026 - lane) // 3, 8 + fraction_bits, 1
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
    return dict(
        shapes=len(_SHAPES), lanes=lanes, alpha_bits=alpha_bits, DSP=0,
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


def payload_polar_calendar(frames, *, word_cycles, clock_hz, stall_until=0):
    """Schedule two ingress slots and a single ordered held output.

    Completion refers to the unchecked Polar output boundary, not MAC delivery.
    B1 input cannot pause. A word with no free slot on its first beat is lost;
    earlier owned words and later words remain fully accounted for.
    """
    if clock_hz not in (100_000_000, 160_000_000, 200_000_000):
        raise ValueError("Use a modeled clock: 100, 160 or 200 MHz")
    if type(stall_until) is not int or stall_until < 0:
        raise ValueError("Downstream release edge must be non-negative")
    period = Fraction(clock_hz, 4_000_000)
    # At most two live ingress reservations; expired slots retain no state.
    slots = [0, 0]
    engine_until = 0
    next_boundary = 0
    peak = 0
    for frame_no, frame in enumerate(frames):
        bps = frame.bits_per_symbol
        if type(frame.first_edge) is not int or frame.first_edge < next_boundary:
            raise ValueError("Payload symbol intervals must be non-overlapping and non-negative")
        if bps not in (2, 3) or (type(frame.payload_with_crc_bits) is not int
                or not 24 <= frame.payload_with_crc_bits <= 2047 * 8
                or frame.payload_with_crc_bits % 8):
            raise ValueError("Need legal-range B1 length and QPSK/8PSK")
        blocks = b1_code_block_plan(frame.payload_with_crc_bits, frame.rate, standard=True)
        physical_bit = 0
        for block_no, block in enumerate(blocks):
            for word_no, (n, k) in enumerate(block.codewords):
                first = frame.first_edge + ceil(physical_bit // bps * period)
                physical_bit += n
                last = frame.first_edge + ceil((physical_bit - 1) // bps * period)
                cost = word_cycles[n, k]
                if type(cost) is not int or cost <= 0:
                    raise ValueError("Need positive executed Polar edge counts")
                slot = next((i for i, avail in enumerate(slots) if avail <= first), None)
                if slot is None:
                    yield dict(frame=frame_no, block=block_no, word=word_no, shape=(n,k),
                               first=first, last=last, lost=True, peak_ingress=peak)
                    continue
                start = max(last + 1, engine_until)
                done = start + cost
                output_start = max(done, stall_until)
                # Two single-read O banks, fixed registered read; downstream
                # takes at most two bits per edge once it becomes ready.
                release_output = output_start + (k + 1) // 2
                engine_until = release_output + 1
                # The full word input stays attached through decode (conservative
                # ownership); last + 1 retires the essential ingress write.
                slots[slot] = done
                peak = max(peak, sum(end > first for end in slots))
                yield dict(frame=frame_no, block=block_no, word=word_no, shape=(n,k),
                           first=first, last=last, launch=start, decoded=done,
                           output_start=output_start, taken=release_output,
                           lost=False, ingress_slot=slot, peak_ingress=peak,
                           queue_wait=start-last-1, output_wait=output_start-done,
                           ordered_result_edge=release_output)
        next_boundary = frame.first_edge + ceil(ceil(Fraction(physical_bit, bps)) * period)
