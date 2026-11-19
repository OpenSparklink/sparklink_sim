"""Finite control-derived frame geometry,not payload decoding (§6.3/6.9).

Two FF read operands,one register write,one registered constant/table port.
Division performs23 real restoring steps; products perform10 real shift/add steps.
No payload array,implicit variable division,precomputed length answer or future IQ.
"""

from nearlink_sdr.common.code_block_seg import _SEG_TABLE_20, _get_k_for_segmentation
from nearlink_sdr.common.mcs import get_mcs
from nearlink_sdr.phy.control_length import control_data_length
from nearlink_sdr.phy.psk_fields import psk_field_layout
from nearlink_sdr.phy.sync_fit_word import signed, unsigned

__all__ = ["HeaderGeometryService"]


def _program(frame_type, crc_len, contract):
    standard_b1 = frame_type != 2 and contract == "standard"
    p = [("copy", "payload", "length", 0), ("copy", "present", 1, 0)]
    empty_branch = None
    if standard_b1:
        empty_branch = len(p)
        p.append(("jz", "", "length", 0))
        p += [("sub", "payload", "length", crc_len // 8), ("shl", "B", "length", 3)]
    else:
        p += [("shl", "B", "length", 3), ("add", "B", "B", "crc")]
    branch = len(p)
    p.append(("uncoded", "", 0, 0))
    if frame_type == 2:
        p += [
            ("shl", "tmp", "B", "qshift"),
            ("gt", "cond", "tmp", "p1920"),
            ("sub", "tmp", "tmp", "p904"),
            ("max0", "tmp", "tmp", 0),
            ("div", "tmp", "tmp", "p1024"),
            ("select", "n1024", "tmp", "cond"),
            ("mul", "tmp", "n1024", "Kcb"),
            ("sub", "km", "B", "tmp"),
            ("sub", "tmp", "km", 1),
            ("max0", "tmp", "tmp", 0),
            ("shl", "tmp", "tmp", "qshift"),
            ("div", "index", "tmp", "p128"),
            ("min14", "index", "index", 0),
            ("table", "n512", "index", 0),
            ("table", "n256", "index", 1),
            ("table", "n128", "index", 2),
            ("mul", "tmp", "n512", "K512"),
            ("sub", "km", "km", "tmp"),
            ("mul", "tmp", "n256", "K256"),
            ("sub", "km", "km", "tmp"),
            ("mul", "tmp", "n128", "K128"),
            ("sub", "km", "km", "tmp"),
            ("max0", "km", "km", 0),
            ("add", "tmp", "km", "K64minus1"),
            ("div", "n64", "tmp", "K64"),
            ("shl", "coded", "n1024", 10),
            ("shl", "tmp", "n512", 9),
            ("add", "coded", "coded", "tmp"),
            ("shl", "tmp", "n256", 8),
            ("add", "coded", "coded", "tmp"),
            ("shl", "tmp", "n128", 7),
            ("add", "coded", "coded", "tmp"),
            ("shl", "tmp", "n64", 6),
            ("add", "coded", "coded", "tmp"),
        ]
    else:
        p += [
            ("gt", "cond", "B", "Kcb"),
            ("add", "tmp", "B", "Kcbminus25"),
            ("div", "blocks", "tmp", "Kcbminus24"),
            ("select", "tmp", "blocks", "cond"),
            ("max1", "blocks", "tmp", 0),
            ("shl", "coded", "blocks", 10),
        ]
    jump = len(p)
    p.append(("jump", "", 0, 0))
    p[branch] = ("uncoded", "", len(p), 0)
    p.append(("copy", "coded", "B", 0))
    p[jump] = ("jump", "", len(p), 0)
    end_jump = len(p)
    if standard_b1:
        p.append(("jump", "", 0, 0))
        p[empty_branch] = ("jz", "", "length", len(p))
        p += [("copy", "present", 0, 0), ("copy", "coded", 0, 0)]
        p[end_jump] = ("jump", "", len(p), 0)
    p += [
        ("add", "tmp", "coded", "bpsminus1"),
        ("div", "data", "tmp", "bps"),
        ("sub", "tmp", "data", 1),
        ("max0", "tmp", "tmp", 0),
        ("shr", "tmp", "tmp", "pilotshift"),
        ("select", "tmp", "tmp", "pilotenable"),
        ("add", "wire", "data", "tmp"),
        ("add", "end", "wire", "data_start"),
    ]
    assert len(p) <= 64
    return tuple(p)


class HeaderGeometryService:
    """Compiled FT2/3/4 geometry after real header CRC and field policy pass.

    This component checks legal scalar dimensions,not CRC. It never grants MAC
    delivery. All registers are signed24,division inputs unsigned23. FF operands
    are read combinationally;results register at each modeled edge (no timing proof).
    """

    def __init__(self, frame_type, crc_len, pilot_interval, *,
                 length_contract="standard", observer=None):
        if (
            type(frame_type) is not int
            or frame_type not in (2, 3, 4)
            or type(crc_len) is not int
            or crc_len not in (24, 32)
            or type(pilot_interval) is not int
            or pilot_interval not in (0, 4, 8, 16)
        ):
            raise ValueError("Need compiled PSK frame,CRC24/32 and legal pilot interval")
        if observer is not None and not callable(observer):
            raise ValueError("Observer must be callable")
        control_data_length(0, "A2" if frame_type == 2 else "B1", crc_len,
                            contract=length_contract)
        self.length_contract = length_contract
        self.frame_type, self.crc_len, self.pilot = frame_type, crc_len, pilot_interval
        self.program, self.observer = _program(frame_type, crc_len, length_contract), observer
        self.rows = []
        for mcs in range(13):
            entry = get_mcs(mcs)
            p, q = entry.code_rate.numerator, entry.code_rate.denominator
            table = (
                {}
                if p == q
                else _get_k_for_segmentation(str(entry.code_rate), use_table24=frame_type == 2)
            )
            Kcb = table.get(1024, 0)
            row = dict(
                uncoded=int(p == q),
                qshift=q.bit_length() - 1,
                p1920=p * 1920,
                p904=p * 904,
                p1024=p * 1024,
                p128=p * 128,
                Kcb=Kcb,
                K512=table.get(512, 0),
                K256=table.get(256, 0),
                K128=table.get(128, 0),
                K64=table.get(64, 0),
                K64minus1=table.get(64, 1) - 1,
                Kcbminus25=Kcb - 25,
                Kcbminus24=Kcb - 24,
                bps=entry.bits_per_symbol,
                bpsminus1=entry.bits_per_symbol - 1,
            )
            self.rows.append(row)
        nsync = {2: 32, 3: 62, 4: 126}[frame_type]
        head_bits, head_bps, interval = (
            (64, 2, 16) if frame_type == 2 else (256, 1 if frame_type == 4 else 2, 4)
        )
        positions, size = psk_field_layout(head_bits, head_bps, interval, pilot_interval == 0)
        self.header_end, self.data_start = nsync + int(positions[-1]) + 1, nsync + size
        self.mode, self.result, self.engine = "idle", None, None
        self.registers, self.params = {}, {}
        self.pc = self.cycle = self.mcs = 0
        self.stats = dict.fromkeys(
            (
                "accepted",
                "completed",
                "aborted",
                "ALU_edges",
                "division_steps",
                "product_steps",
                "ROM_reads",
                "input_stalls",
                "output_stalls",
            ),
            0,
        )

    def _count(self, key):
        self.stats[key] = unsigned(self.stats[key] + 1, 64)

    def _emit(self, event, **fields):
        if self.observer:
            self.observer(dict(cycle=self.cycle, event=event, **fields))

    @property
    def ready(self):
        return self.mode == "ready"

    @property
    def input_ready(self):
        return self.mode == "idle"

    def issue(self, length, mcs):
        if (
            type(length) is not int
            or not 0 <= length <= 2047
            or type(mcs) is not int
            or not 0 <= mcs <= 12
        ):
            raise ValueError("Need validated byte length0..2047 and MCS0..12")
        # Input contract validation only; no host-computed payload/geometry answer
        # is retained or passed to the ALU. The actual subtraction is an edge below.
        control_data_length(length, "A2" if self.frame_type == 2 else "B1", self.crc_len,
                            contract=self.length_contract)
        if not self.input_ready:
            self._count("input_stalls")
            return False
        unsigned(self.stats["completed"] + 1, 64)
        self._count("accepted")
        self.registers = dict(
            length=length,
            crc=self.crc_len,
            data_start=self.data_start,
            pilotshift=max(0, self.pilot.bit_length() - 1),
            pilotenable=int(self.pilot != 0),
        )
        self.pc, self.mcs, self.mode = 0, mcs, "rom"
        self.result = None
        self._emit("accept", length=length, mcs=mcs, input_edge=self.cycle)
        return True

    def _read(self, operand):
        if type(operand) is int:
            return operand
        if operand in self.registers:
            return self.registers[operand]
        return self.params[operand]

    def _write(self, name, value):
        self.registers[name] = signed(value, 24)
        if len(self.registers) > 24:
            raise RuntimeError("Finite geometry register file exhausted")
        self._emit("register", name=name, value=value, register_edge=self.cycle + 1)

    def step(self):
        unsigned(self.cycle + 1, 64)
        if self.mode == "rom":
            self.params = self.rows[self.mcs].copy()
            self._emit("parameter_read", mcs=self.mcs, parameter_bits=len(self.params) * 24,
                       latch_edge=self.cycle + 1)
            self._count("ROM_reads")
            self.mode = "run"
        elif self.mode == "run":
            if self.engine is not None:
                e = self.engine
                self._count("ALU_edges")
                if e["op"] == "div":
                    bit = e["remaining"] - 1
                    remainder = unsigned((e["remainder"] << 1) | ((e["a"] >> bit) & 1), 24)
                    hit = int(remainder >= e["b"])
                    e["remainder"] = unsigned(remainder - e["b"] if hit else remainder, 23)
                    e["value"] = unsigned((e["value"] << 1) | hit, 23)
                    self._count("division_steps")
                else:
                    bit = 10 - e["remaining"]
                    if (e["b"] >> bit) & 1:
                        e["value"] = signed(e["value"] + (e["a"] << bit), 24)
                    self._count("product_steps")
                self._emit("engine", opcode=e["op"], remaining=e["remaining"], value=e["value"],
                           remainder=e["remainder"])
                e["remaining"] -= 1
                if not e["remaining"]:
                    self._write(e["dest"], e["value"])
                    self.engine = None
                    self.pc += 1
            else:
                op, dest, left, right = self.program[self.pc]
                a, b = self._read(left), self._read(right)
                self._emit("instruction", pc=self.pc, opcode=op, a=a, b=b)
                self._count("ALU_edges")
                if op in ("div", "mul"):
                    unsigned(a, 23)
                    unsigned(b, 23 if op == "div" else 10)
                    if op == "div" and b == 0:
                        raise ZeroDivisionError("Invalid compiled geometry divisor")
                    self.engine = dict(
                        op=op,
                        dest=dest,
                        a=a,
                        b=b,
                        value=0,
                        remainder=0,
                        remaining=23 if op == "div" else 10,
                    )
                elif op == "jz":
                    self.pc = b if a == 0 else self.pc + 1
                elif op in ("jump", "uncoded"):
                    self.pc = a if op == "jump" or self.params["uncoded"] else self.pc + 1
                else:
                    if op == "add":
                        value = a + b
                    elif op == "sub":
                        value = a - b
                    elif op == "shl":
                        value = a << b
                    elif op == "shr":
                        value = a >> b
                    elif op == "gt":
                        value = int(a > b)
                    elif op == "max0":
                        value = max(0, a)
                    elif op == "max1":
                        value = max(1, a)
                    elif op == "min14":
                        value = min(14, a)
                    elif op == "select":
                        value = a if b else 0
                    elif op == "table":
                        value = _SEG_TABLE_20[a][b]
                        self._count("ROM_reads")
                    elif op == "copy":
                        value = a
                    else:
                        raise RuntimeError("Corrupt compiled geometry opcode")
                    self._write(dest, value)
                    self.pc += 1
            if self.pc == len(self.program):
                self.result = dict(
                    payload_length_bytes=self.registers["payload"],
                    data_crc_present=bool(self.registers["present"]),
                    coded_bits=self.registers["coded"],
                    data_symbols=self.registers["data"],
                    wire_data_symbols=self.registers["wire"],
                    header_end_symbol=self.header_end,
                    data_start_symbol=self.data_start,
                    frame_end_symbol=self.registers["end"],
                )
                self.mode = "ready"
                self._count("completed")
        elif self.mode == "ready":
            self._count("output_stalls")
        self.cycle += 1

    def take(self):
        if not self.ready:
            return None
        result, self.result, self.mode = self.result, None, "idle"
        return result

    def abort(self):
        if self.mode not in ("idle", "aborted"):
            self._count("aborted")
            self.mode, self.result, self.engine = "aborted", None, None

    def reset(self):
        if self.mode not in ("idle", "aborted"):
            raise RuntimeError("Drain or abort geometry before reset")
        self.mode = "idle"

    def cost(self):
        """Concrete register/ROM/operator estimate;no synthesized resource or Fmax."""
        columns = set()
        for key in self.rows[0]:
            for bit in range(24):
                columns.add(
                    sum(((row[key] & 0xFFFFFF) >> bit & 1) << i for i, row in enumerate(self.rows))
                )
        columns.discard(0)
        state = dict(
            register_file=24 * 24,
            parameter_latches=len(self.rows[0]) * 24,
            serial_engine=106,
            counters=len(self.stats) * 64,
            cycle=64,
            controller=32,
            compiled_frame_context=64,
        )
        rom = dict(
            parameter_rows=13 * len(self.rows[0]) * 24,
            instructions=len(self.program) * 32,
            segmentation_table=15 * 6,
        )
        return dict(
            registers_bits=state,
            constant_ROM_bits=rom,
            logical_total_bits=sum(state.values()) + sum(rom.values()),
            RAMB18=0,
            DSP=0,
            constant_implementation="distributed LUT truth tables",
            constant_data_read_ports=1,
            parameter_row_read_bits=len(self.rows[0]) * 24,
            parameter_latch_write_bits=len(self.rows[0]) * 24,
            instruction_ROM_read_ports=1,
            instruction_ROM_read_bits=32,
            instruction_fetch="combinational LUT ROM into the ALU register edge",
            constant_port_sharing="parameter row at start;segmentation lookup later;one context",
            LUT6_truth_table_upper=len(columns) + 32 + 2,
            LUT6_truth_table_only=True,
            total_LUT=None,
            total_FF=None,
            read_operand_ports=2,
            register_write_ports=1,
            input_bus_bits=15,
            input_capture="one length-register write plus4-bit MCS controller capture",
            constant_initialization="compiled context;not packet-time multiport RAM writes",
            register_bits=24,
            FF_read_mux2_upper=2 * 24 * 63,
            add_sub_bits=24,
            compare_bits=24,
            variable_shift_bits=24,
            division_steps=23,
            product_steps=10,
            program_instructions=len(self.program),
            length_contract=self.length_contract,
            input_length_bits=11,
            payload_length_bits=11,
            data_presence_bits=1,
            output_storage="aliases stable registers until take;no separate FIFO",
            input_contract_validation="caller must provide a valid scalar after real head CRC",
            additional_B1_subtraction="one real24-bit ALU edge;not a host-computed result",
            upstream_validation_cycles=None,
            overflow_rule="raise model error;no wrap/saturation;legal operand bounds must hold",
            runtime_fault_detector_synthesized=False,
            clock_assumption="combinational instruction/FF mux/ALU ->register one edge",
            accepted_clock_hz=None,
            synthesis=False,
            whole_RX16MSps=False,
        )
