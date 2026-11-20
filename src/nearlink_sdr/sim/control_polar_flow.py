"""Value-carrying control SSC with registered physical banks and four credits.

Hard,already descrambled bits only; CRC/fields/MAC remain unchecked. Four-node
RX acceptance and FPGA synthesis are separate from this finite codec service.
"""

from itertools import islice

from nearlink_sdr.phy.control_polar_word import ControlPolarFormat, compile_control_rom
from nearlink_sdr.phy.sync_fit_word import signed, unsigned

__all__ = ["ControlPolarService", "control_polar_cost"]


class _Memory:
    """Two-port data banks,one-port ROM; one-edge registered reads,no hidden queue."""

    def __init__(self, fmt, layout):
        self.fmt, self.layout = fmt, layout
        N, n = fmt.N, fmt.n
        self.limit, self.native_bits = {}, {}
        self.cells = {}
        if layout == "dense":
            cap = 1024 if N == 64 else 2048
            for begin in range(0, (n + 1) * N, cap):
                name = "L" + str(begin // cap)
                self.cells[name] = [None] * min(cap, (n + 1) * N - begin)
                self.native_bits[name] = 18432 if cap == 1024 else 36864
            self.cells["B0"] = [None] * ((n + 1) * N)
            self.native_bits["B0"] = 18432
        else:
            for parity in (0, 1):
                length = len(range(parity, n + 1, 2)) * N
                self.cells["L" + str(parity)] = [None] * length
                self.cells["B" + str(parity)] = [None] * length
                self.native_bits["L" + str(parity)] = 18432 if length <= 1024 else 36864
                self.native_bits["B" + str(parity)] = 18432
        self.cells["ROM"] = list(compile_control_rom(fmt))
        self.native_bits["ROM"] = 18432
        self.limit = {name: 1 if name == "ROM" else 2 for name in self.cells}
        self.pending = []
        self.used = dict.fromkeys(self.cells, 0)
        self.peak = self.used.copy()
        self.written = set()

    def address(self, plane, depth, position):
        if not (plane in ("L", "B") and type(depth) is int and 0 <= depth <= self.fmt.n
                and type(position) is int and 0 <= position < self.fmt.N):
            raise ValueError("Address outside compiled control plane")
        if self.layout == "depth-parity":
            return plane + str(depth & 1), (depth >> 1) * self.fmt.N + position
        address = depth * self.fmt.N + position
        if plane == "B":
            return "B0", address
        cap = 1024 if self.fmt.N == 64 else 2048
        return "L" + str(address // cap), address % cap

    def begin(self):
        replies, self.pending = self.pending, []
        self.used = dict.fromkeys(self.cells, 0)
        self.written.clear()
        return replies

    def available(self, requests):
        counts = self.used.copy()
        for bank, address in requests:
            if bank not in self.cells or not 0 <= address < len(self.cells[bank]):
                raise ValueError("Unallocated physical bank/address")
            # A read must never depend on device-specific read-during-write mode.
            if (bank, address) in self.written:
                return False
            counts[bank] += 1
            if counts[bank] > self.limit[bank]:
                return False
        return True

    def _use(self, bank):
        self.used[bank] += 1
        self.peak[bank] = max(self.peak[bank], self.used[bank])

    def read(self, slot, operand, bank, address):
        value = self.cells[bank][address]
        if value is None:
            raise RuntimeError("Control RAM read before deterministic write")
        self._use(bank)
        # Valid pipeline,not a variable-latency response queue or a stored timestamp.
        self.pending.append((slot, operand, value))
        if len(self.pending) > 3:
            raise RuntimeError("Only one operation with at most three reads may issue")

    def write(self, bank, address, value):
        if bank == "ROM":
            raise RuntimeError("Compiled control ROM is read only")
        signed(value, 10) if bank.startswith("L") else unsigned(value, 1)
        self._use(bank)
        self.cells[bank][address] = value
        self.written.add((bank, address))


class ControlPolarService:
    """One hard codeword,one ordered unchecked output;no floating decoder call.

    Each block uses bounded context credits: read→registered operands/ALU→write.
    Reads and writes reserve actual bank ports. Nodes/stages drain before the
    bounded stack changes. Rate-1 work aliases the leaf B plane after beta is saved.
    """

    def __init__(self, fmt, *, layout="dense", observer=None):
        if not isinstance(fmt, ControlPolarFormat) or layout not in ("dense", "depth-parity"):
            raise ValueError("Need explicit control format and dense or depth-parity banks")
        if observer is not None and not callable(observer):
            raise ValueError("Observer must be callable or absent")
        self.fmt, self.layout, self.observer = fmt, layout, observer
        self.memory = self._make_memory(fmt, layout)
        self.cycle = 0
        self.mode, self.block, self.result = "idle", None, None
        self.slots = [None] * 4
        self.stack = [None] * fmt.stack_frames
        self.depth = 0
        self.input_bits = [0] * fmt.N
        self.output_bits = [0] * fmt.K
        self.output_index = self.info_address = 0
        self.first_cycle = self.ready_cycle = None
        self.taken_this_cycle = False
        self.stats = dict.fromkeys(
            ("accepted", "completed", "aborted", "input_stalls", "output_stalls", "port_stalls",
             "ram_reads", "ram_writes", "f_operations", "g_operations", "xor_operations",
             "hard_decisions", "node_visits"), 0,
        )

    def _make_memory(self, fmt, layout):
        return _Memory(fmt, layout)

    def _count(self, key, amount=1):
        self.stats[key] = unsigned(self.stats[key] + amount, 64)

    def _emit(self, event_kind, **fields):
        if self.observer is not None:
            self.observer(dict(cycle=self.cycle, event=event_kind, **fields))

    @property
    def input_ready(self):
        return self.mode == "idle" and not self.taken_this_cycle

    @property
    def ready(self):
        return self.mode == "ready"

    def issue(self, bits):
        values = list(islice(bits, self.fmt.N + 1))
        if len(values) != self.fmt.N or any(type(v) is not int or v not in (0, 1) for v in values):
            raise ValueError("Need exactly one compiled binary Python-integer codeword")
        if self.mode == "aborted":
            raise RuntimeError("Aborted context requires explicit reset")
        if not self.input_ready:
            self._count("input_stalls")
            return False
        unsigned(self.stats["completed"] + 1, 64)
        self._count("accepted")
        self.input_bits[:] = values
        self.output_bits[:] = [0] * self.fmt.K
        self.first_cycle, self.ready_cycle = self.cycle, None
        self.mode, self.block, self.result = "load", None, None
        self.output_index, self.depth = 0, 0
        return True

    def _push(self, start, length, depth, node):
        if self.depth >= len(self.stack):
            raise OverflowError("Finite control traversal stack exhausted")
        unsigned(start, self.fmt.n)
        unsigned(length, self.fmt.n + 1)
        unsigned(node, self.fmt.n + 1)
        self.stack[self.depth] = dict(start=start, length=length, depth=depth, node=node,
                                      phase="type", step=length)
        self.depth += 1

    def _pop(self):
        self.depth -= 1
        self.stack[self.depth] = None
        if not self.depth:
            self.mode = "info"
        else:
            parent = self.stack[self.depth - 1]
            if parent["phase"] == "left":
                parent["phase"] = "g"
            elif parent["phase"] == "right":
                parent["phase"] = "combine"
            else:
                raise RuntimeError("Unexpected control traversal return")

    def _open_block(self):
        if self.mode == "load":
            kind, count = "load", self.fmt.N
        elif self.mode in ("info", "extract"):
            kind, count = self.mode, 1
        elif self.mode == "walk":
            f = self.stack[self.depth - 1]
            kind = f["phase"]
            count = (1 if kind == "type" else f["length"] // 2
                     if kind in ("f", "g", "combine", "transform") else f["length"])
        else:
            raise RuntimeError("No active finite control block")
        self.block = dict(kind=kind, count=count, issued=0, completed=0)
        # One registered controller/address-base setup edge,not free block work.
        self._emit("block", **self.block)

    def _command(self, index):
        kind, N, n = self.block["kind"], self.fmt.N, self.fmt.n
        M = self.memory
        if kind == "load":
            return dict(reads=[], writes=[M.address("L", 0, index)], aux=self.input_bits[index])
        if kind == "info":
            return dict(reads=[("ROM", 2 * N - 1 + self.output_index)], writes=[])
        if kind == "extract":
            return dict(reads=[M.address("B", n, self.info_address)], writes=[])
        f = self.stack[self.depth - 1]
        start, length, d = f["start"], f["length"], f["depth"]
        half = length // 2
        if kind == "type":
            return dict(reads=[("ROM", f["node"])], writes=[])
        if kind == "zero":
            return dict(reads=[], writes=list(dict.fromkeys(
                [M.address("B", d, start + index), M.address("B", n, start + index)])))
        if kind == "hard":
            return dict(reads=[M.address("L", d, start + index)], writes=list(dict.fromkeys(
                [M.address("B", d, start + index), M.address("B", n, start + index)])))
        if kind == "transform":
            h = f["step"] // 2
            position = start + (index // h) * f["step"] + index % h
            return dict(reads=[M.address("B", n, position), M.address("B", n, position + h)],
                        writes=[M.address("B", n, position)])
        if kind == "combine":
            return dict(reads=[M.address("B", d + 1, start + index),
                               M.address("B", d + 1, start + half + index)],
                        writes=[M.address("B", d, start + index),
                                M.address("B", d, start + half + index)])
        reads = [M.address("L", d, start + index), M.address("L", d, start + half + index)]
        if kind == "g":
            reads.append(M.address("B", d + 1, start + index))
        return dict(reads=reads,
                    writes=[M.address("L", d + 1, start + index + (half if kind == "g" else 0))])

    def _compute(self, job):
        kind, v = job["kind"], job["operands"]
        if any(x is None for x in v):
            raise RuntimeError("RAM result was not registered on the preceding issue edge")
        if kind == "load":
            out = [1 - 2 * job["aux"]]
        elif kind == "zero":
            out = [0] * len(job["writes"])
        elif kind == "hard":
            self._count("hard_decisions")
            out = [int(v[0] < 0)] * len(job["writes"])
        elif kind in ("transform", "combine"):
            self._count("xor_operations")
            out = [v[0] ^ v[1]] + ([v[1]] if kind == "combine" else [])
        elif kind == "f":
            self._count("f_operations")
            out = [(-1 if (v[0] < 0) != (v[1] < 0) else 1) * min(abs(v[0]), abs(v[1]))]
        elif kind == "g":
            self._count("g_operations")
            out = [(-v[0] if v[2] else v[0]) + v[1]]
        else:
            out = [v[0]]
        for value in out:
            signed(value, 10)
        job["output"], job["stage"] = out, "write"
        self._emit("compute", opcode=kind, operands=v.copy(), values=out.copy(),
                   registered_write_edge=self.cycle + 1)

    def _finish_block(self, final):
        kind, self.block = self.block["kind"], None
        f = self.stack[self.depth - 1] if self.depth else None
        if kind == "load":
            self._push(0, self.fmt.N, 0, 0)
            self.mode = "walk"
        elif kind == "info":
            self.info_address, self.mode = final[0], "extract"
        elif kind == "extract":
            self.output_bits[self.output_index] = unsigned(final[0], 1)
            self.output_index += 1
            if self.output_index == self.fmt.K:
                self._count("completed")
                self.result = dict(N=self.fmt.N, K=self.fmt.K,
                                   information_bits=tuple(self.output_bits),
                                   crc_status="not_checked")
                self.mode, self.ready_cycle = "ready", self.cycle + 1
            else:
                self.mode = "info"
        elif kind == "type":
            self._count("node_visits")
            if final[0] not in (0, 1, 2):
                raise RuntimeError("Corrupt compiled node type")
            f["phase"] = ("zero", "hard", "f")[final[0]]
        elif kind in ("zero", "combine"):
            self._pop()
        elif kind == "hard":
            if f["length"] == 1:
                self._pop()
            else:
                f["phase"], f["step"] = "transform", f["length"]
        elif kind == "transform":
            f["step"] //= 2
            if f["step"] == 1:
                self._pop()
        elif kind in ("f", "g"):
            half = f["length"] // 2
            f["phase"] = "left" if kind == "f" else "right"
            self._push(f["start"] + (half if kind == "g" else 0), half, f["depth"] + 1,
                       2 * f["node"] + (2 if kind == "g" else 1))

    def step(self):
        unsigned(self.cycle + 1, 64)
        replies = self.memory.begin()
        for slot, operand, value in replies:
            self.slots[slot]["operands"][operand] = value
        if self.mode == "ready":
            self._count("output_stalls")
        finished = False
        # At most one issue/ALU/write per edge;priority writes reserve ports before reads.
        for slot, job in enumerate(self.slots):
            if job is None or job["stage"] != "write":
                continue
            if not self.memory.available(job["writes"]):
                raise RuntimeError("One registered result exceeded write-port budget")
            unsigned(self.stats["ram_writes"] + len(job["writes"]), 64)
            values = job["output"] if job["writes"] else []
            for (bank, address), value in zip(job["writes"], values, strict=True):
                self.memory.write(bank, address, value)
                self._emit("write", bank=bank, address=address, value=value)
            self._count("ram_writes", len(job["writes"]))
            self.slots[slot] = None
            self.block["completed"] += 1
            if self.block["completed"] == self.block["count"]:
                assert all(x is None for x in self.slots) and not self.memory.pending
                self._finish_block(job["output"])
                finished = True
        for job in self.slots:
            if job is not None and job["stage"] == "read":
                self._compute(job)
        if self.mode not in ("idle", "aborted", "ready") and not finished:
            if self.block is None:
                self._open_block()
            elif self.block["issued"] < self.block["count"]:
                command = self._command(self.block["issued"])
                if None not in self.slots or not self.memory.available(command["reads"]):
                    self._count("port_stalls")
                else:
                    slot = self.slots.index(None)
                    unsigned(self.stats["ram_reads"] + len(command["reads"]), 64)
                    self.slots[slot] = command | dict(kind=self.block["kind"], stage="read",
                                                     operands=[None] * len(command["reads"]))
                    for operand, (bank, address) in enumerate(command["reads"]):
                        self.memory.read(slot, operand, bank, address)
                        self._emit("read", bank=bank, address=address, slot=slot, operand=operand)
                    self._count("ram_reads", len(command["reads"]))
                    self.block["issued"] += 1
        self.cycle += 1
        self.taken_this_cycle = False

    def take(self):
        if not self.ready:
            return None
        result, self.result, self.mode = self.result, None, "idle"
        self.taken_this_cycle = True
        return result

    def abort(self):
        if self.mode in ("idle", "aborted"):
            return
        self._count("aborted")
        self.mode, self.block, self.result = "aborted", None, None
        self.slots[:] = [None] * 4
        self.memory.pending.clear()
        self.stack[:] = [None] * len(self.stack)
        self.depth = 0

    def reset(self):
        if self.mode not in ("idle", "aborted"):
            raise RuntimeError("Drain or abort before reset")
        self.mode = "idle"


def control_polar_cost(fmt, *, layout="dense"):
    """Capacity/primitive packing estimates, not LUT/FF synthesis or timing proof.

    The issue API presents a complete N-bit register word in parallel. Serial
    demap/descramble ingress, CRC and fields are NOT included or zero-cycle.
    ROM and data physical padding are charged. No bank or RX/TX sharing credited.
    """
    service = ControlPolarService(fmt, layout=layout)
    memory = service.memory
    N, n = fmt.N, fmt.n
    state = {
        "codeword_register": N,
        "held_information_register": fmt.K,
        "stack_padded_frames": fmt.stack_frames * 64,
        # Each slot needs <=146 bits: valid/stage/opcode/read/write counts,
        # three read addresses and operands,two write addresses/results,aux.
        "four_contexts_padded": 4 * 192,
        "registered_read_replies": 3 * 16,
        "statistics": len(service.stats) * 64,
        "cycle_first_ready": 3 * 64 + 2,
        "block_counts_and_opcode_padded": 32,
        "control_and_pointers_padded": 64,
        "current_command_padded": 128,
        "bank_used_peak_and_collision_tags": 4 * len(memory.cells) + 32,
    }
    ram = (n + 1) * N * 11
    rom = len(compile_control_rom(fmt)) * n
    logical = ram + rom + sum(state.values())
    equivalents = sum(memory.native_bits.values()) // 18432
    separate_sites = sum((v + 36863) // 36864 for v in memory.native_bits.values())
    return dict(
        N=N, K=fmt.K, layout=layout, LLR_bits=10, LLR_absolute_bound=N,
        arithmetic="signed exact min-sum;zero tie ->0;no rounding,saturation or wrap",
        logical_RAM_bits=ram, logical_ROM_bits=rom, state_components_bits=state,
        logical_total_bits=logical, native_bank_bits=memory.native_bits.copy(),
        native_total_bits=sum(memory.native_bits.values()),
        RAMB18_equivalents=equivalents,
        BRAM36_sites_lower=(equivalents + 1) // 2, BRAM36_sites_upper=separate_sites,
        physical_data_ports=memory.limit.copy(), DSP=0,
        address_adders=dict(count=4, bits=12), minsum_ALU_bits=10,
        macro_latency_cycles=dict(RAM_read=1, minsum_ALU=1, registered_writeback=1),
        input_bus_bits=N, contexts=4, runtime_stack_frames=fmt.stack_frames,
        runtime_recursion=False, unbounded_queue=False,
        RAM_read_during_write="same-address new reads stall;no device mode assumed",
        LUT=None, FF=None, actual_synthesis=False, verified_clock_hz=None,
        unmodeled_head_demap_descramble_CRC_fields_cycles=None,
        full_RX_acceptance=False,
        prospective_capacity_budget_met=(logical <= 65536 and equivalents <= 6
                                         and separate_sites <= 5),
    )
