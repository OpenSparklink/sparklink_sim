"""Finite ALU32 carry steps and native25x18 DSP pipeline for PSK sync (§6.2.3).

Value and schedule model, not RTL or synthesis. Operands are explicitly typed;
wide Python integers hold wires/registers, never a precomputed wide ALU answer.
There is one issue port, one result port, and one reserved mailbox per context.
"""

from dataclasses import dataclass

__all__ = ["WordCommand", "WordFabric", "WordFabricAssumptions", "WordReply", "word_fabric_cost"]

_CODES = frozenset(("add", "sub", "neg", "abs", "mov", "cmp", "rne", "tile"))
_MASK = (1 << 32) - 1


@dataclass(frozen=True)
class WordFabricAssumptions:
    """Compiled operators; concurrency is bounded independently of host speed."""

    alu_lanes: int = 1
    contexts: int = 9

    def __post_init__(self):
        if type(self.alu_lanes) is not int or self.alu_lanes not in (1, 2, 4):
            raise ValueError("Need1,2 or4 ALU32 lanes")
        if type(self.contexts) is not int or not 1 <= self.contexts <= 9:
            raise ValueError("Need1..9 finite contexts")


@dataclass(frozen=True)
class WordCommand:
    """Distinct input/output widths and signedness; tag is a compiled7-bit ID."""

    code: str
    a: int
    b: int
    a_bits: int
    b_bits: int
    a_signed: bool
    b_signed: bool
    bits: int
    signed: bool
    tag: int = 0

    def __post_init__(self):
        if self.code not in _CODES:
            raise ValueError("Unknown finite operator")
        for width in (self.a_bits, self.b_bits, self.bits):
            if type(width) is not int or not 1 <= width <= 208:
                raise ValueError("Need input/output widths1..208")
        for flag in (self.a_signed, self.b_signed, self.signed):
            if type(flag) is not bool:
                raise ValueError("Need explicit signedness")
        for value, width, signed in (
            (self.a, self.a_bits, self.a_signed),
            (self.b, self.b_bits, self.b_signed),
        ):
            if type(value) is not int or not -(1 << (width - 1)) * signed <= value < 1 << (
                width - signed
            ):
                raise ValueError("Operand exceeds its explicit format")
        if type(self.tag) is not int or not 0 <= self.tag < 128:
            raise ValueError("Need compiled7-bit tag")
        if self.code == "tile" and (
            (self.a_bits, self.b_bits, self.a_signed, self.b_signed)
            not in ((24, 17, False, False), (22, 18, True, True))
            or self.bits != 43 or not self.signed
        ):
            raise ValueError("Native tile is positive24x17 or signed22x18 in signed25x18 footprint")
        if self.code == "rne" and (
            (self.a_bits, self.b_bits, self.bits, self.b) != (33, 4, 22, 10)
            or not self.a_signed
            or self.b_signed
            or not self.signed
        ):
            raise ValueError("Compiled RNE is signed33,shift10,to signed22")
        if self.code == "cmp" and (not self.signed or self.bits < 2):
            raise ValueError("Comparison needs a signed result with at least2 bits")


@dataclass(frozen=True)
class WordReply:
    """Stable private mailbox;completed_cycle is host evidence,not retained HW state."""

    context: int
    token: int
    tag: int
    value: int
    overflow: bool
    completed_cycle: int


class _Job:
    """One ALU lane: at most one32-bit segment is evaluated per step."""

    def __init__(self, command, context, token):
        self.c, self.context, self.token = command, context, token
        self.extent = max(command.a_bits, command.b_bits, command.bits) + 2
        self.cursor = self.result = self.stage = 0
        self.carry = int(
            command.code in ("sub", "neg") or (command.code == "abs" and command.a < 0)
        )
        self.comparison = (command.a >= 0) - (command.b >= 0)
        self.low = self.quotient = self.remainder = 0

    def _rounded_step(self):
        # Fixed low32/high1 magnitude,23-bit quotient+10-bit remainder,
        # one24-bit increment and one signed restoration; no variable divider.
        c = self.c
        negative = c.a < 0
        if self.stage == 0:
            raw = c.a & _MASK
            value = ((_MASK ^ raw) + 1) if negative else raw
            self.low, self.carry = value & _MASK, value >> 32
        elif self.stage == 1:
            raw = (c.a >> 32) & 1
            high = ((raw ^ 1) + self.carry) & 1 if negative else raw
            self.quotient = (self.low >> 10) | (high << 22)
            self.remainder = self.low & 1023
        elif self.stage == 2:
            self.quotient += int(
                self.remainder > 512 or (self.remainder == 512 and self.quotient & 1)
            )
        else:
            self.result = -self.quotient if negative else self.quotient
            return True
        self.stage += 1
        return False

    def step(self):
        c = self.c
        if c.code == "rne":
            return self._rounded_step()
        if c.code == "mov" or (c.code == "abs" and not c.a_signed):
            self.result = c.a  # Registered wire copy, not a wide arithmetic operation.
            return True
        if c.code == "cmp":
            width = min(32, self.extent - self.cursor * 32)
            offset = self.extent - self.cursor * 32 - width
            mask = (1 << width) - 1
            a, b = (c.a >> offset) & mask, (c.b >> offset) & mask
            if not self.comparison:
                self.comparison = (a > b) - (a < b)
            self.cursor += 1
            self.result = self.comparison
            return self.cursor * 32 >= self.extent
        width = min(32, self.extent - self.cursor * 32)
        mask = (1 << width) - 1
        offset = self.cursor * 32
        a, b = (c.a >> offset) & mask, (c.b >> offset) & mask
        if c.code == "sub":
            partial = a + (b ^ mask) + self.carry
        elif c.code in ("abs", "neg"):
            negate = c.code == "neg" or c.a < 0
            partial = (a ^ mask) + self.carry if negate else a
        else:
            partial = a + b + self.carry
        self.carry = partial >> width
        self.result |= (partial & mask) << offset
        self.cursor += 1
        if self.cursor * 32 >= self.extent:
            # Interpret the assembled wire; extra sign bits detect overflow.
            if self.result & (1 << (self.extent - 1)):
                self.result -= 1 << self.extent
            return True
        return False


class WordFabric:
    """Finite value-level fabric with registered feedback and finite tokens.

    A take may hold a mailbox indefinitely. A completed context cannot reissue
    on the same cycle even after take. No operand/result history is retained.
    DSP input → native product → pipeline register → mailbox takes3 cycles,II1.
    """

    def __init__(self, assumptions=None, *, initial_cycle=0, sink=None):
        self.a = WordFabricAssumptions() if assumptions is None else assumptions
        if not isinstance(self.a, WordFabricAssumptions):
            raise ValueError("Need WordFabricAssumptions")
        if type(initial_cycle) is not int or not 0 <= initial_cycle < 2**63:
            raise ValueError("Need bounded nonnegative initial clock")
        if sink is not None and not callable(sink):
            raise ValueError("Need diagnostic callback")
        self.cycle, self.sink = initial_cycle, sink
        self.lanes = [None] * self.a.alu_lanes
        self.dsp = [None] * 3
        self.mailboxes = [None] * self.a.contexts
        self.busy = [False] * self.a.contexts
        self.cooldown = [False] * self.a.contexts
        self.next_token = [0] * self.a.contexts
        self.issued_this_cycle = self.taken_this_cycle = False
        self.stats = dict(issued=0, completed=0, taken=0, alu_steps=0)

    def _emit(self, event, **fields):
        if self.sink is not None:
            self.sink(dict(event=event, cycle=self.cycle, **fields))

    def _count(self, key):
        if self.stats[key] == 2**64 - 1:
            raise OverflowError("Finite diagnostic counter exhausted")
        self.stats[key] += 1

    def _context(self, context):
        if type(context) is not int or not 0 <= context < self.a.contexts:
            raise ValueError("Invalid context")

    def issue(self, command, context, token):
        """False means explicit backpressure; malformed/reused tokens raise."""
        if not isinstance(command, WordCommand):
            raise ValueError("Need a typed command")
        self._context(context)
        if type(token) is not int or not 0 <= token < 65536 or token != self.next_token[context]:
            raise ValueError("Need the next unique16-bit token;reset before exhaustion")
        if self.issued_this_cycle or self.busy[context] or self.cooldown[context]:
            return False
        if command.code == "tile":
            if self.dsp[0] is not None:
                return False
            self.dsp[0] = (_Job(command, context, token), None)
        else:
            free = next((k for k, job in enumerate(self.lanes) if job is None), None)
            if free is None:
                return False
            self.lanes[free] = _Job(command, context, token)
        self.busy[context] = self.issued_this_cycle = True
        self.next_token[context] += 1
        self._count("issued")
        self._emit("word_issue", context=context, token=token, command=vars(command))
        return True

    def _complete(self, job, value):
        c, ctx = job.c, job.context
        # Sign-extension/range checker of the assembled result, not arithmetic.
        overflow = not -(1 << (c.bits - 1)) * c.signed <= value < 1 << (c.bits - c.signed)
        if self.mailboxes[ctx] is not None:
            raise RuntimeError("Reserved result mailbox overwritten")
        self.mailboxes[ctx] = WordReply(ctx, job.token, c.tag, value, overflow, self.cycle)
        self.cooldown[ctx] = True
        self._count("completed")
        self._emit("word_complete", **vars(self.mailboxes[ctx]))

    def tick(self):
        """Advance exactly one physical clock; no precomputed answer countdown."""
        if self.cycle == 2**63 - 1:
            raise OverflowError("Finite clock exhausted")
        self.cycle += 1
        self.cooldown = [False] * self.a.contexts
        self.issued_this_cycle = self.taken_this_cycle = False
        for k, job in enumerate(self.lanes):
            if job is not None:
                self._count("alu_steps")
                if job.step():
                    self._complete(job, job.result)
                    self.lanes[k] = None
        if self.dsp[2] is not None:
            job, product = self.dsp[2]
            self._complete(job, product)
        stage0 = self.dsp[0]
        if stage0 is not None:
            job, _ = stage0
            # Both compiled formats fit one signed25x18 primitive. No wide product.
            stage0 = (job, job.c.a * job.c.b)
        self.dsp = [None, stage0, self.dsp[1]]

    def take(self, context):
        """One successful take per cycle; other stable mailboxes remain held."""
        self._context(context)
        if self.taken_this_cycle or self.mailboxes[context] is None:
            return None
        result, self.mailboxes[context] = self.mailboxes[context], None
        self.busy[context] = False
        self.taken_this_cycle = True
        self._count("taken")
        self._emit("word_take", context=context, token=result.token)
        return result

    @property
    def idle(self):
        return not any(self.busy)

    def flush(self):
        """One-cycle synchronous cancel; erase all tokens/results and pipelines."""
        self.tick()
        self.lanes = [None] * self.a.alu_lanes
        self.dsp = [None] * 3
        self.mailboxes = [None] * self.a.contexts
        self.busy = [False] * self.a.contexts
        self.cooldown = [False] * self.a.contexts
        self.next_token = [0] * self.a.contexts
        self.issued_this_cycle = self.taken_this_cycle = False
        self._emit("word_flush")


def word_fabric_cost(assumptions=None):
    """Register/port reservations; ALU LUT/control timing remain unsynthesized."""
    a = WordFabricAssumptions() if assumptions is None else assumptions
    if not isinstance(a, WordFabricAssumptions):
        raise ValueError("Need WordFabricAssumptions")
    fields = dict(
        alu_lanes=a.alu_lanes * 832,
        dsp3_stages=3 * 80,
        private_mailboxes=a.contexts * 256,
        per_context_control=a.contexts * 24,
        global_clock_counters_arbitration=384,
    )
    return dict(
        assumptions=vars(a),
        state_components=fields,
        state_bits=sum(fields.values()),
        DSP25x18=1,
        ALU32=a.alu_lanes,
        DSP_latency=3,
        DSP_II=1,
        issue_ports=1,
        result_take_ports=1,
        registered_feedback=True,
        max_operand_bits=208,
        extended_result_bits=210,
        RAM_copies=0,
        LUT_FF_timing=(
            'register bit reservation only;decode,mux,rangecheck,monitor reduction '
            'not synthesized'
        ),
        synthesis_or_timing_evidence=False,
    )
