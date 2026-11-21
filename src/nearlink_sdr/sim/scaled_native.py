"""Native signed DSP and finite registered scaled-score pipelines (§6.2.3).

Only25x18 products are evaluated by the DSP model. Wide products are assembled
by subsequent registered adders. Constants use finite shift/add trees,not DSP
or host-wide multiplication. All clocks are hypotheses,not synthesis evidence.
"""

from dataclasses import dataclass

from nearlink_sdr.phy.acquisition_flow import signed_tile_terms
from nearlink_sdr.phy.banked_words import DifferenceWord

__all__ = ["DifferencePipeline", "DifferenceQueue", "FoldResult", "Q12Pipeline", "ScorePipeline"]


def _count(stats, key, amount=1):
    value = stats[key] + amount
    if not 0 <= value < 2**64:
        raise OverflowError("Unsigned64 pipeline statistic exhausted")
    stats[key] = value


def _unsigned(value, bits):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise OverflowError("Exact unsigned datapath bound violated")
    return value


def _native(a, b):
    if not -(1 << 24) <= a < 1 << 24 or not -(1 << 17) <= b < 1 << 17:
        raise OverflowError("Operand exceeds native signed25x18 footprint")
    return a * b


class _NativeBank:
    """Four dedicated DSPs,three registered edges,one4-product issue/edge."""

    def __init__(self):
        self.cells = [None] * 3
        self.stats = dict(issued=0, products=0)

    def issue(self, operands, token):
        if self.cells[0] is not None:
            return False
        if len(operands) != 4 or len(token) != 2:
            raise ValueError("Need four native operands and(tag,component)")
        for a, b in operands:
            if not -(1 << 24) <= a < 1 << 24 or not -(1 << 17) <= b < 1 << 17:
                raise ValueError("Native signed operand range")
        self.cells[0] = tuple(operands), token
        _count(self.stats, "issued")
        return True

    def step(self):
        result = self.cells[2]
        first = self.cells[0]
        if first is not None:
            operands, token = first
            first = tuple(_native(a, b) for a, b in operands), token
            _count(self.stats, "products", 4)
        self.cells = [None, first, self.cells[1]]
        return result

    @property
    def idle(self):
        return not any(cell is not None for cell in self.cells)


@dataclass(frozen=True)
class FoldResult:
    tag: int
    real: int
    imag: int
    energy: int
    discarded: int


class DifferenceQueue:
    """64 total credits,64 RAM words and one registered156-bit front word.

    A pending read owns its credit. ``bool`` reports a valid front,``len`` all
    occupied credits. Front retirement and the next RAM read are distinct from
    the single RAM write port. Registering the read precedes front validity.
    No RAM data initialization or unbounded deque is used.
    """

    capacity = 64

    def __init__(self):
        self.memory = [None] * 64
        self.read_pointer = self.write_pointer = self.backing_count = 0
        self.front_word = None
        self.front_valid = self.read_pending = self.wrote = False
        self.last_read_address = None  # Current-edge port observation.
        self.stats = dict(writes=0, read_issues=0, takes=0, empty_edges=0, alias_holds=0, stalls=0)

    def __len__(self):
        return self.backing_count + int(self.front_valid or self.read_pending)

    def __bool__(self):
        return self.front_valid

    @property
    def can_write(self):
        return (
            not self.wrote
            and len(self) < self.capacity
            and self.write_pointer != self.last_read_address
        )

    def note_stall(self):
        _count(self.stats, "stalls")
        if self.write_pointer == self.last_read_address:
            _count(self.stats, "alias_holds")

    def append(self, item):
        if not self.can_write:
            raise OverflowError("Registered difference queue has no write credit")
        tag, word = item
        if type(tag) is not int or not 0 <= tag < 2**63 or not isinstance(word, DifferenceWord):
            raise ValueError("Need a63-bit tag and93-bit difference")
        self.memory[self.write_pointer] = item
        self.write_pointer = (self.write_pointer + 1) & 63
        self.backing_count += 1
        self.wrote = True
        _count(self.stats, "writes")

    def __getitem__(self, index):
        if index != 0 or not self.front_valid:
            raise IndexError("Only a completed registered front word is readable")
        return self.front_word

    def popleft(self):
        result = self[0]
        self.front_valid = False
        _count(self.stats, "takes")
        return result

    def advance(self):
        self.wrote = False
        self.last_read_address = None
        if self.read_pending:
            self.read_pending, self.front_valid = False, True
        if not self.front_valid and not self.read_pending and self.backing_count:
            word = self.memory[self.read_pointer]
            if word is None:
                raise RuntimeError("A positive FIFO occupancy referenced unwritten RAM")
            self.front_word = word
            self.last_read_address = self.read_pointer
            self.read_pointer = (self.read_pointer + 1) & 63
            self.backing_count -= 1
            self.read_pending = True
            _count(self.stats, "read_issues")
        elif not len(self):
            _count(self.stats, "empty_edges")

    def estimate(self):
        return dict(
            credit_capacity=64,
            RAM_words=64,
            word_bits=156,
            logical_RAM_bits=64 * 156,
            registered_front_bits=156,
            extra_unusable_word_capacity_bits=156,
            controller_bits=6 + 6 + 7 + 3,
            statistics_bits=6 * 64,
            read_ports=1,
            write_ports=1,
            read_latency_edges=1,
            front_visibility="registered read then input handoff;pending read counts toward64",
            runtime_RAM_clear_words=0,
            LUTRAM_primitive="RAM64M64x3SDP",
            LUTRAM_primitive_count=52,
            LUTRAM_LUT_estimate=208,
            SLICEM_slice_estimate=52,
            allocated_LUTRAM_bit_capacity=52 * 256,
            FIFO_single_address_conflict_compare_bits=6,
            same_address_read_write="prohibited;front credit leaves a backing slot",
            alternate_register_backing_bits=64 * 156,
            alternate_register_read_MUX2_upper=63 * 156,
            alternate_mapping_scope="same1R1W/registered-front contract;estimate,not synthesis",
            actual_synthesis_or_Fmax=False,
        )


class DifferencePipeline:
    """Lag4 registers→4 input DSPs→adds→4 square DSPs→power.

    Ten registered stages and a distinct pending-FIFO handoff. A full sink
    freezes the entire pipeline through clock enables;no input is consumed.
    The4-word IQ lag is read before its register write,with no RAM collision.
    """

    def __init__(self):
        self.cells = [None] * 10
        self.iq = [(0, 0, 0)] * 4
        self.next_tag = 0
        self.stats = dict(accepted=0, completed=0, native_products=0, stalled_edges=0)

    def _stage(self, index, value):
        if index == 1:
            tag, (r, i, e), (pr, pi, pe) = value
            products = (_native(r, pr), _native(i, pi), _native(i, pr), _native(r, pi))
            _count(self.stats, "native_products", 4)
            return tag, e, pe, products
        if index == 4:
            tag, e, pe, (a, b, c, d) = value
            return tag, a + b, c - d, e + pe
        if index == 5:
            tag, r, i, e = value
            terms = signed_tile_terms(r, r, 21, 21) + signed_tile_terms(i, i, 21, 21)
            if [shift for _, _, shift in terms] != [0, 17, 0, 17]:
                raise RuntimeError("Compiled21-bit square decomposition changed")
            _count(self.stats, "native_products", 4)
            return tag, r, i, e, tuple(_native(a, b) for a, b, _ in terms)
        if index == 8:
            tag, r, i, e, (a, b, c, d) = value
            return tag, r, i, e, _unsigned(a + (b << 17), 42), _unsigned(c + (d << 17), 42)
        if index == 9:
            tag, r, i, e, a, b = value
            return tag, DifferenceWord(r, i, _unsigned(a + b, 43), e)
        return value

    def step(self, incoming, pending, pending_capacity):
        if self.cells[-1] is not None and (
            len(pending) >= pending_capacity or not getattr(pending, "can_write", True)
        ):
            _count(self.stats, "stalled_edges")
            if hasattr(pending, "note_stall"):
                pending.note_stall()
            return
        if self.cells[-1] is not None:
            pending.append(self.cells[-1])
            _count(self.stats, "completed")
        old = self.cells
        new = [None] * len(old)
        for j in range(1, len(old)):
            if old[j - 1] is not None:
                new[j] = self._stage(j, old[j - 1])
        if incoming:
            tag, r, i, e = incoming[0]
            if tag != self.next_tag or not 0 <= tag < 2**63 - 5:
                raise ValueError("Difference producer lost a contiguous source tag")
            incoming.popleft()
            phase = tag % 4
            new[0] = tag, (r, i, e), self.iq[phase]
            self.iq[phase] = r, i, e
            self.next_tag += 1
            _count(self.stats, "accepted")
        self.cells = new

    @property
    def idle(self):
        return not any(cell is not None for cell in self.cells)

    def estimate(self):
        return dict(
            DSP=8,
            DSP_allocation="4 input products+4 square tiles,dedicated",
            native_tile_products_per_input=8,
            native_pipeline_edges=3,
            DSP_internal_pipeline_bit_capacity=8 * (25 + 18 + 43 + 48),
            IQ_lag_register_bits=108,
            fabric_stage_bits_upper=[117, 77, 77, 77, 113, 113, 113, 113, 197, 156],
            valid_bits=10,
            source_tag_counter_bits=63,
            statistics_bits=4 * 64,
            latency_edges=11,
            unstalled_II=1,
            lag_mux_input_bits=4 * 27,
            real_adders_per_edge_upper=dict(difference21=2, exponent8=1, square42=2, power43=1),
            physical_FF_or_Fmax_verified=False,
        )


class _ConstantTree:
    """One registered level/edge;reused adders within this fixed context.

    Shifted input wires are not leaf registers. The maximum first-level
    register bank is reused on following levels;the mux/add cost is explicit.
    """

    def __init__(self, coefficient, bits):
        if type(coefficient) is not int or not 1 <= coefficient < 2**32:
            raise ValueError("Need a compiled positive32-bit coefficient")
        self.positions = tuple(i for i in range(32) if coefficient & (1 << i))
        self.bits, self.nodes = bits, None

    @property
    def ready(self):
        return self.nodes is not None and len(self.nodes) == 1

    def step(self, value):
        if self.ready:
            return
        source = [value << i for i in self.positions] if self.nodes is None else self.nodes
        self.nodes = [
            _unsigned(source[i] + (source[i + 1] if i + 1 < len(source) else 0), self.bits)
            for i in range(0, len(source), 2)
        ]

    @property
    def result(self):
        if not self.ready:
            raise ValueError("Constant tree result has not crossed its registered levels")
        return self.nodes[0]

    def estimate(self):
        leaves = len(self.positions)
        slots = (leaves + 1) // 2
        return dict(
            constant_nonzero_bits=leaves,
            adder_units=leaves // 2,
            adder_width=self.bits,
            additions_per_value=leaves - 1,
            state_bits=slots * self.bits + 6,
            registered_levels=max(1, (leaves - 1).bit_length()),
            shared_between_contexts=False,
            constant_shifts="wiring;first-level input fanout and selection still map to fabric",
        )


class ScorePipeline:
    """Two contexts share4 native DSPs;constant trees have dedicated contexts.

    Numerator squares use two consecutive4-tile issues. Every reconstruction,
    sum,threshold compare and output handoff crosses a modeled edge.
    """

    def __init__(self, widths, coefficient_energy):
        self.w, self.coefficient_energy = dict(widths), coefficient_energy
        self.native = _NativeBank()
        self.jobs = [None, None]
        self.reply = self.pairs = self.square = None
        self.stats = dict(accepted=0, completed=0, pair_adds=0, square_adds=0, numerator_adds=0)

    def _find(self, tag):
        result = next(
            (job for job in self.jobs if job is not None and job["source"].tag == tag), None
        )
        if result is None:
            raise ValueError("Native score reply lost its owning context")
        return result

    def step(self, incoming, normalizer):
        # Snapshot old states first. Newly computed stages cannot feed another
        # dependent action on the same edge.
        for index, job in enumerate(self.jobs):
            if job is None:
                continue
            source = job["source"]
            if job["phase"] == "output" and normalizer.offer(
                source, job["numerator"], job["denominator"].result, job["hit"]
            ):
                self.jobs[index] = None
                _count(self.stats, "completed")
                continue
            if job["phase"] == "numerator" and job["right"].ready and job["denominator"].ready:
                job["hit"] = source.energy > 0 and (job["numerator"] << 24 >= job["right"].result)
                job["phase"] = "output"
            elif job["phase"] == "squares" and all(v is not None for v in job["squares"]):
                job["numerator"] = _unsigned(sum(job["squares"]), self.w["numerator"])
                _count(self.stats, "numerator_adds")
                job["phase"] = "numerator"
            job["right"].step(source.energy)
            job["denominator"].step(source.energy)
        if self.square is not None:
            tag, component, value = self.square
            self._find(tag)["squares"][component] = _unsigned(value, 2 * self.w["correlation"])
        next_square = None
        if self.pairs is not None:
            tag, component, a, b = self.pairs
            next_square = tag, component, a + b
            _count(self.stats, "square_adds")
        next_pairs = None
        if self.reply is not None:
            values, (tag, component) = self.reply
            a, b, c, d = values
            next_pairs = tag, component, a + (b << 17), (c << 24) + (d << 41)
            _count(self.stats, "pair_adds", 2)
        eligible = [job for job in self.jobs if job is not None and job["issued"] < 2]
        if eligible:
            job = min(eligible, key=lambda item: item["source"].tag)
            component = job["issued"]
            value = job["source"].real if component == 0 else job["source"].imag
            terms = signed_tile_terms(value, value, self.w["correlation"], self.w["correlation"])
            if [shift for _, _, shift in terms] != [0, 17, 24, 41]:
                raise RuntimeError("Compiled28..30-bit square decomposition changed")
            if not self.native.issue([(a, b) for a, b, _ in terms], (job["source"].tag, component)):
                raise RuntimeError("Score dispatcher issued twice on its single native port")
            job["issued"] += 1
        self.square, self.pairs, self.reply = next_square, next_pairs, self.native.step()
        free = next((i for i, job in enumerate(self.jobs) if job is None), None)
        if incoming and free is not None:
            source = incoming.popleft()
            if not isinstance(source, FoldResult):
                raise ValueError("Need the completed registered C/E ticket")
            for value in (source.real, source.imag):
                if (
                    not -(1 << (self.w["correlation"] - 1))
                    <= value
                    < 1 << (self.w["correlation"] - 1)
                ):
                    raise OverflowError("Correlation word exceeded compiled width")
            _unsigned(source.energy, self.w["window_energy"])
            self.jobs[free] = dict(
                source=source,
                issued=0,
                squares=[None, None],
                numerator=None,
                hit=False,
                phase="squares",
                right=_ConstantTree(self.w["threshold_integer"], self.w["compare"]),
                denominator=_ConstantTree(
                    self.coefficient_energy,
                    self.w["window_energy"] + (self.coefficient_energy - 1).bit_length(),
                ),
            )
            _count(self.stats, "accepted")

    @property
    def idle(self):
        return (
            not any(job is not None for job in self.jobs)
            and self.native.idle
            and self.reply is self.pairs is self.square is None
        )

    def estimate(self):
        c, e, n = [self.w[key] for key in ("correlation", "window_energy", "numerator")]
        d = e + (self.coefficient_energy - 1).bit_length()
        right = _ConstantTree(self.w["threshold_integer"], self.w["compare"]).estimate()
        denominator = _ConstantTree(self.coefficient_energy, d).estimate()
        context = 63 + 2 * c + e + 7 + 2 + 4 * c + 2 + n + 1 + 1 + 3
        context += right["state_bits"] + denominator["state_bits"]
        return dict(
            DSP=4,
            contexts=2,
            native_tile_products_per_value=8,
            DSP_internal_pipeline_bit_capacity=4 * (25 + 18 + 43 + 48),
            native_metadata_register_bits=3 * 64,
            native_valid_bits=3,
            context_register_bits=2 * context,
            context_valid_bits=2,
            reply_pair_square_register_bits=(4 * 43 + 64) + (4 * c + 64) + (2 * c + 64),
            reply_pair_square_valid_bits=3,
            statistics_bits=5 * 64 + 2 * 64,
            reconstruction_adders=[2 * c, 2 * c, 2 * c],
            numerator_adder_bits=n,
            threshold_compare_bits=self.w["compare"],
            numerator_left_shift="constant24-bit wiring",
            constant_tree_per_context=dict(threshold=right, denominator=denominator),
            no_free_constant_multiplier=True,
            shared_DSP_scope="two consecutive component issues;prefix DSPs are separate",
            whole_receiver_acceptance=False,
        )


class Q12Pipeline:
    """Twelve registered restoring stages plus explicit output handoff.

    Commit sink has one write/edge. Zero/unity flags follow the same pipeline;
    no data-dependent fast path reorders outputs or consumes a CRC answer.
    """

    def __init__(self, widths, terms, coefficient_energy):
        self.w, self.terms, self.coefficient_energy = dict(widths), terms, coefficient_energy
        self.input = None
        self.cells = [None] * 12
        self.stats = dict(accepted=0, completed=0, subtracts=0)

    def offer(self, source, numerator, denominator, hit):
        if self.input is not None:
            return False
        self.input = dict(
            source=source,
            numerator=numerator,
            denominator=denominator,
            hit=hit,
            mode=1 if source.energy == 0 else 2 if numerator >= denominator else 0,
            remainder=0 if source.energy == 0 or numerator >= denominator else numerator,
            rank=4096 if source.energy > 0 and numerator >= denominator else 0,
        )
        _count(self.stats, "accepted")
        return True

    def _bit(self, cell):
        result = dict(cell)
        if result["mode"] == 0:
            remainder = result["remainder"] << 1
            bit = remainder >= result["denominator"]
            if bit:
                remainder -= result["denominator"]
                _count(self.stats, "subtracts")
            result["remainder"] = remainder
            result["rank"] = (result["rank"] << 1) | int(bit)
        return result

    def step(self):
        output = self.cells[-1]
        if output is not None:
            _count(self.stats, "completed")
        old = [self.input, *self.cells[:-1]]
        self.cells = [self._bit(cell) if cell is not None else None for cell in old]
        self.input = None
        return output

    @property
    def idle(self):
        return self.input is None and not any(cell is not None for cell in self.cells)

    def estimate(self):
        c, e, n = [self.w[key] for key in ("correlation", "window_energy", "numerator")]
        d = e + (self.coefficient_energy - 1).bit_length()
        # Carry full FoldResult for exact point observations,not just the rank.
        source = 63 + 2 * c + e + 7
        cell = source + n + d + (d + 1) + 13 + 2 + 1
        return dict(
            DSP=0,
            registered_stages=12,
            latency_edges=13,
            initiation_interval=1,
            input_register_bits=cell,
            pipeline_register_bits=12 * cell,
            valid_bits=13,
            statistics_bits=3 * 64,
            wide_subtract_compare_units=12,
            fractional_subtract_width=d + 1,
            initial_zero_unity_compare_width=max(n, d),
            fractional_iterations=12,
            divide="registered restoring,one bit/stage;zero/unity retained",
            full_receiver16MSps=False,
        )
