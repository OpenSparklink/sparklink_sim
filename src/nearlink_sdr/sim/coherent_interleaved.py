"""Exact finite coherent words with signed native products and finite interleaving.

One DSP25x18,one issue/take port;four contexts share two actual ALU32 lanes.
The original CORDIC,NCO ROM,wide tail and threshold program are reused (§6.2.3).
No candidate,phase grid,CRC,expected data or floating answer drives scheduling.
"""

from nearlink_sdr.phy.sync_coherent import phase_rotation_words
from nearlink_sdr.sim.coherent_captured_frame import WordConfirmedCapturedFrameReplay
from nearlink_sdr.sim.coherent_word_flow import CoherentWordService
from nearlink_sdr.sim.word_arithmetic import (
    WordCommand,
    WordFabric,
    WordFabricAssumptions,
    word_fabric_cost,
)

__all__ = ["InterleavedCoherentWordService", "ScheduledCoherentCapturedFrameReplay",
           "SignedCoherentWordService"]


class _ArithmeticContext:
    """One compiled worker and pending command;private result mailbox in fabric."""

    _abs = CoherentWordService._abs
    _sum = CoherentWordService._sum
    _tile = CoherentWordService._tile
    _square22 = CoherentWordService._square22

    def __init__(self, owner, context):
        self.owner, self.context = owner, context

    def _command(self, code, a, b=0, *, ab, bb=1, asigned=False, bsigned=False,
                 bits, signed_result=False):
        if max(ab, bb, bits) > 110:
            raise ValueError("Compiled coherent pending command exceeds110 bits")
        c = WordCommand(code, a, b, ab, bb, asigned, bsigned, bits, signed_result)
        fabric, ctx = self.owner.fabric, self.context
        token = fabric.next_token[ctx]
        while not fabric.issue(c, ctx, token):
            yield
        if code == "tile":
            self.owner._count("native_products")
        while True:
            yield
            reply = fabric.take(ctx)
            if reply is not None:
                if reply.overflow:
                    raise OverflowError("Coherent worker exceeded its compiled word")
                return reply.value

    def native(self, value, coefficient):
        return (yield from self._command("tile", value, coefficient, ab=22, bb=18,
                                          asigned=True, bsigned=True, bits=43, signed_result=True))

    def rotate_accumulate(self, yr, yi, c, s, previous, *, imaginary):
        a = yield from self.native(yr, s if imaginary else c)
        b = yield from self.native(yi, c if imaginary else s)
        value = yield from self._sum(a, b, 48, signed_result=True, subtract=not imaginary)
        return (yield from self._sum(previous, value, 48, signed_result=True))


class SignedCoherentWordService(CoherentWordService):
    """First single-factor candidate:one context,original program with signed DSP."""

    def _accumulate(self, source, step):
        worker = _ArithmeticContext(self, 0)
        phase, cr, ci, energy = 0, 0, 0, 0
        for j, code in enumerate(self.codes):
            self._emit("coherent_cache_read", id=source.ident, index=j)
            self._count("cache_reads")
            yield
            yr, yi = source.words[j]
            combined = (phase + (code << 16)) & (2**20 - 1)
            for _ in range(4):
                yield
            c, s = phase_rotation_words(combined, phase_bits=20)
            self._count("coefficient_outputs")
            phase = (phase + step) & (2**20 - 1)
            ar, ai = (yield from worker._abs(yr, 22)), (yield from worker._abs(yi, 22))
            cr = yield from worker.rotate_accumulate(yr, yi, c, s, cr, imaginary=False)
            ci = yield from worker.rotate_accumulate(yr, yi, c, s, ci, imaginary=True)
            er, ei = (yield from worker._square22(ar)), (yield from worker._square22(ai))
            energy = yield from worker._sum(energy, er, 52)
            energy = yield from worker._sum(energy, ei, 52)
        return cr, ci, energy

    @staticmethod
    def estimate():
        old = CoherentWordService.estimate()
        fields = old["register_fields"] | dict(native_input_mode=6)
        return old | dict(
            register_fields=fields, register_bits_upper=old["register_bits_upper"] + 6,
            compiled_structure="serial-signed22x18", signed_input_mux2_bits=43,
            native_formats=["unsigned24x17", "signed22x18"],
            extra_DSP_for_signed_products=0, signed_mode_flags_in_pipeline_reserved=True,
        )


class InterleavedCoherentWordService(SignedCoherentWordService):
    """Four compiled workers;round-robin requests,registered native/ALU replies.

    Pending commands/results are bounded by four,not a future event calendar.
    A blocked context keeps its command;one actual issue and take per clock.
    """

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.fabric = WordFabric(WordFabricAssumptions(2, 4),
                                 initial_cycle=self.started, sink=self.sink)
        self._cursor, self._workers = 0, []

    def _parallel(self, workers):
        if len(workers) not in (2, 4) or self._workers:
            raise RuntimeError("Need one compiled set of two/four worker programs")
        self._workers = list(workers)
        active, values = [True] * len(workers), [None] * len(workers)
        try:
            while any(active):
                for offset in range(len(workers)):
                    k = (self._cursor + offset) % len(workers)
                    if not active[k]:
                        continue
                    try:
                        next(workers[k])
                    except StopIteration as reply:
                        values[k], active[k] = reply.value, False
                self._cursor = (self._cursor + 1) % 4
                if any(active):
                    yield
            return tuple(values)
        finally:
            for worker in self._workers:
                worker.close()
            self._workers = []

    def _accumulate(self, source, step):
        workers = [_ArithmeticContext(self, j) for j in range(4)]
        phase, cr, ci, energy = 0, 0, 0, 0
        for j, code in enumerate(self.codes):
            self._emit("coherent_cache_read", id=source.ident, index=j)
            self._count("cache_reads")
            yield
            yr, yi = source.words[j]
            combined = (phase + (code << 16)) & (2**20 - 1)
            for _ in range(4):
                yield
            c, s = phase_rotation_words(combined, phase_bits=20)
            self._count("coefficient_outputs")
            phase = (phase + step) & (2**20 - 1)
            ar, ai = yield from self._parallel((workers[0]._abs(yr, 22), workers[1]._abs(yi, 22)))
            cr, ci, er, ei = yield from self._parallel((
                workers[0].rotate_accumulate(yr, yi, c, s, cr, imaginary=False),
                workers[1].rotate_accumulate(yr, yi, c, s, ci, imaginary=True),
                workers[2]._square22(ar), workers[3]._square22(ai),
            ))
            energy = yield from self._sum(energy, er, 52)
            energy = yield from self._sum(energy, ei, 52)
        return cr, ci, energy

    @staticmethod
    def estimate():
        old = SignedCoherentWordService.estimate()
        fabric = word_fabric_cost(WordFabricAssumptions(2, 4))
        fields = old["register_fields"] | dict(scheduler_control=128, pending_commands=4 * 275)
        return old | dict(
            fabric=fabric, register_fields=fields,
            register_bits_upper=fabric["state_bits"] + sum(fields.values()),
            compiled_structure="interleaved-signed22x18", ALU32=2,
            pending_command_slots=4, pending_command_bits=275, command_operand_bound_bits=110,
            pending_command_mux4_bits=275, request_and_take_arbiter_contexts=4,
            result_mailbox_mux4_bits=256,
            issue_ports=1, result_take_ports=1, registered_mailboxes=4,
            tail_and_worker0_command_port_reuse=True,
            no_extra_sample_or_coefficient_ports=True,
        )


class ScheduledCoherentCapturedFrameReplay(WordConfirmedCapturedFrameReplay):
    """Explicit compiled experiment;uses the same actual receipt/head/raw owner."""

    def __init__(self, *args, structure, **options):
        choices = {"serial-signed22x18": SignedCoherentWordService,
                   "interleaved-signed22x18": InterleavedCoherentWordService}
        if type(structure) is not str or structure not in choices:
            raise ValueError("Need an explicit compiled coherent structure")
        self.structure, self._service_class = structure, choices[structure]
        super().__init__(*args, **options)

    def _new_word_service(self, selection, cycle):
        return self._service_class(selection, self.codes, initial_cycle=cycle)

    def _word_service_cost(self):
        return self._service_class.estimate()

    def _terminal(self, event):
        event["coherent_schedule_structure"] = self.structure
        return super()._terminal(event)

    def estimate(self):
        return super().estimate() | dict(compiled_coherent_structure=self.structure)
