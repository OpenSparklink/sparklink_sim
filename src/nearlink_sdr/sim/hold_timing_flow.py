"""Bounded exact known-sync timing:registered cache→DSP energy→power feedback.

One cache pass,no energy side RAM. The private power recurrence may run ahead
by four credits;only accepting an ordered reply commits the sole packet NCO.
Wide RNE/add/compare and constant-rate commit are explicit one-edge macros.
"""

from collections import deque
from copy import copy, deepcopy

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.sync_fit_constants import GAIN_FLOOR
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, signed, unsigned
from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState, compare_scaled, positive
from nearlink_sdr.sim.initial_interpolant_flow import InitialInterpolantPort
from nearlink_sdr.sim.timing_word_flow import (
    TimingWordService,
    two_real_products,
    validate_divider,
)

__all__ = ["HoldTimingPipeline", "PipelinedPacketTimingService"]


class HoldTimingPipeline:
    """Four total contexts,two output slots;no CRC,payload,truth or alternative phase."""

    def __init__(self, state, cache, clock_origin=0, *, word_sink=None):
        if (
            not isinstance(state, WordTimingState)
            or state.count != 0
            or state.fmt != TimingWordFormat(30, 32)
            or state.omega_q32 != 4 << 32
            or state.error_word != 0
            or state.previous_word is not None
            or state.previous_q32 != state.next_q32
        ):
            raise ValueError("Need frozen30/32 packet timing before first known-sync word")
        if (
            not isinstance(cache, InitialInterpolantPort)
            or cache.pending is not None
            or cache.cache.state != "sealed"
            or cache.cache.read != 0
            or cache.cache.symbols != state.hold_symbols
            or cache.cache.fmt != INPUT_FORMAT
            or cache.cache.phase >= 65536
            or state.next_q32 != ((cache.cache.first_cell << 32) + (cache.cache.phase << 16))
        ):
            raise ValueError("Need one matching sealed ordered sync cache")
        if type(clock_origin) is not int or not 0 <= clock_origin < 2**63 - 1:
            raise ValueError("Need bounded actual cache epoch")
        self.state, self.cache, self.clock_origin = state, cache, clock_origin
        self.word_sink = word_sink
        self.cycle = 0
        self.first_cycle = self.ready_cycle = None
        self.start_q32 = state.next_q32
        self.power = state.power_word
        self.requested = self.computed = 0
        self.energy = self.power_job = None
        self.outputs = deque()
        self.prepared_state = self.commit_ready = None
        self.taken_this_cycle = False
        self.closed = None
        self.stats = dict.fromkeys(
            [
                "cache_requests",
                "cache_returns",
                "native_products",
                "power_updates",
                "accepted",
                "output_stall_edges",
                "credit_stall_edges",
                "aborted",
            ],
            0,
        )

    def _count(self, key, amount=1):
        self.stats[key] = unsigned(self.stats[key] + amount, 64)

    @property
    def ready(self):
        return self.prepared_state is not None and not self.closed

    @property
    def mode(self):
        if self.closed:
            return self.closed
        return "idle" if self.drained else "hold_pipeline"

    @property
    def drained(self):
        return (
            self.state.count == self.state.hold_symbols
            and not self.outputs
            and self.energy is None
            and self.power_job is None
            and self.cache.pending is None
            and self.prepared_state is None
        )

    def peek(self):
        return self.outputs[0] | dict(ready=self.commit_ready) if self.ready else None

    def validate_take(self):
        """Read-only owner/counter preflight before a joint symbol transfer."""
        if not self.ready or self.taken_this_cycle:
            return False
        j = self.outputs[0]
        s = self.state
        if (
            j["index"] != s.count
            or s.next_q32 != self.start_q32 + (s.count << 34)
            or s.omega_q32 != 4 << 32
            or s.error_word
        ):
            raise RuntimeError("Known-sync packet state changed before ordered acceptance")
        unsigned(self.stats["accepted"] + 1, 64)
        return True

    def take(self):
        if not self.validate_take():
            return None
        j, s = self.outputs[0], self.state
        # Transfer the actual registered result. No NCO/power recomputation in take.
        accepted = unsigned(self.stats["accepted"] + 1, 64)
        result = deepcopy(self.prepared_state)
        for name, value in result.items():
            setattr(
                s,
                "power_word" if name == "power" else name,
                tuple(value) if name == "power" else value,
            )
        s.previous_word = j["word"]
        self.outputs.popleft()
        self.first_cycle, self.ready_cycle = j["issued"], self.commit_ready
        self.prepared_state = self.commit_ready = None
        self.stats["accepted"] = accepted
        self.taken_this_cycle = True
        return result

    def _energy(self):
        j = self.energy
        if j is None:
            return
        age = j["age"]
        if age == 0:
            self._count("native_products", 4)
        elif age == 2:
            r, i, _ = unpack_interpolated_word(j["word"], INPUT_FORMAT)
            j["products"] = [signed(a * b, 43) for a, b, _ in two_real_products(r, r, i, i)]
        elif age == 3:
            p = j.pop("products")
            j["pairs"] = [signed(p[0] + (p[1] << 17), 45), signed(p[2] + (p[3] << 17), 45)]
        elif age == 4:
            j["energy"] = unsigned(sum(j.pop("pairs")), 44)
        if age >= 4 and self.power_job is None:
            self.power_job = j | dict(age=0)
            self.energy = None
        else:
            j["age"] = min(age + 1, 5)

    def _power(self):
        j = self.power_job
        if j is None:
            return
        age = j["age"]
        m, e = self.power
        if age == 0:
            j["decay"] = unsigned((m << 6) - m, 36)
        elif age == 1:
            j["dm"], j["de"] = positive(j["decay"], e, 44)
            j["em"], j["ee"] = positive(j["energy"], 2 * j["exponent"], 44)
        elif age == 2:
            j["common"] = max(ex for value, ex in [(j["dm"], j["de"]), (j["em"], j["ee"])] if value)
            j["a"] = rne_shift(j["dm"], j["common"] - j["de"])
            j["b"] = rne_shift(j["em"], j["common"] - j["ee"]) if j["em"] else 0
            j["erased"] = int(j["a"] == 0 and j["decay"] != 0) + int(
                j["b"] == 0 and j["energy"] != 0
            )
        elif age == 3:
            j["sum"] = unsigned(j["a"] + j["b"], 45)
        elif age == 4:
            j["power"] = positive(j["sum"], j["common"] - 6, 30)
        elif age == 5:
            if len(self.outputs) == 2:
                self._count("output_stall_edges")
                return
            floor = positive(*GAIN_FLOOR, 30)
            floored = compare_scaled(*j["power"], *floor) < 0
            power = floor if floored else j["power"]
            self.outputs.append(
                dict(
                    index=j["index"],
                    word=j["word"],
                    power=power,
                    erased=j["erased"],
                    floored=floored,
                    issued=j["issued"],
                    ready=self.cycle + 1,
                )
            )
            self.power = power
            self.computed = unsigned(self.computed + 1, 7)
            self._count("power_updates")
            self.power_job = None
            return
        j["age"] += 1

    def step(self):
        c = unsigned(self.clock_origin + self.cycle, 63)
        if not self.closed:
            if self.outputs and self.prepared_state is None:
                j = self.outputs[0]
                if j["index"] != self.state.count:
                    raise RuntimeError("One registered state preparation per ordered result")
                # Frozen hold:one actual macro edge for96bit +constant4 and
                # parallel64bit counters. Shadow is not another live packet owner.
                shadow = copy(self.state)
                self.prepared_state = shadow.commit(
                    j["word"], j["power"], 0, False, j["erased"], 0, j["floored"], 0, 0
                )
                self.commit_ready = self.cycle + 1
            if self.energy is None and (self.word_sink is None or self.word_sink.can_issue):
                reply = self.cache.take(c)
                if reply is not None:
                    index, word = reply
                    if index != self.stats["cache_returns"]:
                        raise RuntimeError("Registered sync-cache reply changed order")
                    if self.word_sink is not None and not self.word_sink.issue(word):
                        raise RuntimeError("Broadcast consumed a reply without both input credits")
                    _, _, exponent = unpack_interpolated_word(word, INPUT_FORMAT)
                    self.energy = dict(
                        index=index, word=word, exponent=exponent, issued=self.cycle, age=0
                    )
                    self._count("cache_returns")
            self._power()  # Feedback commits before a waiting next energy enters stage0.
            self._energy()
            credits = self.requested - self.state.count
            if (
                self.requested < self.state.hold_symbols
                and self.energy is None
                and self.cache.pending is None
                and credits < 4
            ):
                key = self.cache.cache
                if self.cache.request(key.first_cell + 4 * self.requested, key.phase, c):
                    self.requested = unsigned(self.requested + 1, 7)
                    self._count("cache_requests")
            elif self.requested < self.state.hold_symbols and credits == 4:
                self._count("credit_stall_edges")
            if not 0 <= self.requested - self.state.count <= 4 or len(self.outputs) > 2:
                raise RuntimeError("Finite hold credits escaped their allocation")
        self.cycle = unsigned(self.cycle + 1, 64)
        self.taken_this_cycle = False

    def abort(self, *, release_port=True):
        if not self.closed:
            self._count("aborted")
            self.outputs.clear()
            self.prepared_state = self.commit_ready = None
            self.energy = self.power_job = None
            if release_port:
                self.cache.abort(self.clock_origin + self.cycle)
            self.closed = "aborted"

    def release(self):
        if self.closed:
            raise RuntimeError("Closed hold context cannot transfer twice")
        if not self.drained:
            raise RuntimeError("Known-sync pipeline must drain before transfer")
        self.closed = "released"

    def estimate(self):
        parts = dict(
            geometry_and_progress=102,
            private_power=40,
            energy_context=225,
            power_context=609,
            two_output_slots=460,
            statistics=512,
            clocks_and_epochs=256,
            control=32,
            prepared_state=686,
            prepared_state_valid_and_edge=65,
        )
        return dict(
            additional_state_parts=parts,
            additional_state_bits_upper=sum(parts.values()),
            shared_DSP25x18=4,
            new_raw_RAM_bits=0,
            additional_cache_read_ports=0,
            cache_reads=self.state.hold_symbols,
            cache_read_latency=1,
            output_slots=2,
            total_credits=4,
            DSP_latency=3,
            energy_tree_edges=2,
            power_feedback_edges=6,
            no_stall_output_II=6,
            constant_rate_NCO_commit_edges=1,
            shared_bank_switch_edges=1,
            input_configuration="known32/62/126,rate4,power30,error32 only",
            assumptions="RNE/normalize/wide compare/add/counters1edge;not synthesis",
            normalization_carrier_decoder_cycles=None,
            full_receiver16MSps=False,
        )


class PipelinedPacketTimingService:
    """One4DSP bank:hold pipeline drains before original data feedback activates."""

    def __init__(self, state, cache, clock_origin, *, word_sink=None, divider="full"):
        self.divider = validate_divider(divider)
        self.state = state
        self.hold = HoldTimingPipeline(state, cache, clock_origin, word_sink=word_sink)
        self.normal = None
        self.cycle = 0
        self.first_cycle = self.ready_cycle = None

    @property
    def mode(self):
        return self.normal.mode if self.normal is not None else self.hold.mode

    @property
    def stats(self):
        normal = (
            self.normal.stats
            if self.normal is not None
            else dict.fromkeys(
                [
                    "accepted",
                    "completed",
                    "native_products",
                    "divider_steps",
                    "backpressure",
                    "output_stall_edges",
                    "aborted",
                ],
                0,
            )
        )
        h = self.hold.stats
        return normal | dict(
            accepted=normal["accepted"] + h["cache_requests"],
            completed=normal["completed"] + h["accepted"],
            native_products=normal["native_products"] + h["native_products"],
            aborted=normal["aborted"] + h["aborted"],
        )

    def issue(self, current, midpoint):
        if self.normal is None:
            if not self.hold.drained or midpoint is None:
                raise RuntimeError("Hold contexts must drain before switching the DSP bank")
            if self.hold.taken_this_cycle:
                self.hold._count("credit_stall_edges")
                return False  # One real bank-selection edge,not same-edge handoff.
            self.hold.release()
            self.normal = TimingWordService(self.state, divider=self.divider)
            self.normal.cycle = self.cycle
        return self.normal.issue(current, midpoint)

    def take(self):
        service = self.hold if self.normal is None else self.normal
        reply = service.take()
        if reply is not None:
            self.first_cycle, self.ready_cycle = service.first_cycle, service.ready_cycle
        return reply

    def step(self):
        service = self.hold if self.normal is None else self.normal
        if service.cycle != self.cycle:
            raise RuntimeError("Exactly one owner advances the shared DSP bank")
        service.step()
        self.cycle = service.cycle

    def abort(self):
        if self.normal is None:
            self.hold.abort(release_port=False)
        else:
            self.normal.abort()

    def release(self):
        (self.hold if self.normal is None else self.normal).release()

    def estimate(self):
        normal = TimingWordService(self.state, divider=self.divider).estimate()
        extra = self.hold.estimate()
        return normal | dict(
            known_sync_pipeline=extra,
            state_register_bits_upper=normal["state_register_bits_upper"]
            + extra["additional_state_bits_upper"],
            hold_and_data_share_DSP_by_drained_phase=True,
            old4096_state_budget_passed=normal["state_register_bits_upper"]
            + extra["additional_state_bits_upper"]
            <= 4096,
        )
