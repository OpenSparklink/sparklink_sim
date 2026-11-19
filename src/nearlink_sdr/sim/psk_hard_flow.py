"""Registered hard symbol decision and bounded real-CRC header assembly.

One owned symbol / one owned header;no input FIFO,no hidden sample history.
The observer is a host-only diagnostic sink and never steers a decision.
"""

from copy import deepcopy

from nearlink_sdr.phy.psk_hard_word import _decision, _project, _validate
from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.header_word_flow import HeaderWordService

__all__ = ["HardPSKWordService", "HeaderFromWordsService"]


class HardPSKWordService:
    """Input capture→registered23bit projections→registered hard result.

    Output is visible after two edges. A transfer edge is not reused for a new
    input:input_ready returns on the following edge,so the unstalled II is3.
    """

    def __init__(self, observer=None):
        if observer is not None and not callable(observer):
            raise ValueError("Observer must be callable")
        self.observer = observer
        self.cycle = 0
        self.mode = "idle"
        self.word = self.index = self.first = self.second = 0
        self.modulation = "BPSK"
        self.zero = False
        self.result = None
        self.last_take = None
        self.stats = dict.fromkeys(
            (
                "accepted",
                "completed",
                "aborted",
                "input_stalls",
                "output_stalls",
                "add23",
                "sign22",
                "sign23",
                "ties",
                "zeros",
            ),
            0,
        )

    def _count(self, key, amount=1):
        self.stats[key] = unsigned(self.stats[key] + amount, 64)

    def _emit(self, event, **fields):
        if self.observer:
            self.observer(dict(cycle=self.cycle, event=event, **fields))

    @property
    def input_ready(self):
        return self.mode == "idle" and (self.last_take is None or self.cycle > self.last_take)

    @property
    def ready(self):
        return self.mode == "ready"

    def issue(self, word, modulation, symbol_index):
        _validate(word, modulation, symbol_index)
        if self.mode == "aborted":
            raise RuntimeError("Reset aborted hard decision before reuse")
        if not self.input_ready:
            self._count("input_stalls")
            return False
        self._count("accepted")
        self.word, self.modulation, self.index = int(word), modulation, symbol_index
        self.mode = "project"
        self._emit("accept", word=self.word, index=self.index, modulation=modulation)
        return True

    def step(self):
        unsigned(self.cycle + 1, 64)
        if self.mode == "project":
            real, imag = _validate(self.word, self.modulation, self.index)
            self.first, self.second = _project(real, imag, self.modulation, bool(self.index & 1))
            self.zero = real == imag == 0
            if self.modulation == "QPSK" and self.index & 1:
                self._count("add23", 2)
            self._emit(
                "projection", first=self.first, second=self.second, register_edge=self.cycle + 1
            )
            self.mode = "decide"
        elif self.mode == "decide":
            label, bits, tie = _decision(self.first, self.second, self.modulation)
            self._count(
                "sign23" if self.modulation == "QPSK" and self.index & 1 else "sign22", len(bits)
            )
            self._count("completed")
            self._count("ties", int(tie))
            self._count("zeros", int(self.zero))
            self.result = dict(
                index=self.index, label=label, bits=bits, mathematical_tie=tie, zero_word=self.zero
            )
            self.mode = "ready"
            self._emit("decision", **self.result, register_edge=self.cycle + 1)
        elif self.mode == "ready":
            self._count("output_stalls")
        self.cycle += 1

    def take(self):
        if not self.ready:
            return None
        result, self.result, self.mode = self.result, None, "idle"
        self.last_take = self.cycle
        self._emit("transfer", **result)
        return result

    def abort(self):
        if self.mode not in ("idle", "aborted"):
            self._count("aborted")
            self.mode, self.result = "aborted", None

    def reset(self):
        if self.mode not in ("idle", "aborted"):
            raise RuntimeError("Drain or abort before resetting hard decision")
        self.mode = "idle"

    @staticmethod
    def cost():
        state = dict(
            input_word=55,
            absolute_index=17,
            modulation=1,
            projections=46,
            output_bits=2,
            tie_zero_flags=2,
            mode=3,
            last_take=64,
            last_take_valid=1,
            statistics=10 * 64,
            cycle=64,
        )
        return dict(
            state_bits=state,
            logical_total_bits=sum(state.values()),
            DSP=0,
            RAMB18=0,
            input_bus_bits=55 + 17 + 1,
            input_register_ports=1,
            projection_write_ports=2,
            projection_write_bits_each=23,
            add_subtract23_parallel=2,
            sign_tests_max=2,
            zero_format_check_bits=44 + 11,
            multiplications=0,
            divisions=0,
            square_roots=0,
            latency_edges=2,
            unstalled_II=3,
            rounding="none",
            saturation="none",
            overflow="invalid word rejected",
            tie="maximum exact projection;lowest numeric Gray label",
            clock_assumption="one23bit add/sub stage and one sign stage",
            verified_Fmax=None,
            LUT=None,
            FF=None,
            synthesis=False,
            observer_trace_is_hardware_RAM=False,
            whole_RX16MSps=False,
        )


class HeaderFromWordsService:
    """Actual55bit head wire words→hard bits→existing finite real CRC/geometry.

    begin() owns one compiled ordinary A2+LQI/B1 head. offer() starts at the
    profile's sync-end index;known sync words are routed by the upstream owner.
    All wire indices,including discarded pilots,must be contiguous.
    """

    def __init__(self, profile, *, layout="dense", observer=None):
        if observer is not None and not callable(observer):
            raise ValueError("Observer must be callable")
        self.observer = observer
        self.header = HeaderWordService(profile, layout=layout, observer=self._header_event)
        self.hard = HardPSKWordService(observer=self._hard_event)
        profile = self.header.profile
        self.modulation = "BPSK" if profile.config.frame_type == 4 else "QPSK"
        self.bps = 1 if self.modulation == "BPSK" else 2
        self.interval = 16 if profile.config.frame_type == 2 else 4
        self.start = len(profile.sync_symbols())
        self.end = self.header.geometry.header_end
        self.size = self.header.format.N
        self.bits = [0] * self.size
        self.cycle = self.offered = self.committed = self.bit_count = self.since_pilot = 0
        self.mode, self.result, self.eof, self.last_take = "idle", None, False, None
        self.stats = dict.fromkeys(
            (
                "begun",
                "completed",
                "aborted",
                "input_stalls",
                "output_stalls",
                "word_accepts",
                "word_commits",
                "data_words",
                "pilot_words",
                "bit_writes",
                "header_issues",
            ),
            0,
        )

    def _count(self, key, amount=1):
        self.stats[key] = unsigned(self.stats[key] + amount, 64)

    def _emit(self, event, **fields):
        if self.observer:
            self.observer(dict(cycle=self.cycle, event=event, **fields))

    def _hard_event(self, event):
        self._emit("hard", local=event)

    def _header_event(self, event):
        self._emit("header", local=event)

    @property
    def input_ready(self):
        return self.mode == "idle" and (self.last_take is None or self.cycle > self.last_take)

    @property
    def word_input_ready(self):
        return (
            self.mode == "collect"
            and not self.eof
            and self.hard.input_ready
            and self.offered < self.end - self.start
        )

    @property
    def ready(self):
        return self.mode == "ready"

    def begin(self):
        if self.mode == "aborted":
            raise RuntimeError("Reset aborted head bridge before reuse")
        if not self.input_ready:
            self._count("input_stalls")
            return False
        self._count("begun")
        self.offered = self.committed = self.bit_count = self.since_pilot = 0
        self.mode, self.result, self.eof = "collect", None, False
        self._emit("begin", first_index=self.start, end_index=self.end)
        return True

    def offer(self, word, symbol_index):
        _validate(word, self.modulation, symbol_index)
        if self.eof:
            raise RuntimeError("EOF head input cannot resume before a new context")
        if not self.word_input_ready:
            self._count("input_stalls")
            return False
        if symbol_index != self.start + self.offered:
            raise ValueError("Head wire index is repeated,missing or from another field")
        if not self.hard.issue(word, self.modulation, symbol_index):
            raise RuntimeError("Bridge must reserve the actual hard-decision input credit")
        self.offered = unsigned(self.offered + 1, 9)
        self._count("word_accepts")
        return True

    def _complete(self, row):
        self._count("completed")
        self.result, self.mode = row, "ready"
        self._emit("head_result", status=row["status"], register_edge=self.cycle + 1)

    def step(self):
        unsigned(self.cycle + 1, 64)
        if self.hard.cycle != self.cycle or self.header.cycle != self.cycle:
            raise RuntimeError("All head components must advance on the same registered edges")
        mode = self.mode
        decision = self.hard.take() if mode == "collect" and self.hard.ready else None
        result = self.header.take() if mode == "header" and self.header.ready else None
        # New issues below occur after these advances:never consume a same-edge reply.
        self.hard.step()
        self.header.step()
        if mode == "collect" and decision is not None:
            if decision["index"] != self.start + self.committed:
                raise RuntimeError("Registered hard decision belongs to another input")
            pilot = self.since_pilot == self.interval
            if pilot:
                self.since_pilot = 0
                self._count("pilot_words")
            else:
                if self.bit_count + self.bps > self.size:
                    raise RuntimeError("Compiled head layout would overrun its finite bit buffer")
                self.bits[self.bit_count : self.bit_count + self.bps] = decision["bits"]
                self.bit_count = unsigned(self.bit_count + self.bps, 9)
                self.since_pilot = unsigned(self.since_pilot + 1, 5)
                self._count("data_words")
                self._count("bit_writes", self.bps)
            self.committed = unsigned(self.committed + 1, 9)
            self._count("word_commits")
            self._emit(
                "collect",
                index=decision["index"],
                pilot=pilot,
                bit_count=self.bit_count,
                register_edge=self.cycle + 1,
            )
            if self.committed == self.end - self.start:
                if self.bit_count != self.size:
                    raise RuntimeError("Incomplete coded head after its compiled wire end")
                self.mode = "start_header"
        elif mode == "start_header":
            if not self.header.issue(self.bits):
                raise RuntimeError("Bridge owns one drained finite header context")
            self._count("header_issues")
            self.mode = "header"
        elif mode == "header" and result is not None:
            self._complete(result)
        elif mode == "ready":
            self._count("output_stalls")
        if self.mode == "collect" and self.eof and self.hard.mode == "idle":
            self._complete(
                dict(
                    status="truncated_header",
                    header_crc=None,
                    data_crc=None,
                    payload=None,
                    length_ok=None,
                )
            )
        self.cycle += 1

    def end_input(self):
        if self.mode not in ("collect", "start_header", "header"):
            raise RuntimeError("EOF requires an owned unfinished head context")
        self.eof = True

    def take(self):
        if not self.ready:
            return None
        row, self.result, self.mode = self.result, None, "idle"
        self.last_take = self.cycle
        return row

    def abort(self):
        if self.mode not in ("idle", "aborted"):
            self._count("aborted")
            self.hard.abort()
            self.header.abort()
            self.mode, self.result = "aborted", None

    def reset(self):
        if self.mode not in ("idle", "aborted"):
            raise RuntimeError("Drain or abort before resetting head bridge")
        self.hard.reset()
        self.header.reset()
        self.mode, self.eof = "idle", False

    def snapshot(self):
        return deepcopy(
            dict(
                mode=self.mode,
                cycle=self.cycle,
                offered=self.offered,
                committed=self.committed,
                bit_count=self.bit_count,
                statistics=self.stats,
                hard_statistics=self.hard.stats,
                header_statistics=self.header.stats,
            )
        )

    def cost(self):
        header, hard = self.header.cost(), self.hard.cost()
        own = dict(
            coded_bit_buffer=self.size,
            offered_committed_bit_indices=3 * 9,
            pilot_counter=5,
            control_and_EOF=4,
            compiled_field_context=35,
            statistics=11 * 64,
            cycle=64,
            last_take=64,
            last_take_valid=1,
        )
        extra = sum(own.values()) + hard["logical_total_bits"]
        return dict(
            header=header,
            hard=hard,
            extra_state_bits=own,
            additional_logic_bits=extra,
            logical_total_bits=header["logical_total_bits"] + extra,
            output_receipt_storage="alias child receipt;begin blocked until take",
            input_FIFO=0,
            input_bus_bits=55 + 17,
            input_index_compare_bits=17,
            packed_bit_write_bits_per_edge=self.bps,
            packed_bit_write_implementation="FF bit/pair enable decoder;not free RAM ports",
            packed_bit_read_bus_bits=self.size,
            raw_sample_history=0,
            DSP=header["DSP"],
            RAMB18_equivalents=header["RAMB18_equivalents"],
            BRAM36_sites_upper=header["BRAM36_sites_upper"],
            source_sync_routing_comparisons=2,
            source_or_payload_buffer_cycles=None,
            payload_cycles=None,
            extra_component_candidate_budget_bits=3072,
            extra_candidate_capacity_met=extra <= 3072,
            no_Rust_word_implementation=True,
            LUT=None,
            FF=None,
            synthesis=False,
            whole_RX16MSps=False,
        )
