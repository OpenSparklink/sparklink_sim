"""Value-carrying serial normalization baseline; four dedicated DSP25x18 and one ROM.

Gain preparation is a30-step restoring divider. Every ROM/DSP return and
registered NCO commit consumes real edges. No whole-RX throughput claim.
"""

from copy import deepcopy

from nearlink_sdr.phy.interpolation_word import rne_shift, unpack_interpolated_word
from nearlink_sdr.phy.normalization_word import (
    MAX_SYMBOLS,
    WordNormalizationState,
    _integer,
    normalize_complex,
    pack_normalized_word,
)
from nearlink_sdr.phy.sync_coherent import _COSINE_QUARTER
from nearlink_sdr.phy.sync_fit_word import INPUT_FORMAT, signed, unsigned

__all__ = ["SymbolNormalizationService"]


class SymbolNormalizationService:
    """One symbol credit; startup overlaps ROM work, not other symbols or DSP banks."""

    def __init__(self, state):
        if not isinstance(state, WordNormalizationState):
            raise ValueError("Need one untouched finite normalization state")
        self.state = state.claim()
        self.cycle = 0
        self.first_cycle = self.ready_cycle = None
        self.last_issue_cycle = self.last_take_cycle = None
        self.closed = None
        self.job = self.registered = None
        self.setup = dict(stage="normalize", bit=0)
        self.preparation_ready_cycle = None
        self.stats = dict.fromkeys(
            ["accepted", "completed", "native_products", "rom_reads",
             "divider_steps", "output_stall_edges", "aborted"], 0
        )

    def _count(self, name, amount=1):
        self.stats[name] = unsigned(self.stats[name] + amount, 64)

    @property
    def mode(self):
        if self.closed:
            return self.closed
        if self.registered is not None:
            return "ready"
        if self.job is not None:
            return self.job["stage"]
        return "idle" if self.state.inverse is not None else "prepare_gain"

    @property
    def can_issue(self):
        return (not self.closed and self.job is None and self.state.count < MAX_SYMBOLS
                and self.last_issue_cycle != self.cycle and self.last_take_cycle != self.cycle)

    def issue(self, word):
        if self.closed:
            raise RuntimeError("Closed normalization packet cannot resume")
        word = unsigned(_integer(word), 52)
        r, i, e = unpack_interpolated_word(word, INPUT_FORMAT)
        if not self.can_issue:
            return False
        accepted = unsigned(self.stats["accepted"] + 1, 64)
        self.job = dict(word=word, real=r, imag=i, exponent=e,
                        index=self.state.count, phase=self.state.phase,
                        issued=self.cycle, stage="phase")
        self.stats["accepted"] = accepted
        self.last_issue_cycle = self.cycle
        return True

    def _prepare_gain(self, next_cycle):
        s = self.setup
        if s["stage"] == "normalize":
            m, e = self.state.gain
            shift = 54 - m.bit_length()
            s.update(divisor=unsigned(m << shift, 54), exponent=signed(e - shift, 11),
                     stage="divide", remainder=1 << 53, quotient=0)
            if s["remainder"] == s["divisor"]:
                s["quotient"], s["remainder"] = 1, 0
        elif s["stage"] == "divide":
            shifted = unsigned(s["remainder"] << 1, 55)
            bit = int(shifted >= s["divisor"])
            remainder = shifted - (s["divisor"] if bit else 0)
            quotient = unsigned((s["quotient"] << 1) | bit, 31)
            self._count("divider_steps")
            s.update(remainder=unsigned(remainder, 54), quotient=quotient, bit=s["bit"] + 1)
            if s["bit"] == 30:
                s["stage"] = "round"
        elif s["stage"] == "round":
            twice = unsigned(2 * s["remainder"], 55)
            increment = twice > s["divisor"] or (
                twice == s["divisor"] and s["quotient"] & 1
            )
            s["quotient"] = unsigned(s["quotient"] + int(increment), 31)
            s["stage"] = "register_inverse"
        elif s["stage"] == "register_inverse":
            q, e = s["quotient"], -83 - s["exponent"]
            if q == 1 << 30:
                q >>= 1
                e += 1
            self.state._accept_inverse((unsigned(q, 30), signed(e, 11)))
            self.preparation_ready_cycle = next_cycle
            s["stage"] = "done"

    @staticmethod
    def _rom_descriptor(address):
        quadrant, offset = divmod(address, 1024)
        return (offset, 0 if offset == 0 else 1024 - offset, quadrant, offset == 0)

    @staticmethod
    def _sign(words, descriptor):
        c, s = words
        q, zero = descriptor[2:]
        s = 0 if zero else s
        return [(c, -s), (-s, -c), (-c, s), (s, c)][q]

    def _issue_products(self, pairs, following):
        operands = tuple((signed(a, 25), signed(b, 18)) for a, b in pairs)
        if not 1 <= len(operands) <= 4:
            raise RuntimeError("One DSP issue may occupy at most four lanes")
        due = unsigned(self.cycle + 3, 64)
        self._count("native_products", len(operands))
        self.job.update(operands=operands, due=due, following=following, stage="DSP_wait")

    def _symbol(self, inverse, next_cycle):
        j = self.job
        stage = j["stage"]
        if stage == "phase":
            if self.state.fmt.mode == "nearest":
                j.update(address=rne_shift(j["phase"], 28) & 4095, fraction=0)
            else:
                phase28 = rne_shift(j["phase"], 12) & ((1 << 28) - 1)
                j["address"], j["fraction"] = divmod(phase28, 1 << 16)
            j["stage"] = "addresses"
        elif stage == "addresses":
            j["first"] = self._rom_descriptor(j["address"])
            j["second"] = self._rom_descriptor((j["address"] + 1) & 4095)
            j["stage"] = "ROM_first_request"
        elif stage == "ROM_first_request":
            j["rom_pending"] = j["first"][:2]
            self._count("rom_reads", 2)
            j["stage"] = "ROM_first_return"
        elif stage == "ROM_first_return":
            j["first_words"] = tuple(_COSINE_QUARTER[a] for a in j.pop("rom_pending"))
            if self.state.fmt.mode == "linear":
                self._count("rom_reads", 2)
                j["rom_pending"] = j["second"][:2]
                j["stage"] = "ROM_second_return"
            else:
                j["stage"] = "sign"
        elif stage == "ROM_second_return":
            j["second_words"] = tuple(_COSINE_QUARTER[a] for a in j.pop("rom_pending"))
            j["stage"] = "sign"
        elif stage == "sign":
            j["coefficients"] = self._sign(j.pop("first_words"), j["first"])
            if self.state.fmt.mode == "linear":
                j["neighbor"] = self._sign(j.pop("second_words"), j["second"])
                j["stage"] = "interpolate_issue"
            else:
                j["stage"] = "gain_issue"
        elif stage == "interpolate_issue":
            pairs = [(signed(b - a, 9), j["fraction"])
                     for a, b in zip(j["coefficients"], j["neighbor"], strict=True)]
            self._issue_products(pairs, "interpolate_round")
        elif stage == "DSP_wait":
            if self.cycle >= j["due"]:
                j["products"] = [signed(a * b, 43) for a, b in j.pop("operands")]
                j["stage"] = j.pop("following")
        elif stage == "interpolate_round":
            j["coefficients"] = tuple(
                signed(rne_shift(signed((c << 16) + p, 35), 16), 18)
                for c, p in zip(j["coefficients"], j.pop("products"), strict=True)
            )
            j["stage"] = "gain_issue"
        elif stage == "gain_issue":
            if inverse is not None:
                lo, hi = inverse[0] & ((1 << 17) - 1), inverse[0] >> 17
                self._issue_products([(c, part) for c in j["coefficients"]
                                      for part in (lo, hi)], "gain_combine")
        elif stage == "gain_combine":
            p = j.pop("products")
            j["fused"] = (signed(p[0] + (p[1] << 17), 48),
                          signed(p[2] + (p[3] << 17), 48))
            j["stage"] = "gain_normalize"
        elif stage == "gain_normalize":
            j["c"], j["s"], j["ce"] = normalize_complex(*j.pop("fused"), inverse[1] - 16, 18)
            j["stage"] = "rotate_issue"
        elif stage == "rotate_issue":
            self._issue_products([(j["real"], j["c"]), (j["imag"], j["s"]),
                                  (j["real"], j["s"]), (j["imag"], j["c"])], "rotate_sum")
        elif stage == "rotate_sum":
            p = j.pop("products")
            j["sum"] = signed(p[0] - p[1], 41), signed(p[2] + p[3], 41)
            j["stage"] = "output_normalize"
        elif stage == "output_normalize":
            r, i, e = normalize_complex(*j["sum"], j["exponent"] + j["ce"], 22)
            j["output"] = pack_normalized_word(r, i, e)
            j["erased"] = j["sum"][0] != 0 and r == 0, j["sum"][1] != 0 and i == 0
            j["stage"] = "register_commit"
        elif stage == "register_commit":
            if j["phase"] != self.state.phase or j["index"] != self.state.count:
                raise RuntimeError("Normalization owner changed while the symbol was in flight")
            self.registered = self.state.prepare_result(j["word"], j["output"], j["erased"])
            self.ready_cycle = next_cycle
            j["stage"] = "ready"
        elif stage == "ready":
            self._count("output_stall_edges")
        else:
            raise RuntimeError("Unknown finite normalization operation")

    def step(self):
        next_cycle = unsigned(self.cycle + 1, 64)
        if not self.closed:
            inverse = self.state.inverse  # No same-edge use of a just registered reciprocal.
            self._prepare_gain(next_cycle)
            if self.job is not None:
                self._symbol(inverse, next_cycle)
        self.cycle = next_cycle

    def peek(self):
        return deepcopy(self.registered)

    def take(self):
        if self.registered is None:
            return None
        completed = unsigned(self.stats["completed"] + 1, 64)
        result = deepcopy(self.registered)
        self.state.accept_registered(self.registered)
        self.first_cycle = self.job["issued"]
        self.stats["completed"] = completed
        self.job = self.registered = None
        self.last_take_cycle = self.cycle
        return result

    def release(self):
        if self.closed or self.mode != "idle":
            raise RuntimeError("Only a drained prepared packet may release normalization")
        self.closed = "released"

    def abort(self):
        if not self.closed:
            self._count("aborted")
            self.job = self.registered = None
            self.closed = "aborted"

    @staticmethod
    def estimate(mode="linear"):
        if mode not in ("linear", "nearest"):
            raise ValueError("Need a compiled phase structure")
        parts = dict(
            packet_owner=40 + 42 + 64 + 41 + 18 + 52 + 55 + 7 * 64 + 3,
            raw_symbol=52 + 44 + 8 + 18 + 40,
            ROM_addresses_flags_and_values=24 + 40 + 6 + 68 + 72 + 16,
            DSP_external_operands_and_returns=4 * (25 + 18) + 4 * 43,
            coefficient_work=96 + 36 + 11,
            output_work=82 + 55 + 2,
            registered_commit=18 + 40 + 52 + 55 + 40 + 18 + 7 * 64,
            restoring_divider=54 + 55 + 31 + 11 + 6,
            service_statistics=7 * 64,
            clocks_and_held_tags=6 * 64,
            control=32,
        )
        return dict(
            state_register_bits_upper=sum(parts.values()), state_parts=parts,
            additional_DSP25x18=4, DSP_internal_pipeline_bits=4 * (25 + 18 + 43 + 48),
            DSP_latency_edges=3, native_products_per_symbol=10 if mode == "linear" else 8,
            coefficient_ROM_bits=1024 * 17, coefficient_ROM_RAMB18_estimate=1,
            coefficient_ROM_BRAM36_sites_upper=1, coefficient_ROM_read_ports=2,
            ROM_reads_per_symbol=4 if mode == "linear" else 2,
            new_original_source_RAM_bits=0, new_source_or_cache_read_ports=0,
            gain_restoring_steps_once_per_packet=30, symbol_credits=1,
            gain_preparation_registered_edge=33, first_result_when_issued_at_reset_edge=46,
            prepared_symbol_latency_edges=24 if mode == "linear" else 18,
            input_handoff_edges=1,
            scalar_formats="phase40/frequency42;inverse30+11;coeff18+11;output22+22+11",
            division_per_symbol=0, sqrt_per_symbol=0, runtime_trigonometry=0,
            rounding="RNE with shared exponent and explicit carry correction;no saturation",
            overflow="reject width/counter exhaustion;phase modulo2^40 is intentional",
            native_products="30x18 split into17+13;22x18 fits one25x18;"
                            "two phase interpolation products",
            wide_macros="LZC/RNE/variable shift48/41,35bit fixed-RNE add,55bit compare/sub,"
                        "40bit phase and18bit count plus64bit counters assume one edge",
            nonlinear_and_ALU_mapping="explicit operation graph,not LUT/FF synthesis",
            sharing_with_other_RX_TX_decoder_or_MAC_proven=False,
            interpolator_timing_and_head_decoder_cycles_included=False,
            complete_receiver16MSps=False, synthesis_evidence=False,
        )
