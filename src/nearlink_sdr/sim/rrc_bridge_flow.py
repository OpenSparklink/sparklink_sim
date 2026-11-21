"""One-credit registered RRC→scaled-word bridge (§6.2.1.2/6.2.3).

Actual range/RNE words flow through three assumed fabric stages. Raw-energy
alignment and scaled-bank arithmetic remain separate,nonzero-cost work.
"""

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat
from nearlink_sdr.phy.rrc_bridge import _clip, _normalize, _round_components
from nearlink_sdr.phy.sync_fit_word import unsigned
from nearlink_sdr.sim.capture_fine import pack_filter_word, unpack_filter_word

__all__ = ["RRCScaledBridgeService"]


class RRCScaledBridgeService:
    """One held word/tag;normalize→RNE→clip,then one held output credit.

    No same-edge take/issue fallthrough. Counters are independent64-bit state;
    their updates and wide fabric stage timing are hypotheses,not synthesis.
    """

    def __init__(self, fmt=None, *, start_cycle=0):
        self.fmt = ScaledBankFormat() if fmt is None else fmt
        if not isinstance(self.fmt, ScaledBankFormat):
            raise ValueError("Need existing scaled-bank format")
        if type(start_cycle) is not int or not 0 <= start_cycle < 2**63 - 3:
            raise ValueError("Need signed64 service clock with stage headroom")
        self.cycle, self.last_take_cycle, self.mode = start_cycle, None, "idle"
        self._clear()
        self.stats = dict.fromkeys(
            (
                "accepted",
                "completed",
                "retired",
                "held_edges",
                "aborted",
                "exponent_underflows",
                "exponent_overflows",
                "input_saturations",
                "components_erased",
            ),
            0,
        )

    def _clear(self):
        self.real = self.imag = self.exponent = self.tag = self.native = self.shift = 0
        self.nonzero = (False, False)
        self.saturations = self.erased = self.underflow = self.overflow = 0

    def _updates(self, changes):
        updates = {key: unsigned(self.stats[key] + value, 64) for key, value in changes.items()}
        self.stats.update(updates)  # Only after every updated counter has been checked.

    @property
    def input_ready(self):
        return self.mode == "idle" and (
            self.last_take_cycle is None or self.cycle > self.last_take_cycle
        )

    @property
    def ready(self):
        return self.mode == "ready"

    @property
    def result(self):
        if not self.ready:
            return None
        # A view of the same word/tag registers,no second stored result or FIFO.
        return dict(
            tag=self.tag,
            word=(self.real, self.imag, self.exponent),
            events=dict(
                exponent_underflows=self.underflow,
                exponent_overflows=self.overflow,
                input_saturations=self.saturations,
                components_erased=self.erased,
            ),
        )

    def issue(self, word, tag):
        if not self.input_ready:
            return False
        r, i, exponent = unpack_filter_word(word)
        if type(tag) is not int or not 0 <= tag < 2**63 or not -96 <= exponent <= 48:
            raise ValueError("Need matching signed64 sample tag and RRC exponent[-96,48]")
        self._updates(dict(accepted=1))
        self.real, self.imag, self.exponent, self.tag = r, i, exponent, tag
        self.nonzero = (r != 0, i != 0)
        self.mode = "normalize"
        return True

    def step(self):
        if self.mode == "aborted":
            raise RuntimeError("Aborted bridge cannot return a stale word")
        edge = unsigned(self.cycle + 1, 63)
        if self.mode == "normalize":
            _, _, self.exponent, self.shift, self.native = _normalize(
                pack_filter_word(self.real, self.imag, self.exponent), self.fmt
            )
            if not -16 <= self.shift <= 48 or not -128 <= self.native < 128:
                raise RuntimeError("RRC bridge exceeded its proven stage bounds")
            self.mode = "round"
        elif self.mode == "round":
            self.real, self.imag, self.saturations = _round_components(
                self.real, self.imag, self.shift, self.fmt.mantissa_bits
            )
            self.mode = "clip"
        elif self.mode == "clip":
            word, events = _clip(
                self.real,
                self.imag,
                self.exponent,
                self.native,
                self.saturations,
                self.nonzero,
                self.fmt.mantissa_bits,
            )
            self._updates(dict(completed=1, **events))
            self.real, self.imag, self.exponent = word
            self.underflow = events["exponent_underflows"]
            self.overflow = events["exponent_overflows"]
            self.saturations, self.erased = events["input_saturations"], events["components_erased"]
            self.mode = "ready"
        elif self.ready:
            self._updates(dict(held_edges=1))
        self.cycle = edge

    def take(self):
        if not self.ready:
            return None
        self._updates(dict(retired=1))
        result = self.result
        self._clear()
        self.mode, self.last_take_cycle = "idle", self.cycle
        return result

    def abort(self):
        if self.mode == "aborted":
            return
        if self.mode != "idle":
            self._updates(dict(aborted=1))
        self._clear()
        self.mode = "aborted"

    def estimate(self):
        fields = dict(
            real=19,
            imag=19,
            exponent=8,
            sample_tag=64,
            native_exponent=9,
            shift=7,
            nonzero_flags=2,
            saturation_count=2,
            erased_count=2,
            range_flags=2,
            mode=3,
            clock=64,
            last_take_clock=64,
            last_take_valid=1,
        )
        return dict(
            compiled_mantissa_bits=self.fmt.mantissa_bits,
            registered_stages=3,
            initiation_interval=4,
            active_contexts=1,
            output_credits=1,
            register_fields=fields,
            counter_bits=9 * 64,
            register_bits_upper=sum(fields.values()) + 9 * 64,
            input_word_bits=46,
            output_word_bits=2 * self.fmt.mantissa_bits + 7,
            sample_tag_bits=64,
            new_sample_RAM=0,
            DSP=0,
            coefficient_ROM=0,
            operations=dict(
                abs19=4,
                max_compare19=1,
                LZC19=1,
                add_sub9=3,
                range_compare9=2,
                RNE_shifts19=2,
                left_bound_shifts=2,
                left_bound_compare19=2,
                clip_compare19=4,
                zero_detect19=4,
                RNE_guard_sticky_increment19=2,
                sign_restore19=2,
                flag_OR2=1,
                flag_sum2=2,
                divide=0,
                sqrt=0,
                multiply=0,
            ),
            independent_counter_adders64_upper=6,
            mapping="fabric shifts/adds/comparators;one register set reused in-place",
            timing="one assumed registered stage each;counter updates also need timing proof",
            LUT_FF_or_Fmax=None,
            synthesis=False,
            full_receiver16MSps=False,
            raw_energy_alignment_cost=None,
            scaled_bank_cycle_cost=None,
            old_fixed22_18DSP_cost_applicable=False,
        )
