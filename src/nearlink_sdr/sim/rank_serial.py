"""Single-credit exact Q12 normalization alternatives (§6.2.3,experimental).

No division,square root or floating arithmetic. Both variants pay a registered
prepare edge and fixed digit count,including zero/unity cases. The existing
score producer owns N,D and the immutable C/E ticket;no duplicate square/tree
is evaluated here. This is not a complete receiver or synthesis result.
"""

from collections.abc import Mapping
from types import MappingProxyType

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat
from nearlink_sdr.sim.scaled_native import FoldResult, _count, _unsigned

__all__ = ["Q12Serial"]


class Q12Serial:
    """One registered context,one or two quotient bits per fixed iteration.

    The sink must consume the output when ``will_emit`` is true,as with the
    baseline Q12Pipeline. Input backpressure holds the score producer's credit.
    Out-of-width values and broken producer contracts raise before acceptance;
    they never wrap or silently saturate. Only the Q12 unity clamp is intended.
    """

    def __init__(self, widths, terms, coefficient_energy, *, digit_bits=2):
        if (
            type(terms) is not int
            or terms not in (31, 61, 125)
            or type(coefficient_energy) is not int
            or coefficient_energy not in (terms, 2 * terms)
            or type(digit_bits) is not int
            or digit_bits not in (1, 2)
            or not isinstance(widths, Mapping)
            or dict(widths) != FixedBankFormat(10).widths(terms, coefficient_energy, 0.6)
        ):
            raise ValueError("Need compiled scaled10 target widths and one/two-bit digits")
        self.w = MappingProxyType(dict(widths))
        self.terms, self.coefficient_energy = terms, coefficient_energy
        self.digit_bits, self.iterations = digit_bits, 12 // digit_bits
        self.d = self.w["window_energy"] + (coefficient_energy - 1).bit_length()
        self.cell, self.phase, self.digits = None, 0, 0
        self.stats = dict(accepted=0, completed=0, subtracts=0)

    def offer(self, source, numerator, denominator, hit):
        """Register one owned ticket;False means no input was consumed."""
        if self.cell is not None:
            return False
        if not isinstance(source, FoldResult) or type(hit) is not bool:
            raise ValueError("Need an immutable C/E ticket and Boolean gate")
        c, e = self.w["correlation"], self.w["window_energy"]
        _unsigned(source.tag, 63)
        _unsigned(source.energy, e)
        _unsigned(source.discarded, 7)
        for value in (source.real, source.imag):
            if type(value) is not int or not -(1 << (c - 1)) <= value < 1 << (c - 1):
                raise OverflowError("Signed C/E ticket exceeds its compiled port")
        _unsigned(numerator, self.w["numerator"])
        _unsigned(denominator, self.d)
        if bool(source.energy) != bool(denominator):
            raise ValueError("Producer denominator must be zero exactly when energy is zero")
        _count(self.stats, "accepted")
        self.cell = dict(
            source=source,
            numerator=numerator,
            denominator=denominator,
            hit=hit,
            mode=0,
            remainder=0,
            rank=0,
            triple=0,
        )
        self.phase, self.digits = 1, 0
        return True

    def step(self):
        """Advance one registered edge;return the old completed context once."""
        if self.cell is None:
            if self.phase != 0:
                raise RuntimeError("Idle serial rank lost its context ownership")
            return None
        cell = self.cell
        if self.phase == 1:
            n, d = cell["numerator"], cell["denominator"]
            mode = 1 if d == 0 else 2 if n >= d else 0
            triple = _unsigned(d + (d << 1), self.d + 2) if self.digit_bits == 2 else 0
            cell.update(
                mode=mode,
                triple=triple,
                remainder=n if mode == 0 else 0,
                rank=4096 if mode == 2 else 0,
            )
            self.phase = 2
        elif self.phase == 2:
            if cell["mode"] == 0:
                expanded = _unsigned(cell["remainder"] << self.digit_bits, self.d + self.digit_bits)
                d = cell["denominator"]
                if self.digit_bits == 1:
                    digit = int(expanded >= d)
                    subtract = d if digit else 0
                else:
                    # Three parallel comparisons and a fixed4-way selection.
                    ge1, ge2, ge3 = expanded >= d, expanded >= d << 1, expanded >= cell["triple"]
                    digit = 3 if ge3 else 2 if ge2 else 1 if ge1 else 0
                    subtract = (0, d, d << 1, cell["triple"])[digit]
                remainder = _unsigned(expanded - subtract, self.d)
                if remainder >= d:
                    raise OverflowError("Restoring remainder exceeded its proven bound")
                rank = _unsigned((cell["rank"] << self.digit_bits) | digit, 13)
                if digit:
                    _count(self.stats, "subtracts")
                cell.update(remainder=remainder, rank=rank)
            self.digits += 1
            if self.digits == self.iterations:
                self.phase = 3
        elif self.phase == 3:
            _count(self.stats, "completed")
            result = {
                key: cell[key]
                for key in (
                    "source",
                    "numerator",
                    "denominator",
                    "hit",
                    "mode",
                    "remainder",
                    "rank",
                )
            }
            self.cell, self.phase, self.digits = None, 0, 0
            return result
        else:
            raise RuntimeError("Invalid serial rank control state")
        return None

    @property
    def will_emit(self):
        return self.cell is not None and self.phase == 3

    @property
    def idle(self):
        return self.cell is None and self.phase == 0

    def estimate(self):
        c, e, n, d = (self.w["correlation"], self.w["window_energy"], self.w["numerator"], self.d)
        fields = dict(
            source_ticket=63 + 2 * c + e + 7,
            numerator=n,
            denominator=d,
            triple=d + 2 if self.digit_bits == 2 else 0,
            remainder=d,
            rank=13,
            mode=2,
            digit_counter=self.iterations.bit_length(),
            phase=2,
            hit=1,
        )
        context = sum(fields.values())
        total = context + 1 + 3 * 64
        return dict(
            DSP=0,
            register_fields=fields,
            registered_stages=1,
            input_register_bits=context,
            pipeline_register_bits=0,
            valid_bits=1,
            statistics_bits=3 * 64,
            total_register_bits=total,
            research_register_cap_bits=640,
            research_register_cap_pass=total <= 640,
            contexts=1,
            input_queue_added=0,
            new_RAM=0,
            digit_bits=self.digit_bits,
            latency_edges=self.iterations + 2,
            initiation_interval=self.iterations + 2,
            prepare_edges=1,
            fractional_iterations=self.iterations,
            output_handoff_edges=1,
            prepare_adder_width=d + 2 if self.digit_bits == 2 else 0,
            prepare_adder_units=1 if self.digit_bits == 2 else 0,
            digit_compare_units=3 if self.digit_bits == 2 else 1,
            digit_compare_width=d + self.digit_bits,
            digit_subtract_units=1,
            digit_subtract_width=d + self.digit_bits,
            digit_select_MUX2_upper=(3 if self.digit_bits == 2 else 1) * (d + self.digit_bits),
            remainder_range_assertion_width=d,
            initial_zero_unity_compare_width=max(n, d),
            worst_operations_per_word=dict(
                zero_compare=1,
                unity_compare=1,
                prepare_add=1 if self.digit_bits == 2 else 0,
                digit_compares=self.iterations * (3 if self.digit_bits == 2 else 1),
                digit_subtracts=self.iterations,
                variable_multiply=0,
                divide=0,
                sqrt=0,
            ),
            context_ports=(
                "FF context:one input write/retire read;one remainder/rank update per digit"
            ),
            statistics_adders64_upper=2,
            divider_or_sqrt_units=0,
            variable_multiplier_units=0,
            denominator_times2="wiring",
            denominator_times3="one registered prepare add" if self.digit_bits == 2 else "absent",
            rounding="exact floorQ12 and unity clamp;no extra rounding or saturation",
            statistics_subtracts="one per nonzero digit,not bit-popcount",
            input_validation="typed-port and producer-invariant simulation assertions",
            invalid_producer_fault_controller_cycles=None,
            full_receiver16MSps=False,
            actual_synthesis=False,
            LUT_FF_Fmax=None,
        )
