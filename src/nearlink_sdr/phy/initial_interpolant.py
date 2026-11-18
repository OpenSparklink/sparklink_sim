"""Finite reuse of the identical local sync interpolant (§6.2.1.2 implementation).

Only32/62/126 known-sync symbols at SPS4 are compiled. This cache owns actual
interpolant words,not the global-spline sync vector or raw sample references.
"""

from nearlink_sdr.phy.interpolation_word import LagrangeWordFormat

__all__ = ["InitialInterpolantCache"]


class InitialInterpolantCache:
    """One fill then one ordered replay;no alternate answer or associative search.

    Logical methods have no clock claim. The separate registered port model
    accounts for read response storage and timing. Release invalidates ownership
    without a128-cycle RAM clear;stale bytes can never become a valid reply.
    """

    def __init__(self, fmt, symbols):
        if not isinstance(fmt, LagrangeWordFormat):
            raise ValueError("Need explicit compiled interpolant format")
        if type(symbols) is not int or symbols not in (32, 62, 126):
            raise ValueError("Need32/62/126 sync symbols at SPS4")
        self.fmt, self.symbols = fmt, symbols
        self.words = [0] * 128
        self.first_cell = self.phase = None
        self.written = self.read = 0
        self.state = "empty"

    def _key(self, cell, phase, index):
        if type(cell) is not int or not 0 <= cell < 2**63:
            raise ValueError("Need nonnegative signed64 stencil center")
        if type(phase) is not int or not 0 <= phase <= 2**self.fmt.phase_bits:
            raise ValueError("Need compiled fractional phase word")
        if self.first_cell is not None and (
            cell != self.first_cell + 4 * index or phase != self.phase
        ):
            raise RuntimeError("Initial interpolant identity/order changed;no fallback")

    def store(self, cell, phase, word):
        if self.state not in ("empty", "writing"):
            raise RuntimeError("Initial cache is sealed or released")
        if type(word) is not int or not 0 <= word < 2 ** (2 * self.fmt.output_bits + 8):
            raise ValueError("Need actual compiled interpolant word")
        self._key(cell, phase, self.written)
        if self.first_cell is None:
            if cell + 4 * (self.symbols - 1) >= 2**63:
                raise OverflowError("Initial cache index progression exceeds signed64")
            self.first_cell, self.phase = cell, phase
        self.words[self.written] = word
        self.written += 1
        self.state = "sealed" if self.written == self.symbols else "writing"

    def reuse(self, cell, phase):
        if self.state not in ("sealed", "reading"):
            raise RuntimeError("Initial cache is incomplete or released")
        self._key(cell, phase, self.read)
        word = self.words[self.read]
        self.read += 1
        self.state = "released" if self.read == self.symbols else "reading"
        return word

    def release(self):
        self.state = "released"

    def diagnostics(self):
        return dict(
            state=self.state,
            written=self.written,
            reused=self.read,
            valid_words=0 if self.state == "released" else self.written,
        )

    def estimate(self):
        bits = 2 * self.fmt.output_bits + 8
        # cell64,phase incl1,length7,write7,read7,state3,key-valid1.
        control = 64 + self.fmt.phase_bits + 1 + 7 * 3 + 3 + 1
        return dict(
            capacity_words=128,
            word_bits=bits,
            payload_bits=128 * bits,
            controller_bits=control,
            bram36=1,
            BRAM_mapping="one512x72 SDP;serialized fill/read phases;unused space not shared",
            alternative_LUT6=2 * bits,
            alternative_MUXF7=bits,
            alternative_mapping="128x1 single-port perbit;serialized phases,not extra replica",
            writes_per_clock=1,
            reads_per_clock=1,
            simultaneous_fill_and_replay=False,
            key_add64_paths=1,
            key_compare64_paths=1,
            phase_compare_bits=self.fmt.phase_bits + 1,
            counter_increment7_paths=1,
            shared_key_ALU_basis="fill and replay states cannot overlap",
            controller_and_address_LUTs=None,
            extra_DSP=0,
            associative_search=False,
            physical_registered_response_and_clock_state_included=False,
            Python_object_overhead_included=False,
            synthesized=False,
            full_receiver16msps=False,
        )
