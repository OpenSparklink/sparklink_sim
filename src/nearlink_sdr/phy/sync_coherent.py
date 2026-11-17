"""Known-sync CFO-compensated confirmation (§6.2.1.2/6.2.3.2).

Timing/CFO comes from the already selected differential maximum. Confirmation
cannot choose another peak, frequency or phase using CRC or expected data.
Only the optional NCO coefficient node has finite word arithmetic here.
"""

import math
from collections import deque

import numpy as np

__all__ = ["SyncConfirmationNCO", "SyncConfirmationNCOFlow", "coherent_sync_confirmation"]

_PHASE_BITS = 20
_PHASE_MASK = (1 << _PHASE_BITS)-1
_COEFFICIENT_SCALE = 1 << 16
_COSINE_QUARTER = tuple(round(math.cos(math.pi*k/2048)*_COEFFICIENT_SCALE)
                       for k in range(1024))


class SyncConfirmationNCO:
    """Bounded20-bit modulo phase, RNE12-bit address, signed18/Q16 coefficients.

    The integer entry consumes an already converted phase-increment word. The
    floating atan2/frequency-to-word bridge is deliberately outside this node.
    Each candidate starts at zero; at most128 outputs, with caller-owned arrays.
    """

    def __init__(self, step_word):
        if type(step_word) is not int or not 0 <= step_word <= _PHASE_MASK:
            raise ValueError("Need unsigned20-bit phase increment")
        self.step = step_word
        self.phase = self.produced = 0

    def advance(self, count):
        if type(count) is not int or count < 1 or self.produced+count > 128:
            raise ValueError("Need1..128 total NCO outputs per candidate")
        real, imag = np.empty(count, np.int32), np.empty(count, np.int32)
        for k in range(count):
            address, remainder = divmod(self.phase, 256)
            address += int(remainder > 128 or (remainder == 128 and address & 1))
            quadrant, offset = divmod(address % 4096, 1024)
            cosine = _COSINE_QUARTER[offset]
            sine = 0 if offset == 0 else _COSINE_QUARTER[1024-offset]
            c, s = [(cosine, sine), (-sine, cosine), (-cosine, -sine),
                    (sine, -cosine)][quadrant]
            real[k], imag[k] = c, -s  # CFO compensation is exp(-j*omega*k).
            self.phase = (self.phase+self.step) & _PHASE_MASK
        self.produced += count
        return real, imag

    @staticmethod
    def estimate():
        """One explicit ROM/pipeline allocation; not synthesized or full sync."""
        components = dict(quarter_cosine_rom=1024*17, phase=20, increment=20, count=8,
                          four_pipeline_tags=4*8, rounded_address=12,
                          two_rom_addresses=2*10, sign_zero_pipeline_flags=8,
                          two_rom_words=2*17, two_output_words=2*18,
                          pipeline_valid=4, target_count=8)
        return dict(
            logical_bits=sum(components.values()), components=components,
            coefficient_rom="1024x17 unsignedQ16; endpoints bypassed; oneRAMB18 TDP2read estimate",
            rom_read_ports=2, rom_writes_in_service=0, phase_bits=20,
            address_bits=12, coefficient_signed_bits=18, coefficient_fractional_bits=16,
            rounding="nearest ties even at phase address and constant coefficient generation",
            phase_overflow="unsigned modulo2^20, intentional; reject excess output count",
            multipliers=0, divides=0, sqrt=0, runtime_trig=0,
            arithmetic="21-bit phase add;21-bit address RNE carry;11-bit reflected address;18-bit sign negate",
            ii=1, latency_cycles=4,
            pipeline="round address;reflect/decode;registeredROM;sign/output;one-cycle phase feedback",
            assumptions="separate pipelined add/compare units and synchronous dual-read ROM; no placement/timing evidence",
            excluded="floating atan2/word bridge, input-symbol alignment, coherent sums/normalization, all frontend/decoder/TX concurrency",
            whole_system_accepted=False, synthesis_evidence=False,
        )


class SyncConfirmationNCOFlow:
    """Bounded four-credit ready/valid calendar for one coefficient block.

    This is a token/calendar model, not a synthesized implementation of each
    pipeline stage. A stall stops input-word admission and phase advancement.
    ADC/history/confirmation deadlines and sample alignment are not simulated.
    """

    def run(self, step_word, count, *, stalls=(), max_cycles=65536):
        if (type(count) is not int or not 1 <= count <= 128
                or type(max_cycles) is not int or not 1 <= max_cycles <= 65536
                or not isinstance(stalls, (tuple, list)) or len(stalls) > 64):
            raise ValueError("Need bounded block, cycle limit and at most64 stall intervals")
        previous = -1
        for start, stop in stalls:
            if (type(start) is not int or type(stop) is not int
                    or not 0 <= start < stop or start < previous):
                raise ValueError("Need ordered, disjoint nonnegative stalls")
            previous = stop
        nco = SyncConfirmationNCO(step_word)
        pipe, output = deque(), []
        accepted = blocked = peak = 0
        for cycle in range(max_cycles):
            ready = not any(start <= cycle < stop for start, stop in stalls)
            if ready and pipe and pipe[0]["due"] <= cycle:
                token = pipe.popleft()
                output.append(dict(tag=token["tag"], real=token["real"], imag=token["imag"],
                                   accepted_cycle=token["cycle"], emitted_cycle=cycle))
            if accepted < count and len(pipe) < 4:
                real, imag = nco.advance(1)
                pipe.append(dict(tag=accepted, cycle=cycle, due=cycle+4,
                                 real=int(real[0]), imag=int(imag[0])))
                accepted += 1
            elif accepted < count:
                blocked += 1
            peak = max(peak, len(pipe))
            if len(output) == count:
                break
        return dict(accepted=accepted, emitted=output, input_backpressure_cycles=blocked,
                    peak_pipeline_tokens=peak, pending_tokens=len(pipe),
                    incomplete=len(output) != count, final_phase=nco.phase,
                    max_latency_cycles=max((r["emitted_cycle"]-r["accepted_cycle"]
                                           for r in output), default=None),
                    scope="one bounded NCO block;resolved output arrays are caller-owned diagnostics",
                    whole_system_accepted=False, synthesis_evidence=False)


def coherent_sync_confirmation(symbols, reference, omega, threshold, *, fixed_nco=False):
    """Confirm the chosen sync with Q; gate by squared quantities, without division.

    ``omega`` is the selected differential correlation phase, not an injected
    truth or a fitted payload value. ``quality`` division is diagnostic only.
    The optional fixed coefficient path leaves input/accumulators floating.
    """
    y, ref = np.asarray(symbols, dtype=complex), np.asarray(reference, dtype=complex)
    if (y.ndim != 1 or ref.ndim != 1 or len(y) != len(ref) or not 2 <= len(y) <= 128
            or not np.isfinite(y).all() or not np.isfinite(ref).all()
            or type(fixed_nco) is not bool or isinstance(omega, (bool, np.bool_))
            or not isinstance(omega, (int, float, np.integer, np.floating))
            or not np.isfinite(omega) or abs(omega) > np.pi
            or isinstance(threshold, (bool, np.bool_)) or not np.isscalar(threshold)
            or not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold) or not 0 < threshold <= 1):
        raise ValueError("Need finite2..128 matching sync symbols, aliased CFO and explicit Q threshold")
    if max(np.max(abs(y.real)), np.max(abs(y.imag)),
           np.max(abs(ref.real)), np.max(abs(ref.imag))) > 1e70:
        raise OverflowError("Coherent confirmation exceeds finite fourth-power range")
    nco = None
    if fixed_nco:
        signed_step = round(float(omega)/(2*np.pi)*(1 << _PHASE_BITS))
        oscillator = SyncConfirmationNCO(signed_step & _PHASE_MASK)
        real, imag = oscillator.advance(len(y))
        rotation = (real+1j*imag)/_COEFFICIENT_SCALE
        nco = dict(step_word=oscillator.step, phase_bits=_PHASE_BITS,
                   quantized_cycles_per_symbol=signed_step/(1 << _PHASE_BITS),
                   coefficient_signed_bits=18, coefficient_fractional_bits=16,
                   generated=len(y), final_phase=oscillator.phase,
                   scope="finite NCO coefficients only; floating CFO bridge and coherent accumulators")
    else:
        rotation = np.exp(-1j*float(omega)*np.arange(len(y)))
    correlation = np.sum(y*ref.conj()*rotation)
    numerator = float(correlation.real**2+correlation.imag**2)
    signal_energy = float(np.sum(y.real**2+y.imag**2))
    reference_energy = float(np.sum(ref.real**2+ref.imag**2))
    denominator = signal_energy*reference_energy
    if not np.isfinite([numerator, denominator]).all():
        raise OverflowError("Coherent normalization overflow")
    quality = numerator/denominator if denominator > 0 else 0.
    return dict(statistic="known-sync-CFO-compensated-coherent-Q", quality=quality,
                numerator=numerator, denominator=denominator, threshold=float(threshold),
                accepted=bool(denominator > 0 and numerator >= threshold*denominator),
                zero_energy=denominator == 0, omega_rad_per_symbol=float(omega),
                cycles_per_symbol=float(omega)/(2*np.pi), nco=nco,
                source="already selected differential sync correlation;no frequency/phase retry")
