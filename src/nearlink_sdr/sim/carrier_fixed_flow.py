"""Value-carrying carrier schedule; primitive delays are assumptions, not synthesis.

ROM4,DSP3,wideadd/RNE1,parallel3-ALU CORDIC1iteration/clock and power-node8
are explicit proposals. Unlike a packet-latency placeholder, rotations,each
CORDIC iteration and individual PI updates carry exact values between clocks.
"""

from collections import deque

from nearlink_sdr.phy.carrier_fixed import (
    _ATAN_WORDS,
    _FREQUENCY_BOUNDS,
    CarrierWordFormat,
    FixedCarrierLoop,
    _rne,
    _word,
)
from nearlink_sdr.phy.sync_coherent import phase_rotation_words

__all__ = ["CarrierWordEngine", "carrier_word_cost", "simulate_carrier_words"]


class CarrierWordEngine:
    """One owned active word;41-clock service with optional state-forwarded reissue."""

    def __init__(self, modulation, start, *, fraction_bits=10):
        self.node = FixedCarrierLoop(modulation, start, fmt=CarrierWordFormat(fraction_bits))
        self.active = None
        self.done = False

    def begin(self, real, imag):
        if self.active is not None:
            raise RuntimeError("Carrier engine already owns a word")
        if self.node.closed or self.node.used >= 2**17:
            raise RuntimeError("Closed or exhausted carrier packet")
        self.active = dict(
            a=_word(real),
            b=_word(imag),
            phase=self.node.phase,
            frequency=self.node.frequency,
            m=self.node.m,
            cycle=0,
        )
        self.done = False

    def tick(self):
        if self.active is None or self.done:
            raise RuntimeError("Need an active unfinished carrier word")
        s = self.active
        s["cycle"] += 1
        c = s["cycle"]
        if c == 3:
            s["power_products"] = (s["a"] * s["a"], s["b"] * s["b"])
        elif c == 4:
            s["raw_power"] = sum(s["power_products"])
            s["cos"], s["sin"] = phase_rotation_words(s["phase"] & (2**32 - 1))
        elif c == 5:
            s["power"] = _rne(s["raw_power"], 2 * self.node.fmt.fraction_bits - 16)
            self.node.stats["power_erased"] += int(s["raw_power"] > 0 and s["power"] == 0)
        elif c == 6:
            previous = self.node.power.power
            s["low"], s["high"] = s["power"] * 16 < previous, s["power"] > 4 * previous
            self.node.stats["low_power_holds"] += int(s["low"])
            self.node.stats["high_power_holds"] += int(s["high"])
        elif c == 7:
            a, b, co, si = s["a"], s["b"], s["cos"], s["sin"]
            s["products"] = (a * co, b * si, a * si, b * co)
        elif c == 8:
            a, b, d, e = s["products"]
            s["sums"] = (a - b, d + e)
        elif c == 9:
            raw = [_rne(v, 16) for v in s["sums"]]
            self.node.stats["output_saturations"] += sum(not -(2**17) <= v < 2**17 for v in raw)
            s["real"], s["imag"] = (min(2**17 - 1, max(-(2**17), v)) for v in raw)
        elif c == 10:
            peak = max(abs(s["real"]), abs(s["imag"]))
            s["zero"] = peak == 0
            s["shift"] = 0 if peak == 0 else 21 - peak.bit_length()
        elif c == 11:
            s["x"], s["y"] = s["real"] << s["shift"], s["imag"] << s["shift"]
        elif c == 12:
            s["angle"] = 0
            if s["x"] < 0:
                s["x"], s["y"], s["angle"] = -s["x"], -s["y"], -(2**31)
        if c == 13:
            # Existing bounded integer power primitive:8-cycle macro, input
            # ready5/output13. Gate used below was separately registered at6.
            self.node.power.advance_words([s["power"]])
        if 13 <= c <= 30 and not (s["zero"] or s["low"] or s["high"]) and s["y"] != 0:
            k = c - 13
            x, y = s["x"], s["y"]
            direction = 1 if y > 0 else -1
            s["x"], s["y"] = x + direction * (y >> k), y - direction * (x >> k)
            s["angle"] += direction * _ATAN_WORDS[k]
            if (
                not -(2**23) <= s["x"] < 2**23
                or not -(2**23) <= s["y"] < 2**23
                or not -(2**33) <= s["angle"] < 2**33
            ):
                self.node.stats["cordic_overflows"] += 1
                raise ArithmeticError("Carrier calendar CORDIC scratch overflow")
        elif c == 31:
            angle = (s["angle"] + 2**31) % 2**32 - 2**31
            if (self.node.start + self.node.used) & 1:
                angle += 2**31 // s["m"]
            s["angle"] = angle
        elif c == 32:
            sector = 2**32 // s["m"]
            s["limit"] = sector // 4
            s["error"] = (
                0 if s["zero"] or s["low"] or s["high"] else s["angle"] % sector - sector // 2
            )
            if s["zero"] and not (s["low"] or s["high"]):
                self.node.stats["zero_vector_holds"] += 1
        elif c == 33:
            self.node.stats["error_clamps"] += int(abs(s["error"]) > s["limit"])
            s["error"] = min(s["limit"], max(-s["limit"], s["error"]))
        elif c == 34:
            s["fi"] = _rne(s["error"], 11)
        elif c == 35:
            s["next_f"] = s["frequency"] + s["fi"]
        elif c == 36:
            bound = _FREQUENCY_BOUNDS[s["m"]]
            self.node.stats["frequency_clamps"] += int(abs(s["next_f"]) > bound)
            s["next_f"] = min(bound, max(-bound, s["next_f"]))
        elif c == 37:
            s["pi"] = _rne(s["error"], 5)
        elif c == 38:
            s["next_phase"] = s["phase"] + s["next_f"]
        elif c == 39:
            s["next_phase"] += s["pi"]
        elif c == 40:
            self.node.stats["phase_wraps"] += int(not -(2**31) <= s["next_phase"] < 2**31)
            self.node.phase = (s["next_phase"] + 2**31) % 2**32 - 2**31
            self.node.frequency = s["next_f"]
        elif c == 41:
            self.node.used += 1
            self.done = True
        return dict(
            cycle=c,
            phase=s["phase"],
            x=s.get("x"),
            y=s.get("y"),
            angle=s.get("angle"),
            error=s.get("error"),
            frequency=s.get("next_f"),
            done=self.done,
        )

    def take(self):
        if not self.done:
            raise RuntimeError("Carrier word not complete")
        s = self.active
        result = {k: s[k] for k in ("real", "imag", "phase", "frequency", "error")}
        result["held"] = s["low"] or s["high"]
        self.active = None
        self.done = False
        return result


def carrier_word_cost(input_depth=4):
    if type(input_depth) is not int or input_depth not in (4, 8):
        raise ValueError("Need fixed input depth4/8 study")
    state = dict(
        input_FIFO=input_depth * 56,
        output_FIFO=4 * 154,
        owned_word_and_rotation_products=2 * 18 + 4 * 36 + 2 * 37,
        CORDIC_xy=2 * 24,
        CORDIC_angle=34,
        atan_ROM=18 * 32,
        NCO_cosine_ROM=1024 * 17,
        NCO_ROM_read_words=2 * 17,
        NCO_addresses_and_sign=13 + 2 * 11 + 8,
        phase_frequency_error=3 * 32,
        power_state=33,
        power_scratch=3 * 41,
        counters=18 * 19,
        index_and_field_context=4 * 18 + 8,
        pipeline_controls=64,
        coefficient_pipeline=4 * (32 + 18) + 2 * 18,
        PI_scratch=4 * 34,
        power_products_and_raw_words=2 * 36 + 37 + 32,
        active_old_state_snapshot=2 * 32,
    )
    return dict(
        evidence="value-carrying executable word schedule plus allocation estimate;not synthesis",
        input_depth=input_depth,
        service_clocks=41,
        NCO_rounding_ALU_bits=33,
        NCO_reflection_ALU_bits=11,
        NCO_sign_bits=18,
        DSP25x18_tiles=6,
        real_product_operand_bits=[18, 18],
        CORDIC_parallel_adders_bits=[24, 24, 34],
        rotation_parallel_adders_bits=[37, 37],
        PI_ALU_bits=34,
        power_ALU_bits=42,
        counter_parallel_adders_bits=[19] * 18,
        counter_bound="at most2events/component-pair per symbol,131072symbols;19bits cover262144",
        leading_zero_encoder_bits=18,
        normalization_barrel_shifters=[24, 24],
        shifter_reuse="normalization leftshift11 and CORDIC rightshift13..30 do not overlap",
        bidirectional_shifter_operand_and_amount_mux_bits=48 + 10,
        shifter_steering_LUT_and_direction_cost=None,
        NCO_ROM="existing1024x17,2read/0write;one additional RAMB18,not free shared sync ROM",
        atan_ROM="18x32,one combinational read per cycle;LUT allocation unverified",
        FIFO_ports="each1R/1W;input4/8 and output4 register proposals;word tags/state charged",
        logical_bits=sum(state.values()),
        state=state,
        dedicated_ROM_BRAM18_estimate=1,
        other_FIFO_and_state_LUT_FF=None,
        assumptions="NCO4,DSP3,wideadd/RNE1,parallel3ALU CORDIC1iteration/cycle,power macro8;18dedicated counter adders;write-forwarded state reissue",
        full_receiver_verified=False,
        actual_synthesis_or_timing=False,
        actual_gain_normalization_or_header_decode_schedule_included=False,
        all_original_resource_failures_preserved=True,
        same_cycle_state_forwarding_assumed=True,
        observer_arrival_timestamps_not_receiver_functional_state=True,
        hardware_latency_debug_instrumentation_cost=None,
    )


def simulate_carrier_words(
    real,
    imag,
    *,
    modulation="QPSK",
    start=0,
    header_symbols=0,
    header_modulation="QPSK",
    clock_hz=200_000_000,
    symbol_hz=4_000_401,
    stalls=(),
    source_batch=1,
    input_depth=4,
):
    """One packet stream;ADC-like arrivals cannot pause,queue overflow invalidates packet."""
    if (
        len(real) != len(imag)
        or not 1 <= len(real) <= 2**17
        or type(header_symbols) is not int
        or not 0 <= header_symbols < len(real)
        or any(type(v) is not int or v <= 0 for v in (clock_hz, symbol_hz))
        or clock_hz not in (100_000_000, 160_000_000, 200_000_000)
        or symbol_hz not in (4_000_000, 4_000_401)
        or type(source_batch) is not int
        or source_batch not in (1, 4, 8)
        or type(input_depth) is not int
        or input_depth not in (4, 8)
        or not isinstance(stalls, (list, tuple))
        or len(stalls) > 64
    ):
        raise ValueError("Need finite word packet and positive at-most1word/clock source")
    stop = -1
    for a, b in stalls:
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= 2**24 or a < stop:
            raise ValueError("Need ordered disjoint stalls")
        stop = b
    engine = CarrierWordEngine(header_modulation if header_symbols else modulation, start)
    pending, output = deque(), deque()
    emitted, dropped = [], []
    active_tag = active_arrival = None
    index = cycle = peak_in = peak_out = 0
    invalid = False

    def source_arrival(tag):
        return (tag // source_batch) * source_batch * clock_hz // symbol_hz + tag % source_batch

    horizon = max(source_arrival(len(real) - 1), stop) + 41 * (len(real) + 16)
    while index < len(real) or pending or output or engine.active is not None:
        if cycle > horizon:
            raise AssertionError("Finite carrier calendar failed to drain")
        blocked = any(a <= cycle < b for a, b in stalls)
        if output and not blocked:
            item = output.popleft()
            item["emitted_cycle"] = cycle
            emitted.append(item)
        if engine.active is not None and not engine.done:
            engine.tick()
        if engine.done and len(output) < 4:
            result = engine.take()
            output.append(
                dict(tag=active_tag, arrival=active_arrival, completed_cycle=cycle, result=result)
            )
        if index < len(real) and cycle == source_arrival(index):
            if invalid:
                dropped.append(dict(tag=index, cycle=cycle, reason="packet_already_invalid"))
            elif len(pending) == input_depth:
                dropped.append(dict(tag=index, cycle=cycle, reason="input_FIFO_full"))
                invalid = True
                while pending:
                    tag, _, _, _ = pending.popleft()
                    dropped.append(dict(tag=tag, cycle=cycle, reason="flushed_invalid_packet"))
                while output:
                    item = output.popleft()
                    dropped.append(
                        dict(tag=item["tag"], cycle=cycle, reason="flushed_output_invalid_packet")
                    )
                if engine.active is not None:
                    dropped.append(
                        dict(tag=active_tag, cycle=cycle, reason="aborted_active_invalid_packet")
                    )
                engine.active = None
                engine.done = False
            else:
                pending.append((index, _word(real[index]), _word(imag[index]), cycle))
            index += 1
        if engine.active is None and pending and not invalid:
            tag, a, b, arrival = pending.popleft()
            if tag == header_symbols:
                engine.node.switch_modulation(modulation)
            engine.begin(a, b)
            active_tag, active_arrival = tag, arrival
        peak_in = max(peak_in, len(pending))
        peak_out = max(peak_out, len(output))
        cycle += 1
    return dict(
        emitted=emitted,
        dropped=dropped,
        cycles=cycle,
        resource_packet_invalid=invalid,
        input_sequence_intact=not dropped,
        peak_input=peak_in,
        peak_output=peak_out,
        max_latency=max((v["emitted_cycle"] - v["arrival"] for v in emitted), default=None),
        final_state=engine.node.finish(),
        runtime_owned_word_bound=input_depth + 4 + 1,
        input_depth=input_depth,
        source_batch=source_batch,
        source_hz=symbol_hz,
        clock_hz=clock_hz,
        whole_rx16msps_verified=False,
    )
