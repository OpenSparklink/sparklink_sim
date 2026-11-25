"""Selected normalized word stream, actual timed B1 header and carrier join."""

from collections import deque
from dataclasses import replace

from nearlink_sdr.phy.interpolation_word import rne_shift
from nearlink_sdr.phy.normalization_word import unpack_normalized_word
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.carrier_fixed_flow import CarrierWordEngine
from nearlink_sdr.sim.psk_hard_flow import HeaderFromWordsService
from nearlink_sdr.sim.psk_sync_gain import PSKSyncGain, _q14
from nearlink_sdr.sim.psk_word_demap import (
    B1SymbolFrame,
    TimedPSKWord,
    replay_b1_symbol_frames,
)

__all__ = ["replay_selected_b1_carrier"]

_CLOCKS = (100_000_000, 160_000_000, 200_000_000)
_DEPTHS = (8, 64, 128)
_END = object()


def _signed18(value, shift):
    if value == 0 or shift <= -22:
        return 0, False
    if shift >= 18:
        return (131071 if value > 0 else -131072), True
    scaled = rne_shift(value, -shift)
    clipped = min(131071, max(-131072, scaled))
    return clipped, clipped != scaled


class _SelectedFrame:
    """One arrival register and one ordered suffix FIFO with fixed-rate drain."""

    def __init__(self, words, profile, first_edge, clock_hz, depth, layout, observer):
        self.source = iter(words)
        self.edge, self.first = first_edge, first_edge
        self.period = clock_hz // 4_000_000
        self.depth = depth
        self.observer = observer
        self.head = HeaderFromWordsService(profile, layout=layout)
        assert self.head.begin()
        self.nsync = self.head.start
        self.data_start = self.head.header.geometry.data_start
        template = self.head.header.profile.sync_symbols()
        self.template = tuple((_q14(z.real)[0], _q14(z.imag)[0]) for z in template)
        self.gain = PSKSyncGain(self.nsync)
        self.carrier = CarrierWordEngine(self.head.modulation, self.nsync)
        self.profile = self.head.header.profile
        self.next_input = first_edge
        self.index = 0
        self.pending = None
        self.buffer = deque()
        self.eof = False
        self.truncated = False
        self.active_index = None
        self.first_payload_launch = self.first_payload_output = self.next_launch = None
        self.header_edge = self.header_row = self.payload_cfg = None
        self.failure = self.failure_edge = self.missed_launch_edge = None
        self.peak = self.warm_outputs = self.payload_outputs = 0
        self.source_saturations = self.output_saturations = self.post_frame_words = 0

    def _send(self, kind, **fields):
        if self.observer is not None:
            self.observer(dict(kind=kind, edge=self.edge, **fields))

    def _fail(self, reason):
        self.failure, self.failure_edge = reason, self.edge
        self._send("front_failed", reason=reason)

    def _receive(self):
        if self.eof or self.edge != self.next_input:
            return
        if self.header_row is not None and self.index == self.header_row["frame_end_symbol"]:
            self.eof = True
            return
        word = next(self.source, _END)
        if word is _END:
            self.eof = True
            if self.index < self.nsync:
                self._fail("truncated_sync")
            elif self.head.mode == "collect" and self.head.offered < self.head.end - self.nsync:
                self.head.end_input()
            return
        r, i, exp = unpack_normalized_word(word)
        index = self.index
        self.index += 1
        self.next_input += self.period
        self.pending = index, r, i, exp
        if self.nsync <= index < self.head.end and not self.head.offer(word, index):
            raise RuntimeError("Nominal head stream exceeded hard-demapper II")

    def _convert(self):
        if self.pending is None:
            return
        index, r, i, exp = self.pending
        self.pending = None
        if self.payload_cfg is not None and index >= self.header_row["frame_end_symbol"]:
            self.post_frame_words += 1
            return
        shift = exp + (14 if index < self.nsync else 10)
        real, cr = _signed18(r, shift)
        imag, ci = _signed18(i, shift)
        self.source_saturations += cr + ci
        if index < self.nsync:
            self.gain.accept(self.edge, (real, imag), self.template[index])
        else:
            if len(self.buffer) == self.depth:
                self._fail("carrier_fifo_full")
                return
            self.buffer.append((index, real, imag))

    def _take_header(self):
        if not self.head.ready:
            return
        self.header_row = self.head.take()
        self.header_edge = self.edge
        row = self.header_row
        self._send("header_visible", status=row["status"], header_crc=row["header_crc"])
        if row["status"] != "header_valid":
            self._fail(row["status"])
            return
        cfg = replace(self.profile.config, mcs_index=row["mcs"])
        if cfg.is_uncoded or cfg.mod_str not in ("QPSK", "8PSK"):
            self._fail("unsupported_payload_path")
            return
        if row["data_start_symbol"] != self.data_start or not row["data_crc_present"]:
            raise RuntimeError("Verified header disagrees with owned B1 wire context")
        self.payload_cfg = cfg
        while self.buffer and self.buffer[-1][0] >= row["frame_end_symbol"]:
            self.buffer.pop()
            self.post_frame_words += 1
        if self.index >= row["frame_end_symbol"]:
            self.eof = True

    def _take_carrier(self):
        if not self.carrier.done:
            return None
        result = self.carrier.take()
        index, self.active_index = self.active_index, None
        if index < self.data_start:
            self.warm_outputs += 1
            return None
        if index != self.data_start + self.payload_outputs:
            raise RuntimeError("Unexpected B1 carrier wire output order")
        if self.first_payload_output is None:
            self.first_payload_output = self.edge
        if self.edge != self.first_payload_output + self.payload_outputs * self.period:
            raise RuntimeError("Paced B1 carrier output missed its nominal edge")
        r, cr = _signed18(result["real"], 4)
        i, ci = _signed18(result["imag"], 4)
        self.output_saturations += cr + ci
        self.payload_outputs += 1
        self._send("payload_word", index=index)
        return TimedPSKWord(self.edge, r, i)

    def _launch(self):
        if not self.buffer or self.carrier.active is not None:
            return
        index, real, imag = self.buffer[0]
        if index >= self.data_start:
            if self.payload_cfg is None or self.edge <= self.header_edge:
                return
            if self.first_payload_launch is None:
                if self.gain.result is None or self.gain.result.ready_edge > self.edge:
                    self._fail("late_sync_gain")
                    return
                self.first_payload_launch, self.next_launch = self.edge, self.edge
                self.carrier.node.switch_modulation(self.payload_cfg.mod_str)
                self._send("payload_launch", index=index, modulation=self.payload_cfg.mod_str)
            if self.edge < self.next_launch:
                return
            if self.edge != self.next_launch:
                self.missed_launch_edge = self.next_launch
                self._fail("carrier_pace_missed")
                return
            self.next_launch += self.period
        self.buffer.popleft()
        self.carrier.begin(real, imag)
        self.active_index = index

    def _step(self):
        out = self._take_carrier()
        self._take_header()
        if self.failure is None:
            self.gain.advance(self.edge)
            if self.carrier.active is not None:
                self.carrier.tick()
            # An older FIFO word can retire before a same-edge register commit.
            self._launch()
            if self.failure is None:
                self._convert()
            if self.failure is None:
                self._launch()  # forward a new committed word if B2 is still idle
                self._receive()
                if self.header_row is None:
                    self.head.step()
        self.peak = max(self.peak, len(self.buffer))
        self.edge += 1
        return out if self.failure is None else None

    def next_payload(self):
        while self.failure is None:
            if (self.header_row is not None
                    and self.payload_outputs == self.header_row["wire_data_symbols"]):
                return None
            if (self.eof and self.pending is None and not self.buffer
                    and self.carrier.active is None and self.header_row is not None):
                self.truncated = True
                self._fail("truncated_payload")
                return None
            out = self._step()
            if out is not None:
                return out
        return None

    def result(self, delivery):
        status = self.failure
        if status is None:
            status = "delivered" if delivery.get("mac_complete") else "payload_rejected"
        return dict(status=status, header=self.header_row, header_ready_edge=self.header_edge,
                    gain_ready_edge=(None if self.gain.result is None
                                     else self.gain.result.ready_edge),
                    first_payload_launch=self.first_payload_launch,
                    first_payload_output=self.first_payload_output,
                    wire_inputs=self.index, carrier_warm_outputs=self.warm_outputs,
                    carrier_payload_outputs=self.payload_outputs, peak_suffix_words=self.peak,
                    suffix_capacity=self.depth, failure_edge=self.failure_edge,
                    missed_launch_edge=self.missed_launch_edge,
                    truncated_payload=self.truncated,
                    post_frame_buffered_words=self.post_frame_words,
                    input_saturations=self.source_saturations,
                    output_saturations=self.output_saturations, delivery=delivery)


def replay_selected_b1_carrier(
    words, profile, *, clock_hz, word_cycles, first_edge=0, suffix_depth=128,
    head_layout="dense", observer=None, front_observer=None, **decoder_options,
):
    """Replay one selected normalized frame with actual clocked subservices.

    Arrival is one normalized word per nominal symbol. The 55-bit source owns
    acquisition, selection, sample timing and normalization, whose time is
    outside this composition. A live header alone selects payload geometry.
    """
    if (not isinstance(profile, PSKReceiveProfile) or profile.control_format != "B1"
            or profile.config.frame_type not in (3, 4)
            or profile.config.b1_code_block_mode != "standard"
            or profile.config.symbol_rate_mhz != 4 or profile.length_contract != "standard"
            or profile.payload_bitorder != "big"):
        raise ValueError("Need standard 4 Msym/s FT3/4 B1 profile")
    if (type(clock_hz) is not int or clock_hz not in _CLOCKS
            or type(first_edge) is not int or first_edge < 0
            or type(suffix_depth) is not int or suffix_depth not in _DEPTHS):
        raise ValueError("Need bounded selected-frame clock, origin and suffix FIFO")
    front = _SelectedFrame(words, profile, first_edge, clock_hz, suffix_depth,
                           head_layout, front_observer)
    first = front.next_payload()
    if first is None:
        return front.result({})
    row = front.header_row
    frame = B1SymbolFrame.from_config(
        first.edge, row["payload_length_bytes"], front.payload_cfg,
        symbol_index=row["data_start_symbol"], prepared_gain=front.gain.result,
    )
    if frame.coded_bits != row["coded_bits"]:
        raise RuntimeError("B1 payload service bit extent differs from actual header")

    def payload():
        yield first
        while (word := front.next_payload()) is not None:
            yield word

    delivery = replay_b1_symbol_frames(
        [frame], payload(), clock_hz=clock_hz, word_cycles=word_cycles,
        observer=observer, **decoder_options,
    )
    return front.result(delivery)
