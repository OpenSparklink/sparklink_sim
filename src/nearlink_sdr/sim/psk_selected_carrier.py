"""Selected normalized word stream, actual timed B1 header and carrier join."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace

from nearlink_sdr.phy.interpolation_word import rne_shift
from nearlink_sdr.phy.normalization_word import unpack_normalized_word
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.carrier_fixed_flow import CarrierWordEngine
from nearlink_sdr.sim.carrier_pi_flow import ParallelPICarrierEngine
from nearlink_sdr.sim.psk_hard_flow import HeaderFromWordsService
from nearlink_sdr.sim.psk_sync_gain import PSKSyncGain, _q14
from nearlink_sdr.sim.psk_word_demap import (
    B1SymbolFrame,
    TimedPSKWord,
    replay_b1_symbol_frames,
)

__all__ = ["B1SelectedInput", "replay_selected_b1_carrier", "replay_selected_b1_train"]

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


def _scaled_iq(real, imag, exponent, fraction):
    shift = exponent + fraction
    real, clipped_real = _signed18(real, shift)
    imag, clipped_imag = _signed18(imag, shift)
    return real, imag, clipped_real + clipped_imag


class _SyncGain:
    """Own the gain pipeline until its immutable result is registered."""

    def __init__(self, profile):
        self.template = tuple((_q14(z.real)[0], _q14(z.imag)[0])
                              for z in profile.sync_symbols())
        self.engine = PSKSyncGain(len(self.template))
        self.result = None

    def advance(self, edge):
        if self.engine is not None:
            self.result = self.engine.advance(edge)
            if self.result is not None:
                self.engine = None

    def accept(self, edge, index, real, imag, exponent):
        real, imag, clipped = _scaled_iq(real, imag, exponent, 14)
        self.engine.accept(edge, (real, imag), self.template[index])
        return clipped


class _SyncOffer:
    """A candidate consuming sync words only, without head or suffix storage."""

    def __init__(self, candidate, offer, period):
        self.candidate, self.offer = candidate, offer
        self.source = iter(offer.words)
        self.gain = _SyncGain(offer.profile)
        self.period = period
        self.next_input = offer.first_edge
        self.header_edge = self.next_input + period * len(self.gain.template)
        self.index = self.saturations = 0
        self.pending = None

    def step(self, edge):
        self.gain.advance(edge)
        if self.pending is not None:
            self.saturations += self.gain.accept(edge, *self.pending)
            self.pending = None
        if edge != self.next_input:
            return True
        if edge >= self.header_edge:
            raise RuntimeError("Staged B1 source reached an unowned header")
        word = next(self.source, _END)
        if word is _END:
            return False
        self.pending = self.index, *unpack_normalized_word(word)
        self.index += 1
        self.next_input += self.period
        return True


class _SuffixRAM:
    """One shared synchronous read request and one write per clock edge."""

    def __init__(self, depth):
        self.words = [0] * depth
        self.edge = -1
        self.read_addr = self.write_addr = None
        self.reads = self.writes = self.paired = 0

    def _advance(self, edge):
        if edge < self.edge:
            raise RuntimeError("Shared B1 suffix RAM clock moved backwards")
        if edge != self.edge:
            self.edge = edge
            self.read_addr = self.write_addr = None

    def read(self, edge, address):
        self._advance(edge)
        if self.read_addr is not None or self.write_addr == address:
            raise RuntimeError("Shared B1 suffix RAM read port conflict")
        self.read_addr = address
        self.reads += 1
        self.paired += int(self.write_addr is not None)
        return self.words[address]

    def write(self, edge, address, word):
        self._advance(edge)
        if self.write_addr is not None or self.read_addr == address:
            raise RuntimeError("Shared B1 suffix RAM write port conflict")
        self.write_addr = address
        self.writes += 1
        self.paired += int(self.read_addr is not None)
        self.words[address] = word


class _SuffixIQ:
    """Contiguous Q10 I/Q ring with one registered RAM read and an empty bypass."""

    def __init__(self, depth, first_index, ram=None):
        self.ports = ram if isinstance(ram, _SuffixRAM) else None
        self.ram = self.ports.words if self.ports is not None else (
            [0] * depth if ram is None else ram
        )
        self.base = 0
        self.address_mask = depth - 1
        self.first_index = first_index
        self.count = 0
        self.head_word = self.read_pending = None

    def __len__(self):
        return self.count

    def address(self, index):
        return (self.base + index) & self.address_mask

    def advance(self, edge=None):
        # Read requested when the old head retired. A clipped suffix has no live request.
        if not self.count or self.head_word is not None:
            return
        if self.ports is None:
            self.head_word = self.ram[self.address(self.first_index)]
        elif self.read_pending is not None and self.read_pending[0] <= edge:
            _, self.head_word = self.read_pending
            self.read_pending = None

    def append(self, index, real, imag, edge=None):
        if self.count == len(self.ram) or index != self.first_index + self.count:
            raise RuntimeError("Noncontiguous or full B1 suffix ring")
        word = (real & 0x3FFFF) | ((imag & 0x3FFFF) << 18)
        if self.count:
            address = self.address(index)
            if self.ports is None:
                self.ram[address] = word
            else:
                self.ports.write(edge, address, word)
        else:
            self.head_word = word
        self.count += 1

    def popleft(self, edge=None):
        word = self.head_word
        if word is None:
            raise RuntimeError("B1 suffix head used before registered read")
        self.head_word = None
        self.first_index += 1
        self.count -= 1
        if self.ports is not None and self.count:
            self.read_pending = edge + 1, self.ports.read(edge, self.address(self.first_index))
        return ((word & 0x3FFFF) ^ 0x20000) - 0x20000, ((word >> 18) ^ 0x20000) - 0x20000

    def discard_from(self, end):
        kept = min(self.count, max(0, end - self.first_index))
        removed = self.count - kept
        self.count = kept
        if not kept:
            self.head_word = self.read_pending = None
        return removed


class _SelectedFrame:
    """One arrival register and one ordered suffix FIFO with fixed-rate drain."""

    def __init__(
        self, words, profile, first_edge, clock_hz, depth, layout, observer, *, engine_type,
        suffix_ram=None, staged_sync=None, head_owner=None,
    ):
        self.source = iter(words) if staged_sync is None else staged_sync.source
        self.edge = first_edge
        self.period = clock_hz // 4_000_000
        self.depth = depth
        self.observer = observer
        self.owns_head = head_owner is None
        self.head = HeaderFromWordsService(profile, layout=layout) if self.owns_head else head_owner
        if self.owns_head:
            assert self.head.begin()
        self.nsync = self.head.start
        self.data_start = self.head.header.geometry.data_start
        self.gain = _SyncGain(profile) if staged_sync is None else staged_sync.gain
        self.carrier = (
            None if engine_type is None else engine_type(self.head.modulation, self.nsync)
        )
        self.profile = self.head.header.profile
        self.next_input = first_edge
        self.index = 0
        self.pending = None
        self.buffer = _SuffixIQ(depth, self.nsync, suffix_ram)
        self.prior_suffix = None
        self.eof = False
        self.truncated = False
        self.active_index = None
        self.first_payload_launch = self.first_payload_output = self.next_launch = None
        self.header_edge = self.header_row = self.payload_cfg = None
        self.failure = self.failure_edge = self.missed_launch_edge = None
        self.peak = self.warm_outputs = self.payload_outputs = 0
        self.source_saturations = self.output_saturations = self.post_frame_words = 0
        if staged_sync is not None:
            self.next_input = staged_sync.next_input
            self.index, self.pending = staged_sync.index, staged_sync.pending
            self.source_saturations = staged_sync.saturations

    def use_suffix_after(self, previous):
        # The older validated range only drains; its physical tail cannot move.
        self.prior_suffix = previous.buffer
        tail = previous.buffer.address(previous.buffer.first_index + len(previous.buffer))
        self.buffer.base = (tail - self.nsync) & self.buffer.address_mask

    def start_carrier(self, engine_type):
        self.prior_suffix = None
        self.carrier = engine_type(self.head.modulation, self.nsync)

    @property
    def input_released(self):
        return self.eof or (self.payload_cfg is not None
                            and self.index >= self.header_row["frame_end_symbol"])

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
        if index < self.nsync:
            self.source_saturations += self.gain.accept(self.edge, index, r, i, exp)
        else:
            real, imag, clipped = _scaled_iq(r, i, exp, 10)
            self.source_saturations += clipped
            reserved = len(self.buffer)
            if self.prior_suffix is not None:
                reserved += len(self.prior_suffix)
            if reserved == self.depth:
                self._fail("carrier_fifo_full" if self.prior_suffix is None
                           else "selected_shared_suffix_full")
                return
            self.buffer.append(index, real, imag, self.edge)

    def _take_header(self):
        if not self.head.ready:
            return
        self._install_header(self.head.take())

    def _install_header(self, row):
        self.header_row, self.header_edge = row, self.edge
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
        self.post_frame_words += self.buffer.discard_from(row["frame_end_symbol"])
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
        index = self.buffer.first_index
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
        real, imag = self.buffer.popleft(self.edge)
        self.carrier.begin(real, imag)
        self.active_index = index

    def _step(self):
        out = None
        if self.carrier is not None:
            self.buffer.advance(self.edge)
            out = self._take_carrier()
        if self.header_row is None:
            self._take_header()
        if self.failure is None:
            self.gain.advance(self.edge)
            if self.carrier is not None:
                if self.carrier.active is not None:
                    self.carrier.tick()
                # Retire an older FIFO word before a same-edge register commit.
                self._launch()
            if self.failure is None:
                self._convert()
            if self.failure is None:
                if self.carrier is not None:
                    self._launch()  # forward a newly committed word if B2 remains idle
                self._receive()
                if self.header_row is None and self.owns_head:
                    self.head.step()
        self.peak = max(self.peak, len(self.buffer))
        self.edge += 1
        return out if self.failure is None else None

    def next_payload(self, active_edge=None):
        while self.failure is None:
            if (self.header_row is not None
                    and self.payload_outputs == self.header_row["wire_data_symbols"]):
                return None
            if (self.eof and self.pending is None and not self.buffer
                    and self.carrier.active is None and self.header_row is not None):
                self.truncated = True
                self._fail("truncated_payload")
                if active_edge is not None:
                    active_edge(self, self.edge, False)
                return None
            prior_index = self.index
            out = self._step()
            if active_edge is not None:
                active_edge(self, self.edge - 1, self.index != prior_index)
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


def _engine_type(carrier_schedule):
    if carrier_schedule == "serial":
        return CarrierWordEngine
    if carrier_schedule == "parallel-pi":
        return ParallelPICarrierEngine
    raise ValueError("Need serial or parallel-pi carrier schedule")


def _validate_front(profile, clock_hz, first_edge, suffix_depth):
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


def _payload_frame(front, first):
    row = front.header_row
    frame = B1SymbolFrame.from_config(
        first.edge, row["payload_length_bytes"], front.payload_cfg,
        symbol_index=row["data_start_symbol"], prepared_gain=front.gain.result,
    )
    if frame.coded_bits != row["coded_bits"]:
        raise RuntimeError("B1 payload service bit extent differs from actual header")
    return frame


def _payload_words(front, first, active_edge=None):
    yield first
    while (word := front.next_payload(active_edge)) is not None:
        yield word


def replay_selected_b1_carrier(
    words, profile, *, clock_hz, word_cycles, first_edge=0, suffix_depth=128,
    head_layout="dense", carrier_schedule="serial", observer=None, front_observer=None,
    **decoder_options,
):
    """Replay one selected normalized frame with actual clocked subservices.

    Arrival is one normalized word per nominal symbol. The 55-bit source owns
    acquisition, selection, sample timing and normalization, whose time is
    outside this composition. A live header alone selects payload geometry.
    The explicit parallel-pi schedule retires the same B2 arithmetic earlier.
    """
    _validate_front(profile, clock_hz, first_edge, suffix_depth)
    engine_type = _engine_type(carrier_schedule)
    front = _SelectedFrame(words, profile, first_edge, clock_hz, suffix_depth,
                           head_layout, front_observer, engine_type=engine_type)
    first = front.next_payload()
    if first is None:
        return front.result({})
    delivery = replay_b1_symbol_frames(
        [_payload_frame(front, first)], _payload_words(front, first),
        clock_hz=clock_hz, word_cycles=word_cycles, observer=observer, **decoder_options,
    )
    return front.result(delivery)


@dataclass(frozen=True)
class B1SelectedInput:
    """A candidate-local normalized source offered at its absolute first-word edge."""

    first_edge: int
    profile: PSKReceiveProfile
    words: Iterable[int]


def replay_selected_b1_train(
    offers, *, clock_hz, word_cycles, suffix_depth=128, head_layout="dense",
    carrier_schedule="serial", observer=None, front_observer=None, sync_overlap=False,
    shared_suffix=False, **decoder_options,
):
    """Share one selected front lane and one downstream service across ordered offers.

    The caller owns the normalized candidate boundaries. An offer before the
    prior front retires is rejected at its arrival edge by default.
    With sync_overlap, one early offer may begin its sync work after the older
    lane releases normalized input and the gain engine. It needs the older
    lane to retire by its first non-sync input edge. Independently framed
    inputs can continue after a truncated prefix. With shared_suffix, a younger
    head can proceed if the older input and head have retired. Both ranges
    share one bounded IQ RAM, while the older frame keeps the carrier lane
    until it retires.
    Front events tag the candidate index. The payload_selected event associates
    that index with the frame index used by the downstream observer. Compare
    timestamps, since lazy lookahead can report an earlier rejected offer after
    the older owner's last event.
    """
    if clock_hz not in _CLOCKS or suffix_depth not in _DEPTHS:
        raise ValueError("Need bounded selected-frame clock and suffix FIFO")
    if type(sync_overlap) is not bool:
        raise ValueError("Need a boolean B1 sync overlap option")
    if type(shared_suffix) is not bool:
        raise ValueError("Need a boolean B1 shared suffix option")
    sync_overlap = sync_overlap or shared_suffix
    engine_type = _engine_type(carrier_schedule)
    outcomes = Counter()
    peak = shared_peak = shared_pairs = 0
    suffix_ram = _SuffixRAM(suffix_depth) if shared_suffix else [0] * suffix_depth

    def report(candidate, kind, edge, **fields):
        if front_observer is not None:
            front_observer(dict(candidate=candidate, kind=kind, edge=edge, **fields))

    def checked_offers():
        last_edge = -1
        for candidate, offer in enumerate(offers):
            if not isinstance(offer, B1SelectedInput):
                raise ValueError("Need B1 selected input offers")
            _validate_front(offer.profile, clock_hz, offer.first_edge, suffix_depth)
            if offer.first_edge < last_edge:
                raise ValueError("B1 candidate arrival edges must be nondecreasing")
            if sync_overlap and offer.first_edge == last_edge:
                raise ValueError("B1 sync overlap requires at most one offer per edge")
            last_edge = offer.first_edge
            yield candidate, offer

    def reject(candidate, edge, reason, **fields):
        outcomes[reason] += 1
        report(candidate, "front_failed", edge, reason=reason, **fields)

    def start_frame(number, offer, edge, prefix, engine):
        def observe(row):
            front_observer(dict(candidate=number, **row))

        return _SelectedFrame(
            offer.words, offer.profile, edge, clock_hz, suffix_depth, head_layout,
            None if front_observer is None else observe,
            engine_type=engine, suffix_ram=suffix_ram, staged_sync=prefix,
        )

    def retire(number, front, edge):
        nonlocal peak
        status = front.failure or "payload_complete"
        outcomes[status] += 1
        peak = max(peak, front.peak)
        final = front.result({})
        final["status"] = status
        del final["delivery"]
        report(number, "front_retired", edge, result=final)

    def frames():
        nonlocal shared_peak, shared_pairs
        period = clock_hz // 4_000_000
        arrivals = checked_offers()
        incoming = staged = pending = None
        free_at, next_frame = 0, 0

        def active_edge(front, edge, raw_received):
            nonlocal incoming, staged, pending, shared_peak, shared_pairs
            if staged is not None:
                if raw_received:
                    raise RuntimeError("B1 overlap reused the same normalized input port")
                if edge == staged.header_edge:
                    if (shared_suffix and front.input_released and front.payload_cfg is not None
                            and front.pending is None):
                        number = staged.candidate
                        younger = start_frame(number, staged.offer, edge, staged, None)
                        younger.use_suffix_after(front)
                        pending = number, younger
                        report(number, "suffix_overlap_started", edge, old_words=len(front.buffer))
                    else:
                        reject(staged.candidate, edge, "selected_suffix_busy",
                               sync_words=staged.index)
                    staged = None
                elif not staged.step(edge):
                    reject(staged.candidate, edge, "truncated_sync", sync_words=staged.index)
                    staged = None

            occupied = pending is not None
            if occupied:
                number, younger = pending
                if raw_received or younger.edge != edge:
                    raise RuntimeError("Shared B1 head did not own this input edge")
                previous_pairs = suffix_ram.paired
                younger._step()
                shared_pairs += suffix_ram.paired - previous_pairs
                shared_peak = max(shared_peak, len(front.buffer)+len(younger.buffer))
                if younger.failure is not None:
                    retire(number, younger, edge)
                    pending = None

            if incoming is None or incoming[1].first_edge > edge:
                return
            number, next_offer = incoming
            if next_offer.first_edge != edge:
                raise RuntimeError("Missed the original B1 sync offer edge")
            if occupied or staged is not None:
                reject(number, edge, "selected_sync_busy")
            elif raw_received or not front.input_released:
                reject(number, edge, "selected_input_busy")
            elif front.gain.result is None:
                reject(number, edge, "selected_gain_busy")
            else:
                staged = _SyncOffer(number, next_offer, period)
                report(number, "sync_overlap_started", edge, header_edge=staged.header_edge)
                if not staged.step(edge):
                    reject(number, edge, "truncated_sync", sync_words=staged.index)
                    staged = None
            incoming = next(arrivals, None)

        while True:
            if pending is not None:
                candidate, front = pending
                pending = None
                if front.edge != free_at:
                    raise RuntimeError("Shared B1 suffix was not promoted on its original timeline")
                front.start_carrier(engine_type)
                report(candidate, "suffix_overlap_promoted", free_at, words=len(front.buffer))
            else:
                if staged is None and incoming is None:
                    incoming = next(arrivals, None)
                    if incoming is None:
                        break
                prefix = staged
                if prefix is not None:
                    candidate, offer = prefix.candidate, prefix.offer
                    staged = None
                    edge = free_at
                    if edge > prefix.header_edge:
                        raise RuntimeError("Staged B1 prefix outlived its header deadline")
                    report(candidate, "sync_overlap_promoted", edge, sync_words=prefix.index)
                else:
                    candidate, offer = incoming
                    edge = offer.first_edge
                    incoming = next(arrivals, None) if sync_overlap else None
                    if edge < free_at:
                        reject(candidate, edge, "selected_front_busy", free_at=free_at)
                        continue
                front = start_frame(candidate, offer, edge, prefix, engine_type)

            callback = active_edge if sync_overlap else None
            first = front.next_payload(callback)
            if first is not None:
                report(candidate, "payload_selected", first.edge, frame=next_frame)
                next_frame += 1
                yield _payload_frame(front, first), _payload_words(front, first, callback)
            free_at = front.edge + int(front.edge == front.failure_edge)
            retire(candidate, front, free_at - 1)

    delivery = replay_b1_symbol_frames(
        frames(), clock_hz=clock_hz, word_cycles=word_cycles,
        observer=observer, **decoder_options,
    )
    result = dict(front=dict(outcomes), peak_suffix_words=peak, delivery=delivery)
    if shared_suffix:
        result["shared_suffix"] = dict(
            capacity=suffix_depth, peak_words=max(peak, shared_peak), peak_while_shared=shared_peak,
            reads=suffix_ram.reads, writes=suffix_ram.writes, paired_edges=suffix_ram.paired,
            paired_while_shared=shared_pairs,
        )
    return result
