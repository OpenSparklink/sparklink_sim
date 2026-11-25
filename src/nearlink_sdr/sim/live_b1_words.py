"""One actual matched-filtered sample owner through finite B1 carrier and MAC."""

from nearlink_sdr.phy.normalization_word import NormalizationFormat, unpack_normalized_word
from nearlink_sdr.phy.timing_word import ONE, TimingWordFormat
from nearlink_sdr.sim.carrier_pi_flow import ParallelPICarrierEngine
from nearlink_sdr.sim.packed_head_replay import PackedHeadSamplingReplay
from nearlink_sdr.sim.psk_selected_carrier import (
    _payload_frame,
    _payload_words,
    _SelectedFrame,
    _SuffixRAM,
    _validate_front,
)
from nearlink_sdr.sim.psk_word_demap import replay_b1_symbol_frames
from nearlink_sdr.sim.split_sampling_replay import _SplitReceiver

__all__ = ["replay_live_b1_words"]

_END = object()
_STARTUP_EDGES = 12
_ADC_HZ = 16_000_000


class _LiveWordReceiver(_SplitReceiver):
    """Retain only the bounded sync diagnostics and one registered output word."""

    output_word = None

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self._values[:] = self.reference  # Only completed sync entries are diagnostic samples.

    def _symbol_trace_size(self):
        return len(self.reference)

    def _event(self, row, *, end_padding=False):
        complete_sync = self._active["count"] >= len(self.reference)
        event = super()._event(row, end_padding=end_padding)
        if not complete_sync:
            event["sync_evm"] = None
        return event

    def _store_normalized_word(self, source_word, output_word, after_samples):
        if self.output_word is not None:
            raise RuntimeError("The B1 owner did not retire the registered output")
        active = self._active
        index = active["count"]
        if index < len(self.reference):
            super()._store_normalized_word(source_word, output_word, after_samples)
        else:
            self._observe(after_samples)
            active.update(count=index + 1, previous_t=active["next"],
                          next=self.word_timing.next_q32 / ONE)
            self._offer_header_word(index, output_word)
        self.output_word = index, output_word

    def _check_selected_boundary(self):
        raise RuntimeError("The finite B1 owner must resolve payload, without reference decoding")


class _Sampling(PackedHeadSamplingReplay):
    _consumer_type = _LiveWordReceiver

    def _join(self):
        if self.consumer.output_word is None:
            super()._join()

    def complete_transfer(self):
        if not self.pending_service or self.pending_service["kind"] != "payload":
            raise RuntimeError("B1 transfer ended before the strict source extent")
        row = self.consumer.header_result | dict(status="payload_transferred", data_crc=None)
        self._close(self.consumer._event(row))


class _LiveB1Frame(_SelectedFrame):
    """Drive the actual source and B2 on the same edge, without a second header."""

    def __init__(self, flow, words, *, clock_hz, continuous, depth, front_observer):
        super().__init__(
            words, flow.consumer.profile, flow.cycle, clock_hz, depth, None,
            front_observer, engine_type=ParallelPICarrierEngine, suffix_ram=_SuffixRAM(depth),
            head_owner=flow.head_owner,
        )
        self.flow = flow
        self.source_start = self.edge
        self.clock_hz, self.continuous = clock_hz, continuous
        self.source_words = 0
        self.source_failed = None
        self.source_exhausted = False
        self.sync_wait = None
        self.sync_wait_peak = 0
        self.receipt = None
        self.first_data_edge = None
        self.output_hold_edges = 0

    def _drive_sampling(self):
        flow = self.flow
        if flow.done or flow.pending_service is not None:
            return
        if flow.cycle != self.edge:
            raise RuntimeError("B1 and its source lost the shared clock")
        if not self.source_exhausted:
            due = ((self.edge - self.source_start) * _ADC_HZ >= self.source_words * self.clock_hz
                   if self.continuous else flow.wants_replay_input)
            if due:
                word = next(self.source, _END)
                if word is _END:
                    flow.end_input()
                    self.source_exhausted = True
                elif flow.write_input(word):
                    self.source_words += 1
                else:
                    self.source_failed = (self.edge, flow.history.end)
        if not flow.done:
            if self.head.ready and flow.consumer.boundary_kind == "header":
                self.receipt = dict(self.head.result)  # One broadcast of the registered receipt.
            flow.step()
        if flow.pending_service is not None and flow.pending_service["kind"] != "payload":
            raise RuntimeError("Unexpected unclocked operation in live B1 sampling")

    def _take_header(self):
        if self.receipt is not None:
            row, self.receipt = self.receipt, None
            if row["status"] == "header_valid" and not row["data_crc_present"]:
                row["status"] = "unsupported_payload_path"
            self._install_header(row)

    def _receive(self):
        consumer = self.flow.consumer
        if consumer.output_word is None:
            self.eof = self.flow.pending_service is not None
            return
        index, word = consumer.output_word
        if index >= self.nsync and len(self.buffer) == self.depth:
            self.output_hold_edges += 1
            return
        consumer.output_word = None
        if index != self.index:
            raise RuntimeError("B1 source lost its ordered normalized word")
        self.index += 1
        self.pending = index, *unpack_normalized_word(word)
        self._send("normalized_word", index=index, word=word)
        if index == self.data_start:
            self.first_data_edge = self.edge

    def _convert(self):
        engine = self.gain.engine
        if self.sync_wait is not None and engine.input_ready:
            self.source_saturations += self.gain.accept(self.edge, *self.sync_wait)
            self.sync_wait = None
        if self.pending is not None and self.pending[0] < self.nsync and not engine.input_ready:
            if self.sync_wait is not None:
                self._fail("sync_gain_fifo_full")
                return
            self.sync_wait, self.pending = self.pending, None
            self.sync_wait_peak = 1
            return
        super()._convert()

    def _launch(self):
        if (self.buffer and self.buffer.first_index >= self.data_start
                and self.first_payload_launch is None
                and self.edge < self.first_data_edge + _STARTUP_EDGES):
            return
        super()._launch()

    def _step(self):
        edge = self.edge
        self._drive_sampling()
        output = super()._step()
        if self.failure is None and self.flow.done:
            self._fail(self.flow.event["frame"]["status"])
        if self.failure is None and self.next_launch == edge:
            self.missed_launch_edge = edge
            self._fail("carrier_pace_missed")
        return output if self.failure is None else None


def replay_live_b1_words(
    profile, prepared, words, *, selection, clock_hz, word_cycles, continuous,
    suffix_depth=8, front_observer=None, **decoder_options,
):
    """Consume one chosen matched-filtered B1 frame and its actual timed source.

    Raw input is already filtered and the owned sync/fit prefix is prepared.
    `continuous` writes further words at 16 MS/s and records finite overflow.
    With `False`, requested words alone are replayed. The iterator retains any
    untouched suffix. A rejected continuous write records its absolute address.
    """
    _validate_front(profile, clock_hz, 0, suffix_depth)
    if type(continuous) is not bool:
        raise ValueError("Need an explicit continuous or pauseable source")
    flow = _Sampling(
        profile, prepared, error_age=1, normalization_format=NormalizationFormat(),
        word_timing_format=TimingWordFormat(30, 32), reuse_stencil=True,
        compact_query=True, start_stencil_on_reply=True, **selection,
    )
    front = _LiveB1Frame(
        flow, words, clock_hz=clock_hz, continuous=continuous, depth=suffix_depth,
        front_observer=front_observer,
    )
    first = front.next_payload()
    delivery = {}
    if first is not None:
        delivery = replay_b1_symbol_frames(
            [_payload_frame(front, first)], _payload_words(front, first),
            clock_hz=clock_hz, word_cycles=word_cycles, **decoder_options,
        )
    if not flow.done:
        if front.failure is None:
            flow.complete_transfer()
        else:
            flow.abort("cancelled")
    return front.result(delivery) | dict(
        input_from_prepared_filtered_source=True, continuous_filtered_source=continuous,
        post_prepare_raw_words=front.source_words, first_rejected_raw=front.source_failed,
        first_data_input_edge=front.first_data_edge, startup_margin_edges=_STARTUP_EDGES,
        sync_wait_peak=front.sync_wait_peak, head_decodes=flow.head_owner.stats["header_issues"],
        normalized_output_hold_edges=front.output_hold_edges,
        source_history_peak=flow.history.stats["peak_live"],
        source_terminal_status=flow.event["frame"]["status"],
        normalized_outputs=flow.stats["joined_symbols"],
        suffix_ram_reads=front.buffer.ports.reads, suffix_ram_writes=front.buffer.ports.writes,
        reference_trace_symbols=len(flow.consumer._values),
        functional_resolutions=flow.stats["functional_resolutions"],
        whole_receiver16MSps=False,
    )
