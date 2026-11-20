"""Real head word/CRC clocks inside selected replay;no whole continuous-RX claim."""

from nearlink_sdr.phy.word_head_rx import WordHeadPSKReceiver
from nearlink_sdr.sim.overlapped_normalization_replay import OverlappedNormalizationFrameReplay
from nearlink_sdr.sim.psk_hard_flow import HeaderFromWordsService

__all__ = ["HeadWordsFrameReplay"]


class HeadWordsFrameReplay(OverlappedNormalizationFrameReplay):
    """One physical head context shares the transport epoch;no new raw suffix bank."""

    def __init__(
        self,
        profile,
        prepared,
        *,
        head_layout="dense",
        head_observer=None,
        timing_divider="bounded-prefix",
        **options,
    ):
        self.head_owner = HeaderFromWordsService(
            profile, layout=head_layout, observer=head_observer
        )
        self.head_owner.begin()
        super().__init__(profile, prepared, timing_divider=timing_divider, **options)
        self.stats.update(dict.fromkeys(["finite_head_edges", "finite_head_completions"], 0))

    def _make_consumer(self, profile, initial_fit, word_timing_format, selection):
        return WordHeadPSKReceiver(
            profile,
            initial_fit=initial_fit,
            normalization_format=self.normalization_format,
            word_timing_format=word_timing_format,
            head_owner=self.head_owner,
            **selection,
        )

    def _plan(self):
        if self.timing_service is not None and self.consumer.boundary_kind == "header":
            self.mode = "word_header"
            self.pending_service = None
        else:
            super()._plan()

    def step(self):
        if self.done:
            return
        if self.head_owner.cycle != self.cycle - self.service_epoch:
            raise RuntimeError("Head and transport must use the same actual edge")
        if self.mode == "word_header":
            self._count("finite_head_edges")
            if self.head_owner.ready:
                event = self.consumer.accept_header_result(after_samples=self.history.end)
                self._count("finite_head_completions")
                self._count("header_operations")
                if event is not None:
                    self._close(event)
                else:
                    self._plan()
            self.kernel.advance()
            self.timing_service.step()
            self.query_service.step()
            self.normalization_service.step()
        else:
            super().step()
        self.head_owner.step()

    def _close(self, event):
        state = self.head_owner.snapshot()
        cancelled = self.head_owner.mode != "idle"
        if cancelled:
            self.head_owner.abort()
        super()._close(event)
        event["finite_word_head"] = dict(
            state=state,
            cancelled=cancelled,
            exact_edges_executed=True,
            new_raw_history=0,
            payload_cycles=None,
            full_receiver16MSps=False,
        )

    def estimate(self):
        result = super().estimate()
        # Both new64bit counters are already counted through len(self.stats).
        result["new_controller_state_bits"] += 1  # One receipt-ownership flag.
        return result | dict(
            finite_word_head=self.head_owner.cost(),
            head_controller_increment_bits=2 * 64 + 1,
            head_increment_counters_already_in_controller=True,
            head_mode_reuses_existing4bit_field=True,
            new_raw_history=0,
            payload_cycles=None,
            whole_receiver16MSps=False,
        )
