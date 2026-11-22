"""Known word-head latency on the actual captured-frame calendar (§6.3/6.9).

This replays the measured finite HeaderWordService receipt, not a second decoder.
Tracking/demap still resolve off-clock at the hard-input boundary. The positive
service then blocks consumption and public CRC/length/terminal availability.
Unknown arithmetic is not assigned zero cycles; absolute RX latency stays unknown.
"""

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.capture_frame import CapturedFrameReplay
from nearlink_sdr.sim.header_word_flow import HeaderWordService

__all__ = ["ClockedHeaderCapturedFrameReplay"]


class ClockedHeaderCapturedFrameReplay(CapturedFrameReplay):
    """One selected frame, one borrowed head receipt and finite registered wait.

    Hard bits are modeled as available together after functional demap. No credit
    is taken for overlapping descramble/SSC/CRC with their arrival. The reference
    receiver's private early receipt is not a scheduling decision: no further raw
    word or public head/terminal can pass this gate until the known service ends.
    """

    def __init__(self, profile, **options):
        if not isinstance(profile, PSKReceiveProfile) or profile.head_backend not in (
            "word-dense",
            "word-depth-parity",
        ):
            raise ValueError("Need an actual finite word-head backend,not unknown reference timing")
        super().__init__(profile, **options)
        self.header_clock_stats = dict(issued=0, completed=0, cancelled=0)
        self._clear_header_clock()

    def _clear_header_clock(self):
        self._head_issue = self._head_ready = self._head_available = None
        self._head_edges = 0
        self._held_head = self._held_terminal = None

    def _count_header(self, name):
        if self.header_clock_stats[name] == 2**64 - 1:
            raise OverflowError("Header calendar statistic exhausted")
        self.header_clock_stats[name] += 1

    @property
    def header_waiting(self):
        return self._held_head is not None

    @property
    def needs_word(self):
        return super().needs_word and not self.header_waiting

    def _accept_reference_results(self, events):
        row = self.rx.header_result
        if self.header is None and row is not None and "head_word_model" in row:
            if self._head_issue is not None or len(events) > 1:
                raise RuntimeError("One selected frame may own only one actual header receipt")
            edges = row["head_word_model"]["service_edges"]
            if type(edges) is not int or not 0 < edges <= 10000:
                raise RuntimeError("Actual word-head receipt needs positive bounded service edges")
            if self.cycle + edges >= 2**63 - 1:
                raise OverflowError("Header availability exceeds finite clock")
            if row.get("data_crc") is not None or row.get("payload") is not None:
                raise RuntimeError("Header boundary must precede payload resolution")
            self._count_header("issued")
            self._head_issue, self._head_edges = self.cycle, edges
            self._head_ready = self.cycle + edges
            self._held_head, self._held_terminal = row, events[0] if events else None
            return
        super()._accept_reference_results(events)

    def tick(self, cycle, *, blocked=False):
        if not self.header_waiting:
            return super().tick(cycle, blocked=blocked)
        self._edge(cycle)
        if type(blocked) is not bool:
            raise ValueError("Need explicit frame consumer readiness")
        if cycle < self._head_ready or blocked:
            return
        self._count_header("completed")
        self._head_available = cycle
        terminal, self._held_terminal = self._held_terminal, None
        self._held_head = None
        # The same receipt is still owned by rx; no head re-decode or answer retry.
        super()._accept_reference_results([] if terminal is None else [terminal])

    def end_input(self, cycle):
        if self.header_waiting:
            raise RuntimeError("Drain actual known header service before EOF")
        return super().end_input(cycle)

    def abort(self, reason, cycle):
        if reason not in ("tail_history_expired", "source_epoch_aborted", "cancelled"):
            raise ValueError("Need explicit bounded-source abort reason")
        if self.active and not self.done and self.header_waiting:
            self._edge(cycle)
            self._count_header("cancelled")
            self._held_head = self._held_terminal = None
        return super().abort(reason, cycle)

    def _terminal(self, event):
        event["header_service_calendar"] = dict(
            issue_cycle=self._head_issue,
            nominal_ready_cycle=self._head_ready,
            available_cycle=self._head_available,
            service_edges=self._head_edges,
            execution="latency replay of actual integer header receipt;parallel hard input",
            head_demap_cycles=None,
            complete_cycle_evidence_valid=False,
        )
        super()._terminal(event)

    def release(self):
        super().release()
        self._clear_header_clock()

    def estimate(self):
        result = super().estimate()
        fields = dict(
            issue_ready_available=3 * 63, service_edges=14, valid_flags=2, statistics=3 * 64
        )
        bits = sum(fields.values())
        base = result["register_bits_upper"]
        result.update(
            reference_owner_register_bits=base,
            register_fields=result["register_fields"]
            | {"header_calendar_" + name: width for name, width in fields.items()},
            register_bits_upper=base + bits,
            header_calendar_state_bits=fields,
            header_calendar_register_bits=bits,
            combined_owner_register_bits=base + bits,
            header_calendar_research_cap_bits=512,
            header_calendar_capacity_met=bits <= 512,
            header_calendar_control=dict(
                add_bits=63, compare_bits=63, control_edge_assumption=1, Fmax_verified=False
            ),
            header_service=HeaderWordService(
                self.profile, layout=self.profile.head_backend.removeprefix("word-")
            ).cost(),
            header_service_execution="actual receipt latency replay;no duplicate physical decoder",
            borrowed_receipt_and_terminal=True,
            header_demap_cycles=None,
            additional_raw_RAM_DSP_ports=0,
        )
        return result
