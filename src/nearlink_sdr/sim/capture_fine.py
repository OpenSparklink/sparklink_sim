"""Owned raw46-bit capture windows → existing exact fine microtrace (§6.2.3).

The host invokes the bounded microtrace at handoff and reserves its returned
latency. No result is released early. Continuous writes/copying proceed on the
outer calendar while the owned source image remains immutable. This is not a
complete RX,downstream replay port schedule,or new timing/synthesis evidence.
"""

import operator

import numpy as np

from nearlink_sdr.sim.capture_transport import CaptureRequest, FineService
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SplineScoreFlow,
    SyncSignTemplate,
    spline_score_cost,
)

__all__ = ["CaptureFineWorker", "pack_filter_word", "unpack_filter_word"]


def _signed(value, bits):
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def pack_filter_word(real, imag, exponent):
    """Literal signed19+19+8 fields; never quantize or wrap an out-of-range input."""
    values = []
    for v, b in zip((real, imag, exponent), (19, 19, 8), strict=True):
        if isinstance(v, (bool, np.bool_)):
            raise ValueError("Need integer filter word fields")
        try:
            v = operator.index(v)
        except TypeError as error:
            raise ValueError("Need integer filter word fields") from error
        if not -(1 << (b - 1)) <= v < 1 << (b - 1):
            raise ValueError("Filter field exceeds signed width")
        values.append(v & ((1 << b) - 1))
    r, i, e = values
    return r | (i << 19) | (e << 38)


def unpack_filter_word(word):
    """Decode the serialized actual46-bit source interface,independent of a profile."""
    if isinstance(word, (bool, np.bool_)):
        raise ValueError("Need unsigned46-bit filter word")
    try:
        word = operator.index(word)
    except TypeError as error:
        raise ValueError("Need unsigned46-bit filter word") from error
    if not 0 <= word < 1 << 46:
        raise ValueError("Need unsigned46-bit filter word")
    return (
        _signed(word & ((1 << 19) - 1), 19),
        _signed((word >> 19) & ((1 << 19) - 1), 19),
        _signed(word >> 38, 8),
    )


class CaptureFineWorker:
    """One configured sync template,one compiled window,one active fine engine.

    `sink` records host evidence outside hardware,not a retained request history.
    No payload,CRC,true timing or true carrier is an input to this worker.
    Raw RAM is the borrowed immutable owned slot; workingY/M banks are additional.
    """

    def __init__(
        self, template, window, *, confirmation_required, sink=None, reuse_symbol_power=False
    ):
        if not isinstance(template, SyncSignTemplate) or (len(template.words) + 1, window) not in (
            (32, 256),
            (62, 376),
            (126, 632),
        ):
            raise ValueError("Need actual FT2/3/4 sync/window geometry")
        if sink is not None and not callable(sink):
            raise ValueError("Need a host diagnostic sink")
        if type(confirmation_required) is not bool:
            raise ValueError("Need explicit confirmation-required mode")
        self.template, self.window, self.sink = template, window, sink
        self.confirmation_required = confirmation_required
        self.engine = SplineScoreFlow(
            SplineScoreAssumptions(capacity=window, reuse_symbol_power=reuse_symbol_power)
        )

    def __call__(self, event, owned_words):
        request = event.get("geometry")
        if (
            not isinstance(request, CaptureRequest)
            or len(owned_words) != self.window
            or event["window_samples"] != self.window
            or event["after_samples"] != request.after_samples
            or event["path"] != request.path
        ):
            raise ValueError("Need the same admitted bounded request and owned source image")
        words = [unpack_filter_word(w) for w in owned_words]
        result = self.engine.run(
            words,
            self.template,
            request.peaks,
            minimum_tick=request.minimum_tick,
            lookahead_ticks=request.lookahead_ticks,
        )
        if self.sink is not None:
            self.sink(event, result)
        if result.get("resource_drop"):
            raise RuntimeError("Fine microtrace resource abort;not a numerical/CRC failure")
        best = result["best"]
        if result.get("numerical_rejection"):
            disposition, tick = "numerical_reject", None
        elif best is None:
            disposition, tick = "no_candidate", None
        else:
            tick = best["start_tick"]
            disposition = (
                "needs_future"
                if best["disposition"] != "ready"
                else "confirm_pending"
                if self.confirmation_required
                else "ready"
                if best["threshold_pass"]
                else "below_threshold"
            )
        return FineService(result["cycles"], disposition, tick)

    def estimate(self):
        cost = spline_score_cost(self.engine.a)
        return dict(
            fine=cost,
            raw_source_is_same_owned_slot=True,
            raw_read_ports=1,
            raw_read_latency=1,
            raw_word_bits=46,
            raw_write_after_copy=False,
            no_copy_fine_port_overlap_same_slot=True,
            preparation_DSP25x18_estimate=cost["shared_preparation_banks"][
                "unrolled_row_peak_scenario"
            ]["dsp25x18_product_tiles"],
            scoring_DSP25x18_estimate=cost["shared_DSP25x18_count"],
            multiplier_sharing_credit=0,
            note="dedicated preparation and scoring estimate;row macro timing still assumed;do not claim one DSP for both phases without a shared datapath proof",
            selected_result_scalar_bits=32 + 13 + 4,
            no_claim_on_downstream_replay_ports=True,
            confirmation_required=self.confirmation_required,
            coherent_confirmation_scheduled=False,
            complete_receiver_verified=False,
            actual_synthesis_or_timing=False,
        )
