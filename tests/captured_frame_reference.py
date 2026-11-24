"""Independent analytic IQ→actual source ports;expected bytes never enter receiver."""

import math
from dataclasses import asdict, replace
from unittest.mock import patch

import numpy as np
from scripts.model_coarse_ingress import legacy_flags, shared

from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_frame import CapturedFrameReplay
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.coarse_capture_source import CoarseCaptureSource
from nearlink_sdr.sim.spline_score import SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions
from tests.maximum_reference import channel
from tests.maximum_reference import packet as field_packet


def independent_preamble(profile):
    count = {2: 40, 3: 48, 4: 64}[profile.config.frame_type]
    return np.exp(1j * np.tile([np.pi / 4, 0], count // 2))


def packet(profile, payload, **damage):
    # Old research reference uses40 symbols for every FT. Keep its frozen bytes,
    # replace only this new generator's preamble with standard10/12/16us.
    fields = field_packet(profile, payload, **damage)
    return np.r_[independent_preamble(profile), fields[40:]]


def fixture(ft=2, *, size=31, mcs=None, pilots=0, bad_head=False, bad_data=False):
    cfg = shared.configuration(ft, (8 if ft == 2 else 12) if mcs is None else mcs, pilots, size)
    profile = replace(PSKReceiveProfile.ordinary(cfg), allowed_mcs=(cfg.mcs_index,))
    payload = bytes((k * 109 + 37) & 255 for k in range(size))
    symbols = packet(profile, payload, bad_head=bad_head, bad_data=bad_data)
    iq, quiet, _, step, _ = channel(
        symbols,
        offset=0.283,
        ppm=25,
        cfo=-33000,
        prefix=37,
        amplitude=0.14,
        family="clean",
        seed=108371,
        snr_db=None,
    )
    return profile, iq, quiet, payload, len(symbols), step


def receive(
    profile,
    iq,
    quiet,
    *,
    groups=(),
    stalls=(),
    defect=None,
    decoder_observer=None,
    retire_before_fine=False,
    retire_queued=False,
    overlap_fine=False,
    retirement_observer=None,
    fine_observer=None,
    copied_window_observer=None,
    coherence_observer=None,
):
    if coherence_observer is not None and not callable(coherence_observer):
        raise ValueError("Need a read-only coherence observer")
    filt, parts, pos, group = FullMantissaRRC(FixedRRCFormat(12, 18)), [], 0, 0
    while pos < len(iq):
        n = groups[group % len(groups)] if groups else len(iq)
        end = min(pos + n, len(iq))
        for j in range(pos, end, 32):
            parts.append(filt.feed_output_words(iq[j : min(j + 32, end)]))
        pos, group = end, group + 1
    components = [np.concatenate([x[k] for x in parts]) for k in range(3)]
    words = [pack_filter_word(*(int(x[j]) for x in components)) for j in range(len(iq))]
    flags = legacy_flags(
        PSKStreamReceiver(profile, quiet, quiet_batching=False, defer_quiet_fir=False), iq
    )
    producer = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=200000000)).simulate(
        len(words), trace=True
    )
    releases = [
        (e["cycle"], e["sample"], None if e["invalid"] else words[e["sample"]])
        for e in producer["trace"]
        if e["event"] == "output"
    ]
    if defect is not None:
        cycle, tag, _ = releases[defect]
        releases[defect] = cycle, tag, None
    source = CoarseCaptureSource(
        profile.sync_symbols(),
        iter(releases),
        iter((math.ceil(j * 200 / 16) + 10, j, bool(f)) for j, f in enumerate(flags)),
        sample_limit=len(words),
        source_horizon=producer["cycles"] + 8192,
        lanes=16 if profile.config.frame_type == 4 else 8,
    )
    worker = CaptureFineWorker(
        SyncSignTemplate.from_symbols(profile.sync_symbols()),
        source.flow.core.window,
        confirmation_required=True,
        reuse_symbol_power=True,
        defer_all_future=True,
        word_fabric=WordFabricAssumptions(2),
        gauss_difference=True,
        symbol_pipeline=True,
        capture_selection=True,
        sink=fine_observer,
    )
    frames, reads, outcomes = [], [], []
    observed_coherent = 0  # Caller-owned diagnostic cursor,not receiver/hardware state.

    def read(row):
        assert row["word"] == words[row["index"]], row
        reads.append(row)

    def outcome(event, mac):
        nonlocal observed_coherent
        if coherence_observer is not None:
            current = frame.stats["coherent_accepted"]
            delta = current - observed_coherent
            if delta not in (0, 1):
                raise RuntimeError("Coherence observer missed an actual terminal decision")
            coherence_observer(dict(id=frame.selection.ident, accepted=bool(delta)))
            observed_coherent = current
        frames.append(
            (
                event
                | dict(
                    upstream_selection=asdict(frame.selection),
                    observed_source_pulses_at_delivery=source.stats["rrc_pulses"],
                ),
                mac,
            )
        )

    frame = CapturedFrameReplay(profile, coherent_threshold=0.45, sink=outcome, raw_observer=read)
    transport = CaptureTransport(
        CaptureTransportAssumptions(paths=1, history=4096, window=source.flow.core.window)
    )

    def transport_outcome(event, window):
        if window is not None:
            end, size = event["after_samples"], source.flow.core.window
            assert window == words[end - size : end]
            if copied_window_observer is not None:
                copied_window_observer(event, window)
        outcomes.append(event)

    def calendar():
        return transport.simulate(
            None,
            (),
            source_owner=source,
            released_requests=True,
            fine_worker=worker,
            frame_replay=frame,
            retire_before_fine=retire_before_fine,
            retire_queued=retire_queued,
            overlap_fine=overlap_fine,
            retirement_observer=retirement_observer,
            stalls=stalls,
            sink=transport_outcome,
        )

    if decoder_observer is None:
        result = calendar()
    else:
        from nearlink_sdr.phy import rx_pipeline

        original = rx_pipeline.decode_payload

        def observe(*args, **kwargs):
            value = original(*args, **kwargs)
            decoder_observer(
                dict(
                    id=frame.selection.ident,
                    bits=value[0].copy(),
                    CRC=value[1],
                    MCS=args[1].mcs_index,
                )
            )
            return value  # The exact original tuple;no reference/decision changes.

        with patch.object(rx_pipeline, "decode_payload", observe):
            result = calendar()
    return result, frames, reads, frame, source, outcomes


def delivered(frames):
    return [p.mac_payload for _, m in frames for p in m.packets]
