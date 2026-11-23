"""Independent ordinary field/pulse reference and read-only strict MAC collection.

Expected payload and timing truth never enter receive(). It accepts only the
preselected profile,IQ,quiet sample statistics and a fixed frontend selection.
"""

from dataclasses import replace

import numpy as np
from scripts.model_scaled_maximum import configuration

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tone_notch import ToneNotchConfig
from tests.captured_frame_reference import packet
from tests.maximum_reference import channel

OPTIONS = dict(quiet_batching=False, defer_quiet_fir=False, correlation_backend="scaled",
               correlation_mantissa_bits=10, sync_search_method="reference", timing_subdivisions=8)


def profile(ft, mcs, pilots, index=3157):
    return replace(PSKReceiveProfile.ordinary(configuration(ft, mcs, pilots, index)),
                   head_backend="word-depth-parity")


def fixture(ft, mcs, pilots, *, size=43, index=3157, damage=None):
    p = profile(ft, mcs, pilots, index)
    payload = bytes(((j*j*17+j*61+ft*23) ^ (j >> 2)) & 255 for j in range(size))
    wire = packet(p, payload, **(damage or {}))
    x, quiet, legacy_truth, step, _ = channel(wire, offset=.193+ft*.137, ppm=100 if mcs == 8 else -100,
        cfo=73000 if ft % 2 else -47000, prefix=47+ft*13, amplitude=.17,
        family="clean", seed=1033127+ft*11+mcs)
    truth = legacy_truth+({2: 40, 3: 48, 4: 64}[ft]-40)*step
    return p, x, quiet, payload, truth, step


def receive(p, iq, quiet, backend, chunks, *, rrc_sink=None):
    if rrc_sink is not None and not callable(rrc_sink):
        raise ValueError("Need a read-only RRC observer")
    options = dict(OPTIONS, tone_notch=ToneNotchConfig(radius=.96875))
    if backend != "reference":
        options["rrc_backend"] = backend
    rx = MacStreamReceiver(p, quiet, **options)
    if rrc_sink is not None and rx.phy._rrc_word_model is not None:
        original = rx.phy._rrc_word_model.feed_output_words

        def observe(samples):
            result = original(samples)
            copies = tuple(a.copy() for a in result)
            for a in copies:
                a.setflags(write=False)
            rrc_sink(copies)
            return result

        rx.phy._rrc_word_model.feed_output_words = observe
    events, packets, pos, peak = [], [], 0, 0
    rng = np.random.Generator(np.random.Philox(1921321))
    while pos < len(iq):
        n = (len(iq) if chunks == "whole" else 257 if chunks == "fixed" else
             int(rng.integers(1, 510)))
        result = rx.feed(iq[pos:pos+n])
        packets.extend(result.packets)
        events.extend(result.diagnostics["events"])
        peak = max(peak, rx.phy.buffered_samples)
        # The pre-existing notch option keeps both raw and suppressed search
        # histories. Charge each unchanged path;do not apply a one-path bound.
        phy = rx.phy
        if len(phy._filtered) > phy.history_limit or len(phy._tone_filtered) > phy.history_limit:
            raise RuntimeError("One search path exceeded its unchanged history limit")
        limit = 2*phy.history_limit+80+len(phy._box)-1+(phy.quantum-1)
        if phy.buffered_samples > limit:
            raise RuntimeError("Stream sample history exceeded its pre-existing finite bound")
        pos += n
    before = len(packets)
    result = rx.finish()
    packets.extend(result.packets)
    events.extend(result.diagnostics["events"])
    return dict(receiver=rx, events=events, packets=packets, before_EOF=before,
                observed_peak_samples=peak)
