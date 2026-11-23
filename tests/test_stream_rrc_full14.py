"""Opt-in compiled full14 RRC reaches real standard-length headers and strict MAC."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.rrc_full14 import Full14MantissaRRC
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from tests.full14_stream_reference import fixture, receive

MODE = "word-full14-c18"


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("mcs", [8, 12])
@pytest.mark.parametrize("pilots", [0, 4, 8, 16])
def test_standard_profile_unknown_origin_head_crc_length_mcs_payload_and_arbitrary_blocks(ft, mcs, pilots):
    p, x, q, payload, truth, step = fixture(ft, mcs, pilots)
    original = x.tobytes()
    arms = [receive(p, x, q, MODE, chunks) for chunks in ("whole", "fixed", "random")]
    for a in arms:
        assert a["before_EOF"] == 1 and [r.mac_payload for r in a["packets"]] == [payload]
        event, = a["events"]
        frame = event["frame"]
        assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
        assert frame["mcs"] == mcs and frame["payload_length_bytes"] == len(payload)
        assert frame["header"]["data_length"] == len(payload)+(p.config.crc_len//8 if ft >= 3 else 0)
        assert np.max(abs(event["symbol_centers_raw"]-
                          (truth+np.arange(len(event["symbols"]))*step))) <= .25
        rx = a["receiver"].phy
        assert isinstance(rx._rrc_word_model, Full14MantissaRRC) and rx._fir_state.size == 0
        assert rx._rrc_word_model.format.input_mantissa_bits == 14
        assert rx._rrc_word_model.product_bits == 31 and rx.fir_array_bytes == 1696
        assert rx._rrc_word_model.stats["samples"] == len(x)+rx.future_samples
        assert rx._rrc_word_model.stats["zero_flush_samples"] == rx.future_samples == 42
        assert rx._rrc_word_model.closed and not a["receiver"].finish().packets
    assert x.tobytes() == original
    for a in arms[1:]:
        assert a["events"][0]["frame"] == arms[0]["events"][0]["frame"]
        assert a["receiver"].phy.capture_stats == arms[0]["receiver"].phy.capture_stats
        for name in ["symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"]:
            np.testing.assert_array_equal(a["events"][0][name], arms[0]["events"][0][name])
        assert a["receiver"].phy._rrc_word_model.stats == arms[0]["receiver"].phy._rrc_word_model.stats


@pytest.mark.parametrize("change", [dict(quiet_batching=True), dict(defer_quiet_fir=True),
    dict(sps=8), dict(symbol_rate_mhz=2)])
def test_node_wiring_keeps_unverified_geometry_and_silence_skip_guards(change):
    p, _, q, *_ = fixture(2, 8, 0)
    config = {k: v for k, v in change.items() if k in ("sps", "symbol_rate_mhz")}
    p = replace(p, config=replace(p.config, **config))
    options = dict(rrc_backend=MODE, quiet_batching=False, defer_quiet_fir=False)
    options.update({k: v for k, v in change.items() if k not in config})
    with pytest.raises(ValueError, match="SPS4/4Msym"):
        PSKStreamReceiver(p, q, **options)


def test_original_default_and_full12_api_are_not_replaced_by_full14():
    p, _, q, *_ = fixture(2, 8, 0)
    default = PSKStreamReceiver(p, q)
    assert default.rrc_backend == "reference" and default._rrc_word_model is None
    old = PSKStreamReceiver(p, q, quiet_batching=False, defer_quiet_fir=False,
                            rrc_backend="word-full12-c18")
    assert old._rrc_word_model.format.input_mantissa_bits == 12
    assert not isinstance(old._rrc_word_model, Full14MantissaRRC)


def test_evidence_observer_sees_actual_readonly_words_and_cannot_change_receive_answers():
    p, x, q, payload, *_ = fixture(2, 8, 0)
    seen = []

    def observe(words):
        seen.append(words)
        assert all(not a.flags.writeable for a in words)
        with pytest.raises(ValueError):
            words[0][0] = 1

    a = receive(p, x, q, MODE, "random", rrc_sink=observe)
    assert [v.mac_payload for v in a["packets"]] == [payload]
    fields = tuple(np.concatenate([row[k] for row in seen]) for k in range(3))
    actual = np.ldexp(fields[0].astype(float), fields[2])+1j*np.ldexp(
        fields[1].astype(float), fields[2])
    independent_instance = Full14MantissaRRC()
    # Observe the stream's42virtual samples,not standalone finish's80zeros.
    expected = independent_instance.feed(np.r_[x, np.zeros(42)])
    np.testing.assert_array_equal(actual, expected)
    assert sum(len(row[0]) for row in seen) == len(x)+42
