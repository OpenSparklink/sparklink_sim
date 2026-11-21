"""Actual unknown IQ→MAC and coarse words,including the Q12 ranked bank."""

import numpy as np
import pytest

from nearlink_sdr.phy import rrc_bridge, stream_rx
from nearlink_sdr.phy.acquisition_scaled import quantize_scaled_words
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


def floating_reference(real, imag, exponents, fmt, *, stats):
    values = np.ldexp(real.astype(float), exponents) + 1j * np.ldexp(imag.astype(float), exponents)
    return quantize_scaled_words(values, fmt, stats=stats)


def consume(profile, iq, quiet, ranked, bits, chunks):
    rx = MacStreamReceiver(
        profile,
        quiet,
        quiet_batching=False,
        adaptive_quiet=False,
        defer_quiet_fir=False,
        rrc_backend="word-full12-c18",
        correlation_backend="scaled",
        correlation_mantissa_bits=bits,
        timing_backend="auto",
        integer_screen_backend="python",
        sync_search_method="bounded-grid" if ranked else "reference",
        normalized_coarse=ranked,
        sync_coherent_threshold=0.45 if ranked else None,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Actual raw RRC/prescreen path used a floating round trip")

    rx.phy._rrc_word_model.feed = forbidden
    rx.phy.correlation_bank.quantize = forbidden
    packets, events, pos = [], [], 0
    rng = np.random.default_rng(372719)
    while pos < len(iq):
        size = len(iq) if chunks == "whole" else int(rng.integers(1, 258))
        result = rx.feed(iq[pos : pos + size])
        packets += result.packets
        events += result.diagnostics["events"]
        pos += size
    result = rx.finish()
    return packets + result.packets, events + result.diagnostics["events"], rx


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (3, 12, 8), (4, 10, 4)])
@pytest.mark.parametrize("ranked", [False, True])
@pytest.mark.parametrize("bits", [10, 14])
def test_actual_standard_frame_and_all_coarse_states_match_old_adapter(
    ft, mcs, pilot, ranked, bits, monkeypatch
):
    profile = PSKReceiveProfile.ordinary(
        TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            symbol_rate_mhz=4,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=0x315791 if ft != 2 else None,
            crc_seed=0x316171,
            whitening_seed=0x53,
        )
    )
    wire, payload = transmitted(profile, length=43)
    iq, quiet, _, _ = iq_record(wire, cfo=-27000, ppm=100, offset=0.329)
    with monkeypatch.context() as context:
        context.setattr(rrc_bridge, "rrc_to_scaled_words", floating_reference)
        old, old_events, old_rx = consume(profile, iq, quiet, ranked, bits, "whole")
    new, new_events, new_rx = consume(profile, iq, quiet, ranked, bits, "random")
    assert [p.mac_payload for p in old] == [p.mac_payload for p in new] == [payload]
    assert len(old_events) == len(new_events) == 1
    assert old_events[0]["frame"] == new_events[0]["frame"]
    frame = new_events[0]["frame"]
    assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
    assert frame["mcs"] == mcs
    for key in ("symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(old_events[0][key], new_events[0][key])
    assert old_rx.phy.correlation_bank.stats == new_rx.phy.correlation_bank.stats
    assert old_rx.phy._rrc_word_model.stats == new_rx.phy._rrc_word_model.stats
    assert new_rx.phy.timing_backend == ("rust" if stream_rx._RustStreamTiming else "python")
    for name, old_value in old_rx.phy.correlation_bank.__dict__.items():
        if isinstance(old_value, np.ndarray):
            np.testing.assert_array_equal(old_value, getattr(new_rx.phy.correlation_bank, name))
    assert not new_rx.finish().packets


@pytest.mark.parametrize(
    "damage,status",
    [(dict(bad_head=True), "bad_header_crc"), (dict(bad_data=True), "bad_data_crc")],
)
def test_bad_actual_CRCs_remain_failures_with_no_MAC_delivery(damage, status):
    profile = PSKReceiveProfile.ordinary(
        TxConfig(
            mcs_index=12,
            pilot_interval=0,
            symbol_rate_mhz=4,
            crc_seed=0x316171,
            whitening_seed=0x53,
        )
    )
    wire, _ = transmitted(profile, length=47, **damage)
    iq, quiet, _, _ = iq_record(wire)
    packets, events, _ = consume(profile, iq, quiet, True, 10, "random")
    assert not packets and len(events) == 1 and events[0]["frame"]["status"] == status
    assert events[0]["frame"]["payload"] is None
