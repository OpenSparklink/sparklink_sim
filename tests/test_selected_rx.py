"""One upstream choice,realCRC boundary and causal filtered-port regression (§6.3/6.9)."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.phy.mac_interface import MacSelectedReceiver
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig, head_crc_parameters
from tests.test_strict_receive import iq_record, transmitted


def ordinary(ft=2, mcs=8, pilot=0):
    return PSKReceiveProfile.ordinary(
        TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            symbol_rate_mhz=4,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=0x234567 if ft != 2 else None,
        )
    )


def selected_input(profile, *, length=37, **damage):
    """Independent Tx pulse;one actual differential maximum,without payload/CRC selection.

    Whole-record upstream search is a test fixture only,not a streaming capture claim.
    The tested port starts AFTER selection. Filtering uses the existing matched RRC.
    """
    wire, expected = transmitted(profile, length=length, **damage)
    iq, _, _, _ = iq_record(wire, cfo=27000, ppm=75, offset=0.319)
    values = np.convolve(iq, rrc_filter(0.4, 4))
    found = search_psk_sync(values, 4, profile.sync_symbols(), 8)
    assert found is not None
    offset = found["offset"]
    selected_sync = found["phase_symbols"][offset : offset + len(profile.sync_symbols())]
    selection = dict(
        start_tick=round(8 * (found["phase"] + offset * 4)),
        delta=found["delta"],
        differential_score=found["score"],
        selected_sync=selected_sync,
        sample_limit=len(values),
        coherent_threshold=0.45,
    )
    return values, expected, selection


def consume(profile, values, selection, pattern, backend="python"):
    rx = SelectedPSKReceiver(profile, **selection, timing_backend=backend)
    events, pos, j = [], 0, 0
    while pos < len(values) and not rx.done:
        n = int(pattern[j % len(pattern)])
        events.extend(rx.feed_filtered(values[pos : pos + n]))
        pos += rx.last_input_consumed
        j += 1
    events.extend(rx.finish())
    return events, rx


def equal(a, b, *, backend=False):
    assert a["frame"] == b["frame"]
    assert a["delivered_after_filtered_samples"] == b["delivered_after_filtered_samples"]
    assert a["sync_start_filtered"] == b["sync_start_filtered"]
    for key, tolerance in (
        ("symbol_centers_filtered", 2e-9),
        ("symbols", 2e-9),
        ("timing_rate_ppm", 2e-6),
        ("timing_detector_error", 2e-9),
    ):
        if backend:
            np.testing.assert_allclose(a[key], b[key], atol=tolerance, rtol=0)
        else:
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize(
    "ft,mcs,pilot,length",
    [
        (2, 8, 0, 1),
        (2, 12, 8, 137),
        (2, 6, 0, 73),
        (3, 8, 8, 31),
        (3, 12, 0, 227),
        (4, 0, 0, 43),
        (4, 12, 8, 197),
    ],
)
def test_actual_header_boundary_and_arbitrary_blocks(ft, mcs, pilot, length):
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=length)
    values = np.r_[values, np.zeros(10003)]  # Caller-owned irrelevant continuation.
    selection["sample_limit"] = len(values)
    (reference,), rx = consume(profile, values, selection, [len(values)])
    assert reference["frame"]["payload"] == payload
    assert reference["frame"]["header_crc"] and reference["frame"]["data_crc"]
    assert reference["frame"]["header"]["data_length"] == length
    assert reference["frame"]["length_ok"] and reference["frame"]["mcs"] == mcs
    assert len(reference["symbols"]) == rx.frame_end_symbol
    assert rx.input_samples < len(values) - 10003
    assert not reference["used_end_padding"]
    assert rx.feed_filtered(np.ones(3)) == [] and rx.last_input_consumed == 0
    for pattern in ([1], [7, 61], np.random.default_rng(881118).integers(1, 257, 29)):
        (actual,), other = consume(profile, values, selection, pattern)
        equal(reference, actual)
        assert other.input_samples == rx.input_samples
        assert other.reference_memory()["retained_filtered_bytes"] == len(values) * 16


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_real_crc_rejection_never_delivers_or_retries(damage, status):
    profile = ordinary()
    values, _, selection = selected_input(profile, **damage)
    rx = MacSelectedReceiver(profile, **selection, timing_backend="python")
    result = rx.feed_filtered(values)
    assert not result.packets
    assert [r["status"] for r in result.diagnostics["frames"]] == [status]
    row = result.diagnostics["frames"][0]
    assert row["payload"] is None
    if status == "bad_header_crc":
        assert row["header_crc"] is False and row["data_crc"] is None
        assert "header" not in row and rx.phy.frame_end_symbol is None
    assert rx.finish().diagnostics["events"] == []


@pytest.mark.parametrize(
    "mode,status",
    [
        ("wrong_seed", "bad_header_crc"),
        ("length", "invalid_length"),
        ("mcs", "unsupported_mcs"),
    ],
)
def test_untrusted_header_cannot_supply_a_frame_length(mode, status):
    profile = ordinary(3, 8)
    values, _, selection = selected_input(profile, mcs=15 if mode == "mcs" else None)
    if mode == "wrong_seed":
        profile = replace(profile, config=replace(profile.config, head_llid=7))
    elif mode == "length":
        profile = replace(profile, max_payload_bytes=32)
    (event,), rx = consume(profile, values, selection, [13])
    assert event["frame"]["status"] == status and event["frame"]["payload"] is None
    assert rx.frame_end_symbol is None
    if mode != "wrong_seed":
        assert event["frame"]["header_crc"] is True


@pytest.mark.parametrize(
    "kept,status,head",
    [
        (10, "truncated_sync", None),
        (44, "truncated_header", None),
        (90, "truncated_data", True),
    ],
)
def test_eof_does_not_invent_future_samples(kept, status, head):
    profile = ordinary()
    values, _, selection = selected_input(profile, length=255)
    cut = int(selection["start_tick"] / 8 + kept * 4)
    (event,), rx = consume(profile, values[:cut], selection, [1, 71])
    assert event["frame"]["status"] == status
    assert event["frame"]["header_crc"] is head and event["frame"]["payload"] is None
    assert rx.input_samples == cut
    assert not event.get("used_end_padding", False)
    assert rx.finish() == []


def test_retention_limit_is_explicit_and_separate_from_eof():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=255)
    selection["sample_limit"] = round(selection["start_tick"] / 8 + 90 * 4)
    (event,), rx = consume(profile, values, selection, [len(values)])
    assert event["frame"]["status"] == "retention_limit"
    assert event["frame"]["parser_status"] == "truncated_data"
    assert event["frame"]["header_crc"] is True and event["frame"]["payload"] is None
    assert rx.input_samples == selection["sample_limit"] == rx.last_input_consumed
    memory = rx.reference_memory()
    assert memory["legal_symbol_array_bytes"] == memory["legal_symbol_bound"] * 40
    assert not memory["hardware_resource_or_cycles_verified"]
    assert not memory["decoder_FFT_Polar_temporaries_included"]


def test_port_rejects_zero_sync_and_does_not_call_search_or_filter(monkeypatch):
    profile = ordinary()
    values, payload, selection = selected_input(profile)
    rx = MacSelectedReceiver(profile, **selection, timing_backend="python")
    monkeypatch.setattr(rx.phy, "_search", lambda: pytest.fail("new search"))
    result = rx.feed_filtered(values)
    assert [p.mac_payload for p in result.packets] == [payload]
    for method in (rx.phy.feed, rx.phy.iter_feed):
        with pytest.raises(RuntimeError, match="feed_filtered"):
            method(values)
    selection["selected_sync"] = np.zeros(len(profile.sync_symbols()))
    zero = SelectedPSKReceiver(profile, **selection)
    (event,) = zero.feed_filtered(values)
    assert event["frame"]["status"] == "sync_confirmation_rejected"
    assert event["frame"]["header_crc"] is None and zero.input_samples == 0


@pytest.mark.parametrize(
    "change",
    [
        {"sample_limit": True},
        {"sample_limit": 2**20 + 1},
        {"start_tick": -1},
        {"delta": np.nan},
        {"delta": True},
        {"differential_score": 1.01},
        {"selected_sync": np.ones(2)},
        {"selected_sync": np.full(32, np.inf)},
    ],
)
def test_invalid_selection_is_not_silently_accepted(change):
    profile = ordinary()
    _, _, selection = selected_input(profile, length=1)
    with pytest.raises(ValueError):
        SelectedPSKReceiver(profile, **(selection | change))


def test_missing_past_and_closed_bad_inputs_are_explicit():
    profile = ordinary()
    _, _, selection = selected_input(profile, length=1)
    selection["start_tick"] = 0
    rx = SelectedPSKReceiver(profile, **selection)
    (event,) = rx.feed_filtered(np.zeros(32))
    assert event["frame"]["status"] == "insufficient_sync_history"
    assert rx.input_samples == 0
    with pytest.raises(ValueError):
        rx.feed_filtered(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        rx.finish(reason="pad")
    with pytest.raises(ValueError, match="4Msym"):
        SelectedPSKReceiver(replace(profile, config=replace(profile.config, sps=8)), **selection)
    with pytest.raises(ValueError, match="PSKReceiveProfile"):
        SelectedPSKReceiver(None, **selection)


def test_wrong_header_seed_uses_actual_crc_context():
    profile = ordinary()
    values, _, selection = selected_input(profile)
    seed = head_crc_parameters(profile.config)[2]
    wrong = replace(profile, config=replace(profile.config, head_crc_seed=seed ^ 1))
    (event,), _ = consume(wrong, values, selection, [137])
    assert event["frame"]["status"] == "bad_header_crc"


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 6, 0), (3, 8, 8), (4, 12, 0)])
def test_real_native_selected_port_matches_python(ft, mcs, pilot):
    from nearlink_sdr.phy.stream_rx import _RustStreamTiming

    if _RustStreamTiming is None:
        pytest.skip("optional Rust timing extension missing;Rust CI must run this test")
    profile = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(profile, length=139)
    (expected,), _ = consume(profile, values, selection, [len(values)])
    assert expected["frame"]["payload"] == payload
    for pattern in ([1], [17, 257], [len(values)]):
        (event,), _ = consume(profile, values, selection, pattern, "rust")
        assert event["timing_backend"] == "rust"
        equal(expected, event, backend=True)


def test_capacity_can_end_before_selected_position_and_records_resource_failure():
    profile = ordinary()
    values, _, selection = selected_input(profile)
    selection["sample_limit"] = 32
    (event,), rx = consume(profile, values, selection, [len(values)])
    assert event["frame"]["status"] == "retention_limit"
    assert event["frame"]["header_crc"] is None and rx.input_samples == 32


def test_input_is_borrowed_and_only_consumed_prefix_must_be_finite():
    profile = ordinary()
    values, payload, selection = selected_input(profile)
    x = np.r_[values, np.full(123, np.nan)]
    original = values.copy()
    rx = SelectedPSKReceiver(profile, **selection, timing_backend="python")
    (event,) = rx.feed_filtered(x)
    assert event["frame"]["payload"] == payload and rx.input_samples < len(values)
    np.testing.assert_array_equal(values, original)
    with pytest.raises(ValueError, match="NumPy"):
        rx.feed_filtered([1, 2, 3])
    fresh = SelectedPSKReceiver(profile, **selection, timing_backend="python")
    with pytest.raises(ValueError, match="finite"):
        fresh.feed_filtered(np.array([1, np.nan]))
    assert fresh.input_samples == fresh.last_input_consumed == 1
    assert not fresh.done


@pytest.mark.parametrize("length", [1, 2047])
def test_explicit_factory_context_and_legal_maximum(length):
    from tests.test_rx_timing_audit import audit

    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=audit.WORD, data_crc_seed=0x555555, pilot_interval=0
    )
    profile = replace(profile, config=replace(profile.config, mcs_index=12))
    values, payload, selection = selected_input(profile, length=length)
    rx = MacSelectedReceiver(profile, **selection, timing_backend="python")
    result = rx.feed_filtered(values)
    assert [p.mac_payload for p in result.packets] == [payload]
    assert rx.phy.header_result["header"]["data_length"] == length
    assert rx.phy.frame_end_symbol == len(result.diagnostics["events"][0]["symbols"])
