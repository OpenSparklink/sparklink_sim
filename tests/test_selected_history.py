"""Same selected signal/loop/parser with finite raw history;no new sync choices."""

import numpy as np
import pytest

from tests.test_selected_rx import consume, equal, ordinary, selected_input


@pytest.mark.parametrize(
    "ft,mcs,pilot,length",
    [
        (2, 8, 0, 1),
        (2, 10, 8, 137),
        (2, 12, 8, 2047),
        (3, 8, 4, 113),
        (3, 12, 16, 337),
        (4, 0, 0, 37),
        (4, 12, 4, 997),
    ],
)
def test_ring_history_preserves_actual_frame_and_every_symbol(ft, mcs, pilot, length):
    p = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(p, length=length)
    (reference,), full = consume(p, values, selection, [len(values)])
    assert reference["frame"]["payload"] == payload
    selected = {**selection, "history_capacity": 512}
    for pattern in (
        [len(values)],
        [1],
        [37, 11],
        np.random.default_rng(920118).integers(1, 511, 23),
    ):
        (actual,), rx = consume(p, values, selected, pattern)
        equal(reference, actual)
        h = rx.reference_memory()["raw_history"]
        assert h["capacity_samples"] == 512 and h["peak_samples"] <= 512
        assert h["leading_skipped"] == selection["start_tick"] // 8 - 1
        assert h["live_samples"] == 0 and h["logical_writes"] == h["retired_samples"]
        assert rx.input_samples == full.input_samples
        assert rx.reference_memory()["retained_filtered_bytes"] == 2 * 512 * 16
        assert (
            rx.reference_memory()["legal_symbol_array_bytes"]
            == full.reference_memory()["legal_symbol_array_bytes"]
        )


@pytest.mark.parametrize("damage", [{"bad_head": True}, {"bad_data": True}, {"empty": True}])
def test_history_retires_on_real_terminal_failure_without_retry(damage):
    p = ordinary()
    values, _, selection = selected_input(p, **damage)
    (reference,), _ = consume(p, values, selection, [37])
    (actual,), rx = consume(p, values, {**selection, "history_capacity": 512}, [11, 127])
    equal(reference, actual)
    assert actual["frame"]["payload"] is None
    assert rx.reference_memory()["raw_history"]["live_samples"] == 0


def test_insufficient_startup_history_is_resource_failure_not_crc():
    p = ordinary(4, 12)
    values, _, selection = selected_input(p)
    (event,), rx = consume(p, values, {**selection, "history_capacity": 256}, [len(values)])
    assert event["frame"]["status"] == "history_overflow"
    assert event["frame"]["header_crc"] is None and event["frame"]["payload"] is None
    h = rx.reference_memory()["raw_history"]
    assert h["live_samples"] == 0 and h["peak_samples"] == 256
    assert event["raw_history_at_termination"]["live_samples"] == 256
    assert rx.last_input_consumed == rx.input_samples


def test_ring_eof_and_horizon_failures_remain_distinct_and_keep_real_header():
    p = ordinary()
    values, _, selection = selected_input(p, length=255)
    selection = {**selection, "history_capacity": 512}
    cut = int(selection["start_tick"] / 8 + 91 * 4)
    (eof,), rx = consume(p, values[:cut], selection, [1, 47])
    assert eof["frame"]["status"] == "truncated_data" and eof["frame"]["header_crc"] is True
    assert rx.reference_memory()["raw_history"]["peak_samples"] == (
        4 * len(p.sync_symbols()) + int(selection["start_tick"] % 8 != 0)
    )
    selection["sample_limit"] = cut
    (limited,), _ = consume(p, values, selection, [len(values)])
    assert limited["frame"]["status"] == "retention_limit"
    assert limited["frame"]["parser_status"] == "truncated_data"
    assert limited["frame"]["header_crc"] is True


@pytest.mark.parametrize("ft,mcs", [(2, 10), (3, 8), (4, 12)])
def test_actual_native_ring_history_matches_python(ft, mcs):
    from nearlink_sdr.phy.stream_rx import _RustStreamTiming

    if _RustStreamTiming is None:
        pytest.skip("optional native timing absent;RustCI must execute")
    p = ordinary(ft, mcs, 8)
    values, payload, selection = selected_input(p, length=397)
    (expected,), _ = consume(p, values, selection, [len(values)])
    assert expected["frame"]["payload"] == payload
    for pattern in ([1], [71, 257], [len(values)]):
        (event,), _ = consume(p, values, {**selection, "history_capacity": 512}, pattern, "rust")
        assert event["timing_backend"] == "rust"
        equal(expected, event, backend=True)


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("fraction_tick", [0, 1, 7])
def test_independent_startup_extent_counts_existing_fractional_wait(ft, fraction_tick):
    from nearlink_sdr.phy.selected_rx import SelectedPSKReceiver

    p = ordinary(ft, 8)
    n = len(p.sync_symbols())
    start_tick = 80 + fraction_tick
    rx = SelectedPSKReceiver(
        p,
        start_tick=start_tick,
        delta=0,
        differential_score=1,
        selected_sync=p.sync_symbols(),
        sample_limit=1024,
        coherent_threshold=None,
        timing_backend="python",
        history_capacity=512,
    )
    required = int(np.ceil(start_tick / 8 + 4 * (n - 1) + 3))
    assert rx.feed_filtered(np.ones(required, complex)) == []
    expected = 4 * n + int(fraction_tick != 0)
    assert rx.reference_memory()["raw_history"]["peak_samples"] == expected
    assert rx._active["count"] == n


@pytest.mark.parametrize("mcs,length", [(8, 255), (12, 2047)])
def test_ring_preserves_explicit_known_uncoded_factory_context(mcs, length):
    from dataclasses import replace

    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from tests.test_rx_timing_audit import audit

    p = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=audit.WORD, data_crc_seed=0x555555, pilot_interval=0
    )
    p = replace(p, config=replace(p.config, mcs_index=mcs))
    values, payload, selection = selected_input(p, length=length)
    (reference,), _ = consume(p, values, selection, [len(values)])
    (event,), _ = consume(p, values, {**selection, "history_capacity": 512}, [11, 257])
    equal(reference, event)
    assert (
        event["frame"]["payload"] == payload and event["frame"]["header"]["data_length"] == length
    )


def test_eof_terminal_releases_live_validity_and_keeps_failure_descriptor():
    p = ordinary()
    values, _, s = selected_input(p, length=255)
    s["history_capacity"] = 512
    cut = int(s["start_tick"] / 8 + 91 * 4)
    (event,), rx = consume(p, values[:cut], s, [1, 47])
    assert event["frame"]["status"] == "truncated_data"
    assert rx.reference_memory()["raw_history"]["live_samples"] == 0
    assert event["raw_history_at_termination"]["live_samples"] > 0
