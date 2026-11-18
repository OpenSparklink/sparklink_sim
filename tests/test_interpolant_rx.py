"""One initialized symbol/parse port:real header gate,invalid context and storage boundary."""

from copy import deepcopy

import numpy as np
import pytest

from nearlink_sdr.phy.interpolant_rx import InterpolantPSKReceiver
from nearlink_sdr.phy.mac_interface import mac_result_from_events
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def fixture():
    profile = ordinary()
    values, _, selection = selected_input(profile, length=3)
    words = quantize_input(values)
    service, _ = prepare(profile, words, selection)
    return profile, selection, service


@pytest.mark.parametrize(
    "change",
    [
        {"valid": False},
        {"symbols": 31},
        {"phase_turn40": 2**40},
        {"frequency_turn40": True},
        {"gain": (0, 0)},
        {"power": (1, 512)},
        {"gain": ("3", 0)},
    ],
)
def test_missing_or_invalid_initialization_is_never_claimed_as_success(change):
    profile, selection, service = fixture()
    fit = deepcopy(service.fit.result)
    fit.update(change)
    with pytest.raises(ValueError):
        InterpolantPSKReceiver(profile, initial_fit=fit, **selection)
    assert service.ready and not service.fit.output_taken


def test_symbol_input_does_not_store_raw_capture_or_mutate_on_bad_words():
    profile, selection, service = fixture()
    rx = InterpolantPSKReceiver(profile, initial_fit=service.fit.result, **selection)
    assert rx.reference_memory()["retained_filtered_bytes"] == 0
    assert rx.reference_memory()["private_raw_samples"] == 0
    with pytest.raises(RuntimeError):
        rx.feed_filtered(np.ones(3))
    with pytest.raises(RuntimeError):
        rx.parse_boundary(after_samples=service.stencil.history.end)
    for word in [True, -1, 2**52]:
        with pytest.raises(ValueError):
            rx.push_symbol(word, None, after_samples=service.stencil.history.end)
        assert rx.input_samples == 0 and rx._active["count"] == 0
    with pytest.raises(ValueError):
        rx.push_symbol(1, 1, after_samples=10000)
    with pytest.raises(ValueError):
        rx.push_symbol(1, None, after_samples=1)
    (event,) = rx.finish()
    assert event["frame"]["status"] == "truncated_sync" and event["sync_symbols_consumed"] == 0
    assert not mac_result_from_events([event]).packets
    assert rx.finish() == []
    with pytest.raises(RuntimeError):
        rx.push_symbol(1, None, after_samples=10000)
    with pytest.raises(ValueError):
        rx.finish(reason="fake_success")


def test_unconfirmed_or_native_initialization_is_explicitly_unsupported():
    profile, selection, service = fixture()
    for changed in [
        dict(timing_backend="rust"),
        dict(coherent_threshold=None, differential_score=0.1),
    ]:
        with pytest.raises(ValueError):
            InterpolantPSKReceiver(profile, initial_fit=service.fit.result, **(selection | changed))


def test_source_absolute_window_origin_never_fabricates_or_rebases_prior_samples():
    origin = 123_453
    h = ReceiveHistoryPort(8, first_index=origin)
    for k in range(8):
        assert h.write(origin + k, 900 + k, 2 * k)
    assert h.head == origin and h.end == origin + 8
    with pytest.raises(ValueError):
        h.request(origin - 1, 17)
    for k in range(8):
        c = 20 + 2 * k
        assert h.request(origin + k, c)
        assert h.take(c + 1) == (origin + k, 900 + k)
    assert h.stats["accepted_words"] == 8 and h.stats["retired_words"] == 0
    for origin in [-1, True, 2**63 - 1]:
        with pytest.raises(ValueError):
            ReceiveHistoryPort(first_index=origin)
