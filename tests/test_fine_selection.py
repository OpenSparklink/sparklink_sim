"""Typed selected sync port,available clock and unchanged default fine API."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_transport import CaptureRequest
from nearlink_sdr.sim.fine_selection import FineSelection
from nearlink_sdr.sim.spline_score import SyncSignTemplate
from tests.test_coarse_ingress import profile


def selection():
    return FineSelection(3, 256, 256, 11, 9, -18, ((4, 5),) * 32, (3, 4), 25, 100, 31)


def test_float_bridge_is_explicit_and_does_not_change_cache_or_geometry():
    s = selection()
    row = s.reference_selection()
    assert row["start_tick"] == 9 and row["delta"] == np.arctan2(4, 3)
    assert row["differential_score"] == np.sqrt(25 / 3100)
    np.testing.assert_array_equal(row["selected_sync"], np.full(32, (4 + 5j) * 2**-18))
    assert s.source_start_tick == 9 and len(s.words) == 32
    cost = s.estimate()
    assert cost["selected_cache_BRAM36"] == 1 and cost["selected_cache_bits"] == 5632
    assert cost["cache_read_angle_score_and_coherent_cycles"] is None


@pytest.mark.parametrize(
    "changes",
    [
        dict(ident=True),
        dict(source_end=255),
        dict(start_tick=0),
        dict(exponent=512),
        dict(words=((0, 0),) * 31),
        dict(words=((2**21, 0),) * 32),
        dict(correlation=(2**52, 0)),
        dict(numerator=24),
        dict(energy=-1),
        dict(template_energy=30),
        dict(available_cycle=-1),
    ],
)
def test_malformed_actual_selection_never_becomes_success(changes):
    with pytest.raises(ValueError):
        replace(selection(), **changes)


def test_fine_cache_is_one_credit_and_cannot_be_taken_before_actual_service():
    p = profile()
    template = SyncSignTemplate.from_symbols(p.sync_symbols())
    worker = CaptureFineWorker(template, 256, confirmation_required=True, capture_selection=True)
    geometry = CaptureRequest(256, 0, (10,), lookahead_ticks=0)
    event = dict(
        id=7,
        after_samples=256,
        path=0,
        window_samples=256,
        geometry=geometry,
        service_start_cycle=13,
    )
    words = [pack_filter_word((j * 17) % 1000 - 500, (j * 73) % 900 - 450, -10) for j in range(256)]
    service = worker(event, words)
    with pytest.raises(ValueError, match="held"):
        worker(event, words)
    with pytest.raises(ValueError, match="future"):
        worker.take_selection(7, 13 + service.cycles - 1)
    with pytest.raises(ValueError, match="another"):
        worker.take_selection(8, 13 + service.cycles)
    selected = worker.take_selection(7, 13 + service.cycles)
    assert selected.start_tick == service.start_tick and selected.ident == 7
    assert selected.available_cycle == 13 + service.cycles and len(selected.words) == 32
    assert worker.take_selection(7, 13 + service.cycles) is None
    with pytest.raises(ValueError, match="identity"):
        worker(event | dict(service_start_cycle=13 + service.cycles), words)
    with pytest.raises(ValueError, match="clock"):
        worker(event | dict(id=8), words)
    worker(event | dict(id=8, service_start_cycle=13 + service.cycles), words)
    worker.cancel_selection()
    assert worker._selection is None
    ordinary = CaptureFineWorker(template, 256, confirmation_required=True)
    assert "selected_output_port" not in ordinary.estimate()
    with pytest.raises(RuntimeError, match="compiled"):
        ordinary.take_selection(7, 0)


def test_invalid_compiled_cache_choice_and_missing_clock_are_not_success():
    template = SyncSignTemplate.from_symbols(profile().sync_symbols())
    with pytest.raises(ValueError, match="explicit"):
        CaptureFineWorker(template, 256, confirmation_required=True, capture_selection=1)
    worker = CaptureFineWorker(template, 256, confirmation_required=True, capture_selection=True)
    event = dict(
        id=0, after_samples=256, path=0, window_samples=256, geometry=CaptureRequest(256, 0, (10,))
    )
    with pytest.raises(ValueError, match="clock"):
        worker(event, [0] * 256)
    assert worker._selection is None and worker._last_selected_request == -1
