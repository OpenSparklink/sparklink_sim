"""Offline observers accept empty/multiple new peaks without influencing selection."""

import importlib
from pathlib import Path

import pytest

from nearlink_sdr.phy.tone_notch import ToneNotchConfig
from tests.test_strict_receive import iq_record, transmitted
from tests.test_sync_refine import profile


@pytest.mark.parametrize("count", [1, 2])
def test_numeric_observer_handles_new_peak_tuple_and_rejected_windows(tmp_path, monkeypatch, count):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_acquisition_regression")
    p = profile()
    wire, wanted = transmitted(p)
    x, quiet, _, _ = iq_record(wire, offset=.277, cfo=23000, ppm=100)
    rx = module.DiagnosticReceiver(
        p, quiet, diagnostic_file=tmp_path / "numeric.jsonl", quiet_batching=False,
        correlation_backend="scaled", sync_search_method="bounded-grid",
        sync_candidate_count=count, normalized_coarse=True,
        tone_notch=ToneNotchConfig(radius=31/32))
    events = rx.feed(x)+rx.finish()
    assert len(events) == 1 and events[0]["frame"]["payload"] == wanted
    assert events[0]["frame"]["status"] == "valid"
    assert rx.numeric["eligible_rank_queries"] > 0
    assert rx.numeric["max_normalized_rank_loss"] < .005
