"""Real diagnostic wrappers and the optional selected-cache callback coexist."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
from nearlink_sdr.sim.capture_transport import CaptureRequest
from nearlink_sdr.sim.spline_score import SplineScoreFlow, SyncSignTemplate


def case(ft):
    cfg = TxConfig(frame_type=ft, mcs_index=8, pid=3, symbol_rate_mhz=4,
                   ctrl_bits_len=28 if ft == 2 else 27,
                   head_llid=None if ft == 2 else 0x234567)
    sync = PSKReceiveProfile.ordinary(cfg).sync_symbols()
    window = {2: 256, 3: 376, 4: 632}[ft]
    times = 32 + 4 * np.arange(len(sync))
    values = np.interp(np.arange(window), times, sync.real) + 1j * np.interp(
        np.arange(window), times, sync.imag,
    )
    triples = [(round(z.real * 1024), round(z.imag * 1024), 0) for z in values]
    request = CaptureRequest(window, 0, (32,), 8, 128)
    event = dict(id=0, after_samples=window, path=0, window_samples=window,
                 geometry=request, service_start_cycle=13)
    return SyncSignTemplate.from_symbols(sync), triples, event


class LegacyObservedFlow(SplineScoreFlow):
    """A preexisting wrapper owns its sink and accepts only original geometry."""

    def __init__(self, assumptions, trace):
        super().__init__(assumptions)
        self.trace = trace

    def run(self, words, template, peaks, *, minimum_tick, lookahead_ticks):
        return super().run(words, template, peaks, minimum_tick=minimum_tick,
                           lookahead_ticks=lookahead_ticks, sink=self.trace.append)


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("wrapper", ["actual", "legacy"])
def test_default_worker_preserves_actual_wrapper_and_every_result_field(ft, wrapper, monkeypatch):
    template, triples, event = case(ft)
    results, trace = [], []
    worker = CaptureFineWorker(template, len(triples), confirmation_required=True,
                               sink=lambda e, r: results.append(r))
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    actual = importlib.import_module("model_spline_power").ObservedFlow
    engine = actual if wrapper == "actual" else LegacyObservedFlow
    worker.engine = engine(worker.engine.a, trace)
    service = worker(event, [pack_filter_word(*w) for w in triples])
    direct = SplineScoreFlow(worker.engine.a).run(
        triples, template, (32,), minimum_tick=8, lookahead_ticks=128,
    )
    assert results == [direct] and service.cycles == direct["cycles"]
    assert service.start_tick == direct["best"]["start_tick"] and trace
    assert not worker.capture_selection and worker._selection is None


def test_enabled_selected_cache_still_receives_its_one_callback_and_ordered_words():
    template, triples, event = case(2)
    results = []
    worker = CaptureFineWorker(template, len(triples), confirmation_required=True,
                               capture_selection=True, sink=lambda e, r: results.append(r))
    service = worker(event, [pack_filter_word(*w) for w in triples])
    result = worker.take_selection(0, event["service_start_cycle"] + service.cycles)
    assert len(result.words) == 32 and result.start_tick == service.start_tick
    assert result.correlation == tuple(results[0]["best"]["correlation"])
    assert worker.take_selection(0, result.available_cycle) is None


def test_real_observer_exception_is_not_swallowed_as_a_completed_service():
    template, triples, event = case(2)
    worker = CaptureFineWorker(template, len(triples), confirmation_required=True)

    class NotReady:
        def run(self, *args, **options):
            raise NotImplementedError("actual observer unavailable")

    worker.engine = NotReady()
    with pytest.raises(NotImplementedError, match="actual observer"):
        worker(event, [pack_filter_word(*w) for w in triples])
