"""Original-IQ replay cannot silently run old confirmation or bypass config guards."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_strict_receive import iq_record, transmitted
from tests.test_sync_refine import profile


@pytest.mark.parametrize("fixed", [False, True])
def test_replay_forwards_actual_confirmation_and_preserves_event_metadata(tmp_path, monkeypatch, fixed):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_acquisition_regression")
    p = profile()
    wire, wanted = transmitted(p)
    x, quiet, _, _ = iq_record(wire, offset=.277, cfo=23000, ppm=100)
    original = module.DiagnosticReceiver
    called = []

    def receiver(*args, **kwargs):
        assert kwargs["sync_coherent_threshold"] == .45
        assert kwargs["sync_coherent_fixed_nco"] is fixed
        called.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "DiagnosticReceiver", receiver)
    folder = tmp_path / "replay"
    (folder / "evidence").mkdir(parents=True)
    args = SimpleNamespace(backend="scaled", sync_method="bounded-grid", normalized_coarse=True,
                           sync_coherent_threshold=.45, sync_coherent_fixed_nco=fixed)
    replay = SimpleNamespace(args=args, stream_index=0, out=folder, bits=10,
                             full_scale=1., tone_config=None)
    events, cost = module.Replay.receive(replay, x, quiet, p)
    assert called == [True]
    assert len(events) == 1 and events[0]["frame"]["payload"] == wanted
    assert events[0]["sync_confirmation"]["accepted"]
    assert (events[0]["sync_confirmation"]["nco"] is not None) == fixed
    assert cost["capture_stats"]["coherent_accepted"] == 1


@pytest.mark.parametrize("options", [dict(sync_coherent_fixed_nco=True),
    dict(sync_coherent_threshold=.45),
    dict(sync_coherent_threshold=.45, backend="scaled", sync_method="bounded-grid"),
    dict(sync_coherent_threshold=.45, backend="scaled", sync_method="bounded-grid",
         normalized_coarse=True, sync_candidate_count=2)])
def test_invalid_replay_config_rejected_before_quiet_proofs(options, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_acquisition_regression")
    with pytest.raises(ValueError):
        module.Replay(SimpleNamespace(**options))


def test_real_mac_replay_cannot_claim_unexecuted_fixed_sync_nco(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_carrier_mac_real")
    with pytest.raises(ValueError, match="coherent-sync"):
        module.run(SimpleNamespace(experiment="carrier-weights", fixed_nco=True))


def test_reproducer_refuses_changed_frozen_budget_before_output(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("reproduce_sync_coherent")
    (tmp_path / "predevelopment.json").write_text("{}")
    output = tmp_path / "must-not-exist"
    with pytest.raises(AssertionError):
        module.run(SimpleNamespace(evidence=tmp_path, output=output))
    assert not output.exists()
