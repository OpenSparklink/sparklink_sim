"""The two coded diagnostic arms must retain the actual carrier configuration."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("kappa,fixed", [(None, False), (2, False), (2, True)])
def test_coded_hard_soft_replay_forwards_frozen_weights(tmp_path, monkeypatch, kappa, fixed):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_acquisition_regression")
    folder = tmp_path / "coded-replay"
    (folder / "evidence").mkdir(parents=True)
    expected = bytes([1]) * 255
    path = tmp_path / "iq.npz"
    np.savez(path, iq=np.ones(128), quiet=np.zeros(64), expected=np.frombuffer(expected, np.uint8))
    previous = dict(correct=True)
    record = dict(file=str(path), sha256=module.sha(path), mcs=8, id="fixed",
                  arms={arm: dict(frames=[previous]) for arm in ["hard", "maxlog"]})
    (folder / "results.json").write_text(json.dumps([record]))
    np.savez(folder / "evidence/fixed-results.npz", **{
        f"{arm}-0-symbol_centers_raw": np.array([0.]) for arm in ["hard", "maxlog"]})
    monkeypatch.setattr(module, "checked_manifest", lambda folder, name: folder / name)
    options = {} if kappa is None else dict(carrier_weight_kappa=kappa, carrier_weight_fixed=fixed)
    calls = []

    def decode(symbols, profile, **kwargs):
        assert kwargs == options  # Failure exposes a silent fallback to old weights.
        calls.append(profile.soft_payload)
        return dict(status="valid", payload=expected, header=dict(data_length=255), mcs=8,
                    header_crc=True, data_crc=True, length_ok=True)

    monkeypatch.setattr(module, "decode_psk_frame", decode)
    monkeypatch.setattr(module, "diagnostic_bits", lambda *a: np.unpackbits(
        np.frombuffer(expected, np.uint8)))
    receiver = SimpleNamespace(
        args=SimpleNamespace(carrier_weight_kappa=kappa, carrier_weight_fixed=fixed),
        baseline=tmp_path, receive=lambda *a: ([dict(symbols=np.ones(128), sync_start_raw=0)], {}),
        record=lambda *a, **k: None,
    )
    module.Replay.coded(receiver)
    assert calls == [False, True]


def test_fixed_mode_without_kappa_cannot_pass_using_only_quiet_certificates(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("replay_acquisition_regression")
    with pytest.raises(ValueError, match="explicit kappa"):
        module.Replay(SimpleNamespace(carrier_weight_fixed=True))
