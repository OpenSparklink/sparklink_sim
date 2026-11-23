"""Evidence entry guards: a successful packet cannot hide numeric/resource failure."""

import json

import numpy as np
import pytest
from scripts import model_initialized_capture as audit


def test_original_numeric_limits_reject_equal_packets_with_different_trajectory():
    frame = dict(status="valid", header_crc=True, data_crc=True, payload=b"same")
    a = dict(frame=frame, symbols=np.ones(4, complex), symbol_centers_filtered=np.arange(4.),
             timing_rate_ppm=np.zeros(4), timing_detector_error=np.zeros(4))
    b = dict(a, symbol_centers_filtered=a["symbol_centers_filtered"]+.002)
    assert not audit.numeric_pair(a, b)["passed"]
    b = dict(a, symbols=a["symbols"]+.0002)
    assert not audit.numeric_pair(a, b)["passed"]
    assert audit.numeric_pair(a, a)["passed"]
    assert not audit.numeric_pair(a, dict(a, symbols=np.ones(3, complex)))["passed"]


def test_changed_input_container_is_rejected_before_receiver(tmp_path):
    path = tmp_path/"literal.npz"
    np.savez(path, iq=np.ones(3, complex), quiet=np.zeros(4, complex))
    row = dict(file=path.name, SHA256=audit.reference.shared.sha(path))
    np.testing.assert_array_equal(audit.load_input(tmp_path, row)[0], np.ones(3))
    path.write_bytes(path.read_bytes()+b"tamper")
    with pytest.raises(ValueError, match="SHA"):
        audit.load_input(tmp_path, row)


def test_failed_development_never_generates_new_held_input(tmp_path, monkeypatch):
    freeze, out = tmp_path/"freeze.json", tmp_path/"uncreated"
    freeze.write_text(json.dumps(dict(source={}, policy=audit.POLICY,
                                     qualification=dict(passed=False))))
    monkeypatch.setattr(audit.reference.shared, "sources", lambda: {})
    monkeypatch.setattr("sys.argv", ["audit", "--output", str(out), "--split", "validation",
                                     "--freeze", str(freeze)])
    with pytest.raises(SystemExit) as stop:
        audit.main()
    assert stop.value.code == 2 and not out.exists()
