"""Source binding and real RRC calendar integration of the flag alignment node."""

import json
import sys

import numpy as np
import pytest
from scripts import model_raw_gate_join as model


@pytest.mark.parametrize("changed", [None, "policy", "oracle", "value_reference", "lock"])
def test_validation_refuses_changed_or_missing_context_before_generating_inputs(
    tmp_path, monkeypatch, changed
):
    output = tmp_path / "validation"
    args = ["model_raw_gate_join", "--output", str(output), "--split", "validation"]
    if changed is not None:
        frozen = dict(policy=dict(model.POLICY), sources=model.source_bindings())
        if changed == "policy":
            frozen["policy"]["FIFO_depth"] += 1
        else:
            key = {
                "oracle": "tests/test_rrc_bridge.py",
                "value_reference": "scripts/model_rrc_bridge.py",
                "lock": "uv.lock",
            }[changed]
            frozen["sources"][key] = "0" * 64
        path = tmp_path / "freeze.json"
        path.write_text(json.dumps(frozen))
        args.extend(["--freeze", str(path)])
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        model, "inputs", lambda _: pytest.fail("No holdout evaluation before freeze")
    )
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not output.exists()


@pytest.mark.parametrize("frame_type,window", [(2, 256), (3, 376), (4, 632)])
def test_history_geometry_is_derived_from_actual_ordinary_profile(frame_type, window):
    receiver = model.context(frame_type, np.full(513, 0.03 + 0.03j))
    assert receiver.search_samples == window and receiver.delay == 40
    assert receiver.quantum == len(receiver._box) == 32


@pytest.mark.parametrize("latency,complete", [(10, True), (80, False)])
def test_actual_producer_words_flags_and_off_clock_prescreen_are_not_free_service(
    latency, complete
):
    receiver = model.context(2, np.full(513, 0.03 + 0.03j))
    t = np.arange(256)
    samples = np.exp(0.1j * t) * np.where(t < 83, 0.001, 0.5)
    words, _ = model.filter_words(samples)
    flags = model.legacy_flags(receiver, samples)
    expected = model.expected_values(words, receiver)
    row = model.simulate(200, latency, False, words, flags, receiver, expected)
    assert row["continuous_alignment_pass"] == complete
    assert row["peak_flag_FIFO"] <= 16
    assert row["raw_detector_arithmetic_cycles"] is row["scaled_bank_cycles"] is None
    assert row["capture_header_payload_MAC"] is None and row["functional_sink_off_clock"]
    assert not row["four_RX_gates_passed"]
    if complete:
        assert len(row["emitted"]) == 256 and row["functional_score_points"]
        assert row["functional_gate_queries"]
        assert all(p["score_available_cycle"] is None for p in row["functional_score_points"])
        assert all(p["available_cycle"] is None for p in row["functional_gate_queries"])
    else:
        assert row["fault_reason"] and row["bridge_rejected_words"]
        assert row["join_stats"]["flags_discarded"] > 0


def test_compiled_node_source_and_both_independent_references_are_bound():
    bindings = model.source_bindings()
    for name in (
        "src/nearlink_sdr/sim/raw_gate_join.py",
        "scripts/model_raw_gate_join.py",
        "scripts/model_rrc_bridge.py",
        "tests/test_rrc_bridge.py",
        "uv.lock",
    ):
        assert len(bindings[name]) == 64
