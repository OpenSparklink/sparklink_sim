"""Saved CRC/EOF counterexamples remain failures through the finite normalization model."""

import importlib
import json
from pathlib import Path

import pytest

from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def test_model_retains_one_terminal_query_and_separate_numeric_gate(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    model = importlib.import_module("model_normalization_words")
    profile = ordinary(4, 12, 0)
    iq, payload, selection = selected_input(profile, length=49)
    words = quantize_input(iq)[:selection["start_tick"] // 8 + 4 * 448]
    row = model.pair(tmp_path, "EOF", profile, words, selection, payload,
                     json.loads(model.POLICY.read_text()), partitions=True)
    assert row["before"]["status"] == row["after"]["status"] == "truncated_data"
    assert not row["after"]["payload_exact"] and row["after"]["data_crc"] is None
    assert row["numerical"]["numeric_pass"]
    assert row["normalization_word_count"] == 448
    assert row["actual_services"]["state"]["count"] == 448
    assert row["normalization_latency_histogram"] == {24: 447, 33: 1}
    assert row["complete_latency_cycles"] is None and not row["full_receiver16MSps"]
    assert not row["success_loss"] and not row["recovery"] and not row["false_delivery"]


@pytest.mark.parametrize("guard", [0, -1, True, 1025, 2.0])
def test_invalid_host_watchdog_is_rejected_before_input_ownership(monkeypatch, guard):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    model = importlib.import_module("model_timing_words")
    with pytest.raises(ValueError, match="host verification"):
        model.execute(None, None, None, None, None, None, None, host_edges_per_symbol=guard)
