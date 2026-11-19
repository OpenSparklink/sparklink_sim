"""Actual saved words, bad CRC and EOF remain failures after divider reduction."""

import gzip
import importlib
import json
from pathlib import Path

import pytest

from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


@pytest.mark.parametrize("damage,status", [("bad_data", "bad_data_crc"),
                                         ("EOF", "truncated_data")])
def test_model_saves_exact_values_and_never_counts_CRC_or_EOF_failure_as_recovery(
        tmp_path, monkeypatch, damage, status):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    model = importlib.import_module("model_timing_divider")
    p = ordinary(4, 12, 0)
    iq, payload, selection = selected_input(
        p, length=49, **({"bad_data": True} if damage == "bad_data" else {}))
    words = quantize_input(iq)
    if damage == "EOF":
        words = words[:selection["start_tick"] // 8 + 4 * 448]
    row = model.pair(tmp_path, damage, p, words, selection, payload,
                     json.loads(model.POLICY.read_text()), partitions=True)
    assert row["before"]["status"] == row["after"]["status"] == status
    assert not row["after"]["payload_exact"] and not row["recovery"]
    assert row["numerical"]["numeric_pass"] and row["modeled_edge_saving"] > 0
    assert row["modeled_edge_saving"] == 29 * row["prefix_checks"]
    assert row["timing_cost"]["divider"]["steps"] == 34
    assert row["complete_latency_cycles"] is None and not row["full_receiver16MSps"]
    records = []
    for label in ("full", "bounded-prefix"):
        with gzip.open(tmp_path / f"{damage}-{label}-services.jsonl.gz", "rt") as f:
            data = [json.loads(line) for line in f]
        assert len(data) == row["service_records"]
        assert all(r["taken_edge"] >= r["ready_edge"] == r["issued_edge"] + r["latency"]
                   for r in data)
        records.append([r["values"] for r in data])
    assert records[0] == records[1]
