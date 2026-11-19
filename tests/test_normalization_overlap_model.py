"""Model preserves a CRC failure and EOF instead of using identical words as success."""

import gzip
import importlib
import json
from pathlib import Path

import pytest

from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


@pytest.mark.parametrize("damage,status", [("bad_data", "bad_data_crc"),
                                           ("EOF", "truncated_data")])
def test_saved_service_records_and_failure_associations_are_actual(tmp_path, monkeypatch,
                                                                 damage, status):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    model = importlib.import_module("model_normalization_overlap")
    profile = ordinary(4, 12, 0)
    iq, payload, selection = selected_input(profile, length=49,
                                            **({"bad_data": True} if damage == "bad_data" else {}))
    words = quantize_input(iq)
    if damage == "EOF":
        words = words[:selection["start_tick"] // 8 + 4 * 448]
    row = model.pair(tmp_path, damage, profile, words, selection, payload,
                     json.loads(model.POLICY.read_text()), partitions=True)
    assert row["before"]["status"] == row["after"]["status"] == status
    assert not row["after"]["payload_exact"] and not row["recovery"]
    assert row["numerical"]["numeric_pass"] and row["modeled_edge_saving"] > 0
    assert row["cost"]["normalization_service"]["state_register_bits_upper"] == 7825
    assert row["complete_latency_cycles"] is None and not row["full_receiver16MSps"]
    counts = []
    for kind in ["serial", "overlap"]:
        with gzip.open(tmp_path / f"{damage}-{kind}-services.jsonl.gz", "rt") as f:
            records = [json.loads(line) for line in f]
        counts.append(len(records))
        assert all(r["taken_edge"] >= r["ready_edge"] == r["issued_edge"] + r["latency"]
                   for r in records)
    assert counts == [row["service_records"]] * 2
