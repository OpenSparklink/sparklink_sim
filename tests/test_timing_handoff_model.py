"""A completed query without EOF samples is not a committed or successful symbol."""

import importlib
import json
from pathlib import Path

from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


def test_EOF_comparison_retains_failure_and_exact_state_with_one_terminal_query(
    tmp_path, monkeypatch
):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    model = importlib.import_module("model_timing_handoff")
    profile = ordinary(4, 12, 0)
    values, payload, selection = selected_input(profile, length=49)
    words = quantize_input(values)[:selection["start_tick"] // 8 + 4 * 448]
    policy = json.loads(model.POLICY.read_text())
    row = model.pair(tmp_path, "truncated", profile, words, selection, payload, policy,
                     partitions=True)
    assert row["before"]["status"] == row["after"]["status"] == "truncated_data"
    assert row["after"]["data_crc"] is None and not row["after"]["payload_exact"]
    assert row["terminal_queries_without_symbol_commit"] == 1
    assert row["decisions_words_trajectories_and_states_exact"]
    assert not row["success_loss"] and not row["recovery"] and not row["false_delivery"]
    assert row["actual_services"]["timing_state_commits"] == 448
    assert row["actual_services"]["controller_statistics"]["query_replies"] == 449
    assert row["complete_latency_cycles"] is None
