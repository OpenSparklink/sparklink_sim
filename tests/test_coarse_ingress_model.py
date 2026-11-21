"""Driver provenance/real composition smoke;unit tests never generate held data."""

import copy
import json
import sys

import numpy as np
import pytest
from scripts import model_coarse_ingress as model


@pytest.fixture(scope="module")
def development(tmp_path_factory):
    out = tmp_path_factory.mktemp("ingress-development")
    rows = model.inputs(out, "development")
    return out, rows


def test_new_development_has_distinct_payload_configuration_and_controls(development):
    out, rows = development
    assert len(rows) == 13 and sum(r["legal_packets"] for r in rows) == 10
    assert {r["config"]["frame_type"] for r in rows} == {2, 3, 4}
    assert {r["family"] for r in rows} >= {
        "tone",
        "noise",
        "burst",
        "step",
        "wrong-sync-weak",
        "bad_head",
        "bad_data",
        "truncated",
    }
    for row in rows:
        path = out / row["artifact"]
        assert model.sha(path) == row["input_SHA256"]
        with np.load(path) as z:
            for j, expected in enumerate(row["expected"]):
                assert z[f"expected{j}"].tobytes().hex() == expected
    assert len({model.payload(307, j) for j in range(3)}) == 3


@pytest.mark.parametrize(
    "clock,latency,blocked", [(200, 10, False), (100, 10, False), (200, 80, False), (200, 10, True)]
)
def test_real_RRC_release_alignment_core_and_EOF(development, tmp_path, clock, latency, blocked):
    out, rows = development
    row = rows[2]  # FT4;wrong-sync input is merely a fixed development stimulus.
    prepared = model.prepare(out, row)
    result = model.calendar(
        tmp_path, row, prepared, clock, 16, "serial2", latency=latency, blocked=blocked
    )
    assert result["all_committed_integer_and_query_values_exact"]
    assert result["all_source_and_EOF_accounting_exact"] and result["source_samples"] == 3072
    assert result["source_query_prefix_pass"] == (clock == 200 and latency == 10 and not blocked)
    assert result["dedicated_prefix_DSP_upper"] == 20
    assert result["cost"]["own_register_bits_upper"] <= 1536
    assert result["raw_detector_arithmetic_cycles"] is result["fine_packet_decoder_cycles"] is None
    assert result["actual_synthesis"] is result["whole_RX_16MSps"] is False


@pytest.mark.parametrize("damage", ["missing", "policy", "source", "qualification", "hash", "end"])
def test_freeze_guard_runs_before_held_generation(tmp_path, monkeypatch, damage):
    out = tmp_path / "held-must-not-exist"
    dev = tmp_path / "development"
    dev.mkdir()
    sources = {"source": "frozen"}
    frozen = dict(policy=copy.deepcopy(model.POLICY), sources=sources)
    if damage == "policy":
        frozen["policy"]["pending_query_credit"] = 2
    if damage == "source":
        frozen["sources"] = {"source": "changed"}
    freeze = dev / "freeze.json"
    freeze.write_text(json.dumps(frozen))
    receipt = dict(component_qualified=damage != "qualification", freeze_SHA256=model.sha(freeze))
    if damage == "hash":
        receipt["freeze_SHA256"] = "wrong"
    (dev / "model.json").write_text(json.dumps(receipt))
    (dev / "source-bindings-at-end.json").write_text(json.dumps({} if damage == "end" else sources))
    if damage == "missing":
        freeze.unlink()
    monkeypatch.setattr(model, "sources", lambda: sources)
    monkeypatch.setattr(model, "inputs", lambda *a: pytest.fail("Must reject before held inputs"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["model", "--split", "validation", "--freeze", str(freeze), "--output", str(out)],
    )
    with pytest.raises(ValueError):
        model.main()
    assert not out.exists()


def eligible_cases():
    return [
        dict(
            clock_MHz=200,
            blocked=False,
            raw_flag_latency_hypothesis=10,
            frame_type=ft,
            lanes=16 if ft == 4 else 8,
            normalizer=mode,
            source_query_prefix_pass=True,
            cost=dict(
                own_register_research_cap_pass=True,
                own_register_bits_upper=1026,
                coarse=dict(DSP=12, logical_state_bits_upper=103829),
            ),
        )
        for ft in [2, 3, 4]
        for mode in ["pipeline", "serial2"]
    ]


def test_complete_preferred_layout_and_existing_budgets_can_qualify():
    assert model.qualify(eligible_cases())


@pytest.mark.parametrize("damage", ["control", "DSP", "memory", "wordloss", "duplicate", "missing"])
def test_adoption_rejects_numeric_budget_or_missing_layout_even_if_reported_flag_true(damage):
    cases = eligible_cases()
    if damage == "control":
        cases[0]["cost"]["own_register_bits_upper"] = 1537
    elif damage == "DSP":
        cases[0]["cost"]["coarse"]["DSP"] = 19
    elif damage == "memory":
        cases[0]["cost"]["coarse"]["logical_state_bits_upper"] = 131073
    elif damage == "wordloss":
        cases[0]["source_query_prefix_pass"] = False
    elif damage == "duplicate":
        cases[1] = copy.deepcopy(cases[0])
    else:
        cases.pop()
    assert not model.qualify(cases)
