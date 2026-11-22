"""A partial functional closure cannot substitute for four independent RX gates."""

import json
import sys

import pytest
from scripts import model_captured_frame as model


def rows():
    return [
        dict(
            id=k,
            input_SHA256=f"{k + 1:064x}",
            negative=k >= 6,
            samples=4096,
            source_lossless=True,
            requests=3,
            fine_complete=2,
            queue_drops=1,
            source_epoch_drops=0,
            frame_resource_drops=0,
            coherent_accepted=int(k < 6),
            head_CRC_and_length_valid=int(k < 6),
            data_CRC_passes=int(k < 6),
            actual_packets=int(k < 6),
            full_correct=int(k < 6),
            expected_length_matches=1 if k < 6 else None,
            expected_MCS_matches=int(k < 6),
            false_MAC=0,
            duplicate_MAC=0,
            additional_control_bits=2166,
            new_selected_cache_BRAM36=1,
            reference_NumPy_plus_cache_bytes=410352,
            max_source_primitive_operations=2,
            all_original_words_and_chunks_exact=True,
            old_strict_consumer_on_same_actual_reads_exact=True,
            no_private_full_tail_banks=True,
            complete_cycle_evidence_valid=False,
            complete_latency_cycles=None,
        )
        for k in range(10)
    ]


def test_frozen_policy_has_stable_JSON_and_explicit_unmodeled_arithmetic():
    assert json.loads(json.dumps(model.POLICY)) == model.POLICY
    assert model.POLICY["preamble_symbols"] == {"2": 40, "3": 48, "4": 64}
    assert model.POLICY["unknown_confirmation_head_carrier_decoder_cycles"] is None
    assert model.POLICY["new_default"] is model.POLICY["fixed_and_full16MSps"] is False


def test_diagnostic_first_preamble_origin_matches_independent_analytic_channel(tmp_path):
    # audit.reference places symbol0 at512+offset and returns symbol40's center.
    # The channel adds prefix;FT3/4 use their actual48/64-symbol preambles later.
    data = model.make_inputs(tmp_path, "development")
    for row in data[:6]:
        expected_origin = 512 + row["params"]["offset"] + row["params"]["prefix"]
        assert abs(row["true_first_preamble"] - expected_origin) < 1e-9
        assert row["preamble_symbols"] == {2: 40, 3: 48, 4: 64}[row["config"]["frame_type"]]
    assert all(
        row["true_first_preamble"] is None
        for row in data[6:]
        if row["params"]["family"] != "wrong-sync"
    )


@pytest.mark.parametrize(
    "edit",
    [
        dict(false_MAC=1),
        dict(duplicate_MAC=1),
        dict(queue_drops=0),
        dict(source_lossless=False),
        dict(additional_control_bits=4097),
        dict(reference_NumPy_plus_cache_bytes=524289),
        dict(new_selected_cache_BRAM36=2),
        dict(additional_control_bits=-1),
        dict(max_source_primitive_operations=3),
        dict(complete_latency_cycles=0),
        dict(complete_cycle_evidence_valid=True),
        dict(no_private_full_tail_banks=False),
        dict(all_original_words_and_chunks_exact=False),
        dict(old_strict_consumer_on_same_actual_reads_exact=False),
        dict(expected_length_matches=0),
        dict(expected_MCS_matches=0),
        dict(head_CRC_and_length_valid=0),
        dict(data_CRC_passes=0),
        dict(actual_packets=True),
        dict(full_correct=2),
    ],
)
def test_reject_false_delivery_missing_context_unaccounted_drops_and_cost_overrun(edit):
    data = rows()
    assert model.qualify(data)
    data[1].update(edit)
    assert not model.qualify(data)


def test_negative_confirmation_and_empty_or_duplicate_cases_cannot_qualify():
    data = rows()
    data[7]["coherent_accepted"] = 1
    assert not model.qualify(data)
    data = rows()
    data[1]["input_SHA256"] = data[0]["input_SHA256"]
    assert not model.qualify(data)
    assert not model.qualify(rows()[:-1])
    data = rows()
    for r in data:
        r["full_correct"] = 0
    assert not model.qualify(data)


def test_pressure_failure_can_be_retained_without_fabricating_full_chain_acceptance():
    data = rows()
    for r in data[1:6]:
        r.update(actual_packets=0, full_correct=0, data_CRC_passes=0)
    assert model.qualify(data)


def test_validation_checks_freeze_before_generating_any_input(tmp_path, monkeypatch):
    out = tmp_path / "not-created"
    monkeypatch.setattr(
        sys, "argv", ["model_captured_frame", "--split", "validation", "--output", str(out)]
    )
    with pytest.raises(ValueError, match="frozen"):
        model.main()
    assert not out.exists()


@pytest.mark.parametrize("field", ["policy", "sources", "end", "qualified", "SHA"])
def test_validation_rejects_changed_binding_before_opening_new_holdout(
    tmp_path,
    monkeypatch,
    field,
):
    frozen = dict(policy=model.POLICY, sources={"unchanged.py": "ab"})
    old = json.loads(json.dumps(frozen))
    end, qualified = frozen["sources"].copy(), True
    if field in ("policy", "sources"):
        old[field] = {}
    elif field == "end":
        end = {}
    elif field == "qualified":
        qualified = False
    freeze = tmp_path / "freeze.json"
    model.write(freeze, old)
    model.write(tmp_path / "source-bindings-at-end.json", end)
    model.write(
        tmp_path / "model.json",
        dict(
            component_qualified=qualified,
            freeze_SHA256="wrong" if field == "SHA" else model.shared.sha(freeze),
        ),
    )
    monkeypatch.setattr(model.shared, "sources", lambda: frozen["sources"])
    out = tmp_path / "not-created"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "model_captured_frame",
            "--split",
            "validation",
            "--freeze",
            str(freeze),
            "--output",
            str(out),
        ],
    )
    with pytest.raises(ValueError, match="unchanged"):
        model.main()
    assert not out.exists()
