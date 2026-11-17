"""A receiver pass label cannot override a coefficient/quality budget failure."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from nearlink_sdr.phy.sync_coherent import coherent_sync_confirmation


@pytest.fixture
def numeric_case(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("audit_sync_coherent_numeric")
    source = tmp_path / "iq.npz"
    np.savez(source, iq=np.ones(256))
    meta = dict(id="same-packet-label", path=str(source), sha256=module.sha(source))
    paths = [tmp_path / "float", tmp_path / "fixed"]
    for directory, fixed in zip(paths, [False, True], strict=True):
        (directory / "evidence").mkdir(parents=True)
        node = directory / "evidence/sync.npz"
        np.savez(node, symbols=np.ones(32, complex), reference=np.ones(32, complex))
        info = coherent_sync_confirmation(np.ones(32), np.ones(32), 0., .45, fixed_nco=fixed)
        window = dict(query_after_samples=256, start_raw=12., coherent=info,
                      sync_file=node.name, sync_sha256=module.sha(node))
        row = dict(input=meta, frames=[dict(correct=True, sync_start=12.)], windows=[window])
        (directory / "results.json").write_text(json.dumps([row]))
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps(dict(budgets=dict(coherent_quality_fixed_difference=.005,
                                      CFO_fixed_cycles_per_symbol_difference=1e-5))))
    return module, SimpleNamespace(before=paths[0], after=paths[1], budget=budget,
                                   output=tmp_path / "numeric")


def test_numeric_node_acceptance_is_distinct_from_complete_receiver(numeric_case):
    module, args = numeric_case
    assert module.run(args) == 0
    d = json.loads((args.output / "numeric.json").read_text())
    assert d["node_budget_passed"] and not d["whole_system_accepted"]


def test_same_packet_label_cannot_waive_injected_node_quality_error(numeric_case, monkeypatch):
    module, args = numeric_case
    original = module.coherent_sync_confirmation

    def degraded(*a, fixed_nco=False, **kw):
        result = original(*a, fixed_nco=fixed_nco, **kw)
        if fixed_nco:
            result["quality"] -= .01
        return result

    rows = json.loads((args.after / "results.json").read_text())
    rows[0]["windows"][0]["coherent"]["quality"] -= .01
    (args.after / "results.json").write_text(json.dumps(rows))
    monkeypatch.setattr(module, "coherent_sync_confirmation", degraded)
    assert module.run(args) == 1
    d = json.loads((args.output / "numeric.json").read_text())
    assert not d["gates"]["quality"] and d["original_success_losses"] == []


def test_no_evaluated_confirmation_cannot_pass_numeric_gate(numeric_case):
    module, args = numeric_case
    for directory in [args.before, args.after]:
        rows = json.loads((directory / "results.json").read_text())
        rows[0]["windows"] = []
        (directory / "results.json").write_text(json.dumps(rows))
    assert module.run(args) == 1
    assert not json.loads((args.output / "numeric.json").read_text())["gates"]["evaluated_nodes"]


def test_replaced_input_hash_or_node_file_is_rejected(numeric_case):
    module, args = numeric_case
    node = args.before / "evidence/sync.npz"
    node.write_bytes(node.read_bytes()+b"different")
    with pytest.raises(AssertionError):
        module.run(args)
