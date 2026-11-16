"""Cost-only source lineage cannot waive word, schedule or unrelated source changes."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def context(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    driver = importlib.import_module("audit_rrc_word")
    monkeypatch.chdir(tmp_path)
    flow = Path("src/nearlink_sdr/phy/rrc_flow.py")
    flow.parent.mkdir(parents=True)
    original = (
        "class RRCFlow:\n"
        "    def resource_estimate(self):\n        return {'bits': 8}\n"
        "    def simulate(self):\n        return 17\n"
    )
    flow.write_text(original)
    word = flow.with_name("rrc_word.py")
    word.write_text("PORT_BITS = 19\n")

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    selection = dict(sources={str(flow): digest(flow), str(word): digest(word)})
    selected = tmp_path / "selection.json"
    selected.write_text(json.dumps(selection))
    old = tmp_path / "frozen-flow.py"
    old.write_text(original)
    args = SimpleNamespace(selection=selected, cost_lineage=None)
    return driver, flow, word, old, selection, args, digest


def pin_lineage(flow, old, args, digest):
    args.cost_lineage = old.with_name("lineage.json")
    args.cost_lineage.write_text(
        json.dumps(
            dict(
                selection_sha256=digest(args.selection),
                old_source=str(old),
                old_sha256=digest(old),
                new_sha256=digest(flow),
            )
        )
    )


def test_exact_source_needs_no_exception_and_missing_lineage_rejects(context):
    driver, flow, _, _, selection, args, _ = context
    driver.check_selection_sources(selection, args)
    flow.write_text(flow.read_text().replace("'bits': 8", "'bits': 16"))
    with pytest.raises(ValueError, match="Frozen source mismatch"):
        driver.check_selection_sources(selection, args)


def test_zero_reference_and_unimplemented_pair_cycle_are_explicit(context):
    driver = context[0]
    assert driver.relative_norm([0, 0], [0, 0]) == 0
    assert driver.relative_norm([1, 0], [0, 0]) == float("inf")
    assert driver.relative_norm([3, 4], [6, 8]) == 0.5
    driver.check_model_schedule("global", True)
    driver.check_model_schedule("pair", False)
    with pytest.raises(ValueError, match="no reference reuse"):
        driver.check_model_schedule("pair", True)


def test_only_exactly_pinned_cost_method_can_change(context):
    driver, flow, _, old, selection, args, digest = context
    flow.write_text(flow.read_text().replace("'bits': 8", "'bits': 16"))
    pin_lineage(flow, old, args, digest)
    driver.check_selection_sources(selection, args)
    flow.write_text(flow.read_text().replace("'bits': 16", "'bits': 32"))
    with pytest.raises(AssertionError):
        driver.check_selection_sources(selection, args)


@pytest.mark.parametrize("change", ["schedule", "word", "imports", "old-source"])
def test_lineage_never_waives_algorithm_or_old_reference_integrity(context, change):
    driver, flow, word, old, selection, args, digest = context
    flow.write_text(flow.read_text().replace("'bits': 8", "'bits': 16"))
    if change == "schedule":
        flow.write_text(flow.read_text().replace("return 17", "return 18"))
    elif change == "word":
        word.write_text("PORT_BITS = 18\n")
    elif change == "imports":
        flow.write_text("import sys\n" + flow.read_text())
    elif change == "old-source":
        old.write_text(old.read_text().replace("return 17", "return 19"))
    pin_lineage(flow, old, args, digest)
    with pytest.raises((AssertionError, ValueError)):
        driver.check_selection_sources(selection, args)
