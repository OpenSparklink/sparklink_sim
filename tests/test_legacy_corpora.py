"""Published waveform bytes must survive the standard B1 length correction."""

import json
from pathlib import Path

import pytest
from scripts import check_frozen_corpora as check

FROZEN = check.frozen_policy()
MIGRATIONS = json.loads(check.MIGRATION.read_text())["identities"]


@pytest.mark.parametrize("group", tuple(FROZEN["groups"]))
def test_published_recipe_matches_pre_fix_input_member_hashes(group, monkeypatch):
    # Historical command-line studies import shared scripts by their original names.
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    assert list(check.records(group)) == FROZEN["groups"][group]


@pytest.mark.parametrize("key", tuple(MIGRATIONS))
def test_frozen_identity_accepts_only_original_or_exact_audited_migration(key):
    row = MIGRATIONS[key]
    check.require_frozen_identity(key, row["original"], row["original"])
    check.require_frozen_identity(key, row["original"], row["migrated"])
    with pytest.raises(AssertionError):
        check.require_frozen_identity(key, row["original"], "0" * 64)
    with pytest.raises(AssertionError):
        check.require_frozen_identity(key, "0" * 64, row["migrated"])


def test_unrecorded_source_change_is_rejected():
    with pytest.raises(AssertionError):
        check.require_frozen_identity("src/unrecorded.py", "1" * 64, "2" * 64)


@pytest.mark.parametrize("name, operation", [
    ("POLICY", lambda: check.frozen_policy()),
    ("MIGRATION", lambda: check.require_frozen_identity(
        next(iter(MIGRATIONS)), "1" * 64, "2" * 64)),
])
def test_changed_migration_or_original_input_ledger_is_rejected(name, operation, tmp_path,
                                                               monkeypatch):
    path = tmp_path / "altered.json"
    path.write_text('{"groups": {}, "identities": {}}')
    monkeypatch.setattr(check, name, path)
    with pytest.raises(AssertionError):
        operation()
