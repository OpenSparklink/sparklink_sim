"""Verify published IQ recipes against hashes recorded from the pre-B1-fix source."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

POLICY = Path("tests/data/legacy_corpora_sha256.json")
MIGRATION = Path("tests/data/legacy_corpora_migration.json")
POLICY_SHA256 = "10cdb6a6adf57384d5487882fc0a03e43265b54baf35f12aa0c0634787403f9d"
MIGRATION_SHA256 = "69437af6576199c0235524f4cecff5053417835c9574cfd34fb95b520e582fb0"


def frozen_policy():
    assert hashlib.sha256(POLICY.read_bytes()).hexdigest() == POLICY_SHA256
    return json.loads(POLICY.read_text())


def require_frozen_identity(key, original, observed):
    """Accept the original or one explicit byte-audited migration;never rewrite history."""
    if original == observed:
        return
    assert hashlib.sha256(MIGRATION.read_bytes()).hexdigest() == MIGRATION_SHA256
    entry = json.loads(MIGRATION.read_text())["identities"].get(key)
    assert entry is not None, (key, "unrecorded source change")
    assert original == entry["original"] and observed == entry["migrated"], key


def verify_group(group):
    """A migrated historical replay must also retain every original input byte."""
    assert list(records(group)) == frozen_policy()["groups"][group], group


def corpora():
    """Keep already-viewed experiments as regression; this is not new validation."""
    import audit_carrier_fixed as fixed
    import audit_carrier_loop as loop
    import audit_carrier_parallel as parallel
    import audit_carrier_prefix as prefix
    import audit_spline_fixed as spline
    import model_selected_frame as selected
    import model_spline_score as score

    groups = {"prefix-development": prefix.development_cases}
    for name, factory in (
        ("loop", loop.development),
        ("fixed", fixed.cases),
        ("parallel", parallel.cases),
        ("score", score.packets),
        ("spline", spline.streams),
    ):
        for held in (False, True):
            groups[name + ("-validation" if held else "-development")] = (
                lambda factory=factory, held=held: factory(held)
            )
    recipe = json.loads(selected.POLICY.read_text())
    for split in ("development", "validation"):
        groups["selected-" + split] = (
            lambda split=split: selected.fresh_cases(split, recipe)
        )
    return groups


def identity(array):
    value = np.asarray(array)
    return dict(
        dtype=value.dtype.str,
        shape=list(value.shape),
        bytes=value.nbytes,
        sha256=hashlib.sha256(value.tobytes()).hexdigest(),
    )


def records(group):
    for entry in corpora()[group]():
        if len(entry) == 4:
            metadata, profile, iq, payload = entry
            quiet = None  # Selected-frame recipes receive an already-owned window.
        else:
            metadata, profile, iq, quiet, payload = entry
        yield dict(
            id=metadata["id"],
            frame_type=profile.config.frame_type,
            mcs=profile.config.mcs_index,
            # Before the correction the legacy contract was implicit.
            length_contract=getattr(
                profile,
                "length_contract",
                "standard" if profile.config.frame_type == 2 else "legacy-b1-payload-length",
            ),
            iq=identity(iq),
            quiet=None if quiet is None else identity(quiet),
            payload=None if payload is None else identity(np.frombuffer(payload, np.uint8)),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = frozen_policy()
    assert frozen["source_commit"] == "54dd378982c868b2c196dcb76d35fc2fc715c1f0"
    args.output.mkdir(parents=True, exist_ok=False)
    observed = {}
    for group, expected in frozen["groups"].items():
        observed[group] = list(records(group))
        (args.output / (group + ".json")).write_text(
            json.dumps(observed[group], indent=2) + "\n"
        )
        assert observed[group] == expected, group
    result = dict(
        policy_sha256=hashlib.sha256(POLICY.read_bytes()).hexdigest(),
        original_source_commit=frozen["source_commit"],
        groups=len(observed),
        records=sum(map(len, observed.values())),
        exact_original_input_bytes=True,
        expected_payload_changed=False,
        receiver_default_changed=False,
        radio_reception_gain_claimed=False,
    )
    (args.output / "verified.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
