"""Evidence output may live outside the immutable input root, including /tmp."""

import hashlib
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("external", [False, True])
def test_manifest_index_resolves_every_artifact_without_changing_sources(tmp_path, external):
    index = runpy.run_path(
        Path(__file__).resolve().parents[1] / "scripts/summarize_capture_front.py"
    )["manifest_hashes"]
    root = tmp_path / "frozen-evidence"
    output = tmp_path / "external-output" if external else root / "new-output"
    root.mkdir()
    source = root / "original.npz"
    source.write_bytes(b"immutable original IQ")
    expected = {}
    for stage, content in [("original", b'{"result":1}'), ("held-out", b'{"result":2}')]:
        path = output / stage / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        expected[path.resolve()] = hashlib.sha256(content).hexdigest()
    result = index(root, output)
    assert {(root / p).resolve(): h for p, h in result.items()} == expected
    assert len(result) == 2
    assert source.read_bytes() == b"immutable original IQ"
    if not external:
        assert set(result) == {"new-output/original/manifest.json", "new-output/held-out/manifest.json"}
