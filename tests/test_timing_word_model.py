"""Model adoption must retain failures and distinguish unreached nodes from success."""

import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat


def test_development_gate_does_not_require_pressure_recovery_or_pass_unreached_node(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "timing_model_gate", scripts / "model_timing_words.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = json.loads((scripts.parent / "tests/data/timing_word_policy.json").read_text())
    numerical = dict(
        power_relative_max=1e-9,
        TED_absolute_max=1e-6,
        cost=dict(DSP25x18=4, state_register_bits_upper=2510, extra_raw_RAM_bits=0),
    )
    reached = dict(
        handoff_available=True,
        success_loss=False,
        false_delivery=False,
        center_delta_max=1e-5,
        numerical=numerical,
    )
    unavailable = dict(baseline=dict(payload_exact=False), candidates={})
    failed_packet = dict(baseline=dict(payload_exact=False), candidates={"30": reached})
    assert module.development_eligible(
        [unavailable, failed_packet], TimingWordFormat(30, 32), policy
    )
    assert not module.eligible(dict(handoff_available=False), policy)
    assert not module.development_eligible([unavailable], TimingWordFormat(30, 32), policy)
    lost = dict(baseline=dict(payload_exact=True), candidates={})
    assert not module.development_eligible([lost, failed_packet], TimingWordFormat(30, 32), policy)
    for change in [
        dict(success_loss=True),
        dict(false_delivery=True),
        dict(center_delta_max=0.002),
    ]:
        assert not module.eligible(reached | change, policy)


@pytest.fixture
def fixed_input(tmp_path, monkeypatch):
    """Independent container change with unchanged original member bytes."""
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    import model_timing_words as model

    from tests.test_capture_fine import archive_reference

    original = tmp_path / "original.npz"
    np.savez_compressed(original, iq=np.array([1 + 2j]), payload=np.array([5], np.uint8))
    expected = archive_reference(original)
    derived = tmp_path / "stream-input.npz"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(derived, "w") as target:
        for name in reversed(source.namelist()):
            target.writestr(name, source.read(name), compress_type=zipfile.ZIP_STORED)
    inputs, members = tmp_path / "inputs.json", tmp_path / "members.json"
    inputs.write_text('{"fixture":"original input index"}')
    members.write_text(
        json.dumps(dict(source_index_sha256=model.sha(inputs), entries=dict(stream=expected)))
    )
    monkeypatch.setattr(model, "INPUTS", inputs)
    monkeypatch.setattr(model, "MEMBERS", members)
    receipt = model.verify_input_container(derived, expected, require_original=False)
    protocol = dict(
        input_mode="derived-container-original-members-verified",
        input_index_sha256=model.sha(inputs),
        input_members_sha256=model.sha(members),
    )
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "input-containers.json").write_text(json.dumps(dict(stream=receipt)))
    return model, tmp_path, original, receipt, protocol


def test_fixed_input_preserves_original_and_current_container_identity(fixed_input):
    model, root, original, receipt, protocol = fixed_input
    assert receipt["current_archive_sha256"] != receipt["original_archive_sha256"]
    assert model.fixed_input_provenance(root, "stream", model.sha(original)) == receipt
    protocol["input_mode"] = "frozen"
    (root / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="Frozen original archive SHA mismatch"):
        model.fixed_input_provenance(root, "stream", model.sha(original))
    (root / "stream-input.npz").write_bytes(original.read_bytes())
    exact = model.verify_input_container(original, receipt, require_original=True)
    (root / "input-containers.json").write_text(json.dumps(dict(stream=exact)))
    assert model.fixed_input_provenance(root, "stream", model.sha(original)) == exact


@pytest.mark.parametrize("change", ["index", "original", "receipt", "mode", "protocol", "member"])
def test_fixed_input_rejects_changed_provenance_or_bytes(fixed_input, change):
    model, root, original, receipt, protocol = fixed_input
    expected_sha = model.sha(original)
    if change == "index":
        model.INPUTS.write_text("changed index")
    elif change == "original":
        expected_sha = "0" * 64
    elif change == "receipt":
        receipt["current_archive_sha256"] = "0" * 64
        (root / "input-containers.json").write_text(json.dumps(dict(stream=receipt)))
    elif change == "mode":
        protocol["input_mode"] = "unchecked"
    elif change == "protocol":
        protocol["input_members_sha256"] = "0" * 64
    else:
        np.savez(root / "stream-input.npz", iq=np.array([1 + 2j]), payload=np.array([4], np.uint8))
    (root / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        model.fixed_input_provenance(root, "stream", expected_sha)
