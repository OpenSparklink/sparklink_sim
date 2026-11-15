"""Optional convenience and mandatory CI build outcomes must be distinguishable."""

import importlib.util
from pathlib import Path
from subprocess import CompletedProcess

spec = importlib.util.spec_from_file_location(
    "install_accel", Path(__file__).parents[1] / "scripts/install_accel.py"
)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_missing_toolchain_is_not_a_required_build_success(monkeypatch):
    monkeypatch.setattr(installer.shutil, "which", lambda tool: None)
    assert installer.main([]) == 0
    assert installer.main(["--required"]) != 0


def test_failed_required_build_retains_full_diagnostic_and_nonzero_status(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(installer, "RUST_DIR", tmp_path)
    monkeypatch.setattr(installer.shutil, "which", lambda tool: "/test/" + tool)
    message = "ROOT BUILD FAILURE\n" + "diagnostic\n" * 100
    monkeypatch.setattr(
        installer.subprocess, "run", lambda *a, **k: CompletedProcess(a, 2, "", message)
    )
    assert installer.main(["--required"]) == 2
    assert message in capsys.readouterr().err


def test_successful_build_uses_the_lock_and_does_not_claim_runtime_verification(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(installer, "RUST_DIR", tmp_path)
    monkeypatch.setattr(installer.shutil, "which", lambda tool: "/test/" + tool)

    def build(command, **kwargs):
        assert "--locked" in command and "--frozen" in command
        return CompletedProcess(command, 0, "wheel installed", "")

    monkeypatch.setattr(installer.subprocess, "run", build)
    assert installer.main(["--required"]) == 0
    assert "运行时还须核验" in capsys.readouterr().out
