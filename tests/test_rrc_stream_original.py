"""Raw-gate output proof stays valid even with active tone suppression, not state proof."""

import importlib
from pathlib import Path

import numpy as np


def test_quiet_word_output_proof_does_not_assume_rrc_or_tone_inactive(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    module = importlib.import_module("verify_rrc_stream_original")
    from audit_capture_hardware import profile

    rng = np.random.default_rng(510124)
    quiet = 30 * (rng.normal(size=4096) + 1j * rng.normal(size=4096))
    # Coherent input can activate the notch while raw energy remains below
    # the calibration threshold. The proof certifies no search, not no notch.
    tone = 10 * np.exp(0.17j * np.arange(4096))
    proof = module.quiet_output_proof(tone, quiet)
    assert proof is not None and proof["fresh_whole_stream_only"]
    assert not proof["rrc_state_verified"] and not proof["tone_state_verified"]
    assert not proof["range_verified"] and "rrc_component_upper_bound" not in proof
    rx = module.MacStreamReceiver(
        profile(),
        quiet,
        quiet_batching=False,
        defer_quiet_fir=False,
        rrc_backend="word-full12-c18",
        correlation_backend="scaled",
        correlation_mantissa_bits=10,
        tone_notch=module.ToneNotchConfig(radius=0.96875),
    )
    first, last = rx.feed(tone), rx.finish()
    assert not first.packets and not last.packets
    assert not first.diagnostics["events"] and not last.diagnostics["events"]
    assert rx.phy.capture_stats["sync_candidates"] == 0
    assert rx.phy.tone_notch.stats["active_samples"] > 0
    assert module.quiet_output_proof(tone * 100, quiet) is None
