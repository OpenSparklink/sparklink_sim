"""Model adoption must retain failures and distinguish unreached nodes from success."""

import importlib.util
import json
from pathlib import Path

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
