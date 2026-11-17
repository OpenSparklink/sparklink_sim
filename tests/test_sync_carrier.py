"""Independent phase ramps, finite differences and singular/finite-word boundaries."""

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.sync_carrier import initialize_sync_carrier, sync_carrier_work


@pytest.mark.parametrize("n", [2, 32, 62, 126, 128])
def test_independent_clean_phase_ramp(n):
    rng = np.random.default_rng(590118)
    ref = np.exp(1j*rng.uniform(-np.pi, np.pi, n))
    for omega in [-3., -.023, 0., .14, 3.]:
        y = ref*np.exp(1j*(omega*np.arange(n)+.37))
        out = initialize_sync_carrier(y, ref, omega+.001, 2)
        assert abs(out["omega_radians_per_symbol"]-omega) < 1e-8
        assert abs(out["phase_radians"]-.37) < 1e-6


def test_moment_derivatives_match_independent_finite_differences():
    rng = np.random.default_rng(590119)
    y = rng.normal(size=32)+1j*rng.normal(size=32)
    scale = max(max(abs(y.real)), max(abs(y.imag)))
    normalized = [complex(a.real/scale, a.imag/scale) for a in y]
    seed, epsilon = .024, 1e-5

    def objective(w):
        correlation = sum(v*complex(math.cos(w*k), -math.sin(w*k))
                          for k, v in enumerate(normalized))
        return correlation.real**2+correlation.imag**2

    update = initialize_sync_carrier(y, np.ones(32), seed, 1)["updates"][0]
    a, b, c = objective(seed-epsilon), objective(seed), objective(seed+epsilon)
    assert update["gradient"] == pytest.approx((c-a)/(2*epsilon), abs=2e-4)
    assert update["curvature"] == pytest.approx((a-2*b+c)/epsilon**2, abs=.05)


@pytest.mark.parametrize("amplitude", [1e-320, 1e-100, 1., 1e100, 1e300])
def test_finite_extreme_component_normalization(amplitude):
    with np.errstate(all="raise"):
        out = initialize_sync_carrier(np.full(32, amplitude+0j), np.ones(32), 0., 1)
    assert out["initialized"] and out["quality"] == pytest.approx(1.)
    assert out["phase_radians"] == 0.


def test_zero_and_cancelled_correlations_never_initialize():
    assert initialize_sync_carrier(np.zeros(32), np.ones(32), 0., 1) == dict(
        initialized=False, reason="zero_energy")
    out = initialize_sync_carrier([-1., 1.], [1., 1.], 0., 2)
    assert not out["initialized"] and out["reason"] == "zero_coherent_sum"
    assert all(u["disposition"] == "curvature_hold" for u in out["updates"])


@pytest.mark.parametrize("options", [dict(seed=True), dict(seed="0"), dict(seed=4.),
    dict(iterations=True), dict(iterations=0), dict(iterations=3), dict(fixed_nco=1),
    dict(samples=np.zeros(129), reference=np.ones(129)), dict(reference=np.zeros(32)),
    dict(samples=np.full(32, complex(np.inf))), dict(reference=np.ones((1, 32)))])
def test_invalid_nodes_fail_explicitly(options):
    args = dict(samples=np.ones(32), reference=np.ones(32), seed=0., iterations=1)
    args.update(options)
    with pytest.raises(ValueError):
        initialize_sync_carrier(**args)


def test_capped_steps_and_fixed_nco_node():
    ref = np.ones(32)
    for seed in np.linspace(-.2, .2, 21):
        out = initialize_sync_carrier(ref, ref, float(seed), 2)
        assert all(abs(u["step"]) <= np.pi/32 for u in out["updates"])
    omega = .0423
    y = np.exp(1j*(omega*np.arange(32)+.2))
    a = initialize_sync_carrier(y, ref, omega+.001, 2)
    b = initialize_sync_carrier(y, ref, omega+.001, 2, fixed_nco=True)
    assert abs(b["omega_radians_per_symbol"]-a["omega_radians_per_symbol"])/(2*np.pi) < 1e-5
    assert abs(b["phase_radians"]-a["phase_radians"]) < .01
    assert not b["whole_system_accepted"]


def test_cost_keeps_dedicated_state_divisions_and_unknown_formats():
    a, b = sync_carrier_work(1), sync_carrier_work(2)
    assert a["phase_atan2"] == 1 and a["old_phase_atan2_per_sync"] == 32
    assert a["literal_data_divisions"] == 64 and a["dedicated_nco_bits"] == 17610
    assert b["literal_real_product_equivalents"] > a["literal_real_product_equivalents"]
    assert a["extra_symbol_and_accumulator_word_bits"] is None
    for options in [(0, 32), (True, 32), (1, 31)]:
        with pytest.raises(ValueError):
            sync_carrier_work(*options)


def test_empty_numeric_evaluation_fails_even_without_packet_regression(tmp_path, monkeypatch):
    import importlib
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_sync_carrier_numeric")
    (tmp_path / "results.json").write_text("[]")
    budget = tmp_path / "budget.json"
    budget.write_text(json.dumps(dict(frequency_cycles_difference_budget=1e-5,
                                     phase_difference_radians_budget=.01,
                                     quality_difference_budget=.005)))
    output = tmp_path / "out"
    assert audit.run(SimpleNamespace(capture=tmp_path, budget=budget, output=output)) == 1
    assert json.loads((output / "numeric.json").read_text())["gates"]["nonempty"] is False


def test_serial_cost_scenarios_retain_overload_and_unknown_word_formats(tmp_path, monkeypatch):
    import importlib
    import json

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    model = importlib.import_module("model_sync_carrier_work")
    output = tmp_path / "model"
    assert model.run(output) == 1
    report = json.loads((output / "model.json").read_text())
    assert not report["whole_system_accepted"] and len(report["scenarios"]) == 6
    assert report["coefficient_calendar"]["pending_tokens"] == 0
    for scenario in report["scenarios"]:
        assert scenario["known_partial_lower_bits"] > scenario["budget_bits"]
        assert scenario["queue"]["dropped_requests"] > 0
        assert scenario["data_accumulator_and_port_storage"] is None
        stages = scenario["initialization_stages"]
        assert all(a["stop"] <= b["start"] for a, b in pairwise(stages))


def test_held_harness_refuses_unfrozen_steps_or_generic_regression_bypass(tmp_path, monkeypatch):
    import importlib
    import json
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_sync_carrier")
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(dict(initializer_iterations=2)))
    args = SimpleNamespace(iterations=1, fixed_initializer_nco=False, selection=selection,
                           inputs=tmp_path, output=tmp_path / "no-output")
    with pytest.raises(ValueError, match="frozen"):
        audit.run(args)
    args.viewed_regression = True
    with pytest.raises(ValueError, match="original"):
        audit.run(args)
    assert not args.output.exists()


@pytest.mark.parametrize("chunks", ["whole", "random"])
def test_rejected_initializer_cannot_regress_existing_raw_iq_mac_success(chunks):
    import hashlib
    import json

    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tone_notch import ToneNotchConfig
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    folder = Path(__file__).parent / "data"
    path = folder / "sync_carrier_tone_success.npz"
    meta = json.loads((folder / "sync_carrier_tone_success.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["derived_npz_sha256"]
    with np.load(path) as data:
        iq, quiet, wanted = data["iq"], data["quiet"], data["expected"].tobytes()
    assert hashlib.sha256(iq.tobytes()).hexdigest() == meta["iq_bytes_sha256"]
    cfg = TxConfig(mcs_index=6, pid=0, symbol_rate_mhz=4,
                   whitening_seed=0x52, crc_seed=0x123456)
    rx = MacStreamReceiver(PSKReceiveProfile.ordinary(cfg), quiet,
        quiet_batching=False, defer_quiet_fir=False, correlation_backend="scaled",
        normalized_coarse=True, tone_notch=ToneNotchConfig(radius=31/32),
        sync_search_method="bounded-grid", sync_candidate_count=1, sync_coherent_threshold=.45)
    packets, frames, offset = [], [], 0
    rng = np.random.default_rng(600120)
    while offset < len(iq):
        size = len(iq) if chunks == "whole" else int(rng.integers(1, 1001))
        result = rx.feed(iq[offset:offset+size])
        packets.extend(result.packets)
        frames.extend(result.diagnostics["frames"])
        offset += size
    result = rx.finish()
    packets.extend(result.packets)
    frames.extend(result.diagnostics["frames"])
    assert [p.mac_payload for p in packets] == [wanted]
    assert len(frames) == 1
    frame = frames[0]
    assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
    assert frame["header"]["data_length"] == len(wanted) == 511 and frame["mcs"] == 6
