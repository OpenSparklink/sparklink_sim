"""Independent analytic CFO, word oracle and strict MAC confirmation examples."""

import cmath
import math
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.sync_coherent import (
    SyncConfirmationNCO,
    SyncConfirmationNCOFlow,
    coherent_sync_confirmation,
)
from tests.test_stream_receive import equal_events
from tests.test_strict_receive import iq_record, transmitted
from tests.test_sync_refine import profile


def independent_quality(y, s, omega):
    c = sum(complex(a)*complex(b).conjugate()*cmath.exp(-1j*omega*k)
            for k, (a, b) in enumerate(zip(y, s, strict=True)))
    d = sum(abs(complex(a))**2 for a in y)*sum(abs(complex(b))**2 for b in s)
    return abs(c)**2/d if d else 0.


@pytest.mark.parametrize("omega", [-math.pi, -.137, 0., .0467, math.pi])
def test_confirmation_matches_independent_known_sync_and_injected_cfo(omega):
    ref = profile().sync_symbols()
    y = 7.25*ref*np.array([cmath.exp(1j*(omega*k+.619)) for k in range(len(ref))])
    for fixed in [False, True]:
        result = coherent_sync_confirmation(y, ref, omega, .75, fixed_nco=fixed)
        assert result["accepted"] and abs(result["quality"]-1) < 2e-6
        if not fixed:
            assert abs(result["quality"]-independent_quality(y, ref, omega)) < 1e-14
        else:
            assert abs(result["nco"]["quantized_cycles_per_symbol"]-omega/(2*np.pi)) < 1e-5


def test_wrong_template_cfo_and_zero_energy_are_distinct_failures():
    rng = np.random.default_rng(570122)
    ref = profile().sync_symbols()
    y = ref*np.exp(.17j*np.arange(len(ref)))
    assert coherent_sync_confirmation(y, ref, .17, .75)["accepted"]
    assert not coherent_sync_confirmation(y, ref, 0., .75)["accepted"]
    wrong = rng.choice([1., -1.], len(ref)).astype(complex)
    assert not coherent_sync_confirmation(y, wrong, .17, .75)["accepted"]
    for a, b in [(np.zeros(32), ref), (ref, np.zeros(32))]:
        result = coherent_sync_confirmation(a, b, .17, .75)
        assert result["zero_energy"] and result["quality"] == 0 and not result["accepted"]


def test_squared_gate_equality_and_integer_input_cannot_overflow_silently():
    y = np.full(32, 10**10, np.int64)
    r = coherent_sync_confirmation(y, np.ones(32), 0., 1.)
    assert r["accepted"] and r["quality"] == 1.
    assert r["numerator"] == r["denominator"]
    r = coherent_sync_confirmation(np.ones(32), np.ones(32), 0., 1.)
    assert r["accepted"] and r["numerator"] >= r["threshold"]*r["denominator"]


@pytest.mark.parametrize("step", [0, 128, 256, 262144, 524288, 786432, 1048575, 37115])
def test_nco_word_oracle_quadrants_ties_wrap_and_chunk_state(step):
    whole, split = SyncConfirmationNCO(step), SyncConfirmationNCO(step)
    a, b = whole.advance(128)
    parts = [split.advance(n) for n in [1, 17, 32, 3, 75]]
    np.testing.assert_array_equal(a, np.concatenate([p[0] for p in parts]))
    np.testing.assert_array_equal(b, np.concatenate([p[1] for p in parts]))
    for k, (real, imag) in enumerate(zip(a, b, strict=True)):
        phase = (step*k) % 2**20
        address = round(Fraction(phase, 256)) % 4096
        expected = cmath.exp(-2j*math.pi*address/4096)*65536
        assert abs(real-expected.real) <= .50001
        assert abs(imag-expected.imag) <= .50001
        assert -(2**17) <= real < 2**17 and -(2**17) <= imag < 2**17
    assert whole.phase == split.phase == (128*step) % 2**20
    with pytest.raises(ValueError):
        whole.advance(1)


def test_nco_range_and_resource_estimate_is_not_whole_confirmation():
    for step in [-1, 2**20, True, .5]:
        with pytest.raises(ValueError):
            SyncConfirmationNCO(step)
    for count in [0, -1, 129, True]:
        with pytest.raises(ValueError):
            SyncConfirmationNCO(3).advance(count)
    e = SyncConfirmationNCO.estimate()
    assert e["logical_bits"] == sum(e["components"].values())
    assert e["components"]["quarter_cosine_rom"] == 17408
    assert e["rom_read_ports"] == 2 and e["ii"] == 1
    assert not e["synthesis_evidence"] and not e["whole_system_accepted"]


def test_nco_cycle_calendar_backpressures_without_advancing_or_losing_words():
    flow = SyncConfirmationNCOFlow()
    reference = flow.run(37115, 128)
    assert reference["max_latency_cycles"] == 4
    assert reference["input_backpressure_cycles"] == 0
    assert not reference["incomplete"] and reference["peak_pipeline_tokens"] == 4
    blocked = flow.run(37115, 128, stalls=[(8, 73)])
    assert blocked["input_backpressure_cycles"] > 0 and not blocked["incomplete"]
    assert blocked["max_latency_cycles"] > 4
    assert [(r["real"], r["imag"]) for r in reference["emitted"]] == [
        (r["real"], r["imag"]) for r in blocked["emitted"]]
    stopped = flow.run(37115, 128, stalls=[(0, 1000)], max_cycles=100)
    assert stopped["incomplete"] and stopped["accepted"] == 4
    assert stopped["pending_tokens"] == 4 and not stopped["emitted"]
    assert stopped["final_phase"] == (4*37115) % 2**20
    for options in [dict(count=0), dict(max_cycles=0), dict(stalls=[(4, 8), (7, 9)]),
                    dict(stalls=[(-1, 2)]), dict(stalls=[(0, 0)]), dict(stalls=1)]:
        args = dict(count=32)
        args.update(options)
        with pytest.raises(ValueError):
            flow.run(3, **args)


@pytest.mark.parametrize("options", [dict(omega=float("nan")), dict(omega=4.),
    dict(omega=True), dict(omega="invalid"), dict(threshold=0.), dict(threshold=1.01),
    dict(threshold=True), dict(threshold=float("nan")), dict(fixed_nco=1)])
def test_invalid_confirmation_configuration_rejected(options):
    kwargs = dict(omega=.02, threshold=.6, fixed_nco=False)
    kwargs.update(options)
    with pytest.raises(ValueError):
        coherent_sync_confirmation(np.ones(32), np.ones(32), **kwargs)


def test_confirmation_size_nonfinite_and_dynamic_range_contract():
    for y, ref in [(np.ones(1), np.ones(1)), (np.ones(129), np.ones(129)),
                   (np.ones(32), np.ones(31)), (np.ones(32)*np.nan, np.ones(32))]:
        with pytest.raises(ValueError):
            coherent_sync_confirmation(y, ref, 0., .6)
    with pytest.raises(OverflowError):
        coherent_sync_confirmation(np.full(32, 1e100), np.ones(32), 0., .6)


@pytest.mark.parametrize("fixed", [False, True])
def test_actual_mac_has_no_crc_based_retry_and_chunk_state_matches(fixed):
    p = profile()
    wire, wanted = transmitted(p)
    x, quiet, _, _ = iq_record(wire, offset=.277, cfo=23000, ppm=100)
    options = dict(quiet_batching=False, correlation_backend="scaled", normalized_coarse=True,
                   sync_search_method="bounded-grid", sync_coherent_threshold=.6,
                   sync_coherent_fixed_nco=fixed)
    rx = MacStreamReceiver(p, quiet, **options)
    packets, events = [], []
    for start in range(0, len(x), 257):
        r = rx.feed(x[start:start+257])
        packets += r.packets
        events += r.diagnostics["events"]
    r = rx.finish()
    packets += r.packets
    events += r.diagnostics["events"]
    assert [p.mac_payload for p in packets] == [wanted]
    assert events[0]["sync_confirmation"]["accepted"]
    assert (events[0]["sync_confirmation"]["nco"] is not None) == fixed
    whole = PSKStreamReceiver(p, quiet, **options)
    equal_events(events, whole.feed(x)+whole.finish())
    assert rx.phy.capture_stats["coherent_evaluations"] >= 1
    assert rx.phy.capture_stats["coherent_accepted"] == 1


@pytest.mark.parametrize("options", [dict(sync_coherent_fixed_nco=True),
    dict(sync_coherent_threshold=.6), dict(sync_coherent_threshold=True),
    dict(sync_coherent_threshold=-.1), dict(sync_coherent_threshold=.6, normalized_coarse=True),
    dict(sync_coherent_threshold=.6, normalized_coarse=True, sync_search_method="bounded-grid",
         sync_candidate_count=2)])
def test_receiver_requires_explicit_single_factor_configuration(options):
    with pytest.raises(ValueError):
        PSKStreamReceiver(profile(), np.zeros(2048), quiet_batching=False,
                          correlation_backend="scaled", **options)


def test_high_differential_correlation_does_not_bypass_coherent_rejection():
    p = profile()
    wire, _ = transmitted(p)
    x, quiet, start, _ = iq_record(wire, offset=.277)
    # A smooth large phase excursion has small adjacent phase increments but
    # poor coherent accumulation. This is a diagnostic distorted waveform,
    # not an injected truth available to the receiver.
    x *= np.exp(2j*np.sin(2*np.pi*(np.arange(len(x))-start)/128))
    options = dict(quiet_batching=False, correlation_backend="scaled", normalized_coarse=True,
                   sync_search_method="bounded-grid")
    old, new = PSKStreamReceiver(p, quiet, **options), PSKStreamReceiver(
        p, quiet, sync_coherent_threshold=.6, **options)
    old.feed(x)
    old.finish()
    events = new.feed(x)+new.finish()
    assert old.capture_stats["sync_candidates"] > 0
    assert new.capture_stats["coherent_rejections"] > 0
    assert not any(e["frame"]["status"] == "valid" for e in events)


def test_confirmation_cost_keeps_storage_failure_and_queue_loss(tmp_path, monkeypatch):
    import importlib
    import json
    from pathlib import Path
    from types import SimpleNamespace

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("model_sync_coherent_work")
    assert module.run(SimpleNamespace(output=tmp_path / "cost")) == 1
    report = json.loads((tmp_path / "cost/model.json").read_text())
    assert not report["whole_system_accepted"]
    assert report["operations"]["gate_divides"] == 0
    assert report["nco"]["rom_read_ports"] == 2
    for row in report["scenarios"]:
        assert row["known_capture_partial_total_bits"] > row["memory_budget_bits"]
        assert not row["direct_graph_within_partial_product_budget"]
        assert row["queue"]["dropped_requests"] > 0
    assert report["nco_calendars"]["stopped"]["incomplete"]
