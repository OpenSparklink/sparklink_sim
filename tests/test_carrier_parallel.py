"""Independent detector coordinates and exact finite parallel calendar checks."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_fixed import CarrierWordFormat, cordic_phase_word
from nearlink_sdr.phy.carrier_parallel import ParallelAngleCarrierLoop, track_parallel_carrier
from nearlink_sdr.sim.carrier_parallel_flow import (
    ParallelCarrierEngine,
    parallel_carrier_cost,
    simulate_parallel_carrier,
)


def test_raw_coordinate_phase_subtraction_and_fraction_pi_oracle():
    for phase in (-(2**31), -1234567, 0, 981745, 2**31 - 1):
        word = ParallelAngleCarrierLoop("8PSK", 3)
        word.phase = phase
        row = word.feed_words([913], [527])
        angle = cordic_phase_word(913, 527) - phase + 2**28
        sector = 2**29
        expected = min(sector // 4, max(-sector // 4, angle % sector - sector // 2))
        assert int(row["error_word"][0]) == expected
        frequency = round(Fraction(expected, 2048))
        assert word.frequency == frequency
        next_phase = phase + frequency + round(Fraction(expected, 32))
        assert word.phase == (next_phase + 2**31) % 2**32 - 2**31
    with pytest.raises(ValueError):
        ParallelAngleCarrierLoop("QPSK", 0, fmt=CarrierWordFormat(12))


@pytest.mark.parametrize("mod,m", [("BPSK", 2), ("QPSK", 4), ("8PSK", 8)])
def test_independent_carrier_truth_state_across_chunks_fields_and_eof(mod, m):
    rng = np.random.default_rng(780119 + m)
    k = np.arange(4064)
    phases = 0.06 + 2 * np.pi * 0.00017 * k
    angle = np.r_[
        (2 * rng.integers(0, 4, 64) + 1) * np.pi / 4 - ((k[:64] + 3) % 2) * np.pi / 4,
        (2 * rng.integers(0, m, 4000) + 1) * np.pi / m - ((k[64:] + 3) % 2) * np.pi / m,
    ]
    y = np.exp(1j * (angle + phases))
    out, info, trace = track_parallel_carrier(
        y[64:], mod, 67, header_symbols=y[:64], header_mod_type="QPSK"
    )
    error = phases[64:] - np.unwrap(trace["phase_word"] * 2 * np.pi / 2**32)
    assert np.sqrt(np.mean(error[2000:] ** 2)) < 0.08
    assert not np.any(np.floor((error + np.pi / m) / (2 * np.pi / m)))
    assert np.isfinite(out).all() and info["iterations"] == 18
    node = ParallelAngleCarrierLoop("QPSK", 3)
    real, imag = np.rint(y.real * 1024).astype(int), np.rint(y.imag * 1024).astype(int)
    for start in range(0, 64, 7):
        node.feed_words(real[start : min(start + 7, 64)], imag[start : min(start + 7, 64)])
    node.switch_modulation(mod)
    parts = [
        node.feed_words(real[start : start + 11], imag[start : start + 11])
        for start in range(64, len(y), 11)
    ]
    assert all(
        np.array_equal(v, np.concatenate([part[key] for part in parts])) for key, v in trace.items()
    )
    assert node.finish()["final_phase_word"] == info["final_phase_word"]
    with pytest.raises(RuntimeError):
        node.feed_words([1], [1])
    for kwargs in ({"header_symbols": y[:64], "header_mod_type": "QPSK"}, {"fraction_bits": 12}):
        with pytest.raises(ValueError):
            track_parallel_carrier(y[:32], mod, 0, **kwargs)


def test_parallel_cycle_values_snapshots_and_all_protection_states():
    rng = np.random.default_rng(780120)
    engine = ParallelCarrierEngine("8PSK", 3)
    direct = ParallelAngleCarrierLoop("8PSK", 3)
    with pytest.raises(RuntimeError):
        engine.tick()
    cases = [(0, 0), (1, 0), (131071, 131071), (-131072, 131071), (1024, 1), (-1024, -1)]
    cases += [tuple(map(int, r)) for r in rng.integers(-2000, 2001, (64, 2))]
    for index, (real, imag) in enumerate(cases):
        if index == 32:
            engine.node.switch_modulation("QPSK")
            direct.switch_modulation("QPSK")
        engine.begin(real, imag)
        ticks = [engine.tick() for _ in range(32)]
        assert ticks[2]["x"] is not None and ticks[3]["angle"] is not None
        assert not any(t["done"] for t in ticks[:-1]) and ticks[-1]["done"]
        with pytest.raises(RuntimeError):
            engine.tick()
        actual = engine.take()
        expected = direct.feed_words([real], [imag])
        for key, outkey in [
            ("real", "real"),
            ("imag", "imag"),
            ("phase", "phase_word"),
            ("frequency", "frequency_word"),
            ("error", "error_word"),
            ("held", "held"),
        ]:
            assert actual[key] == expected[outkey][0]
        assert engine.node.phase == direct.phase and engine.node.frequency == direct.frequency
        assert engine.node.stats == direct.stats and engine.node.power.power == direct.power.power
    assert engine.node.finish() == direct.finish()


def test_finite_parallel_source_overload_stalls_and_port_allocation():
    k = np.arange(512)
    real = np.rint(1024 * np.cos(k * 0.31)).astype(int)
    imag = np.rint(1024 * np.sin(k * 0.31)).astype(int)
    for clock in (100000000, 160000000, 200000000):
        r = simulate_parallel_carrier(
            real, imag, clock_hz=clock, modulation="8PSK", start=3, header_symbols=64
        )
        assert len(r["emitted"]) + len(r["dropped"]) == 512
        assert r["resource_packet_invalid"] == (clock == 100000000)
        if clock != 100000000:
            node = ParallelAngleCarrierLoop("QPSK", 3)
            expected = []
            for j, (a, b) in enumerate(zip(real, imag, strict=True)):
                if j == 64:
                    node.switch_modulation("8PSK")
                expected.append(node.feed_words([a], [b]))
            assert all(
                r["emitted"][j]["result"]["error"] == v["error_word"][0]
                for j, v in enumerate(expected)
            )
    blocked = simulate_parallel_carrier(real, imag, clock_hz=160000000, stalls=((0, 1000),))
    assert blocked["resource_packet_invalid"] and len(blocked["dropped"]) == 512
    shallow = simulate_parallel_carrier(
        real, imag, clock_hz=160000000, source_batch=8, input_depth=4
    )
    deep = simulate_parallel_carrier(real, imag, clock_hz=160000000, source_batch=8, input_depth=8)
    assert shallow["resource_packet_invalid"] and not deep["resource_packet_invalid"]
    a, b = parallel_carrier_cost(), parallel_carrier_cost(8)
    assert a["service_clocks"] == 32 and a["DSP25x18_tiles"] == 6
    assert b["logical_bits"] - a["logical_bits"] == 224
    assert a["additional_PI_RNE_LUT_FF"] is None and not a["actual_synthesis_or_timing"]


@pytest.mark.parametrize("ft,mcs,pilot", [(2, 8, 0), (3, 6, 8), (4, 12, 8)])
def test_actual_iq_to_mac_and_exact_selected_method_ber_replay(monkeypatch, ft, mcs, pilot):
    import importlib
    from pathlib import Path

    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tx_pipeline import TxConfig
    from tests.test_strict_receive import iq_record

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    driver = importlib.import_module("audit_carrier_loop")
    diagnostic = importlib.import_module("audit_rx_soft_capture")
    independent = importlib.import_module("audit_spline_fixed")

    def selected(y, mod, start, **kwargs):
        kwargs.pop("bandwidth")
        return track_parallel_carrier(y, mod, start, **kwargs)

    monkeypatch.setattr(driver, "track_psk_carrier", selected)
    profile = PSKReceiveProfile.ordinary(
        TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            pid=3,
            symbol_rate_mhz=4,
            whitening_seed=0x52,
            crc_seed=0x123456,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
        )
    )
    payload = bytes((k * 73 + 29) % 256 for k in range(37))
    x, q, _, _ = iq_record(
        independent.independent_packet(profile, payload), cfo=-23000, ppm=75, offset=0.391
    )
    receiver = MacStreamReceiver(profile, q, quiet_batching=False, defer_quiet_fir=False)
    with driver.carrier_adapter("medium-head", profile):
        first, last = receiver.feed(x), receiver.finish()
    events = first.diagnostics["events"] + last.diagnostics["events"]
    valid = [e for e in events if e["frame"]["status"] == "valid"]
    assert len(valid) == 1 and valid[0]["frame"]["payload"] == payload
    assert len(first.packets + last.packets) == 1
    event = valid[0]
    assert np.packbits(diagnostic.diagnostic_bits(event, profile)).tobytes() == payload
    assert event["frame"]["carrier"]["iterations"] == 18
    event["frame"]["carrier"]["fraction_bits"] = 12
    with pytest.raises(ValueError, match="Q10/18iterations"):
        diagnostic.diagnostic_bits(event, profile)


def test_parallel_node_bad_shapes_bounds_saturation_and_closed_packet():
    node = ParallelAngleCarrierLoop("QPSK", 0)
    for real, imag in (([], []), ([1], [1, 2]), ([1] * 33, [1] * 33), ([[1]], [[1]])):
        with pytest.raises(ValueError):
            node.feed_words(real, imag)
    node.phase = -(2**29)
    node.feed_words([131071], [131071])
    assert node.stats["output_saturations"] > 0
    node.used = 2**17
    with pytest.raises(OverflowError):
        node.feed_words([1], [1])
    node.finish()
    with pytest.raises(RuntimeError):
        node.feed_words([1], [1])
