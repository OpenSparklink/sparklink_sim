"""Independent angle reference, real causal state and IQ→MAC diagnostic replay."""

import importlib
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy import stream_rx, strict_rx
from nearlink_sdr.phy.carrier_loop import CarrierLoopConfig, PSKCarrierLoop, track_psk_carrier
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


def independent(mod, n=5000):
    m = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[mod]
    k = np.arange(n)
    points = (k * 37 + k // 13) % m
    ideal = np.exp(1j * ((2 * points + 1) * np.pi / m - ((k + 3) % 2) * np.pi / m))
    phase = 0.07 + 2 * np.pi * 0.00017 * k
    return ideal * np.exp(1j * phase), ideal, phase


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK"])
@pytest.mark.parametrize("bandwidth", ["slow", "medium"])
def test_independent_carrier_convergence_and_parity(mod, bandwidth):
    y, ideal, phase = independent(mod)
    out, info, trace = track_psk_carrier(y, mod, 3, bandwidth=bandwidth)
    assert np.max(abs(np.angle(out[2000:] * ideal[2000:].conj()))) < 2e-8
    assert abs(info["final_frequency_rad_per_symbol"] - 2 * np.pi * 0.00017) < 1e-12
    assert np.max(abs(np.angle(np.exp(1j * (phase[2000:] - trace["phase"][2000:]))))) < 2e-8
    assert info["phase_wraps_are_not_cycle_slips"] and not info["fixed_datapath_verified"]


def test_chunk_independence_future_erasure_burst_and_reset():
    y, *_ = independent("QPSK")
    y[973:980] = 0
    y[2027:2032] *= 100
    expected, info, trace = track_psk_carrier(y, "QPSK", 3)
    node = PSKCarrierLoop("QPSK", 3)
    parts, cursor = [], 0
    while cursor < len(y):
        n = (1, 7, 32)[cursor % 3]
        parts.append(node.feed(y[cursor : cursor + n]))
        cursor += n
    for key in parts[0]:
        actual = np.concatenate([p[key] for p in parts])
        assert np.array_equal(actual, expected if key == "symbols" else trace[key])
    assert node.finish() == info
    assert info["stats"]["low_power_holds"] >= 7 and info["stats"]["high_power_holds"] >= 5
    other = y.copy()
    other[3000:] *= 1.7j
    a, _, _ = track_psk_carrier(other, "QPSK", 3)
    assert np.array_equal(a[:3000], expected[:3000])
    reset, _, _ = track_psk_carrier(y, "QPSK", 3)
    assert np.array_equal(reset, expected)
    with pytest.raises(RuntimeError):
        node.feed([1j])
    empty = PSKCarrierLoop("BPSK", 0).finish()
    assert empty["symbols"] == 0 and empty["status"] == "no_symbols"


def test_closed_error_frequency_limits_and_input_validation():
    for args in (("unknown", 0), ("QPSK", True), ("QPSK", -1), ("QPSK", 1.5), ("QPSK", 131072)):
        with pytest.raises(ValueError):
            PSKCarrierLoop(*args)
    with pytest.raises(ValueError):
        PSKCarrierLoop("QPSK", 0, config=True)
    with pytest.raises(ValueError):
        CarrierLoopConfig("unknown")
    for y in ([], [np.nan] * 32, np.ones((2, 2)), [1e7], np.ones(131073)):
        with pytest.raises(ValueError):
            track_psk_carrier(y, "QPSK", 3)
    node = PSKCarrierLoop("QPSK", 0)
    for y in ([], np.ones(33)):
        with pytest.raises(ValueError):
            node.feed(y)
    # Exercise explicit clamps without allowing an impossible persistent state
    # to silently wrap; valid packet resets begin at zero frequency.
    node.frequency = 0.2
    node.feed([1])
    assert node.stats["frequency_clamps"] == 1 and node.stats["error_clamps"] == 1
    node.used = 131072
    with pytest.raises(OverflowError):
        node.feed([1j])
    assert node.used == 131072


@pytest.mark.parametrize("preroll", [False, True])
def test_actual_mac_and_diagnostic_replay_use_identical_loop(monkeypatch, preroll):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_rx_soft_capture")
    profile = PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=8, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )
    wire, payload = transmitted(profile, length=37)
    x, quiet, _, _ = iq_record(wire, cfo=17000, ppm=50)

    active = {}
    decoder = stream_rx.decode_psk_frame

    def decoding(symbols, profile, **kwargs):
        active["symbols"] = symbols
        return decoder(symbols, profile, **kwargs)

    def adapter(y, mod, start, **kwargs):
        options = (
            {}
            if not preroll
            else dict(
                header_symbols=active["symbols"][len(profile.sync_symbols()) : start],
                header_mod_type="QPSK",
            )
        )
        out, info, _ = track_psk_carrier(y, mod, start, **options)
        return out, info

    monkeypatch.setattr(strict_rx, "refine_psk_carrier", adapter)
    monkeypatch.setattr(stream_rx, "decode_psk_frame", decoding)
    rx = MacStreamReceiver(profile, quiet, quiet_batching=False, defer_quiet_fir=False)
    a, b = rx.feed(x), rx.finish()
    event = (a.diagnostics["events"] + b.diagnostics["events"])[0]
    assert event["frame"]["payload"] == payload
    assert np.packbits(audit.diagnostic_bits(event, profile)).tobytes() == payload
    if preroll:
        event["frame"]["carrier"]["header_modulation"] = "BPSK"
        with pytest.raises(ValueError, match="actual header profile"):
            audit.diagnostic_bits(event, profile)
        event["frame"]["carrier"]["header_modulation"] = "QPSK"
    event["frame"]["carrier"]["symbols"] += 1
    with pytest.raises(ValueError, match="actual field"):
        audit.diagnostic_bits(event, profile)


def test_header_preroll_has_one_causal_state_through_modulation_switch():
    k = np.arange(3064)
    start, header_count = 11, 64
    angle = np.r_[
        np.tile([np.pi / 4, 3 * np.pi / 4], 32), (2 * ((k[64:] * 13) % 8) + 1) * np.pi / 8
    ]
    rotations = np.r_[((k[:64] + start) % 2) * np.pi / 4, ((k[64:] + start) % 2) * np.pi / 8]
    values = np.exp(1j * (angle - rotations + 0.09 + 0.0013 * k))
    out, info, trace = track_psk_carrier(
        values[64:],
        "8PSK",
        start + header_count,
        header_symbols=values[:64],
        header_mod_type="QPSK",
    )
    node = PSKCarrierLoop("QPSK", start)
    node.feed(values[:32])
    node.feed(values[32:64])
    node.switch_modulation("8PSK")
    parts = [node.feed(values[j : j + 32]) for j in range(64, len(values), 32)]
    assert np.array_equal(out, np.concatenate([p["symbols"] for p in parts]))
    assert np.array_equal(trace["phase"], np.concatenate([p["phase"] for p in parts]))
    assert info["header_symbols"] == 64 and info["total_processed_symbols"] == len(values)
    assert info["start_symbol_index"] == start + 64 and info["symbols"] == len(values) - 64
    node.finish()
    with pytest.raises(RuntimeError):
        node.switch_modulation("QPSK")
    with pytest.raises(ValueError):
        PSKCarrierLoop("QPSK", 0).switch_modulation("unknown")
    for options in (
        dict(header_mod_type="QPSK"),
        dict(header_symbols=[]),
        dict(header_symbols=np.ones(513), header_mod_type="QPSK"),
        dict(header_symbols=np.ones(5), header_mod_type="QPSK"),
    ):
        with pytest.raises(ValueError):
            track_psk_carrier(values, "8PSK", 3, **options)


def test_cost_keeps_nonlinear_feedback_and_frame_memory_unverified():
    from nearlink_sdr.sim.carrier_loop_cost import carrier_loop_cost

    d = carrier_loop_cost()
    assert d["logical_bits"] == sum(d["storage_bits"].values())
    assert d["proposed_DSP25x18_tiles"] == 6 and d["FFT_points"] == 0
    assert d["cycles_per_symbol_verified"] is None
    assert not d["accepted_fixed_formats"] and not d["executable_cycle_schedule"]
    assert (
        not d["continuous_16msps_verified"]
        and d["existing_rx_frame_and_decode_memory_still_required"]
    )
