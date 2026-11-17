"""Independent known-CFO waveform, causal fit and actual IQ diagnostic replay."""

import functools
import importlib
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy import carrier_prefix, strict_rx
from nearlink_sdr.phy.carrier_prefix import (
    CarrierPrefixConfig,
    PrefixCarrierFit,
    refine_psk_carrier_prefix,
)
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.psk import refine_psk_carrier
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.carrier_prefix_cost import carrier_prefix_cost
from tests.test_strict_receive import iq_record, transmitted


def independent(mod, n=1901, start=3):
    m = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[mod]
    rng = np.random.default_rng(690119)
    k = np.arange(n)
    base = 1.3 * np.exp(1j * (2 * rng.integers(0, m, n) + 1) * np.pi / m)
    base[(k + start) % 2 == 1] *= np.exp(-1j * np.pi / m)
    f, phi = 0.001371, 0.04
    return base * np.exp(1j * (2 * np.pi * f * k + phi)), base, f, phi


@pytest.mark.parametrize("mod", ["BPSK", "QPSK", "8PSK"])
@pytest.mark.parametrize("limit", [128, 256])
def test_independent_constant_carrier_and_alternating_global_index(mod, limit):
    y, ideal, f, phi = independent(mod)
    out, fit = refine_psk_carrier_prefix(y, mod, 3, prefix_symbols=limit)
    assert abs(fit["residual_cycles_per_symbol"] - f) < 1e-8
    assert abs(fit["residual_phase_radians"] - phi) < 1e-6
    assert np.max(abs(out - ideal)) < 2e-5
    assert fit["estimation_symbols"] == limit and fit["payload_symbols"] == len(y)


def test_prefix_cannot_read_future_outlier_and_complete_short_fit_is_unchanged():
    y, *_ = independent("8PSK")
    other = y.copy()
    other[256:] *= np.exp(0.8j) * 10000
    _, a = refine_psk_carrier_prefix(y, "8PSK", 3)
    _, b = refine_psk_carrier_prefix(other, "8PSK", 3)
    assert a == b
    for n in (32, 65, 128):
        baseline, info = refine_psk_carrier(y[:n], "8PSK", 3)
        result, actual = refine_psk_carrier_prefix(y[:n], "8PSK", 3)
        assert np.array_equal(result, baseline)
        assert all(actual[k] == v for k, v in info.items())


@pytest.mark.parametrize("pattern", [1, 7, 32])
def test_bounded_prefix_collector_chunks_and_eof_are_not_full_frame_buffer(pattern):
    y, *_ = independent("QPSK", n=5000)
    node = PrefixCarrierFit(len(y), "QPSK", 3, carrier_weight_kappa=4)
    offset = 0
    while node.estimate is None:
        row = node.feed(y[offset : offset + pattern])
        offset += row["consumed"]
    _, expected = refine_psk_carrier_prefix(y, "QPSK", 3, carrier_weight_kappa=4)
    assert row["estimate"] == expected and offset == 256
    assert node.buffer.shape == (256,) and node.buffer.nbytes == 4096
    assert node.feed(y[256:257])["consumed"] == 0
    assert node.finish()["status"] == "ready"
    with pytest.raises(RuntimeError):
        node.feed(y[:1])
    short = PrefixCarrierFit(80, "QPSK", 3)
    short.feed(y[:31])
    end = short.finish()
    assert end["status"] == "truncated_prefix" and end["estimate"] is None and end["observed"] == 31


def test_actual_mac_and_ber_replay_forward_prefix_metadata(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_rx_soft_capture")
    p = PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=8, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )
    wire, payload = transmitted(p, length=37)
    x, q, _, _ = iq_record(wire, cfo=17000, ppm=50)
    original = refine_psk_carrier_prefix
    monkeypatch.setattr(
        strict_rx, "refine_psk_carrier", functools.partial(original, prefix_symbols=128)
    )
    rx = MacStreamReceiver(p, q, quiet_batching=False, defer_quiet_fir=False)
    a, b = rx.feed(x), rx.finish()
    event = (a.diagnostics["events"] + b.diagnostics["events"])[0]
    assert (
        event["frame"]["payload"] == payload and event["frame"]["carrier"]["prefix_symbols"] == 128
    )
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs["prefix_symbols"])
        return original(*args, **kwargs)

    monkeypatch.setattr(carrier_prefix, "refine_psk_carrier_prefix", spy)
    assert np.packbits(audit.diagnostic_bits(event, p)).tobytes() == payload and calls == [128]
    event["frame"]["carrier"]["estimation_symbols"] = 1
    with pytest.raises(ValueError, match="actual estimation"):
        audit.diagnostic_bits(event, p)


def test_invalid_context_and_unknown_methods_cannot_fallback():
    y, *_ = independent("QPSK")
    for kwargs in (
        dict(prefix_symbols=True),
        dict(prefix_symbols=64),
        dict(method="phase_regression"),
        dict(carrier_weight_fixed=True),
        dict(carrier_weight_kappa=3),
    ):
        with pytest.raises(ValueError):
            refine_psk_carrier_prefix(y, "QPSK", 3, **kwargs)
    for symbols in ([], np.ones(31), np.ones(65537), np.array([np.nan] * 32), np.ones(32) * 1e71):
        with pytest.raises(ValueError):
            refine_psk_carrier_prefix(symbols, "QPSK", 3)
    for args in (("unknown", 3), ("QPSK", True), ("QPSK", -1), ("QPSK", 1.2), ("QPSK", 65536)):
        with pytest.raises(ValueError):
            refine_psk_carrier_prefix(y, *args)
    for n in (True, 31, 65537):
        with pytest.raises(ValueError):
            PrefixCarrierFit(n, "QPSK", 3)
    with pytest.raises(ValueError):
        PrefixCarrierFit(100, "QPSK", 3, config=True)
    node = PrefixCarrierFit(100, "QPSK", 3, config=CarrierPrefixConfig(128))
    for x in ([], np.ones(33)):
        with pytest.raises(ValueError):
            node.feed(x)
    for options in (dict(frame_symbols=0), dict(complex_bits=32), dict(clock_hz=50_000_000)):
        with pytest.raises(ValueError):
            carrier_prefix_cost(**options)


def test_resource_roles_charge_snapshots_fifos_and_unverified_formats():
    small = carrier_prefix_cost(128, complex_bits=36)
    large = carrier_prefix_cost(256, complex_bits=64)
    slow = carrier_prefix_cost(256, complex_bits=64, clock_hz=100_000_000)
    assert small["fft_points"] == 2048 and large["fft_points"] == 4096
    assert large["total_bram36"] == sum(large["storage_bram36"].values())
    assert large["storage_bram36"]["raw_prefix_snapshot"] > 0
    assert small["real_product_DSP25x18_tiles_upper"] == 1
    assert large["real_product_DSP25x18_tiles_upper"] == 4
    assert large["butterfly_DSP25x18_tiles"] == 16
    assert large["butterfly_II_scenario"] == 2
    assert large["storage_bram36"]["fft_twiddle_rom"] == 4
    assert large["twiddle_entries"] == 2048
    assert slow["allocated_fifo_words_scenario"] > large["allocated_fifo_words_scenario"]
    assert not large["fixed_format_accepted"] and not large["service_is_verified_bound"]
    assert not large["complete16msps_verified"] and not large["synthesis_or_timing_evidence"]
