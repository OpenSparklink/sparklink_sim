"""Independent angle/coefficient/Fraction oracles for finite carrier transitions."""

import cmath
import math
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_fixed import (
    CarrierWordFormat,
    FixedCarrierLoop,
    cordic_phase_word,
    track_fixed_carrier,
)
from nearlink_sdr.phy.carrier_loop import track_psk_carrier
from nearlink_sdr.phy.sync_coherent import SyncConfirmationNCO, phase_rotation_words


def test_variable_phase_rom_matches_independent_rounding_and_old_capped_nco():
    words = [0, 1, 2**31, 2**32 - 1, *range(2**19 - 2, 2**19 + 3)]
    words += np.random.default_rng(730119).integers(0, 2**32, 200).tolist()
    for word in words:
        c, s = phase_rotation_words(word)
        address = round(Fraction(word, 2**20)) % 4096
        ref = cmath.exp(-2j * math.pi * address / 4096) * 65536
        assert abs(c - ref.real) <= 0.50001 and abs(s - ref.imag) <= 0.50001
    node = SyncConfirmationNCO(37115)
    c, s = node.advance(128)
    assert all(
        (int(c[k]), int(s[k])) == phase_rotation_words((37115 * k) % 2**20, phase_bits=20)
        for k in range(128)
    )
    with pytest.raises(ValueError):
        node.advance(1)
    for args in ((True, 32), (2**32, 32), (-1, 32), (1, 24), (1, True)):
        with pytest.raises(ValueError):
            phase_rotation_words(args[0], phase_bits=args[1])


def test_cordic_zero_axes_quadrants_extremes_and_common_scale():
    rng = np.random.default_rng(730120)
    cases = [(x, y) for x in (-131072, -1, 0, 1, 131071) for y in (-131072, -1, 0, 1, 131071)]
    cases += [tuple(map(int, p)) for p in rng.integers(-131072, 131072, (300, 2))]
    for x, y in cases:
        actual = cordic_phase_word(x, y)
        if not x and not y:
            assert actual is None
        else:
            ref = math.atan2(y, x)
            difference = cmath.phase(cmath.exp(1j * (actual * 2 * math.pi / 2**32 - ref)))
            assert abs(difference) <= 0.0002
    assert cordic_phase_word(3, 5) == cordic_phase_word(3 * 128, 5 * 128)
    for a, b in ((True, 0), (1.1, 0), (131072, 0)):
        with pytest.raises(ValueError):
            cordic_phase_word(a, b)


@pytest.mark.parametrize("fraction", [10, 12])
def test_independent_known_carrier_and_integer_state_chunks(fraction):
    k = np.arange(4064)
    m = 8
    phase = 0.06 + 2 * math.pi * 0.00017 * k
    angle = np.r_[
        (2 * (k[:64] % 4) + 1) * math.pi / 4 - ((k[:64] + 3) % 2) * math.pi / 4,
        (2 * (k[64:] % m) + 1) * math.pi / m - ((k[64:] + 3) % 2) * math.pi / m,
    ]
    y = np.exp(1j * (angle + phase))
    fixed, info, trace = track_fixed_carrier(
        y[64:], "8PSK", 67, fraction_bits=fraction, header_symbols=y[:64], header_mod_type="QPSK"
    )
    float_out, _, float_trace = track_psk_carrier(
        y[64:], "8PSK", 67, bandwidth="medium", header_symbols=y[:64], header_mod_type="QPSK"
    )
    assert np.sqrt(np.mean(abs(fixed - float_out) ** 2)) < 0.005
    assert (
        np.max(
            abs(
                np.unwrap(trace["phase_word"] * 2 * math.pi / 2**32)
                - np.unwrap(float_trace["phase"])
            )
        )
        < 0.02
    )
    assert info["stats"]["input_saturations"] == info["stats"]["output_saturations"] == 0
    real = np.rint(y.real * 2**fraction).astype(int)
    imag = np.rint(y.imag * 2**fraction).astype(int)
    node = FixedCarrierLoop("QPSK", 3, fmt=CarrierWordFormat(fraction))
    for j in range(0, 64, 7):
        node.feed_words(real[j : min(j + 7, 64)], imag[j : min(j + 7, 64)])
    node.switch_modulation("8PSK")
    parts = [node.feed_words(real[j : j + 7], imag[j : j + 7]) for j in range(64, len(y), 7)]
    assert all(np.array_equal(trace[key], np.concatenate([p[key] for p in parts])) for key in trace)
    end = node.finish()
    assert end["final_phase_word"] == info["final_phase_word"]
    with pytest.raises(RuntimeError):
        node.feed_words([1], [1])
    with pytest.raises(RuntimeError):
        node.switch_modulation("QPSK")


def test_rational_pi_updates_saturation_rejection_and_validation():
    node = FixedCarrierLoop("QPSK", 0)
    old = node.phase
    row = node.feed_words([1024], [1024])
    e = int(row["error_word"][0])
    f = round(Fraction(e, 2048))
    assert node.frequency == f
    assert node.phase == old + f + round(Fraction(e, 32))
    out, info, _ = track_fixed_carrier(np.array([1e4 + 1e4j] * 32), "QPSK", 0)
    assert info["stats"]["input_saturations"] == 64 and np.isfinite(out).all()
    node = FixedCarrierLoop("QPSK", 0)
    node.phase = -(1 << 29)
    node.feed_words([131071], [131071])
    assert node.stats["output_saturations"] > 0
    for args in (("unknown", 0), ("QPSK", True), ("QPSK", 131072)):
        with pytest.raises(ValueError):
            FixedCarrierLoop(*args)
    for option in (True, 11):
        with pytest.raises(ValueError):
            CarrierWordFormat(option)
    with pytest.raises(ValueError):
        FixedCarrierLoop("QPSK", 0, fmt=True)
    with pytest.raises(ValueError):
        node.switch_modulation("unknown")
    for a, b in (([], []), ([1], [1, 2]), ([1.1], [0]), ([True], [0]), ([131072], [0])):
        with pytest.raises(ValueError):
            node.feed_words(a, b)
    node.used = 131072
    with pytest.raises(OverflowError):
        node.feed_words([1], [0])
    for kwargs in (
        dict(header_symbols=[]),
        dict(header_mod_type="QPSK"),
        dict(header_symbols=np.ones(513), header_mod_type="QPSK"),
    ):
        with pytest.raises(ValueError):
            track_fixed_carrier(np.ones(32), "QPSK", 0, **kwargs)
    with pytest.raises(ValueError):
        track_fixed_carrier([], "QPSK", 0)


def test_component_erasure_power_erasure_and_profile_preflight(monkeypatch):
    import importlib
    from pathlib import Path

    out, info, _ = track_fixed_carrier(np.full(32, 1e-6 + 1e-6j), "QPSK", 0)
    assert not np.any(out) and info["stats"]["input_erased"] == 64
    node = FixedCarrierLoop("QPSK", 0)
    node.feed_words([1], [0])
    assert node.stats["power_erased"] == 1 and node.stats["low_power_holds"] == 1
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    audit = importlib.import_module("audit_carrier_fixed")
    with pytest.raises(ValueError, match="m-sequence"):
        audit.validate_profiles([(2, 8, 0, 11), (3, 8, 0, 83)], 6)
    assert len(audit.validate_profiles([(2, 8, 0, 11), (3, 8, 0, 83), (4, 12, 8, 79)], 4)) == 3


def test_actual_iq_to_mac_and_faithful_fixed_ber_replay(monkeypatch):
    import importlib
    from pathlib import Path

    from nearlink_sdr.phy.mac_interface import MacStreamReceiver
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tx_pipeline import TxConfig
    from tests.test_strict_receive import iq_record, transmitted

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    driver = importlib.import_module("audit_carrier_loop")
    diagnostic = importlib.import_module("audit_rx_soft_capture")

    def fixed(y, mod, start, **kwargs):
        kwargs.pop("bandwidth")
        return track_fixed_carrier(y, mod, start, **kwargs)

    monkeypatch.setattr(driver, "track_psk_carrier", fixed)
    p = PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=8, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )
    wire, payload = transmitted(p, length=37)
    x, q, _, _ = iq_record(wire, cfo=19000, ppm=-50)
    rx = MacStreamReceiver(p, q, quiet_batching=False, defer_quiet_fir=False)
    with driver.carrier_adapter("medium-head", p):
        a, b = rx.feed(x), rx.finish()
    event = (a.diagnostics["events"] + b.diagnostics["events"])[0]
    assert event["frame"]["payload"] == payload and event["frame"]["carrier"]["state_words_exact"]
    assert np.packbits(diagnostic.diagnostic_bits(event, p)).tobytes() == payload
    event["frame"]["carrier"]["fraction_bits"] = 11
    with pytest.raises(ValueError, match="Q10/Q12"):
        diagnostic.diagnostic_bits(event, p)


def test_legitimate_modulation_switch_clamps_integer_integrator_limits():
    initial = round(Fraction(2**32, 20))
    for sign in (-1, 1):
        for mod, m in (("BPSK", 2), ("QPSK", 4), ("8PSK", 8)):
            node = FixedCarrierLoop("BPSK", 0)
            node.frequency = sign * initial
            node.switch_modulation(mod)
            node.feed_words([0], [0])
            assert node.frequency == sign * round(Fraction(2**32, 10 * m))
            assert node.stats["frequency_clamps"] == int(m != 2)
