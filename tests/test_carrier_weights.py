"""Independent power/RNE and waveform checks; CRC never selects a carrier answer."""

from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights, bounded_carrier_weights
from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.psk import refine_psk_carrier
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.strict_rx import decode_psk_frame as strict_decode
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


@pytest.mark.parametrize("fixed", [False, True])
def test_burst_does_not_instantly_raise_estimator_and_chunks_keep_state(fixed):
    values = np.r_[np.ones(17), np.full(24, 1e6), np.ones(80), np.full(70, .01)]
    whole, info = bounded_carrier_weights(values, 4, fixed=fixed)
    node = BoundedCarrierWeights(4, fixed=fixed)
    pieces = [node.advance_power(values[s:s + 7]) for s in range(0, len(values), 7)]
    np.testing.assert_array_equal(np.concatenate(pieces), whole)
    assert info["stats"] == node.stats
    assert whole[17] == 4 and whole[18] == 4 * 65 / 64
    assert whole[40] <= 4 * (65 / 64)**23 + 1e-3
    prefix, _ = bounded_carrier_weights(values[:24], 4, fixed=fixed)
    np.testing.assert_array_equal(prefix, whole[:24])


def test_unsigned_fixed_words_match_independent_fraction_oracle():
    powers = np.random.default_rng(530118).integers(0, 2**41, 513, dtype=np.uint64)
    powers[:8] = [0, 1, 65536, 65537, 64, 32, 2**41 - 1, 0]
    state, expected = 65536, []
    for word in powers:
        p = int(word)
        expected.append(min(p, 8 * state))
        state = min(2**32, max(1, state + round(Fraction(min(p, 2 * state) - state, 64))))
    node = BoundedCarrierWeights(8, fixed=True)
    actual = np.concatenate([node.advance_words(powers[s:s + 32])
                             for s in range(0, len(powers), 32)])
    np.testing.assert_array_equal(actual, expected)
    assert node.power == state and actual.max() < 2**36


def test_fixed_range_and_zero_power_deadband_are_visible_not_hidden():
    node = BoundedCarrierWeights(2, fixed=True)
    out = node.advance_power([1e100, 1e-100])
    assert np.isfinite(out).all()
    assert node.stats["input_power_saturations"] == 1
    assert node.stats["input_power_erased"] == 1
    for _ in range(200):
        node.advance_words(np.zeros(32, dtype=np.uint64))
    assert node.power == 32  # Q16 RNE feedback deadband; not the float floor1.
    floating = BoundedCarrierWeights(2)
    for _ in range(200):
        floating.advance_power(np.zeros(32))
    assert floating.power == 2**-16 and floating.stats["lower_clamps"] > 0


def test_estimator_upper_clamp_and_saturated_telemetry_do_not_wrap():
    node = BoundedCarrierWeights(2)
    node.power = 2**16
    node.stats["weights_clipped"] = 2**63 - 1
    node.advance_power([1e10])
    assert node.power == 2**16 and node.stats["upper_clamps"] == 1
    assert node.stats["telemetry_saturated"]
    node.stats["samples"] = 2**63 - 2
    with pytest.raises(OverflowError):
        node.advance_power([0, 0])


def test_independent_constellation_outlier_does_not_capture_entire_carrier_fit():
    rng = np.random.default_rng(530119)
    i = np.arange(1024)
    wire = np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, len(i))))
    wire *= np.exp(-1j * np.pi / 4 * ((i + 3) % 2))
    y = wire * np.exp(2j * np.pi * .001 * i)
    # Give the injected outlier the same prescribed alternating rotation;
    # otherwise de-rotation alternates its fourth-power tone out of the band.
    y[300:332] = (20 * np.exp(2j * np.pi * .02 * i[300:332])
                  * np.exp(-1j * np.pi / 4 * ((i[300:332] + 3) % 2)))
    _, original = refine_psk_carrier(y, "QPSK", 3)
    _, bounded = refine_psk_carrier(y, "QPSK", 3, carrier_weight_kappa=4)
    old_error = abs(original["residual_cycles_per_symbol"] - .001)
    new_error = abs(bounded["residual_cycles_per_symbol"] - .001)
    assert new_error < 1e-5 and old_error > .01
    assert bounded["weights"]["stats"]["weights_clipped"] == 32


@pytest.mark.parametrize("mcs", [8, 12])
@pytest.mark.parametrize("pilot", [0, 8])
@pytest.mark.parametrize("fixed", [False, True])
def test_unknown_iq_actual_mac_and_chunked_carrier_weights(mcs, pilot, fixed):
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=mcs, pilot_interval=pilot,
                                                 symbol_rate_mhz=4, whitening_seed=0x52,
                                                 crc_seed=0x123456))
    wire, expected = transmitted(profile, length=73)
    x, quiet, _, _ = iq_record(wire, ppm=100, cfo=23000, offset=.31)
    outputs = []
    for chunk in [len(x), 137]:
        rx = MacStreamReceiver(profile, quiet, carrier_weight_kappa=4,
                               carrier_weight_fixed=fixed)
        events, packets = [], []
        for s in range(0, len(x), chunk):
            result = rx.feed(x[s:s + chunk])
            events += result.diagnostics["events"]
            packets += result.packets
        result = rx.finish()
        events += result.diagnostics["events"]
        packets += result.packets
        assert [p.mac_payload for p in packets] == [expected]
        assert len(events) == 1 and events[0]["frame"]["carrier"]["weights"]["fixed"] == fixed
        outputs.append(events[0])
    assert outputs[0]["frame"] == outputs[1]["frame"]
    np.testing.assert_array_equal(outputs[0]["symbol_centers_raw"], outputs[1]["symbol_centers_raw"])


@pytest.mark.parametrize("values", [[], [np.nan], [-1], np.zeros(33), [[1]]])
def test_invalid_node_power_is_not_accepted(values):
    with pytest.raises(ValueError):
        BoundedCarrierWeights(2).advance_power(values)


@pytest.mark.parametrize("values", [[1.0], [-1], [2**41]])
def test_fixed_interface_does_not_round_invalid_words(values):
    with pytest.raises(ValueError):
        BoundedCarrierWeights(2, fixed=True).advance_words(values)


@pytest.mark.parametrize("kappa,fixed", [(1, False), (True, False), (4, 1)])
def test_unfrozen_configuration_is_not_silently_accepted(kappa, fixed):
    with pytest.raises(ValueError):
        BoundedCarrierWeights(kappa, fixed=fixed)


def test_no_fixed_mode_or_old_phase_regression_without_explicit_valid_configuration():
    with pytest.raises(ValueError):
        refine_psk_carrier(np.ones(40), "QPSK", 0, carrier_weight_fixed=True)
    with pytest.raises(ValueError):
        refine_psk_carrier(np.ones(40), "QPSK", 0, method="phase_regression",
                           carrier_weight_kappa=4)


def test_invalid_experimental_config_fails_even_before_short_header_decode():
    profile = PSKReceiveProfile.ordinary(TxConfig())
    for options in [dict(carrier_weight_fixed=True), dict(carrier_weight_kappa=3),
                    dict(carrier_weight_kappa=4, carrier_method="phase_regression")]:
        with pytest.raises(ValueError):
            strict_decode(np.ones(3), profile, **options)
        with pytest.raises(ValueError):
            MacStreamReceiver(profile, np.zeros(64), **options)
    for power in [[], [[1]]]:
        with pytest.raises(ValueError):
            bounded_carrier_weights(power, 4)
    with pytest.raises(ValueError):
        BoundedCarrierWeights(4).advance_words([1])
