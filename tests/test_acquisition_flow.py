"""Signed tile arithmetic and finite-rate scheduling; deliberately failing resource points."""

import random

import pytest

from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, FixedDifferentialBank
from nearlink_sdr.phy.acquisition_flow import (
    AcquisitionFlow,
    FlowAssumptions,
    laned_correlation,
    multiply_signed_tiles,
    signed_multiplier_tiles,
)
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


@pytest.mark.parametrize('a_bits,b_bits', [(16, 16), (22, 22), (29, 29), (36, 36),
                                          (45, 45), (52, 52), (96, 29)])
def test_signed_partial_products_reconstruct_full_width_without_unsigned_limb_assumptions(a_bits, b_bits):
    rng = random.Random(340121)
    left = [-(1 << (a_bits-1)), (1 << (a_bits-1))-1, 0, -1]
    right = [-(1 << (b_bits-1)), (1 << (b_bits-1))-1, 0, -1]
    for _ in range(12):
        left.append(rng.randrange(-(1 << (a_bits-1)), 1 << (a_bits-1)))
        right.append(rng.randrange(-(1 << (b_bits-1)), 1 << (b_bits-1)))
    for a, b in zip(left, right, strict=True):
        assert multiply_signed_tiles(a, b, a_bits, b_bits) == a*b
    assert signed_multiplier_tiles(a_bits, b_bits) == signed_multiplier_tiles(b_bits, a_bits)


@pytest.mark.parametrize('lanes', [2, 4, 8])
def test_laned_sign_accumulation_and_padding_preserve_exact_integer_dot(lanes):
    rng = random.Random(340122)
    r, i, a, b = [[rng.randrange(-100000, 100001) for _ in range(31)] for _ in range(4)]
    a, b = [1 if v >= 0 else -1 for v in a], [1 if v >= 0 else -1 for v in b]
    actual = laned_correlation(r, i, a, b, lanes)
    assert actual == (sum(x*s+y*t for x, y, s, t in zip(r, i, a, b, strict=True)),
                      sum(y*s-x*t for x, y, s, t in zip(r, i, a, b, strict=True)))


def test_16msps_requires_resources_and_explicit_operator_latency_not_just_fifo_space():
    flow = AcquisitionFlow()
    estimate = flow.estimate
    assert estimate['multiplier_primitives'] == 18
    assert estimate['primitive_multiplies_per_sample'] == 40
    assert flow.initiation_interval == 12
    report = flow.simulate(16384)
    assert report['continuous_rate_feasible'] and report['dropped_sample_tags'] == 0
    assert max(report['peak_waiting_by_stage']) == 0
    assert report['peak_raw_flag_join'] == 4
    assert report['max_sample_latency_cycles'] == 50
    assert report['max_query_latency_from_last_sample_cycles'] == 71
    assert estimate['storage']['total_bits'] <= 16*1024*8
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8))
    bank = FixedDifferentialBank(p.sync_symbols(), 4, 256, input_full_scale=262144.,
                                 format=FixedBankFormat(22))
    assert estimate['storage']['bank_bits'] == bank.logical_storage_bits
    weak_resources = AcquisitionFlow(FlowAssumptions(lanes=2))
    bad = weak_resources.simulate(16384)
    assert not bad['continuous_rate_feasible'] and bad['dropped_sample_tags'] > 0
    assert max(bad['peak_waiting_by_stage']) == 64
    slow_adders = AcquisitionFlow(FlowAssumptions(adder_latency=2))
    assert not slow_adders.simulate(8192)['continuous_rate_feasible']
    slow_clock = AcquisitionFlow(FlowAssumptions(clock_hz=160_000_000))
    assert not slow_clock.simulate(8192)['continuous_rate_feasible']


def test_elaboration_input_bounds_and_real_queue_depth_are_enforced():
    with pytest.raises(ValueError):
        signed_multiplier_tiles(1, 20)
    with pytest.raises(ValueError):
        multiply_signed_tiles(1, 1, 0, 18)
    with pytest.raises(ValueError):
        multiply_signed_tiles(1000, 1, 4, 18)
    with pytest.raises(ValueError):
        laned_correlation([1], [], [], [], 4)
    with pytest.raises(ValueError):
        laned_correlation([1], [1], [2], [0], 4)
    with pytest.raises(TypeError):
        laned_correlation([1.5], [1], [1], [0], 4)
    with pytest.raises(ValueError):
        AcquisitionFlow(False)
    for args in [dict(lanes=1), dict(clock_hz=True), dict(fifo_depth=0), dict(taps=128),
                 dict(window_samples=16), dict(adder_latency=0)]:
        with pytest.raises(ValueError):
            FlowAssumptions(**args)
    with pytest.raises(ValueError, match='power-of-two'):
        FlowAssumptions(fifo_depth=63)
    with pytest.raises(ValueError):
        AcquisitionFlow().simulate(0)
    with pytest.raises(OverflowError):
        AcquisitionFlow().simulate(2**63)
