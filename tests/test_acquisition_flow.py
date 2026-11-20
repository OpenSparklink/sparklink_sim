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


@pytest.mark.parametrize('clock', [100, 160, 200])
def test_explicit_ideal_arrivals_keep_legacy_queues_and_every_reservation(clock):
    flow = AcquisitionFlow(FlowAssumptions(clock_hz=clock * 1_000_000))
    ideal, tagged = [], []
    old = flow.simulate(2048, sink=ideal.append)
    arrivals = ((n, (n * clock + 15) // 16) for n in range(2048))
    new = flow.simulate(2048, arrivals=arrivals, sink=tagged.append)
    assert new == old and ideal == tagged
    assert sum(e['event'] == 'sample_drop' for e in tagged) == new['dropped_sample_tags']
    assert sum(e['event'] == 'query_drop' for e in tagged) == new['query_drops']


def test_external_latency_is_not_erased_and_observer_does_not_steer_schedule():
    count, shift = 128, 719
    flow = AcquisitionFlow()
    a, b = [], []
    ideal = [(n, (n * 25 + 1) // 2) for n in range(count)]
    first = flow.simulate(count, arrivals=ideal, sink=a.append)
    shifted = flow.simulate(count, arrivals=((n, c + shift) for n, c in ideal), sink=b.append)
    assert shifted == dict(
        first, pipeline_drain_end_cycle=first['pipeline_drain_end_cycle'] + shift
    )
    for left, right in zip(a, b, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            assert right[key] == (left[key] + shift if key.endswith('_cycle') else left[key])
    assert flow.simulate(count, arrivals=ideal) == first
    # Query reservations may be returned before a later sample's reservation
    # in observer order. A controller must wait for ready_cycle,not call order.
    query = next(e for e in a if e['event'] == 'query_reservation')
    later = next(e for e in a if e['sample'] == query['sample'] + 1)
    assert later['ready_cycle'] < query['ready_cycle']


def test_actual_rrc_emissions_join_by_tag_without_resetting_arrival_time():
    from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow

    rrc = PairRRCFlow().simulate(1024, trace=True)
    outputs = [(e['sample'], e['cycle']) for e in rrc['trace'] if e['event'] == 'output']
    assert not rrc['dropped'] and not rrc['invalid']
    assert [n for n, _ in outputs] == list(range(1024))
    events = []
    flow = AcquisitionFlow()
    result = flow.simulate(1024, arrivals=iter(outputs), sink=events.append)
    assert not result['dropped_sample_tags'] and not result['query_drops']
    samples = [e for e in events if e['event'] == 'sample_reservation']
    assert [(e['sample'], e['arrived_cycle']) for e in samples] == outputs
    assert samples[0]['arrived_cycle'] == 60
    assert samples[0]['core_ready_cycle'] == 60 + flow.estimate['core_no_queue_latency_cycles']
    assert all(e['ready_cycle'] >= e['arrived_cycle'] for e in events)
    assert result['pipeline_drain_end_cycle'] > rrc['cycles']


@pytest.mark.parametrize('clock', [100, 160])
def test_dropped_rrc_tags_cannot_be_relabeled_or_filled_with_zeros(clock):
    from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions

    rrc = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=clock * 1_000_000)).simulate(1024, trace=True)
    outputs = [(e['sample'], e['cycle']) for e in rrc['trace'] if e['event'] == 'output']
    assert rrc['dropped'] > 0
    with pytest.raises(ValueError, match='sample tag'):
        AcquisitionFlow().simulate(1024, arrivals=outputs)


@pytest.mark.parametrize('arrivals', [
    3, [], [(True, 0)], [(1, 0)], [(0, -1)], [(0, True)], [(0, 1.0)],
    [(0, 2**63)], [(0, 0), None], [(0, 0), (1, 1)],
])
def test_malformed_or_extra_arrival_calendar_is_not_success(arrivals):
    with pytest.raises(ValueError):
        AcquisitionFlow().simulate(1, arrivals=arrivals)


@pytest.mark.parametrize('arrivals', [
    [(0, 0), (0, 13)], [(0, 0), (2, 13)], [(0, 13), (1, 13)], [(0, 13), (1, 12)],
])
def test_duplicate_missing_reordered_or_same_edge_tags_are_explicit(arrivals):
    with pytest.raises(ValueError):
        AcquisitionFlow().simulate(2, arrivals=arrivals)


def test_invalid_observer_rejected_before_reserving_work():
    with pytest.raises(ValueError, match='observer'):
        AcquisitionFlow().simulate(1, sink=True)


def test_irregular_unblocked_arrivals_have_independent_constant_stage_latency():
    rng = random.Random(372714)
    cycle, arrivals = 719, []
    for n in range(512):
        cycle += rng.choice([12, 13, 17, 51])
        arrivals.append((n, cycle))
    events = []
    result = AcquisitionFlow().simulate(512, arrivals=iter(arrivals), sink=events.append)
    assert not result['dropped_sample_tags'] and not result['query_drops']
    assert max(result['peak_waiting_by_stage']) == 0
    assert len(events) == 512 + 16
    for row in events:
        assert row['arrived_cycle'] == arrivals[row['sample']][1]
        # Frozen no-queue contract:50 registered edges;the query adds21.
        latency = 50 if row['event'] == 'sample_reservation' else 71
        assert row['ready_cycle'] - row['arrived_cycle'] == latency
