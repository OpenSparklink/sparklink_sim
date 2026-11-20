"""Actual padded planes,registered port traces and one compiled head choice before CRC."""

import json
import random
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.control_polar_word import ControlPolarFormat
from nearlink_sdr.phy.normalization_word import NormalizationFormat
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.control_polar_flow import ControlPolarService
from nearlink_sdr.sim.packed_head_replay import (
    PackedControlPolarService,
    PackedHeadSamplingReplay,
    control_bank_plan,
)
from nearlink_sdr.sim.split_sampling_replay import SplitSamplingFrameReplay
from tests.test_control_polar_flow import drain
from tests.test_head_word_replay import run
from tests.test_selected_frame_replay import prepare
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input

VECTORS = json.loads((Path(__file__).parent / 'data/control_polar_reference.json').read_text())


@pytest.mark.parametrize("N,K", [(64, 40), (256, 51)])
@pytest.mark.parametrize("layout", ["dense", "depth-parity"])
def test_same_integer_microprogram_and_every_registered_bank_event(N, K, layout):
    fmt = ControlPolarFormat(N, K)
    vectors = [v for v in VECTORS['vectors'] if v['N'] == N]
    randomizer = random.Random(712219 + N)
    patterns = [([0]*N, None), ([1]*N, None), ([i & 1 for i in range(N)], None)]
    patterns += [(v['coded_bits'], v['information_bits']) for v in vectors]
    patterns += [([randomizer.randrange(2) for _ in range(N)], None) for _ in range(3)]
    for word, independent in patterns:
        old_events, new_events = [], []
        old = ControlPolarService(fmt, layout=layout, observer=old_events.append)
        new = PackedControlPolarService(fmt, layout=layout, observer=new_events.append)
        assert old.issue(word) and new.issue(word)
        drain(old)
        drain(new)
        assert old.cycle == new.cycle
        assert old_events == new_events and old.stats == new.stats
        mapping = {b.name: b for b in new.memory.banks}
        per_edge = {}
        for e in new_events:
            if e['event'] in ('read', 'write'):
                per_edge.setdefault(e['cycle'], []).append(e)
        site_peak = 0
        for events in per_edge.values():
            unit_ports, site_ports = {}, {}
            reads, writes = set(), set()
            for e in events:
                bank = mapping[e['bank']]
                key = bank.site, bank.half
                unit_ports[key] = unit_ports.get(key, 0) + 1
                site_ports[bank.site] = site_ports.get(bank.site, 0) + 1
                assert unit_ports[key] <= (1 if bank.name == 'ROM' else 2)
                (reads if e['event'] == 'read' else writes).add((bank.name, e['address']))
            assert not reads & writes
            for site, count in site_ports.items():
                full = any(b.site == site and b.half is None for b in mapping.values())
                assert count <= (2 if full else 4)
                site_peak = max(site_peak, count)
        if layout == 'depth-parity':
            assert site_peak > 2  # Two independent18Kb halves,not a falsely shared two-port36Kb.
        a, b = old.take(), new.take()
        assert a == b and a['crc_status'] == 'not_checked'
        if independent is not None:
            assert b['information_bits'] == tuple(independent)
        assert old.memory.peak == new.memory.peak
        for name, plane in new.memory.cells.items():
            assert list(plane) == old.memory.cells[name]
            assert len(plane.physical_words) >= len(plane)
            assert all(v is None or 0 <= v < 1 << plane.bank.logical_bits
                       for v in plane.physical_words)


@pytest.mark.parametrize("N,K,layout,sites", [(64, 40, 'dense', 2),
    (64, 40, 'depth-parity', 3), (256, 51, 'dense', 3), (256, 51, 'depth-parity', 3)])
def test_explicit_half_sites_modes_widths_and_distinct_ports(N, K, layout, sites):
    plan = control_bank_plan(ControlPolarFormat(N, K), layout=layout)
    occupied = set()
    for bank in plan['banks']:
        assert bank['logical_length'] <= bank['depth']
        assert bank['word_bits'] in (1, 9, 18)
        assert bank['logical_bits'] <= bank['word_bits']
        assert not bank['payload_uses_parity'] and not bank['ECC_enabled']
        assert bank['mode'].startswith('TDP')
        assert bank['ports_used_max'] == (1 if bank['name'] == 'ROM' else 2)
        for half in [0, 1] if bank['half'] is None else [bank['half']]:
            assert (bank['site'], half) not in occupied
            occupied.add((bank['site'], half))
    assert plan['physical_sites'] == sites
    assert plan['configured_native_bits'] + plan['unassigned_reserved_half_bits'] == sites*36864
    assert plan['new_DSP'] == plan['new_logical_state_bits'] == 0
    assert not plan['synthesized'] and plan['LUT_FF_or_Fmax'] is None


@pytest.mark.parametrize("N,K", [(64, 40), (256, 51)])
def test_signed_range_padding_and_unaddressed_native_tail_are_explicit(N, K):
    node = PackedControlPolarService(ControlPolarFormat(N, K), layout='depth-parity')
    for name, plane in node.memory.cells.items():
        values = [-512, -257, -1, 0, 1, 255, 511] if name.startswith('L') else (
            [0, (1 << plane.bank.logical_bits)-1])
        for index in range(len(plane)):
            value = values[index % len(values)]
            plane[index] = value
            assert plane[index] == value
            assert plane.physical_words[index] == value & ((1 << plane.bank.logical_bits)-1)
        before = plane.physical_words.copy()
        with pytest.raises(OverflowError):
            plane[0] = 512 if name.startswith('L') else 1 << plane.bank.logical_bits
        for address in [-1, len(plane)]:
            with pytest.raises(IndexError):
                plane[address] = 0
            with pytest.raises(IndexError):
                _ = plane[address]
        assert plane.physical_words == before
        assert all(v is None for v in plane.physical_words[len(plane):])


@pytest.mark.parametrize("N,K", [(64, 40), (256, 51)])
@pytest.mark.parametrize("cut", [0, 100, 900])
def test_abort_reset_preserves_one_credit_and_no_stale_registered_reply(N, K, cut):
    fmt = ControlPolarFormat(N, K)
    old, new = ControlPolarService(fmt), PackedControlPolarService(fmt)
    for node in (old, new):
        assert node.issue([1]*N)
        for _ in range(cut):
            node.step()
        node.abort()
        assert not node.memory.pending and not any(node.slots)
        node.reset()
        assert node.issue([i & 1 for i in range(N)])
        drain(node)
        for _ in range(11):
            node.step()
    assert old.take() == new.take() and old.stats == new.stats
    assert not new.input_ready and not new.memory.pending


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 0, 19), (3, 12, 8, 47), (4, 10, 4, 23)])
def test_complete_selected_frames_keep_all_sampling_and_CRC_values(ft, mcs, pilot, length):
    p = ordinary(ft, mcs, pilot)
    values, payload, selection = selected_input(p, length=length)
    old, a, _ = run(p, quantize_input(values), selection, partial(
        SplitSamplingFrameReplay, error_age=1))
    new, b, barriers = run(p, quantize_input(values), selection, partial(
        PackedHeadSamplingReplay, error_age=1))
    assert old['frame'] == new['frame'] and new['frame']['payload'] == payload
    for key in ['symbols', 'symbol_centers_filtered', 'timing_rate_ppm', 'timing_detector_error']:
        np.testing.assert_array_equal(old[key], new[key])
    assert a.consumer.word_timing.snapshot() == b.consumer.word_timing.snapshot()
    assert b.head_owner.header.codec.layout == ('dense' if ft == 2 else 'depth-parity')
    assert b.estimate()['head']['BRAM36_sites_upper'] == (2 if ft == 2 else 3)
    assert b.history.capacity == 512 and b.kernel is b.prepared.stencil.interpolator
    assert b.cycle <= a.cycle and barriers == ['payload']
    assert b.take_mac_result().packets[0].mac_payload == payload


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_selected_continuous200MHz_source_reaches_real_header_and_unmodeled_payload(ft):
    p = ordinary(ft, 8 if ft == 2 else 12, 8 if ft == 3 else 4 if ft == 4 else 0)
    values, _, selection = selected_input(p, length=19 if ft == 2 else 47 if ft == 3 else 23)
    words = quantize_input(np.r_[values, np.zeros(4096)])
    selection['sample_limit'] = len(words)
    prepared, pos = prepare(p, words, selection)
    flow = PackedHeadSamplingReplay(p, prepared, error_age=1,
                                   normalization_format=NormalizationFormat(),
                                   word_timing_format=TimingWordFormat(30, 32),
                                   reuse_stencil=True, **selection)
    first, origin = pos, flow.cycle
    while not flow.done and flow.pending_service is None:
        if flow.cycle >= origin + ((pos-first)*25+1)//2:
            assert flow.write_input(words[pos])
            pos += 1
        flow.step()
        assert flow.cycle-origin < 100000
    assert not flow.done and flow.pending_service['kind'] == 'payload'
    assert flow.consumer.head_receipt_taken and flow.head_owner.header.stats['completed'] == 1
    assert not flow.stats['input_overflow'] and flow.history.live < 512
    with pytest.raises(RuntimeError, match='terminal'):
        flow.take_mac_result()
    while not flow.done:  # Now explicit downstream blockage;theADC still cannot pause.
        if flow.cycle >= origin + ((pos-first)*25+1)//2:
            flow.write_input(words[pos])
            pos += 1
        flow.step()
    assert flow.event['frame']['status'] == 'history_overflow'
    assert flow.stats['input_overflow'] == 1 and not flow.take_mac_result().packets


@pytest.mark.parametrize("damage,status", [({'bad_head': True}, 'bad_header_crc'),
                                         ({'bad_data': True}, 'bad_data_crc')])
def test_real_CRC_failure_keeps_one_answer_and_no_MAC(damage, status):
    p = ordinary(3, 12, 8)
    values, _, selection = selected_input(p, length=17, **damage)
    event, flow, _ = run(p, quantize_input(values), selection, partial(
        PackedHeadSamplingReplay, error_age=1))
    assert event['frame']['status'] == status and not flow.take_mac_result().packets
    assert flow.head_owner.stats['header_issues'] == 1
    if damage.get('bad_head'):
        assert flow.query_count == flow.consumer._head_required
        assert flow.consumer.frame_end_symbol is None


@pytest.mark.parametrize("ft,bad", [(2, 'depth-parity'), (3, 'dense')])
def test_layout_cannot_be_selected_after_CRC_or_consume_source_on_rejection(ft, bad):
    p = ordinary(ft, 8, 0)
    values, _, selection = selected_input(p, length=3)
    prepared, _ = prepare(p, quantize_input(values), selection)
    before = deepcopy(prepared.fit.result)
    with pytest.raises(ValueError, match='compiled'):
        PackedHeadSamplingReplay(p, prepared, head_layout=bad, error_age=1,
                                 normalization_format=NormalizationFormat(),
                                 word_timing_format=TimingWordFormat(30, 32),
                                 reuse_stencil=True, **selection)
    assert prepared.ready and prepared.fit.result == before and not prepared.fit.output_taken


@pytest.mark.parametrize("fmt,layout", [(None, 'dense'), (ControlPolarFormat(64, 40), 'unknown')])
def test_uncompiled_primitive_plan_is_rejected(fmt, layout):
    with pytest.raises(ValueError):
        control_bank_plan(fmt, layout=layout)
