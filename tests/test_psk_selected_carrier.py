"""Composition of actual normalized words, timed header and B1 delivery."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.common.scrambler import scramble_sequence
from nearlink_sdr.phy.control_info import ControlInfoB1
from nearlink_sdr.phy.normalization_word import pack_normalized_word
from nearlink_sdr.phy.psk_fields import map_psk_field
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig, encode_head
from nearlink_sdr.sim.psk_selected_carrier import _SuffixIQ, replay_selected_b1_carrier
from nearlink_sdr.sim.psk_word_demap import B1SymbolFrame, PSKWordDemapper, TimedPSKWord
from tests.test_b1_payload_service import _wave
from tests.test_psk_word_demap import _symbols


def _word(x):
    return pack_normalized_word(round(x.real*2**18),round(x.imag*2**18),-18)


def _frame(ft,mcs,payload,*,pilot=8,width=24,bad_head=False,bad_data=None):
    spec,_,service,loads=_symbols(0,payload,mcs=mcs,pilot=pilot,field_index=0,ft=ft)
    cfg=TxConfig(frame_type=ft,mcs_index=8,ctrl_bits_len=27,head_llid=0x173925,
                 whitening_seed=0x74,crc_seed=0x48932D,crc_len=width,
                 pilot_interval=pilot,symbol_rate_mhz=4,b1_code_block_mode="standard")
    p=PSKReceiveProfile.ordinary(cfg)
    parity=ControlInfoB1(0,0,0,mcs,len(payload)+width//8,0,0).pack(cfg.head_llid)
    if bad_head:
        parity[-1]^=1
    sync=p.sync_symbols()
    control=map_psk_field(encode_head(parity,cfg),"BPSK" if ft==4 else "QPSK",
                          len(sync),4,omit_last=not pilot)
    start=len(sync)+len(control)
    tx=replace(cfg,mcs_index=mcs)
    wire_spec=B1SymbolFrame.from_config(0,len(payload),tx,symbol_index=start,noise_variance=.1)
    coded=np.concatenate([np.asarray(x)<0 for _,x in _wave(wire_spec.payload,payload,bad_data)])
    bits=coded ^ scramble_sequence(256+len(coded),cfg.whitening_seed)[256:]
    body=map_psk_field(bits,tx.mod_str,start,pilot)
    if width!=spec.payload.crc_width:
        from nearlink_sdr.sim.payload_polar_flow import decode_payload_polar_word
        service,loads={},{}
        for shape,values in _wave(wire_spec.payload,payload):
            _, r=decode_payload_polar_word(values,*shape,geometry="threshold")
            service[shape],loads[shape]=r["cycles"],r["input_load_cycles"]
    return p,list(map(_word,np.r_[sync,control,body])),start,service,loads


@pytest.mark.parametrize("ft,mcs,pilot,width", [(3,4,8,24),(4,9,8,32),(3,6,0,24)])
def test_actual_header_gate_paced_carrier_and_mac(ft,mcs,pilot,width):
    payload=bytes((i*3+51)%256 for i in range(65))
    ctx,wire,start,service,loads=_frame(ft,mcs,payload,pilot=pilot,width=width)
    events,front=[],[]
    r=replay_selected_b1_carrier(
        iter(wire),ctx,clock_hz=200_000_000,first_edge=37,word_cycles=service,
        input_load_cycles=loads,geometry="threshold",fraction_bits=1,
        observer=events.append,front_observer=front.append,discard_failed_suffix=True,
    )
    assert r["status"]=="delivered"
    assert r["header"]["mcs"]==mcs and r["header"]["header_crc"]
    assert ctx.config.mcs_index==8
    assert r["header"]["data_start_symbol"]==start
    assert r["header_ready_edge"]>37+(start-2)*50
    assert r["gain_ready_edge"]<r["header_ready_edge"]<r["first_payload_launch"]
    assert r["first_payload_output"]-r["first_payload_launch"]==42
    assert r["carrier_payload_outputs"]==len(wire)-start
    assert 100<r["peak_suffix_words"]<=128
    assert bytes(x["value"] for x in events if x["kind"]=="mac_byte")==payload
    hdr,=[v for v in front if v["kind"]=="header_visible"]
    assert all(v["edge"]>hdr["edge"] for v in front if v["kind"]=="payload_word")


@pytest.mark.parametrize("clock,depth", [(100,8),(160,8),(200,8),(200,64)])
def test_fifo_pressure_preserves_failure(clock,depth):
    p,w,start,cycles,_=_frame(3,4,bytes(65))
    r=replay_selected_b1_carrier(iter(w),p,clock_hz=clock*10**6,word_cycles=cycles,
                                  suffix_depth=depth)
    assert r["status"]=="carrier_fifo_full"
    assert r["delivery"]=={} and r["carrier_payload_outputs"]==0
    assert r["peak_suffix_words"]==depth
    assert r["failure_edge"]>=0 and start>=p.sync_symbols().size


@pytest.mark.parametrize("part", ["sync","header","payload"])
def test_eof_retires_only_complete_owned_words(part):
    p,wire,start,service,loads=_frame(3,4,bytes(65))
    cuts=dict(sync=10,header=75,payload=start+100)
    r=replay_selected_b1_carrier(iter(wire[:cuts[part]]),p,clock_hz=200_000_000,
                                 word_cycles=service,geometry="threshold",input_load_cycles=loads)
    assert r["status"]==f"truncated_{part}"
    assert not r["delivery"].get("mac_complete",0)


def test_wrong_head_and_payload_crc_and_unsupported_decoded_mode():
    p,wire,_,service,_=_frame(4,9,bytes(65),bad_head=True)
    r=replay_selected_b1_carrier(wire,p,clock_hz=200_000_000,word_cycles=service)
    assert r["status"]=="bad_header_crc" and r["first_payload_launch"] is None
    p,wire,_,service,loads=_frame(3,4,bytes(65),bad_data="frame")
    r=replay_selected_b1_carrier(wire,p,clock_hz=200_000_000,word_cycles=service,
                                 geometry="threshold",input_load_cycles=loads)
    assert r["status"]=="payload_rejected" and r["delivery"]["frame_failed"]==1
    p,wire,start,service,_=_frame(3,4,bytes(65))
    from tests.test_b1_wire_length import b1_header_wire
    valid,_=b1_header_wire(p,65+3,mcs=8)
    r=replay_selected_b1_carrier([*map(_word,valid),*wire[start:]],p,
                                 clock_hz=200_000_000,word_cycles=service)
    assert r["status"]=="unsupported_payload_path" and r["header"]["mcs"]==8


def test_clock_context_and_tagged_demap_edge_are_checked():
    p,wire,_,service,_=_frame(3,4,bytes(65))
    with pytest.raises(ValueError,match="suffix FIFO"):
        replay_selected_b1_carrier(wire,p,clock_hz=200_000_000,word_cycles=service,suffix_depth=256)
    with pytest.raises(ValueError,match="B1 profile"):
        replay_selected_b1_carrier(wire,replace(p,payload_bitorder="little"),
                                    clock_hz=200_000_000,word_cycles=service)
    with pytest.raises(ValueError,match="signed18"):
        TimedPSKWord(0,131072,0)
    m=PSKWordDemapper(bits_per_symbol=2,n_bits=2,pilot_interval=0,symbol_index=0,
                      first_edge=100,clock_hz=200_000_000,noise_variance=.1)
    with pytest.raises(ValueError,match="carrier output edge"):
        m.feed(TimedPSKWord(99,1,1))
    assert m.feed(TimedPSKWord(100,16384,16384))[0].edge==124


def test_short_frame_pace_failure_is_owned_rejection():
    p,wire,_,service,_=_frame(3,4,b"short")
    r=replay_selected_b1_carrier(iter(wire),p,clock_hz=160_000_000,
                                 word_cycles=service,suffix_depth=128)
    assert r["status"]=="carrier_pace_missed"
    assert r["header"]["header_crc"]
    assert r["missed_launch_edge"]==r["first_payload_launch"]+40
    assert r["failure_edge"]==r["first_payload_launch"]+42
    assert not r["delivery"].get("mac_complete",0)


@pytest.mark.parametrize("depth", [8, 64, 128])
def test_suffix_ring_wrap_and_same_edge_read_write_addresses_are_distinct(depth):
    fifo = _SuffixIQ(depth, 2 * depth - 1)
    original = list(range(fifo.first_index, fifo.first_index + depth))
    for index in original:
        fifo.append(index, index - 131072, 131071 - index)
    assert len(fifo) == depth
    for index in original:
        assert fifo.first_index == index
        assert fifo.popleft() == (index - 131072, 131071 - index)
        fifo.append(index + depth, index - 131072, 131071 - index)
        with pytest.raises(RuntimeError, match="registered read"):
            fifo.popleft()
        fifo.advance()
    assert len(fifo) == depth
    for index in original:
        assert fifo.first_index == index + depth
        assert fifo.popleft() == (index - 131072, 131071 - index)
        fifo.advance()
    assert not fifo
    fifo.append(fifo.first_index, -131072, 131071)
    assert fifo.popleft() == (-131072, 131071)


def test_suffix_ring_trims_across_wrap_and_detaches_discarded_head():
    fifo = _SuffixIQ(8, 23)
    for index in range(23, 31):
        fifo.append(index, index, -index)
    assert fifo.popleft() == (23, -23)
    assert fifo.discard_from(26) == 5
    fifo.advance()
    assert fifo.popleft() == (24, -24)
    assert fifo.discard_from(25) == 1
    fifo.advance()
    assert not fifo
    fifo.append(25, -7, 9)
    assert fifo.popleft() == (-7, 9)
    with pytest.raises(RuntimeError, match="Noncontiguous"):
        fifo.append(27, 0, 0)


def test_extra_input_after_actual_short_frame_is_released():
    profile, frame, _, cycles, loads = _frame(4, 11, b"")
    input_words = iter([*frame, *([_word(3.7 + 3.7j)] * 256)])
    front = []
    result = replay_selected_b1_carrier(
        input_words, profile, clock_hz=200_000_000, word_cycles=cycles,
        geometry="threshold", input_load_cycles=loads, front_observer=front.append,
    )
    assert result["status"] == "delivered"
    assert result["header"]["frame_end_symbol"] == len(frame)
    assert result["wire_inputs"] == len(frame) + result["post_frame_buffered_words"]
    assert 0 < result["post_frame_buffered_words"] < 128
    assert sum(1 for _ in input_words) == 256 - result["post_frame_buffered_words"]
    sent = [event["index"] for event in front if event["kind"] == "payload_word"]
    assert sent == list(range(result["header"]["data_start_symbol"], len(frame)))


def test_compact_b1_banks_reduce_actual_head_wait_and_retain_paced_failure():
    p, wire, _, service, loads = _frame(4, 9, bytes(range(65)), width=32)
    deliver = {}
    for layout in ("dense", "depth-parity", "compact-parity"):
        events = []
        row = replay_selected_b1_carrier(
            iter(wire), p, clock_hz=200_000_000, word_cycles=service,
            geometry="threshold", input_load_cycles=loads, head_layout=layout,
            observer=events.append,
        )
        deliver[layout] = row
        assert row["status"] == "delivered"
        assert bytes(x["value"] for x in events if x["kind"] == "mac_byte") == bytes(range(65))
    compact = deliver["compact-parity"]
    assert compact["header"] == deliver["dense"]["header"] == deliver["depth-parity"]["header"]
    assert compact["header_ready_edge"] <= deliver["depth-parity"]["header_ready_edge"] + 64
    assert deliver["dense"]["header_ready_edge"] - compact["header_ready_edge"] >= 1200

    faster_head = replay_selected_b1_carrier(
        iter(wire), p, clock_hz=160_000_000, word_cycles=service,
        geometry="threshold", input_load_cycles=loads, head_layout="compact-parity",
    )
    assert faster_head["status"] == "carrier_pace_missed"
    assert faster_head["header"]["header_crc"] and faster_head["header"]["length_ok"]
    assert not faster_head["delivery"].get("mac_complete", 0)
    slow_head = replay_selected_b1_carrier(
        iter(wire), p, clock_hz=160_000_000, word_cycles=service,
        geometry="threshold", input_load_cycles=loads, head_layout="dense",
    )
    assert slow_head["status"] == "carrier_fifo_full"
    assert slow_head["header"] is None


def test_parallel_pi_completes_compact_b1_at_160_mhz_using_real_crc():
    payload = bytes(range(65))
    profile, words, _, cycles, loads = _frame(4, 9, payload, width=32)
    opts = dict(clock_hz=160_000_000, word_cycles=cycles, input_load_cycles=loads,
                geometry="threshold", head_layout="compact-parity")
    events = []
    serial = replay_selected_b1_carrier(words, profile, **opts)
    fast = replay_selected_b1_carrier(
        words, profile, **opts, observer=events.append, carrier_schedule="parallel-pi",
    )
    assert serial["status"] == "carrier_pace_missed"
    assert fast["header"] == serial["header"] and fast["header"]["length_ok"]
    assert fast["status"] == "delivered"
    assert fast["first_payload_output"] - fast["first_payload_launch"] == 39
    assert bytes(v["value"] for v in events if v["kind"] == "mac_byte") == payload
    with pytest.raises(ValueError, match="parallel-pi"):
        replay_selected_b1_carrier(words, profile, **opts, carrier_schedule="unknown")
