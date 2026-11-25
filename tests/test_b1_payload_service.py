"""Independent CRC vectors and bounded B1 delivery timing."""

from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from nearlink_sdr.common.code_block_seg import b1_code_block_plan
from nearlink_sdr.common.crc import CRC24A_POLY, CRC24B_POLY, CRC32_POLY
from nearlink_sdr.common.polar import PolarEncoder
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.b1_payload_service import B1PayloadFrame, replay_b1_payload_frames
from nearlink_sdr.sim.payload_polar_flow import decode_payload_polar_word, payload_polar_calendar


def _frame(offset, length, *, ft=3, mcs=10, width=24, seed=0x42C48B12):
    cfg = TxConfig(frame_type=ft, mcs_index=mcs, symbol_rate_mhz=4, crc_len=width,
                   crc_seed=seed, b1_code_block_mode="standard")
    return B1PayloadFrame.from_config(offset, length, cfg)


def _append(bits, width, poly, seed):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    value = ((seed & ((1 << width) - 1)) << len(bits)) ^ (value << width)
    divisor = (1 << width) | poly
    while value.bit_length() > width:
        value ^= divisor << (value.bit_length() - width - 1)
    return (*bits, *(value >> i & 1 for i in range(width - 1, -1, -1)))


def _wave(frame, payload, damage=None):
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8)).tolist()
    width, seed = frame.crc_width, frame.crc_seed
    poly = CRC24A_POLY if width == 24 else CRC32_POLY
    frame_data = list(_append(bits, width, poly, seed))
    if damage == "frame":
        frame_data[-1] ^= 1
    blocks = b1_code_block_plan(len(frame_data), frame.arrival.rate, standard=True)
    offset = 0
    for j, block in enumerate(blocks):
        end = offset + block.data_bits
        data = frame_data[offset:end]
        if block.crc_bits:
            data = list(_append(data, 24, CRC24B_POLY, seed))
            if j == 0 and damage == "block":
                data[-1] ^= 1
        cap = sum(k for _, k in block.codewords)
        padded = [1] * (cap - len(data)) + data
        bit_at = 0
        for n, k in block.codewords:
            source = padded[bit_at:bit_at+k]
            values = PolarEncoder(n, k).encode(np.array(source, dtype=np.int8))
            yield (n, k), list(np.where(values != 0, -3.3, 8.1))
            bit_at += k
        offset = end


def _run(
    frames, items, *, early=True, truncate=0, mac_end=0, polar_end=0, short_tail=False,
    mac_ring=False, short_result=False, prefer_tail=False, packed_tail=False, mac_capacity=3,
):
    values, service, loads = [], {}, {}
    for spec, (payload, damage) in zip(frames, items, strict=True):
        for (n, k), x in _wave(spec, payload, damage):
            values.extend(x)
            if (n, k) not in service:
                _, r = decode_payload_polar_word(x, n, k, geometry="threshold")
                service[n,k], loads[n,k] = r["cycles"], r["input_load_cycles"]
    events = []
    wire = iter(values[:-truncate] if truncate else values)
    report = replay_b1_payload_frames(
        iter(frames), wire, word_cycles=service, input_load_cycles=loads if early else None,
        clock_hz=200_000_000, geometry="threshold", mac_stall_until=mac_end,
        stall_until=polar_end, observer=events.append, discard_failed_suffix=True,
        short_tail_slot=short_tail, mac_ring=mac_ring, short_result_slot=short_result,
        prefer_short_tail=prefer_tail, packed_tail_slot=packed_tail,
        mac_frame_capacity=mac_capacity,
    )
    assert all(a["edge"] <= b["edge"] for a,b in pairwise(events))
    return events, report, service, loads


def _delivered(events, frame):
    return bytes(x["value"] for x in events if x["kind"]=="mac_byte" and x["frame"]==frame)


@pytest.mark.parametrize("width,length", [(24,63), (32,62), (24,0), (32,5)])
def test_b1_crc_and_actual_registered_output(width, length):
    frame = _frame(0,length,width=width,mcs=4)
    payload = bytes((k*17+5) % 256 for k in range(length))
    events,r,cycles,loads = _run([frame], [(payload,None)])
    end = list(payload_polar_calendar([frame.arrival], word_cycles=cycles,
                                      input_load_cycles=loads, clock_hz=200_000_000))[-1]["taken"]
    check, = [x for x in events if x["kind"]=="crc_result"]
    done, = [x for x in events if x["kind"]=="mac_complete"]
    assert check["edge"]==end and check["block_crc_ok"] and check["frame_crc_ok"]
    if length>=62:
        assert check["checked_blocks"]==2
    assert _delivered(events,0)==payload
    assert r["frame_ram_reads"] == r["frame_ram_writes"] == length if length else (
        "frame_ram_reads" not in r and "frame_ram_writes" not in r
    )
    assert done["edge"]==end+1+length
    assert r["peak_buffered_bytes"]==length
    assert r["frame_ram_bytes"]==2044


@pytest.mark.parametrize("damage,block_ok,frame_ok", [
    ("block", False, True), ("frame", True, False),
])
def test_corruption_domains_are_independent(damage, block_ok, frame_ok):
    frame = _frame(0,63,mcs=4)
    events,r,_,_ = _run([frame], [(bytes(range(63)),damage)])
    check, = [x for x in events if x["kind"]=="crc_result"]
    assert check["block_crc_ok"] is block_ok
    assert check["frame_crc_ok"] is frame_ok
    assert r["frame_failed"]==1 and "mac_complete" not in r
    assert _delivered(events,0)==b""


def test_wrong_configured_seed_rejects_payload():
    frame = _frame(0,63,mcs=4,width=32)
    words = list(_wave(frame, bytes(range(63))))
    service, loads = {}, {}
    for (n,k),llr in words:
        _, r = decode_payload_polar_word(llr,n,k)
        service[n,k], loads[n,k] = r["cycles"], r["input_load_cycles"]
    events = []
    status = replay_b1_payload_frames(
        [replace(frame, crc_seed=0x439B04)], (v for _,word in words for v in word),
        word_cycles=service, input_load_cycles=loads, clock_hz=200_000_000,
        observer=events.append,
    )
    check, = [x for x in events if x["kind"]=="crc_result"]
    assert not check["frame_crc_ok"] and not check["block_crc_ok"]
    assert "mac_complete" not in status


def test_mac_busy_rejects_new_frame_and_preserves_verified_ram():
    frames = [_frame(at,3) for at in (0,5000,25000)]
    events, r, _, _ = _run(frames, [(b"abc",None),(b"def",None),(b"ghi",None)], mac_end=20000)
    failures = [x for x in events if x["kind"]=="frame_failed"]
    assert [(x["frame"],x["reason"]) for x in failures]==[(1,"mac_buffer_busy")]
    assert _delivered(events,0)==b"abc" and _delivered(events,1)==b""
    assert _delivered(events,2)==b"ghi"
    assert r["mac_complete"]==2
    assert next(x["edge"] for x in events if x["kind"]=="mac_byte")==20001


def test_eof_poison_does_not_cancel_earlier_complete_queued_word():
    frames = [_frame(0,3), _frame(20000,39)]
    events, r, _, _ = _run(frames, [(b"abc",None),(bytes(39),None)],
                            truncate=4, polar_end=100000)
    assert [(x["frame"],x["reason"]) for x in events if x["kind"]=="frame_failed"]==[
        (1,"truncated"),
    ]
    assert _delivered(events,0)==b"abc"
    assert _delivered(events,1)==b""
    assert r["mac_complete"]==1


def test_resource_dropped_tail_invalidates_frame_and_keeps_following_frame():
    frames = [_frame(0,55), _frame(150000,3)]
    events, r, _, _ = _run(frames, [(bytes(55),None),(b"end",None)], early=False)
    assert [(x["frame"],x["reason"]) for x in events if x["kind"]=="frame_failed"]==[
        (0,"resource_drop"),
    ]
    assert [x["frame"] for x in events if x["kind"]=="crc_result"]==[1]
    assert _delivered(events,1)==b"end"
    assert r["mac_complete"]==1


def test_three_frame_owners_do_not_overlap_output():
    frames = [_frame(0,39), _frame(9000,3), _frame(11300,3)]
    events, r, _, _ = _run(frames, [(bytes(range(39)),None),(b"abc",None),(b"xyz",None)])
    assert r["peak_frame_owners"]==3
    assert r["mac_complete"]==3
    assert _delivered(events,0)==bytes(range(39))
    assert _delivered(events,1)==b"abc" and _delivered(events,2)==b"xyz"


def test_invalid_post_header_context_and_mac_readiness():
    frame = _frame(0,3)
    with pytest.raises(ValueError, match="length"):
        replace(frame,length_bytes=2047)
    with pytest.raises(ValueError, match="CRC context"):
        replace(frame,crc_width=24.0)
    with pytest.raises(ValueError, match="CRC seed"):
        replace(frame,crc_seed=-1)
    with pytest.raises(ValueError, match="FT3/4"):
        B1PayloadFrame.from_config(0,3,TxConfig(frame_type=1))
    with pytest.raises(ValueError, match="readiness"):
        replay_b1_payload_frames([], [], word_cycles={},clock_hz=200_000_000,mac_stall_until=-1)
    with pytest.raises(ValueError, match="descriptions"):
        replay_b1_payload_frames([frame.arrival], [], word_cycles={},clock_hz=200_000_000)


@pytest.mark.parametrize("damage", [None, "frame", "block", "EOF"])
def test_short_tail_storage_keeps_real_crc_and_truncation_checks(damage):
    frame = _frame(0,63,mcs=4) if damage == "block" else _frame(0,55)
    payload = bytes((x * 11 + 47) & 255 for x in range(frame.length_bytes))
    choices = [(payload, damage if damage != "EOF" else None)]
    base, _, _, _ = _run([frame], choices, early=False)
    assert [(x["frame"], x["reason"]) for x in base if x["kind"] == "frame_failed"] == [
        (0, "resource_drop"),
    ]
    result, stats, _, _ = _run([frame], choices, early=False, short_tail=True,
                               truncate=1 if damage == "EOF" else 0)
    if damage is None:
        assert _delivered(result, 0) == payload and stats["mac_complete"] == 1
    else:
        assert not _delivered(result, 0)
        fail, = [x for x in result if x["kind"] == "frame_failed"]
        assert fail["reason"] == ("truncated" if damage == "EOF" else "crc")
        assert "mac_complete" not in stats


def test_short_terminal_owner_and_two_regular_owners_can_overlap_active_decode():
    frames = [_frame(at, size) for at, size in ((0,39), (9000,3), (11300,3), (13450,3))]
    payloads = [bytes([x+51])*f.length_bytes for x,f in enumerate(frames)]
    events,r,_,_ = _run(frames, [(p,None) for p in payloads], short_tail=True)
    assert r["peak_frame_owners"] == 4
    assert [_delivered(events, x) for x in range(4)] == payloads


def test_ring_uses_disjoint_ranges_for_the_previously_blocked_second_frame():
    frames = [_frame(0,55), _frame(150000,3)]
    payloads = [bytes(range(55)), b"xyz"]
    items = [(p,None) for p in payloads]
    common = dict(early=False, short_tail=True, mac_end=200000)
    old,_,_,_ = _run(frames,items,**common)
    assert [(x["frame"],x["reason"]) for x in old if x["kind"]=="frame_failed"] == [
        (1,"mac_buffer_busy"),
    ]
    events,r,_,_ = _run(frames,items,**common,mac_ring=True)
    assert not any(x["kind"]=="frame_failed" for x in events)
    assert [_delivered(events,i) for i in range(2)]==payloads
    assert r["frame_ram_bytes"]==2044 and r["peak_ram_reserved_bytes"]==58
    assert r["peak_payload_reservations"]==2
    byte_edges = [x["edge"] for x in events if x["kind"]=="mac_byte"]
    assert byte_edges==list(range(200001,200056))+list(range(200057,200060))


def test_ring_allows_simultaneous_registered_read_and_distinct_write():
    frames = [_frame(0,154,mcs=4), _frame(125000,3,mcs=10)]
    items = [(bytes((j*13+i) % 256 for j in range(f.length_bytes)),None)
             for i,f in enumerate(frames)]
    observed,_,_,_ = _run(frames,items,short_tail=True)
    checks = [x["edge"] for x in observed if x["kind"]=="crc_result"]
    assert len(checks)==2
    events,stat,_,_ = _run(frames,items,short_tail=True,mac_ring=True,mac_end=checks[1]-24)
    assert [_delivered(events,i) for i in range(2)]==[p for p,_ in items]
    assert stat["ram_same_edge_read_writes"]>0
    assert stat["peak_ram_reserved_bytes"]==157


def test_ring_crc_rollback_reuses_space_and_preserves_wrapped_payload_bytes():
    frames = [_frame(i*95000,154,mcs=4) for i in range(15)]
    payloads = [bytes((j+i*11) % 256 for j in range(154)) for i in range(len(frames))]
    items = [(p,"frame" if i==8 else None) for i,p in enumerate(payloads)]
    events,r,_,_ = _run(frames,items,short_tail=True,mac_ring=True)
    assert [(x["frame"],x["reason"]) for x in events if x["kind"]=="frame_failed"]==[(8,"crc")]
    assert [_delivered(events,i) for i in range(len(frames))]==[
        b"" if i==8 else p for i,p in enumerate(payloads)
    ]
    assert r["ram_wrap_reservations"]==1
    assert r["peak_ram_reserved_bytes"]==154


def test_ring_empty_payloads_use_descriptors_and_never_overwrite():
    frames = [_frame(i*5000,0,mcs=11) for i in range(4)]
    events,r,_,_ = _run(frames,[(b"",None)]*4,short_tail=True,mac_ring=True,mac_end=90000)
    assert r["peak_ram_reserved_bytes"]==0 and r["peak_payload_reservations"]==3
    assert [(x["frame"],x["reason"]) for x in events if x["kind"]=="frame_failed"]==[
        (3,"mac_descriptor_full"),
    ]
    assert [(x["frame"],x["edge"]) for x in events if x["kind"]=="mac_complete"]==[
        (0,90000),(1,90001),(2,90002),
    ]
    assert "frame_ram_writes" not in r


def test_ring_does_not_allow_capacity_overrun():
    frames = [_frame(0,2044,ft=4,mcs=11), _frame(550000,3)]
    items = [(bytes(2044),None),(b"new",None)]
    events,r,_,_ = _run(frames,items,short_tail=True,mac_ring=True,mac_end=700000)
    assert [(x["frame"],x["reason"]) for x in events if x["kind"]=="frame_failed"]==[
        (1,"mac_capacity_full"),
    ]
    assert _delivered(events,0)==items[0][0] and not _delivered(events,1)
    assert r["peak_ram_reserved_bytes"]==2044
    with pytest.raises(ValueError,match="boolean"):
        replay_b1_payload_frames([],[],word_cycles={},clock_hz=200_000_000,mac_ring=1)


def test_ring_truncated_newest_frame_cannot_change_held_mac_bytes():
    frames = [_frame(0,55), _frame(150000,5)]
    wanted = bytes(range(55))
    events,r,_,_ = _run(frames,[(wanted,None),(b"later",None)], early=False,
                        short_tail=True,mac_ring=True,mac_end=300000,truncate=2)
    assert [(x["frame"],x["reason"]) for x in events if x["kind"]=="frame_failed"]==[
        (1,"truncated"),
    ]
    assert _delivered(events,0)==wanted and not _delivered(events,1)
    assert r["mac_complete"]==1



@pytest.mark.parametrize("damage", [None, "frame"])
def test_short_result_moves_actual_older_bits_without_hiding_crc_or_mac_pressure(damage):
    frames = [_frame(at, 2) for at in (0, 4000, 8000, 12000)]
    inputs = [(b"ab", None), (b"cd", damage), (b"ef", None), (b"gh", None)]
    options = dict(polar_end=100_000, mac_end=160_000, mac_ring=True)
    baseline, _, _, _ = _run(frames, inputs, **options)
    events, report, _, _ = _run(frames, inputs, short_result=True, **options)
    reference_failures = [(3, "resource_drop")]
    if damage is not None:
        reference_failures.append((1, "crc"))
    assert [(e["frame"], e["reason"]) for e in baseline
            if e["kind"] == "frame_failed"] == reference_failures
    copied, = [e for e in events if e["kind"] == "result_buffered"]
    release, = [e for e in events if e["kind"] == "result_buffer_released"]
    assert copied["frame"] == release["frame"] == 0
    assert copied["bits"] == 44 and copied["edge"] < 100_000
    assert release["edge"] == copied["until"] == 100_022
    crc, = [e for e in events if e["kind"] == "crc_result" and e["frame"] == 0]
    assert crc["edge"] == release["edge"] and crc["block_crc_ok"] and crc["frame_crc_ok"]
    assert (report["result_buffered"], report["result_buffer_released"]) == (1, 1)
    assert report["peak_payload_reservations"] == 3
    assert _delivered(events, 0) == b"ab" and _delivered(events, 2) == b"ef"
    failures = [(e["frame"], e["reason"]) for e in events if e["kind"] == "frame_failed"]
    assert failures == ([(3, "mac_descriptor_full")] if damage is None else [(1, "crc")])
    if damage is None:
        assert _delivered(events, 1) == b"cd" and _delivered(events, 3) == b""
    else:
        assert _delivered(events, 1) == b"" and _delivered(events, 3) == b"gh"



def test_packed_tail_reaches_true_crc_and_ordered_mac_without_more_data_ram():
    payloads = (bytes(range(39)), b"ab", bytes(range(39,78)), bytes(range(53)))
    frames = [_frame(e,len(p),mcs=m) for e,p,m in zip(
        (0,10000,12000,22000),payloads,(10,10,10,11),strict=True,
    )]
    arguments = dict(polar_end=100_000, short_tail=True, prefer_tail=True, mac_ring=True)
    baseline, _, _, _ = _run(frames, [(p,None) for p in payloads], **arguments)
    events, stats, _, _ = _run(frames, [(p,None) for p in payloads], packed_tail=True, **arguments)
    assert [(x["frame"],x["reason"]) for x in baseline if x["kind"]=="frame_failed"] == [
        (3,"resource_drop"),
    ]
    actual = tuple(_delivered(events,i) for i in range(4))
    assert actual == payloads
    assert all(x["block_crc_ok"] and x["frame_crc_ok"] for x in events if x["kind"]=="crc_result")
    packed, = [x for x in events if x["kind"]=="packed_tail_reserved"]
    assert (packed["frame"], packed["base"], packed["slot"]) == (3,512,1)
    assert stats["mac_complete"] == 4 and stats["frame_ram_bytes"] == 2044



def test_fourth_mac_descriptor_is_separate_from_data_ram_and_does_not_admit_a_fifth():
    payloads = tuple(bytes((i,i+1)) for i in range(5))
    frames = [_frame(i*8000,2) for i in range(5)]
    opts = dict(mac_ring=True, mac_end=60_000)
    base, _, _, _ = _run(frames, [(p,None) for p in payloads], **opts)
    trial, result, _, _ = _run(frames, [(p,None) for p in payloads], mac_capacity=4, **opts)
    assert [x["frame"] for x in base if x["kind"] == "mac_complete"] == [0,1,2]
    assert [x["frame"] for x in trial if x["kind"] == "mac_complete"] == [0,1,2,3]
    assert [_delivered(trial,i) for i in range(4)] == list(payloads[:4])
    fail, = [x for x in trial if x["kind"] == "frame_failed"]
    assert (fail["frame"],fail["reason"]) == (4,"mac_descriptor_full")
    assert result["frame_ram_bytes"] == 2044
    assert result["peak_ram_reserved_bytes"] == 8
    assert result["peak_payload_reservations"] == result["mac_frame_capacity"] == 4
    with pytest.raises(ValueError, match="four with the MAC ring"):
        replay_b1_payload_frames([], [], word_cycles={}, clock_hz=200_000_000, mac_frame_capacity=4)
