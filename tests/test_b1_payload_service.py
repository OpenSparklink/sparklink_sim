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


def _run(frames, items, *, early=True, truncate=0, mac_end=0, polar_end=0, short_tail=False):
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
        short_tail_slot=short_tail,
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
