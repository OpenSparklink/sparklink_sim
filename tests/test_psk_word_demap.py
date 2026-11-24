"""PSK datapath, pilot timing and finite B1 composition witnesses."""

from dataclasses import replace

import numpy as np
import pytest

from nearlink_sdr.common.scrambler import scramble_sequence
from nearlink_sdr.phy.psk_fields import map_psk_field, soft_demap_psk_field
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.payload_polar_flow import (
    B1PayloadArrival,
    decode_payload_polar_word,
    payload_polar_calendar,
)
from nearlink_sdr.sim.payload_polar_ingress import TimedLLR, replay_payload_polar_ingress
from nearlink_sdr.sim.psk_word_demap import (
    B1SymbolFrame,
    PSKWordDemapper,
    replay_b1_symbol_frames,
)
from tests.test_b1_payload_service import _wave


@pytest.mark.parametrize("bps,kind", [(2,"QPSK"),(3,"8PSK")])
@pytest.mark.parametrize("pilot", [0,1,4,8,16])
@pytest.mark.parametrize("fraction", [0,1])
def test_demap_nominal_quantization_pilot_and_whitening(bps,kind,pilot,fraction):
    rng=np.random.default_rng(541+bps+pilot+fraction)
    n=139
    data=rng.integers(0,2,n)
    seed=83
    signs=scramble_sequence(256+n,seed)[256:].astype(int)
    start=5+fraction
    wire=map_psk_field(data ^ signs,kind,start,pilot)
    wire+=.42*(rng.normal(size=len(wire))+1j*rng.normal(size=len(wire)))
    events=[]
    a=PSKWordDemapper(
        bits_per_symbol=bps,n_bits=n,pilot_interval=pilot,symbol_index=start,
        first_edge=61,clock_hz=100_000_000,noise_variance=.067,
        whitening_seed=seed,head_coded_bits=256,fraction_bits=fraction,observer=events.append,
    )
    beat=[v for x in wire for v in a.feed(x)]
    ref=soft_demap_psk_field(wire,kind,start,.067,pilot)[:n]*(1-2*signs)
    scale=1<<fraction
    assert len(beat)==n and a.done
    assert np.max(np.abs(np.array([v.value for v in beat])*scale-np.rint(ref*scale)))<=1
    assert all(v.edge>=85 and (v.edge-85)%25==0 for v in beat)
    pilots=[x for x in events if x["kind"]=="pilot"]
    assert len(pilots)==len(wire)-(n+bps-1)//bps
    assert not {x["edge"]+24 for x in pilots}&{v.edge for v in beat}
    assert events[0]["projection_ready"]==71 and events[0]["maxima_ready"]==79
    assert a.counts["projection_products"]==((n+bps-1)//bps)*(1<<bps)
    with pytest.raises(ValueError,match="Extra"):
        a.feed(0)


def test_extreme_inputs_are_bounded_and_widths_are_checked():
    kwargs=dict(bits_per_symbol=3,n_bits=3,pilot_interval=0,symbol_index=0,
                first_edge=0,clock_hz=200_000_000,fraction_bits=1)
    a=PSKWordDemapper(**kwargs,noise_variance=1e-20)
    result=a.feed(-1e120+1e100j)
    assert all(abs(v.value)<=64 for v in result)
    assert a.counts["clipped_components"]==2 and a.counts["saturated_llrs"]>=1
    b=PSKWordDemapper(**kwargs,noise_variance=1e300)
    assert all(v.value==0 for v in b.feed(1+2j))
    with pytest.raises(ValueError,match="variance"):
        PSKWordDemapper(**kwargs,noise_variance=float("nan"))
    with pytest.raises(ValueError,match="context"):
        PSKWordDemapper(**(kwargs|dict(pilot_interval=3)),noise_variance=.3)
    with pytest.raises(ValueError,match="finite normalized"):
        PSKWordDemapper(**kwargs,noise_variance=.3).feed(float("inf"))


def _symbols(first, payload, *, mcs, pilot, field_index, ft=3):
    cfg=TxConfig(frame_type=ft,mcs_index=mcs,symbol_rate_mhz=4,crc_seed=0x48932D,
                 whitening_seed=0x74,pilot_interval=pilot,b1_code_block_mode="standard")
    frame=B1SymbolFrame.from_config(first,len(payload),cfg,symbol_index=field_index,
                                    noise_variance=.091)
    service,loads,bits={}, {}, []
    for (n,k),v in _wave(frame.payload,payload):
        bits.extend(np.asarray(v)<0)
        if (n,k) not in service:
            _,r=decode_payload_polar_word(v,n,k,geometry="threshold")
            service[n,k],loads[n,k]=r["cycles"],r["input_load_cycles"]
    n=len(bits)
    coded=np.asarray(bits,dtype=np.uint8)^scramble_sequence(n+256,cfg.whitening_seed)[256:]
    wire=map_psk_field(coded,cfg.mod_str,field_index,pilot)
    return frame,wire,service,loads


@pytest.mark.parametrize("gap,deliveries", [(1000,1),(11000,2)])
def test_actual_demap_arrival_pilots_and_two_frames_through_mac(gap, deliveries):
    first,x,cycles,loads=_symbols(25,bytes(range(63)),mcs=4,pilot=8,field_index=314)
    second,y,c2,l2=_symbols(25+len(x)*50+gap,b"hello there",mcs=10,pilot=1,field_index=355,ft=4)
    events,stages=[],[]
    report=replay_b1_symbol_frames(
        iter([first,second]),(v for field in (x,y) for v in field),clock_hz=200_000_000,
        word_cycles=cycles|c2,input_load_cycles=loads|l2,geometry="threshold",
        observer=events.append,demap_observer=stages.append,fraction_bits=1,
    )
    assert report["mac_complete"]==deliveries
    assert [bytes(v["value"] for v in events if v["frame"]==i and v["kind"]=="mac_byte")
            for i in range(deliveries)]==[bytes(range(63)),b"hello there"][:deliveries]
    failures=[x["reason"] for x in events if x["kind"]=="frame_failed"]
    assert failures==(["resource_drop"] if deliveries==1 else [])
    calendar=list(payload_polar_calendar(
        [first.for_llrs().arrival,second.for_llrs().arrival],word_cycles=cycles|c2,
        input_load_cycles=loads|l2,clock_hz=200_000_000,trace_ownership=True,
    ))
    ends=[r["taken"] for i,r in enumerate(calendar)
          if not r["lost"] and (i+1==len(calendar) or calendar[i+1]["frame"]!=r["frame"])]
    assert [v["edge"] for v in events if v["kind"]=="crc_result"]==ends
    assert len(stages)==len(x)+len(y) and any(v["kind"]=="pilot" for v in stages)
    assert calendar[0]["first"]==49 and calendar[0]["pilot_interval"]==8


def test_symbol_eof_tail_and_late_llr_remain_explicit_failures():
    frame,wire,cycles,loads=_symbols(21,b"hello",mcs=10,pilot=4,field_index=238)
    opts=dict(clock_hz=160_000_000,word_cycles=cycles,input_load_cycles=loads,
              geometry="threshold")
    events=[]
    replay_b1_symbol_frames([frame],iter(wire[:-1]),observer=events.append,**opts)
    fail,=[x for x in events if x["kind"]=="frame_failed"]
    assert fail["reason"]=="truncated"
    with pytest.raises(ValueError,match="Trailing"):
        replay_b1_symbol_frames([frame],[*wire,0],**opts)
    with pytest.raises(ValueError,match="description"):
        replay_b1_symbol_frames([frame.payload],wire,**opts)
    with pytest.raises(ValueError,match="context"):
        replay_b1_symbol_frames([replace(frame,whitening_seed=255)],wire,**opts)
    with pytest.raises(ValueError,match="registered ingress"):
        list(replay_payload_polar_ingress(
            [frame.for_llrs().arrival],[TimedLLR(1234,1)],**opts,
        ))
    with pytest.raises(ValueError,match="pilot interval"):
        list(payload_polar_calendar([B1PayloadArrival(0,40,"3/4",3,3)],
                                    word_cycles=cycles,clock_hz=160_000_000))
