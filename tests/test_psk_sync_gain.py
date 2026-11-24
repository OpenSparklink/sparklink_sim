"""Clocked residual/quotient and prepared-gain B1 composition tests."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.sim.psk_sync_gain import PSKSoftGain, PSKSyncGain, prepare_psk_sync_gain
from nearlink_sdr.sim.psk_word_demap import PSKWordDemapper, replay_b1_symbol_frames
from tests.test_psk_word_demap import _symbols


def _reference(energy, count):
    g = min(Fraction(2*count*2**28, energy), 2_000_000_000_000) if energy else Fraction(
        2_000_000_000_000, 1,
    )
    exponent = 0
    while g >= 1:
        g /= 2
        exponent += 1
    while g < Fraction(1,2):
        g *= 2
        exponent -= 1
    mantissa = round(g*2**17)
    return (65536,exponent+1) if mantissa==131072 else (mantissa,exponent)


@pytest.mark.parametrize("n", [1,62,126,256])
def test_integer_square_and_cycle_exact_gain(n):
    rng=np.random.default_rng(n)
    samples=[tuple(map(int,x)) for x in rng.integers(-131072,131072,size=(n,2))]
    reference=[tuple(map(int,x)) for x in rng.integers(-131072,131072,size=(n,2))]
    energy=sum((s[0]-r[0])**2+(s[1]-r[1])**2 for s,r in zip(samples,reference,strict=True))
    events=[]
    model=PSKSyncGain(n,observer=events.append)
    for k,(s,r) in enumerate(zip(samples,reference,strict=True)):
        model.accept(7+k*25,s,r)
    last=7+(n-1)*25
    assert model.advance(last+22) is None
    gain=model.advance(last+23)
    assert gain.ready_edge==last+23
    assert (gain.mantissa,gain.exponent)==_reference(energy,n)
    assert model.energy==energy and model.samples==n
    assert model.squares==2*n and model.division_steps==17
    div=[d["edge"] for d in events if d["kind"]=="quotient"]
    assert div==list(range(last+6,last+23))
    assert events[-1]["kind"]=="ready"


@pytest.mark.parametrize("residue", [0,1,262143])
def test_floor_resolution_and_largest_energy(residue):
    model=PSKSyncGain(256)
    for i in range(256):
        ref=(131071,131071)
        # Only one axis of the final sample has a mismatch.
        sample=(131071-residue if i==255 else 131071,131071)
        if residue==262143:
            sample=(-131072,-131072)
        model.accept(4*i,sample,ref)
    expected=2*256*residue**2 if residue==262143 else residue**2
    gain=model.advance(4*255+23)
    assert model.energy==expected
    assert (gain.mantissa,gain.exponent)==_reference(expected,256)


def test_busy_and_invalid_gain_contexts():
    with pytest.raises(ValueError,match="256"):
        PSKSyncGain(257)
    r=PSKSyncGain(2)
    with pytest.raises(ValueError,match="signed18"):
        r.accept(0,(131072,0),(0,0))
    r.accept(1,(1,0),(0,0))
    with pytest.raises(ValueError,match="busy"):
        r.accept(4,(1,0),(0,0))
    with pytest.raises(ValueError,match="monotonic"):
        r.advance(3)
    r.accept(5,(1,0),(0,0))
    assert r.advance(28)
    with pytest.raises(ValueError,match="complete"):
        r.accept(29,(0,0),(0,0))
    with pytest.raises(ValueError,match="gain words"):
        PSKSoftGain(100,0,0)
    with pytest.raises(ValueError,match="MHz"):
        prepare_psk_sync_gain([0],[0],first_edge=0,clock_hz=150_000_000)
    with pytest.raises(ValueError,match="first sync edge"):
        prepare_psk_sync_gain([0],[0],first_edge=-3,clock_hz=100_000_000)
    with pytest.raises(ValueError,match="longer"):
        prepare_psk_sync_gain([0],[0,0],first_edge=0,clock_hz=100_000_000)
    with pytest.raises(ValueError,match="finite"):
        prepare_psk_sync_gain([float("nan")],[0],first_edge=0,clock_hz=100_000_000)


def test_prepared_gain_wire_replay_and_missed_deadline():
    clock=200_000_000
    start=62*50
    frame,wire,cycles,loads=_symbols(start,b"sync gain",mcs=10,pilot=8,field_index=384)
    ref=np.ones(62,dtype=complex)
    gain,stats=prepare_psk_sync_gain(ref+0.32j,ref,first_edge=0,clock_hz=clock)
    assert gain.ready_edge==61*50+23 and stats["squares"]==124
    assert stats["division_steps"]==17
    prepared=replace(frame,variance=None,prepared_gain=gain)
    opts=dict(clock_hz=clock,word_cycles=cycles,input_load_cycles=loads,geometry="threshold")
    events=[]
    report=replay_b1_symbol_frames([prepared],wire,observer=events.append,**opts)
    assert report["mac_complete"]==1
    assert bytes(x["value"] for x in events if x["kind"]=="mac_byte")==b"sync gain"
    late=replace(prepared,prepared_gain=replace(gain,ready_edge=start+1))
    consumed=[]
    def counted_wire():
        for x in wire:
            consumed.append(x)
            yield x
    with pytest.raises(ValueError,match="missed first"):
        replay_b1_symbol_frames([late],counted_wire(),**opts)
    assert not consumed
    with pytest.raises(ValueError,match="one prepared"):
        replay_b1_symbol_frames([replace(prepared,variance=.04)],wire,**opts)
    service=PSKWordDemapper(bits_per_symbol=3,n_bits=3,pilot_interval=0,symbol_index=0,
                            first_edge=gain.ready_edge,clock_hz=clock,noise_variance=None,
                            prepared_gain=gain)
    assert service.feed(1)[0].edge==gain.ready_edge+24
