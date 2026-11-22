"""One real frame owner credit,registered replies,EOF and configuration identity."""

from dataclasses import replace

import pytest

from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.capture_frame import CapturedFrameReplay
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.fine_selection import FineSelection
from tests.captured_frame_reference import fixture


def selected(p):
    words = tuple((round(z.real * 2**17), round(z.imag * 2**17)) for z in p.sync_symbols())
    energy = len(words) - 1
    return FineSelection(0, 256, 256, 10, 9, -17, words, (energy, 0), energy**2, energy, energy)


def test_bounded_registered_reply_EOF_and_one_terminal_ownership_transfer():
    p, *_ = fixture()
    evidence = []
    owner = CapturedFrameReplay(p, coherent_threshold=0.45, raw_observer=evidence.append)
    choice = selected(p)
    with pytest.raises(ValueError, match="available"):
        owner.begin(choice, 9)
    assert not owner.active
    owner.begin(choice, 10)
    with pytest.raises(ValueError, match="unowned"):
        owner.begin(choice, 10)
    with pytest.raises(RuntimeError, match="terminal"):
        owner.release()
    word = pack_filter_word(1, 2, -10)
    owner.issue_read(0, word, 10)
    with pytest.raises(RuntimeError, match="Drain"):
        owner.end_input(10)
    owner.tick(10)
    assert owner.index == 0
    owner.tick(11, blocked=True)
    assert owner.index == 0
    owner.tick(12)
    assert owner.index == 1 and evidence[0]["index"] == 0
    with pytest.raises(ValueError, match="next"):
        owner.issue_read(3, word, 12)
    owner.end_input(12)
    assert owner.event["frame"]["status"] == "truncated_sync"
    assert not owner.take()[1].packets and owner.take() is None
    assert owner.event["complete_latency_cycles"] is None
    owner.release()
    assert not owner.active and owner.retired_through == 1
    with pytest.raises(ValueError, match="next"):
        owner.issue_read(1, word, 13)


def test_packet_configuration_is_latched_before_delayed_selected_word():
    p, *_ = fixture()
    owner = CapturedFrameReplay(p, coherent_threshold=0.45)
    choice = selected(p)
    expected = owner.profile.config.whitening_seed
    p.config.whitening_seed ^= 0x7F
    assert owner.profile.config.whitening_seed == expected
    owner.begin(choice, 10)
    assert owner.rx.profile.config.whitening_seed == expected
    owner.abort("cancelled", 11)
    assert not owner.take()[1].packets
    owner.release()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(coherent_threshold=True),
        dict(coherent_threshold=0),
        dict(coherent_threshold=float("nan")),
        dict(sample_limit=3),
        dict(sink=3),
        dict(raw_observer=3),
    ],
)
def test_malformed_frame_owner_cannot_start_receive(kwargs):
    p, *_ = fixture()
    with pytest.raises(ValueError):
        CapturedFrameReplay(p, **(dict(coherent_threshold=0.45) | kwargs))


def test_bad_readiness_clock_and_finite_statistics_are_explicit():
    p, *_ = fixture()
    owner = CapturedFrameReplay(p, coherent_threshold=0.45)
    owner.begin(selected(p), 10)
    with pytest.raises(ValueError, match="readiness"):
        owner.tick(11, blocked=1)
    with pytest.raises(ValueError, match="clock"):
        owner.tick(9)
    owner.stats["registered_reads"] = 2**64 - 1
    with pytest.raises(OverflowError, match="statistic"):
        owner.issue_read(0, pack_filter_word(1, 1, 0), 12)
    assert not owner.done and owner.pending is None


def test_zero_sync_is_confirmation_failure_without_a_CRC_retry():
    p, *_ = fixture()
    owner = CapturedFrameReplay(p, coherent_threshold=0.45)
    choice = replace(selected(p), words=((0, 0),) * 32, correlation=(0, 0), numerator=0, energy=0)
    owner.begin(choice, 10)
    assert owner.event["frame"]["status"] == "sync_confirmation_rejected"
    assert not owner.take()[1].packets
    owner.release()


def test_unmatched_transport_modes_fail_before_consuming_source():
    p, *_ = fixture()
    owner = CapturedFrameReplay(p, coherent_threshold=0.45)
    transport = CaptureTransport(CaptureTransportAssumptions(paths=1, history=4096))
    with pytest.raises(ValueError, match="matched"):
        transport.simulate([[0]], (), frame_replay=owner)
    assert not owner.active


@pytest.mark.parametrize(
    "change",
    [
        dict(allowed_mcs=(6,)),
        dict(length_contract="legacy-b1-payload-length"),
    ],
)
def test_unvalidated_configurations_remain_explicit_experimental_boundary(change):
    p, *_ = fixture(3)
    with pytest.raises(ValueError, match="ordinary standard"):
        CapturedFrameReplay(replace(p, **change), coherent_threshold=0.45)
