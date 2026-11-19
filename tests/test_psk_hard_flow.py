"""Registered value-carrying hard head, finite ownership and real CRC negatives."""

from dataclasses import replace

import pytest

from nearlink_sdr.phy.normalization_word import pack_normalized_word
from nearlink_sdr.phy.psk_hard_word import hard_psk_word
from nearlink_sdr.sim.header_word_flow import HeaderWordService
from nearlink_sdr.sim.psk_hard_flow import HardPSKWordService, HeaderFromWordsService
from tests.test_b1_wire_length import b1_header_wire, profile
from tests.test_header_word_flow import drain, hard_header
from tests.test_receive_scope import make_profile
from tests.test_strict_receive import transmitted


def as_word(symbol):
    # Test-only input adapter. Integration evidence must consume actual service words.
    return pack_normalized_word(round(symbol.real * 2**20), round(symbol.imag * 2**20), -20)


def assemble(service, words, *, period=0, eof=True):
    assert service.begin()
    index = 0
    for _ in range(15000):
        if index < len(words) and service.word_input_ready and not service.cycle % (period + 1):
            assert service.offer(words[index], service.start + index)
            index += 1
        if index == len(words) and eof and not service.eof:
            service.end_input()
        if service.ready:
            return service.take()
        service.step()
    pytest.fail("Finite word head did not drain")


def test_hard_service_actual_edges_credit_hold_and_output_value():
    events = []
    flow = HardPSKWordService(observer=events.append)
    word = pack_normalized_word(-(1 << 21), (1 << 21) - 1, 1023)
    assert flow.input_ready and flow.take() is None
    assert flow.issue(word, "QPSK", 1) and not flow.issue(word, "QPSK", 1)
    flow.step()
    assert not flow.ready and (flow.first, flow.second) == (-4194303, -1)
    flow.step()
    assert flow.ready and flow.cycle == 2
    for _ in range(3):
        flow.step()
    assert flow.stats["output_stalls"] == 3
    assert flow.take() == hard_psk_word(word, "QPSK", 1)
    assert flow.take() is None and not flow.issue(0, "BPSK", 0)
    flow.step()
    assert flow.issue(0, "BPSK", 0)
    flow.step()
    flow.step()
    assert flow.take()["zero_word"] and flow.stats["ties"] == 1
    assert flow.stats["add23"] == 2 and flow.stats["sign23"] == 2
    assert any(e["event"] == "projection" and e["register_edge"] == 1 for e in events)
    cost = flow.cost()
    assert cost["latency_edges"] == 2 and cost["unstalled_II"] == 3
    assert cost["DSP"] == cost["RAMB18"] == 0
    assert not cost["whole_RX16MSps"] and cost["verified_Fmax"] is None


def test_hard_invalid_abort_reset_and_overflow_are_explicit():
    with pytest.raises(ValueError):
        HardPSKWordService(observer=3)
    flow = HardPSKWordService()
    with pytest.raises(ValueError):
        flow.issue(0, "8PSK", 0)
    assert flow.stats["accepted"] == 0
    flow.issue(0, "BPSK", 0)
    with pytest.raises(RuntimeError):
        flow.reset()
    flow.abort()
    with pytest.raises(RuntimeError):
        flow.issue(0, "BPSK", 0)
    flow.reset()
    flow.issue(0, "BPSK", 0)
    flow.step()
    flow.step()
    flow.take()
    flow.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        flow.step()
    flow.cycle = 0
    flow.last_take = None
    flow.stats["accepted"] = 2**64 - 1
    with pytest.raises(OverflowError):
        flow.issue(0, "BPSK", 0)
    assert flow.mode == "idle"


@pytest.mark.parametrize("ft,pilot", [(2, 0), (2, 8), (3, 4), (3, 16), (4, 0), (4, 8)])
@pytest.mark.parametrize("bank", ["dense", "depth-parity"])
def test_words_to_real_header_matches_existing_bits_crc_geometry(ft, pilot, bank):
    p = make_profile(ft, 8, pilot)
    p = replace(p, length_contract="standard")
    service = HeaderFromWordsService(p, layout=bank)
    for length, period in [(1, 0), (37, 3), (127, 7)]:
        wire, _ = transmitted(p, length=length)
        bits = hard_header(wire, p)
        expected = HeaderWordService(p, layout=bank)
        assert expected.issue(bits)
        row = drain(expected)
        words = [as_word(z) for z in wire[40 + service.start : 40 + service.end]]
        assert assemble(service, words, period=period) == row
        assert service.bit_count == service.size and service.stats["word_accepts"] > 0
        service.step()  # Transfer edge is not reused for a new context.
    assert service.stats["completed"] == 3
    cost = service.cost()
    assert cost["extra_candidate_capacity_met"] and cost["input_FIFO"] == 0
    assert not cost["whole_RX16MSps"] and cost["payload_cycles"] is None
    assert cost["header"]["RAMB18_equivalents"] == cost["RAMB18_equivalents"]


@pytest.mark.parametrize(
    "damage,status", [({"bad_head": True}, "bad_header_crc"), ({"mcs": 15}, "unsupported_mcs")]
)
def test_real_crc_or_policy_failure_does_not_try_another_answer(damage, status):
    p = profile(3)
    wire, _ = transmitted(p, **damage)
    flow = HeaderFromWordsService(p)
    row = assemble(flow, [as_word(z) for z in wire[40 + flow.start : 40 + flow.end]])
    assert row["status"] == status and row["payload"] is None
    assert flow.stats["header_issues"] == flow.header.stats["accepted"] == 1
    assert flow.header.geometry.stats["accepted"] == 0


@pytest.mark.parametrize("ft,raw", [(3, 0), (4, 0), (3, 1), (4, 3)])
def test_standard_no_data_illegal_length_and_crc_only_remain_distinct(ft, raw):
    p = profile(ft)
    wire, _ = b1_header_wire(p, raw)
    flow = HeaderFromWordsService(p)
    words = [as_word(z) for z in wire[flow.start : flow.end]]
    reference = HeaderWordService(p)
    # Independent existing floating demapper supplies the comparison bits.
    from nearlink_sdr.phy.psk_fields import demap_psk_field

    bits = list(
        map(int, demap_psk_field(wire[flow.start : flow.end], flow.modulation, flow.start, 4))
    )
    reference.issue(bits)
    row = assemble(flow, words)
    assert row == drain(reference)
    assert row["header_crc"] is True
    assert row["status"] == ("invalid_length" if raw == 1 else "header_valid")


@pytest.mark.parametrize("cut", [0, 1, 4, 17, -1])
def test_eof_drains_inflight_word_but_never_validates_incomplete_head(cut):
    p = make_profile(2, 8)
    flow = HeaderFromWordsService(p)
    wire, _ = transmitted(p)
    words = [as_word(z) for z in wire[40 + flow.start : 40 + flow.end]]
    chosen = words[:cut] if cut >= 0 else words[:-1]
    row = assemble(flow, chosen)
    assert row["status"] == "truncated_header" and row["header_crc"] is None
    assert flow.committed == len(chosen) and flow.header.stats["accepted"] == 0


def test_head_ownership_snapshot_bad_indices_hold_and_abort():
    with pytest.raises(ValueError):
        HeaderFromWordsService(make_profile(2, 8), observer=False)
    p = make_profile(2, 8)
    events = []
    flow = HeaderFromWordsService(p, observer=events.append)
    with pytest.raises(RuntimeError):
        flow.end_input()
    assert flow.take() is None and flow.begin() and not flow.begin()
    with pytest.raises(ValueError):
        flow.offer(0, flow.start + 1)
    assert flow.offered == 0
    assert flow.offer(0, flow.start) and not flow.offer(0, flow.start)
    with pytest.raises(RuntimeError):
        flow.reset()
    snapshot = flow.snapshot()
    snapshot["statistics"]["begun"] = 100
    assert flow.stats["begun"] == 1
    flow.abort()
    with pytest.raises(RuntimeError):
        flow.begin()
    flow.reset()
    assert flow.begin()
    flow.end_input()
    with pytest.raises(RuntimeError):
        flow.offer(0, flow.start)
    flow.step()
    assert flow.ready
    for _ in range(3):
        flow.step()
    assert flow.stats["output_stalls"] == 3 and flow.take()["header_crc"] is None
    flow.step()
    assert flow.begin()
    flow.abort()
    flow.reset()
    flow.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        flow.step()
    flow.cycle = 1
    with pytest.raises(RuntimeError, match="same registered edges"):
        flow.step()
    assert any(e["event"] == "hard" for e in events)
