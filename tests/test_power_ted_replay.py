"""Same ordered frame/trajectory and real CRC gates with both natural ready orders."""

from functools import partial

import numpy as np
import pytest

from nearlink_sdr.sim.data_joint_replay import DataJointFrameReplay
from nearlink_sdr.sim.power_ted_replay import PowerTEDFrameReplay
from nearlink_sdr.sim.timing_overlap_flow import ParallelTEDTimingWordService
from tests.test_data_joint_replay import TRACE, at_data_issue
from tests.test_head_word_replay import run
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 8, 8, 19), (3, 12, 4, 47), (4, 10, 16, 23)])
@pytest.mark.parametrize("layout", ["dense", "depth-parity"])
@pytest.mark.parametrize("divider", ["bounded-prefix", "radix4-prefix"])
def test_fixed_frame_source_and_loop_states_exact_under_random_size_blocks(
    ft, mcs, pilot, length, layout, divider
):
    p = ordinary(ft, mcs, pilot)
    values, expected, selection = selected_input(p, length=length)
    words = quantize_input(np.r_[values, np.zeros(1000)])
    selection["sample_limit"] = len(words)
    old_cls = partial(DataJointFrameReplay, head_layout=layout, timing_divider=divider)
    cls = partial(PowerTEDFrameReplay, head_layout=layout, timing_divider=divider)
    a, old, _ = run(p, words, selection, old_cls)
    pattern = tuple(np.random.default_rng(2026112017).integers(1, 33, 23))
    b, new, barriers = run(p, words, selection, cls, pattern)
    assert a["frame"] == b["frame"] and b["frame"]["payload"] == expected
    for key in TRACE:
        np.testing.assert_array_equal(a[key], b[key])
    assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
    assert old.consumer.normalization.snapshot() == new.consumer.normalization.snapshot()
    assert new.history.head == old.history.head and new.history.end == old.history.end
    for k in range(new.history.head, new.history.end):
        assert new.history.words[k % 512] == words[k]
    assert new.estimate()["timing_service"]["state_register_bits_upper"] == (
        old.estimate()["timing_service"]["state_register_bits_upper"] + 104
    )
    assert 0 < old.cycle - new.cycle <= 8 * new.stats["data_joint_completions"]
    assert barriers == ["payload"] and not new.cycle_evidence_valid
    assert [p.mac_payload for p in new.take_mac_result().packets] == [expected]


@pytest.mark.parametrize("zero_TED", [False, True])
def test_natural_ready_orders_hold_until_both_then_commit_and_query_once(zero_TED):
    flow = at_data_issue(PowerTEDFrameReplay)
    if zero_TED:
        flow.words = [0, 0]
    n, q = flow.consumer._active["count"], flow.query_service.stats["accepted"]
    flow.step()
    timing, normal = flow.timing_service.normal, flow.normalization_service
    assert isinstance(timing, ParallelTEDTimingWordService)
    first = None
    while flow.stats["data_joint_completions"] == 0:
        t, r = timing.ready, normal.peek() is not None
        if t != r and first is None:
            first = "timing" if t else "normalization"
        assert flow.consumer._active["count"] == flow.consumer.normalization.count == n
        assert flow.query_service.stats["accepted"] == q
        flow.step()
    assert first == ("timing" if zero_TED else "normalization")
    assert timing.ready_cycle - timing.first_cycle == (21 if zero_TED else 67)
    assert normal.ready_cycle - normal.first_cycle == 24
    assert flow.consumer._active["count"] == flow.consumer.word_timing.count == n + 1
    assert flow.consumer.normalization.count == n + 1
    assert flow.query_service.stats["accepted"] == q + 1
    assert flow.head_owner.hard.stats["accepted"] == 1
    assert flow.abort()["frame"]["header_crc"] is None


@pytest.mark.parametrize(
    "damage,status",
    [
        ({"bad_head": True}, "bad_header_crc"),
        ({"bad_data": True}, "bad_data_crc"),
        ({"empty": True}, "invalid_length"),
    ],
)
def test_bad_real_CRC_and_length_remain_failures(damage, status):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, **damage)
    event, flow, _ = run(p, quantize_input(values), selection, PowerTEDFrameReplay)
    assert event["frame"]["status"] == status and not flow.take_mac_result().packets


@pytest.mark.parametrize("after_sync", [8, 80])
def test_EOF_never_reruns_header_or_invents_data_crc(monkeypatch, after_sync):
    p = ordinary(2, 8, 8)
    values, _, selection = selected_input(p, length=99)
    cut = selection["start_tick"] // 8 + 4 * (len(p.sync_symbols()) + after_sync)
    selection = selection | dict(sample_limit=cut)

    def forbidden(*args, **kwargs):
        pytest.fail("EOF must not run a floating header fallback")

    monkeypatch.setattr("nearlink_sdr.phy.selected_rx.decode_psk_frame", forbidden)
    event, flow, _ = run(p, quantize_input(values)[:cut], selection, PowerTEDFrameReplay)
    assert event["frame"]["status"] == ("truncated_header" if after_sync == 8 else "truncated_data")
    assert event["frame"]["header_crc"] is (None if after_sync == 8 else True)
    assert event["frame"]["data_crc"] is None and not flow.take_mac_result().packets
