"""Actual captured source and strict frame boundary regressions."""

import numpy as np

from tests.captured_frame_reference import delivered, fixture, receive


def test_blocked_consumer_keeps_ADC_running_and_preserves_resource_failure():
    p, iq, quiet, _, _, _ = fixture()
    iq = np.r_[iq, np.zeros(8192, complex)]
    result, frames, reads, _, source, _ = receive(p, iq, quiet, stalls=((0, 200000),))
    assert source.done and result["input_samples"] == len(iq)
    assert delivered(frames) == []
    assert any(e["frame"]["status"] == "tail_history_expired" for e, _ in frames)
    assert result["frame_resource_drops"] and result["peak_owned"] <= 3
    assert result["peak_source_operations_per_primitive"] <= 2
    assert not result["adc_backpressure"] and all(r["index"] < len(iq) for r in reads)


def test_source_epoch_fault_cancels_active_frame_and_accounts_all_later_input():
    p, iq, quiet, _, _, _ = fixture(size=713)
    result, frames, reads, owner, source, _ = receive(p, iq, quiet, defect=3200)
    assert source.done and result["input_samples"] == len(iq)
    assert delivered(frames) == [] and result["source_epoch_drops"] > 0
    assert owner.stats["accepted"] == 1 and owner.stats["resource_aborts"] == 1
    assert result["frame_resource_drops"] == result["frame_terminal"] == 1
    assert any(e["frame"]["status"] == "source_epoch_aborted" for e, _ in frames)
    assert all(r["returned_cycle"] < source.fault_cycle for r in reads)


def test_dense_three_packets_keep_queue_losses_separate_from_delivered_content():
    p, iq, quiet, payload, _, _ = fixture()
    values = np.r_[iq, iq, iq, np.zeros(4096, complex)]
    result, frames, _, owner, source, outcomes = receive(p, values, quiet)
    assert source.done and result["input_samples"] == len(values)
    actual = delivered(frames)
    # Frozen initial three-packet schedule delivers two,not all three.
    assert actual == [payload, payload] and result["queue_drops"] == 6
    assert result["requests"] == result["completed"] + result["queue_drops"]
    assert len({r["id"] for r in outcomes}) == result["requests"]
    assert result["peak_owned"] <= 3 and not owner.active
    assert result["frame_replay_cost"]["private_full_tail_banks"] == 0
