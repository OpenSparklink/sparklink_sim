from collections import deque
from copy import deepcopy
from itertools import pairwise

import pytest

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat
from nearlink_sdr.phy.rrc_bridge import rrc_to_scaled_word
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.raw_gate_join import RawGateJoin
from nearlink_sdr.sim.rrc_bridge_flow import RRCScaledBridgeService


def join(start=0):
    return RawGateJoin(RRCScaledBridgeService(), start_tag=start)


def advance(node, count):
    for _ in range(count):
        node.step()


def issue(node, tag, gate):
    while not node.bridge.input_ready:
        node.step()
    assert node.push_flag(tag, gate)
    word = pack_filter_word(701 + tag % 19, -171 - tag % 11, -15 + tag % 5)
    assert node.bridge.issue(word, tag)
    return word


def state(node):
    return deepcopy({k: v for k, v in vars(node).items() if k != "bridge"}), deepcopy(
        vars(node.bridge)
    )


def test_existing_IQ_credit_is_held_through_one_real_flag_read():
    node = join()
    word = issue(node, 0, True)
    advance(node, 3)
    assert node.bridge.ready and node.mode == "reading" and not node.ready
    assert node.take() is None and node.bridge.stats["retired"] == 0
    node.step()
    result = node.take()
    assert result["word"] == rrc_to_scaled_word(word, ScaledBankFormat())[0]
    assert result["energy_trigger"] and result["window_energy_present"]
    assert node.stats["words_joined"] == node.bridge.stats["retired"] == 1
    assert node.take() is None and not node.bridge.input_ready


@pytest.mark.parametrize("window", [256, 376, 632])
def test_exact_tags_flags_and_profile_gate_history_without_group_delay_shift(window):
    node = RawGateJoin(RRCScaledBridgeService(), start_tag=317, window_samples=window)
    observed, clocks = [], []
    for offset in range(window + 44):
        flag = offset == 0
        word = issue(node, 317 + offset, flag)
        clocks.append(node.cycle)
        advance(node, 4)
        result = node.take()
        assert result["tag"] == 317 + offset
        assert result["word"] == rrc_to_scaled_word(word, ScaledBankFormat())[0]
        assert result["energy_trigger"] == flag
        assert result["window_energy_present"] == (offset < window)
        observed.append(result["tag"])
    assert all(b - a == 5 for a, b in pairwise(clocks))
    assert observed == list(range(317, 317 + window + 44)) and node.count == node.flags == 0
    node.finish()
    assert node.closed and node.stats["closes"] == 1


def test_future_flag_and_same_edge_write_cannot_confirm_an_existing_word():
    node = join()
    assert node.bridge.issue(pack_filter_word(1, 2, -10), 0)
    advance(node, 3)
    assert node.bridge.ready and not node.ready
    assert node.push_flag(0, True)
    assert node.take() is None
    node.step()
    assert node.mode == "reading" and node.take() is None
    node.step()
    assert node.take()["energy_trigger"]


def test_simultaneous_flag_write_and_word_retirement_do_not_erase_the_next_flag():
    node = join()
    issue(node, 0, False)
    advance(node, 4)
    assert node.push_flag(1, True)
    assert not node.take()["energy_trigger"]
    assert node.count == node.flags == 1 and node.head_tag == 1
    assert node.head_visible_cycle == node.cycle + 1
    node.step()
    assert node.bridge.issue(pack_filter_word(7, 9, -7), 1)
    advance(node, 4)
    assert node.take()["energy_trigger"]


@pytest.mark.parametrize("window", [256, 376, 632])
def test_saturating_age_matches_independent_bounded_history_with_repeated_triggers(window):
    node = RawGateJoin(RRCScaledBridgeService(), window_samples=window)
    reference = deque(maxlen=window)
    for tag in range(2 * window + 71):
        flag = tag in (3, 7, window + 9, window + 73) or tag % (window + 3) == 17
        reference.append(flag)
        issue(node, tag, flag)
        advance(node, 4)
        result = node.take()
        assert result["window_energy_present"] == any(reference)
        assert 0 <= node.age <= window


def test_output_blocking_retains_exactly_one_IQ_credit_and_bounded_flags():
    node = join()
    issue(node, 0, True)
    advance(node, 4)
    original = node.bridge.result
    for tag in range(1, 10):
        assert node.push_flag(tag, bool(tag % 2))
        node.step()
        assert node.bridge.result == original and node.ready
        assert not node.bridge.issue(pack_filter_word(0, 0, 0), tag)
    assert node.count == 10 and node.stats["held_edges"] == 9
    assert node.take()["tag"] == 0 and node.count == 9


def test_flag_overflow_invalidates_epoch_but_continuous_input_stays_bounded():
    node = join()
    for tag in range(100):
        accepted = node.push_flag(tag, bool(tag % 3))
        assert accepted == (tag < 16)
        node.step()
        assert 0 <= node.count <= 16
    assert node.mode == "failed" and node.bridge.mode == "aborted"
    assert node.fault_reason == node.reasons["flag_overflow"] and node.fault_tag == 16
    assert node.stats["flags_accepted"] == node.stats["orphan_flags"] == 16
    assert node.stats["flags_discarded"] == 84 and node.stats["words_joined"] == 0
    assert node.count == node.flags == 0 and node.age == node.window and node.take() is None
    node.finish()
    with pytest.raises(RuntimeError, match="EOF"):
        node.push_flag(100, False)


@pytest.mark.parametrize("tag", [9, 11])
def test_missing_or_stale_word_never_compresses_the_stream_index(tag):
    node = join(10)
    assert node.push_flag(10, True)
    assert node.bridge.issue(pack_filter_word(2, 3, -11), tag)
    advance(node, 3)
    assert node.mode == "failed" and node.fault_reason == node.reasons["word_gap"]
    assert node.stats["words_joined"] == 0 and node.stats["orphan_flags"] == 1


@pytest.mark.parametrize("at_take", [False, True])
def test_foreign_word_retirement_is_detected_before_returning_a_stale_flag(at_take):
    node = join()
    issue(node, 0, True)
    advance(node, 4 if at_take else 3)
    assert node.bridge.take() is not None
    if at_take:
        assert node.take() is None
    else:
        node.step()
    assert node.mode == "failed" and node.fault_reason == node.reasons["ownership"]


def test_foreign_clock_ownership_is_rejected():
    node = join()
    node.bridge.step()
    with pytest.raises(RuntimeError, match="clock"):
        node.step()
    assert node.cycle == 0


@pytest.mark.parametrize("tag,flag", [(1, True), (0.0, True), (0, 1), (0, None)])
def test_invalid_flag_port_is_atomic(tag, flag):
    node = join()
    before = state(node)
    with pytest.raises(ValueError):
        node.push_flag(tag, flag)
    assert state(node) == before


def test_one_write_port_refuses_a_second_flag_at_the_same_edge():
    node = join()
    node.push_flag(0, False)
    before = state(node)
    with pytest.raises(ValueError, match="same edge"):
        node.push_flag(1, True)
    assert state(node) == before


@pytest.mark.parametrize("counter", ["words_joined", "bridge_retired"])
def test_joint_retirement_prechecks_both_counter_owners(counter):
    node = join()
    issue(node, 0, False)
    advance(node, 4)
    if counter == "bridge_retired":
        node.bridge.stats["retired"] = 2**64 - 1
    else:
        node.stats[counter] = 2**64 - 1
    before = state(node)
    with pytest.raises(OverflowError):
        node.take()
    assert state(node) == before


def test_flag_counter_and_clock_overflow_do_not_write_or_wrap():
    node = join()
    node.stats["flags_accepted"] = 2**64 - 1
    before = state(node)
    with pytest.raises(OverflowError):
        node.push_flag(0, True)
    assert state(node) == before
    node = join()
    node.cycle = node.bridge.cycle = 2**63 - 1
    before = state(node)
    with pytest.raises(OverflowError):
        node.push_flag(0, True)
    assert state(node) == before
    with pytest.raises(OverflowError):
        node.step()
    assert state(node) == before


@pytest.mark.parametrize("missing", ["flag", "word"])
def test_EOF_preserves_missing_input_failure_and_releases_the_bounded_context(missing):
    node = join()
    if missing == "flag":
        node.bridge.issue(pack_filter_word(3, 7, 1), 0)
        advance(node, 3)
    else:
        node.push_flag(0, False)
    node.finish()
    assert node.closed and node.mode == "failed"
    assert node.fault_reason == node.reasons["EOF_" + missing + "_missing"]
    assert node.stats["closes"] == 1 and node.count == 0
    before = state(node)
    node.finish()
    node.abort()
    assert state(node) == before


@pytest.mark.parametrize("edges", [0, 1, 2, 3, 4])
def test_EOF_requires_existing_real_work_and_output_to_be_drained(edges):
    node = join()
    issue(node, 0, False)
    advance(node, edges)
    with pytest.raises(RuntimeError, match="Drain"):
        node.finish()
    assert not node.closed
    node.abort()
    node.finish()
    assert node.closed and node.fault_reason == node.reasons["abort"]
    with pytest.raises(RuntimeError, match="EOF"):
        node.step()


def test_initial_tag_and_compiled_bridge_contract():
    for tag in (-1, 2**63 - 1, True):
        with pytest.raises(ValueError):
            join(tag)
    with pytest.raises(ValueError):
        RawGateJoin(RRCScaledBridgeService(ScaledBankFormat(8)))
    bridge = RRCScaledBridgeService()
    bridge.issue(pack_filter_word(1, 0, 0), 0)
    with pytest.raises(ValueError):
        RawGateJoin(bridge)
    for window in (0, 255, 257, True):
        with pytest.raises(ValueError):
            RawGateJoin(RRCScaledBridgeService(), window_samples=window)


@pytest.mark.parametrize("window,bits", [(256, 997), (376, 997), (632, 998)])
def test_all_added_state_and_one_registered_read_are_charged(window, bits):
    cost = RawGateJoin(RRCScaledBridgeService(), window_samples=window).estimate()
    assert cost["register_bits_upper"] == bits
    assert bits < 1536
    assert cost["counter_bits"] == 512 and cost["flag_read_edges"] == 1
    assert cost["IQ_contexts_added"] == 0 and cost["IQ_word_held_by_existing_bridge"]
    assert cost["composed_unstalled_bridge_II"] == 5
    assert cost["raw_detector_arithmetic_cycles"] is cost["scaled_bank_cycles"] is None
    assert cost["LUT_FF_Fmax"] is None and not cost["synthesis"]
