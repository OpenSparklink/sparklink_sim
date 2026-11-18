"""Finite word-port proof with continuous attempts,registered replies and stale epochs."""

import pytest

from nearlink_sdr.sim.receive_history import ReceiveHistoryPort


def test_one_credit_real_values_stall_hold_and_many_wraps():
    p = ReceiveHistoryPort(8)
    for index in range(101):
        cycle = index * 4
        word = ((index * 107 + 3) << 19) | index
        assert p.write(index, word, cycle)
        assert not p.request(index, cycle)  # Never rely on same-address device behavior.
        assert p.request(index, cycle + 1)
        assert p.take(cycle + 1) is None
        assert not p.request(index, cycle + 2)
        assert p.take(cycle + 3) == (index, word)
        p.retire_before(index + 1, cycle + 3)
        assert p.live == 0
    assert p.stats["same_address_deferrals"] == 101
    assert p.stats["read_returns"] == p.stats["read_requests"] == 101
    with pytest.raises(ValueError, match="expired"):
        p.request(0, 405)


def test_full_replay_backpressure_never_overwrites_or_silently_drops():
    p = ReceiveHistoryPort(4)
    for i in range(4):
        assert p.write(i, i + 77, i)
    assert not p.write(4, 88, 4)
    assert p.request(0, 4)
    with pytest.raises(ValueError, match="reply"):
        p.retire_before(1, 5)
    assert p.take(7) == (0, 77)
    p.retire_before(1, 7)
    assert p.write(4, 88, 7)
    assert p.stats["backpressure"] == 1 and p.stats["peak_live"] == 4
    assert p.request(4, 8)
    assert p.take(100) == (4, 88)  # Downstream stall holds real word,not a refreshed RAM slot.


def test_one_read_and_write_each_edge_and_wrong_order_are_explicit():
    p = ReceiveHistoryPort()
    assert p.write(0, 3, 0)
    assert not p.write(1, 4, 0)
    assert p.write(1, 4, 1)
    assert p.request(0, 1)
    assert p.take(2) == (0, 3)
    assert p.request(1, 2)
    assert p.take(2) is None
    with pytest.raises(ValueError):
        p.take(1)
    assert p.take(3) == (1, 4)


@pytest.mark.parametrize("capacity", [True, 3, 9, 8192])
def test_invalid_capacity(capacity):
    with pytest.raises(ValueError):
        ReceiveHistoryPort(capacity)


@pytest.mark.parametrize("word", [True, -1, 2**46, 1.5])
def test_invalid_words_never_become_valid_memory(word):
    p = ReceiveHistoryPort()
    with pytest.raises(ValueError):
        p.write(0, word, 0)
    assert p.end == p.live == 0


def test_geometry_and_cost_count_reply_and_diagnostic_state():
    p = ReceiveHistoryPort()
    cost = p.estimate()
    assert cost["logical_bits"] == 512 * 46 and cost["bram36"] == 1
    assert cost["address_timestamp_and_reply_bits"] == 623
    assert cost["statistic_counter_bits"] == 512
    assert cost["interpolation_loop_head_decoder_cycles"] is None
    assert not cost["synthesized"] and not cost["full_receiver16msps_verified"]
    with pytest.raises(ValueError):
        p.request(0, 0)
    with pytest.raises(ValueError):
        p.write(1, 3, 0)
    with pytest.raises(ValueError):
        p.retire_before(1, 0)


def test_repeated_stencil_reads_keep_word_until_the_last_dependency():
    p = ReceiveHistoryPort(8)
    for i in range(4):
        assert p.write(i, 2**45 + i, i)
    for cycle, index in enumerate((0, 1, 2, 3, 0, 1, 2, 3), start=5):
        assert p.request(index, 2 * cycle)
        assert p.take(2 * cycle + 1) == (index, 2**45 + index)
    assert p.live == 4
    p.retire_before(4, 31)
    assert p.live == 0 and p.stats["read_returns"] == 8
