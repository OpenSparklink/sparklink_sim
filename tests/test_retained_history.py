"""Finite ring contents,retirement and counted host-only mirrored view."""

import numpy as np
import pytest

from nearlink_sdr.phy.retained_history import RetainedSampleHistory


def test_many_wraps_preserve_live_values_without_growing_or_copying():
    history = RetainedSampleHistory(8, 17)
    storage = history.storage
    for i in range(1003):
        assert history.append(i, complex(i, -3 * i))
        if i >= 17:
            expected = np.array([complex(k, -3 * k) for k in range(history.origin, i + 1)])
            np.testing.assert_array_equal(history.view, expected)
            assert np.shares_memory(history.view, storage)
            history.retire_before(max(17, i - 2))
        assert history.storage is storage and history.count <= 8
    stats = history.diagnostics()
    assert stats["leading_skipped"] == 17 and stats["host_mirror_writes"] == 2 * (1003 - 17)
    assert stats["host_array_bytes"] == 2 * 8 * 16 and stats["peak_samples"] <= 5
    assert not stats["hardware_port_or_cycle_verified"]


def test_full_history_refuses_without_destroying_unread_word_then_resumes():
    h = RetainedSampleHistory(4, 0)
    for i in range(4):
        assert h.append(i, complex(i, i + 1))
    before = h.view.copy()
    assert not h.append(4, 99) and h.next_input == 4
    np.testing.assert_array_equal(h.view, before)
    h.retire_before(1)
    assert h.append(4, 99)
    np.testing.assert_array_equal(h.view, [1 + 2j, 2 + 3j, 3 + 4j, 99])
    h.retire_before(100)
    assert h.origin == h.end == 5 and h.count == 0
    assert h.append(5, 101)
    np.testing.assert_array_equal(h.view, [101])


@pytest.mark.parametrize(
    "capacity,first", [(True, 0), (3, 0), (9, 0), (8192, 0), (8, -1), (8, 2**20)]
)
def test_invalid_history_is_explicit(capacity, first):
    with pytest.raises(ValueError):
        RetainedSampleHistory(capacity, first)


def test_invalid_append_and_retire_do_not_advance_state():
    h = RetainedSampleHistory(8, 0)
    for index, value in [(1, 0), (True, 0), (0, np.nan), (0, [1, 2])]:
        with pytest.raises(ValueError):
            h.append(index, value)
        assert h.next_input == h.count == 0
    for index in (-1, 2**20 + 1, False):
        with pytest.raises(ValueError):
            h.retire_before(index)
