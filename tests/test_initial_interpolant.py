"""Actual word ownership,identity and lifetime boundaries,without CRC selection."""

import pytest

from nearlink_sdr.phy.initial_interpolant import InitialInterpolantCache
from tests.test_interpolation_word import FORMATS


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("n", [32, 62, 126])
def test_finite_single_fill_and_replay_words_survive_source_retirement(fmt, n):
    c = InitialInterpolantCache(fmt, n)
    words = [(17 * k + 3) | (129 << (2 * fmt.output_bits)) for k in range(n)]
    for k, word in enumerate(words):
        c.store(13 + 4 * k, 2**fmt.phase_bits, word)
    assert c.state == "sealed" and len(c.words) == 128
    with pytest.raises(RuntimeError):
        c.store(13, 0, 0)
    # No raw source is retained in this node;values are the transferred words.
    for k, word in enumerate(words):
        assert c.reuse(13 + 4 * k, 2**fmt.phase_bits) == word
    assert c.diagnostics() == dict(state="released", written=n, reused=n, valid_words=0)
    assert c.words[0] == words[0]  # RAM clear is not required to invalidate ownership.
    with pytest.raises(RuntimeError):
        c.reuse(13, 2**fmt.phase_bits)
    cost = c.estimate()
    assert cost["payload_bits"] == 128 * (2 * fmt.output_bits + 8)
    assert cost["controller_bits"] == 89 + fmt.phase_bits + 1
    assert cost["bram36"] == 1 and cost["extra_DSP"] == 0
    assert not cost["physical_registered_response_and_clock_state_included"]


def test_incomplete_abort_identity_mismatch_never_retries_or_changes_valid_state():
    c = InitialInterpolantCache(FORMATS[-1], 32)
    with pytest.raises(RuntimeError):
        c.reuse(7, 0)
    c.store(7, 5, 17)
    for cell, phase in ((7, 5), (11, 6), (15, 5)):
        with pytest.raises(RuntimeError, match="identity"):
            c.store(cell, phase, 37)
    assert c.written == 1 and c.words[0] == 17
    for k in range(1, 32):
        c.store(7 + 4 * k, 5, k)
    for cell, phase in ((11, 5), (7, 6)):
        with pytest.raises(RuntimeError):
            c.reuse(cell, phase)
    assert c.read == 0
    assert c.reuse(7, 5) == 17
    c.release()
    assert c.diagnostics()["valid_words"] == 0
    with pytest.raises(RuntimeError):
        c.reuse(11, 5)


@pytest.mark.parametrize("n", [True, 0, 31, 128, 129])
def test_only_compiled_sync_lengths(n):
    with pytest.raises(ValueError):
        InitialInterpolantCache(FORMATS[-1], n)
    with pytest.raises(ValueError):
        InitialInterpolantCache(None, 32)


@pytest.mark.parametrize(
    "cell,phase,word",
    [
        (-1, 0, 0),
        (True, 0, 0),
        (0, True, 0),
        (0, -1, 0),
        (0, 65537, 0),
        (0, 0, True),
        (0, 0, -1),
        (0, 0, 2**52),
    ],
)
def test_invalid_keys_words_do_not_create_valid_cache(cell, phase, word):
    c = InitialInterpolantCache(FORMATS[-1], 32)
    with pytest.raises(ValueError):
        c.store(cell, phase, word)
    assert c.state == "empty" and c.written == 0 and c.first_cell is None


def test_index_growth_is_checked_before_first_store():
    c = InitialInterpolantCache(FORMATS[0], 126)
    with pytest.raises(OverflowError):
        c.store(2**63 - 100, 0, 7)
    assert c.state == "empty" and c.written == 0
    c.release()
    assert c.diagnostics()["valid_words"] == 0
