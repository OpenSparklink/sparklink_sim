"""Continuous selected-window input,actual source values,cache/fit shared credits."""

import math

import pytest

from nearlink_sdr.phy.sync_fit_word import WordSyncFit, compile_sync_phases
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from tests.test_initial_interpolant_flow import input_words
from tests.test_selected_rx import ordinary


def service_case(n, hz, *, stall=0):
    history = ReceiveHistoryPort()
    ft = {32: 2, 62: 3, 126: 4}[n]
    codes = compile_sync_phases(ordinary(ft, 8, 0).sync_symbols())
    node = InitialSyncFitService(history, codes, 12345, 1, 32768)
    words = input_words(4 * n + 2048)
    k = 0
    actual = []
    while node.cycle < 20000:
        c = node.cycle
        if k < len(words) and c == math.ceil(k * hz / 16_000_000):
            if not history.write(k, words[k], c):
                node.abort()
                return node, history, actual, k
            k += 1
        reply = node.step(output_ready=c >= stall)
        if reply is not None:
            actual.append(reply["word"])
        assert history.live <= 512 and len(node.stencil.staged) <= 4
        assert node.stencil.interpolator.live <= 7
        if node.ready:
            reference = WordSyncFit(codes, 12345)
            for w in actual:
                reference.feed_word(w)
            assert node.fit.result == reference.finish()
            return node, history, actual, None
    pytest.fail("Finite service did not terminate")


@pytest.mark.parametrize("n", [32, 62, 126])
@pytest.mark.parametrize("hz", [100_000_000, 160_000_000, 200_000_000])
def test_continuous_input_rates_have_explicit_success_or_resource_loss(n, hz):
    node, history, words, loss = service_case(n, hz)
    if loss is None:
        assert node.ready and len(words) == n
        assert history.stats["read_returns"] == 4 * n
        assert node.fit.stats["accepted_words"] == n
        assert node.stencil.output.cache.written == n
        assert node.fit.ready_cycle > node.stencil.ready_cycle
    else:
        assert hz in (100_000_000, 160_000_000) and n == 126
        assert history.stats["backpressure"] == 1 and history.live == 0
        assert node.fit.mode == "aborted" and not node.ready
        assert node.stencil.output.cache.state == "released"
    assert node.estimate()["second_cache_or_source_pass"] is False
    assert node.estimate()["known_sync_replay_PI_carrier_head_decoder_cycles"] is None


def test_stall_overflow_clears_every_owned_credit_and_does_not_emit_stale_fit():
    node, history, _, loss = service_case(126, 200_000_000, stall=14000)
    assert loss is not None and history.live == 0
    assert node.fit.take() is None and node.stencil.history.pending is None
    assert node.stencil.interpolator.live == 0 and node.abort() is None
    with pytest.raises(RuntimeError):
        node.step()


def test_EOF_truncation_drain_and_clock_guards_are_explicit():
    codes = [0] * 32
    history = ReceiveHistoryPort()
    node = InitialSyncFitService(history, codes, 0, 1, 0)
    for k in range(127):
        assert history.write(k, k, k)
        node.step()
    detail = node.end_input()
    assert detail["reason"] == "truncated_sync_future" and not node.ready
    assert history.live == 0 and node.eof
    history = ReceiveHistoryPort()
    node = InitialSyncFitService(history, codes, 0, 1, 0)
    for k in range(129):
        assert history.write(k, 1, k)
        node.step()
    assert node.end_input() is None
    while not node.ready:
        node.step()
    assert node.fit.take()["valid"]
    assert node.estimate()["register_state_bits"] < 8192
    node.fit.cycle += 1
    with pytest.raises(RuntimeError, match="clocks"):
        node.step()
    with pytest.raises(ValueError):
        InitialSyncFitService(ReceiveHistoryPort(), codes, 0, 1, 0).step(output_ready=1)
