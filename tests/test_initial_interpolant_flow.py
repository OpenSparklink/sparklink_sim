"""Real source/cache words,registered credits and explicit continuous-window overflow."""

import math

import pytest

from nearlink_sdr.phy.interpolation_word import interpolate_filter_word
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.initial_interpolant_flow import InitialInterpolantPort, InitialStencilService
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort
from tests.test_interpolation_word import FORMATS


def input_words(n):
    return [
        pack_filter_word((k * 193) % 200003 - 100001, (k * 317) % 170003 - 85001, k % 5 - 17)
        for k in range(n)
    ]


def run_service(n, hz, *, retire=True, blocked=False):
    source = ReceiveHistoryPort()
    node = InitialStencilService(source, FORMATS[-1], n, 1, 32768, retire_after_staging=retire)
    words = input_words(4 * n + 256)
    offered = 0
    activation = math.ceil((node.activation_end - 1) * hz / 16_000_000)
    while node.cycle < 20000:
        c = node.cycle
        if offered < len(words) and c == math.ceil(offered * hz / 16_000_000):
            if not source.write(offered, words[offered], c):
                node.abort()
                return node, source, words, offered
            offered += 1
        node.step(output_ready=not (blocked and activation <= c < activation + 2000))
        assert len(node.staged) <= 4 and node.interpolator.live <= 7
        if node.ready:
            return node, source, words, None
    pytest.fail("service did not reach a defined result")


@pytest.mark.parametrize("n,hz", [(32, 100_000_000), (62, 160_000_000), (126, 200_000_000)])
def test_real_source_staging_early_retirement_and_all_cached_values(n, hz):
    node, source, words, overflow = run_service(n, hz)
    assert overflow is None and node.ready
    assert source.stats["read_returns"] == 4 * n
    assert source.head == 4 * n - 2
    assert source.live >= 2 and source.stats["peak_live"] <= 512
    p = node.output
    c = node.cycle
    for j in range(n):
        expected = interpolate_filter_word(tuple(words[4 * j : 4 * j + 4]), 32768, FORMATS[-1])[0]
        assert p.request(1 + 4 * j, 32768, c)
        assert p.take(c) is None
        assert p.take(c + 3) == (j, expected)
        c += 4
    assert p.cache.diagnostics()["valid_words"] == 0 and p.pending is None
    cost = node.estimate()
    assert cost["incremental_control_bits"] <= 1024
    assert cost["cache"]["payload_bits"] == 6656 and cost["cache"]["bram36"] == 1
    assert cost["source_read_ports"] == 1 and cost["source_write_ports"] == 1
    assert cost["initial_fit_cycles"] is None and cost["PI_carrier_head_decoder_cycles"] is None
    assert not cost["full_receiver16msps"]


@pytest.mark.parametrize("hz", [100_000_000, 160_000_000, 200_000_000])
def test_retaining_raw_for_repeat_remains_an_explicit_overflow_counterfactual(hz):
    node, source, _, overflow = run_service(126, hz, retire=False)
    assert overflow == 512 and source.stats["backpressure"] == 1
    assert node.aborted and source.live == 0
    assert node.output.cache.diagnostics()["valid_words"] == 0
    assert node.history.pending is None and node.interpolator.live == 0
    with pytest.raises(RuntimeError, match="Aborted"):
        node.step()


def test_long_cache_write_block_aborts_instead_of_hiding_lost_input():
    node, source, _, overflow = run_service(126, 100_000_000, blocked=True)
    assert overflow is not None and node.aborted
    assert source.stats["backpressure"] == 1 and node.output.cache.state == "released"


def test_registered_cache_reply_holds_and_owns_last_word_after_cache_release():
    p = InitialInterpolantPort(FORMATS[-1], 32)
    for k in range(32):
        assert p.write(5 + 4 * k, 7, 17 + k, k)
        assert not p.write(5 + 4 * (k + 1), 7, 18 + k, k)
    assert not p.request(5, 7, 31)  # Explicit fill/replay turnaround.
    for k in range(31):
        c = 32 + 4 * k
        assert p.request(5 + 4 * k, 7, c)
        assert not p.request(5 + 4 * k, 7, c + 1)
        assert p.take(c + 2) == (k, 17 + k)
    assert p.request(5 + 4 * 31, 7, 160)
    assert p.cache.state == "released" and p.pending is not None
    assert p.take(160) is None and p.take(190) == (31, 48)
    assert p.take(190) is None
    with pytest.raises(ValueError):
        p.take(180)
    with pytest.raises(RuntimeError):
        p.request(5, 7, 191)
    assert p.estimate()["registered_response_and_clock_bits"] == 317


def test_abort_invalidates_registered_reply_and_incomplete_EOF_stencils():
    p = InitialInterpolantPort(FORMATS[0], 32)
    for k in range(32):
        p.write(1 + 4 * k, 0, k, k)
    assert p.request(1, 0, 33)
    p.abort(34)
    assert p.take(35) is None and p.aborted
    h = ReceiveHistoryPort()
    node = InitialStencilService(h, FORMATS[0], 32, 1, 0)
    for k in range(127):
        assert h.write(k, k, 2 * k)
        node.interpolator.cycle = 2 * k
        node.step()
    assert not node.enabled and not node.ready
    node.abort()
    assert node.output.cache.written == 0 and h.live == 0


def test_source_geometry_future_expiry_clock_and_exclusive_credit_guards():
    h = ReceiveHistoryPort()
    for bad in (None, 0, True):
        with pytest.raises(ValueError):
            InitialStencilService(bad, FORMATS[0], 32, 1, 0)
    with pytest.raises(ValueError):
        InitialStencilService(h, FORMATS[0], 32, 0, 0)
    with pytest.raises(ValueError):
        InitialStencilService(h, FORMATS[0], 32, 1, 0, start_cycle=-1)
    node = InitialStencilService(h, FORMATS[0], 32, 1, 0)
    with pytest.raises(ValueError):
        node.step(output_ready=1)
    for k in range(128):
        assert h.write(k, k, k)
    assert h.request(0, 128)
    with pytest.raises(ValueError):
        InitialStencilService(h, FORMATS[0], 32, 1, 0)
    assert h.take(129) == (0, 0)
    h.retire_before(1, 129)
    node.interpolator.cycle = 130
    with pytest.raises(RuntimeError, match="expired"):
        node.step()


def test_future_counter_and_abort_disposition_do_not_hide_an_owned_reply():
    source = ReceiveHistoryPort()
    node = InitialStencilService(source, FORMATS[-1], 32, 1, 32768)
    node.future_wait_cycles = 2**64 - 1
    with pytest.raises(OverflowError):
        node.step()
    node.future_wait_cycles = 0
    for k in range(129):
        assert source.write(k, k, k)
    node.interpolator.cycle = 129
    node.step()
    assert source.pending is not None
    disposition = node.abort()
    assert disposition == dict(
        interpolation_contexts=0, source_reply=1, staged_words=0, cache_reply=0
    )
    assert node.abort() is None
    assert source.live == 0 and source.pending is None
    assert node.estimate()["incremental_control_bits"] == 841
