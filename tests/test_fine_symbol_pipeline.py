"""Exact independent integer/Fraction oracles and bounded clock ownership."""

from itertools import product

import numpy as np
import pytest

from nearlink_sdr.sim.fine_symbol_pipeline import FineSymbolPipeline, FineSymbolRequest
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions
from tests.test_spline_score import inputs, oracle


def request(token, values, *, phase=0, first=False):
    a, b, c, d, r, i = values
    return FineSymbolRequest(token, phase, (a, b), None if first else (c, d),
                             0 if first else c * c + d * d,
                             0 if first else -23171, 0 if first else 913,
                             0 if first else 47231, (0, 0) if first else (r, i))


def independent(req):
    a, b = req.word
    c, d = req.previous or (0, 0)
    r, i = req.sign
    dr, di = a * c + b * d, b * c - a * d
    power = a * a + b * b
    return (req.word, req.real + dr * r + di * i, req.imag + di * r - dr * i,
            req.energy + power * req.previous_power, power)


def test_extremes_random_signs_and_first_symbols_II1_exact_and_registered():
    cases = [(*row, r, i) for row in product((-2**21, 2**21 - 1), repeat=4)
             for r, i in product((-1, 0, 1), repeat=2)]
    rng = np.random.default_rng(948123)
    cases += [(*map(int, row), (-1, 0, 1)[j % 3], (-1, 0, 1)[(j // 3) % 3])
              for j, row in enumerate(rng.integers(-2**21, 2**21, (72, 4)))]
    node, expected, arrivals = FineSymbolPipeline(), {}, []
    for k in range(len(cases) + 16):
        req = request(k, cases[k], phase=k % 9, first=k % 13 == 0) if k < len(cases) else None
        accepted, reply = node.step(req)
        assert accepted == (req is not None)
        if req is not None:
            expected[k] = independent(req)
        if reply is not None:
            assert reply.state == expected[reply.token] and reply.phase == reply.token % 9
            arrivals.append(k)
            assert k - reply.token == 16
    assert node.idle and arrivals == list(range(16, len(cases) + 16))
    assert node.stats["accepted"] == node.stats["completed"] == len(cases)
    assert node.stats["native_products"] == len(cases) * 16
    assert node.stats["preadds"] == len(cases) * 3
    assert node.stats["adds"] == len(cases) * 18


def test_blocked_sink_freezes_all_slots_without_overwrite_and_cancel_flushes():
    node, rows = FineSymbolPipeline(), []
    for k in range(16):
        assert node.step(request(k, (k, -k, 17, -9, 1, -1)))[0]
    snapshot = repr(node.cells)
    for _ in range(7):
        assert node.step(request(16, (31, -19, 7, 1, -1, 1)), output_ready=False) == (False, None)
        assert repr(node.cells) == snapshot
    for _ in range(16):
        accepted, out = node.step()
        assert not accepted
        rows.append(out.token)
    assert rows == list(range(16)) and node.idle and node.stats["blocked"] == 7
    for k in range(16, 21):
        node.step(request(k, (9, 3, 7, 2, 0, 1)))
    assert node.clear() == 5 and node.idle and node.next_token == 0
    assert node.step(request(0, (11, -2, 0, 0, 0, 0), first=True)) == (True, None)
    for _ in range(15):
        assert node.step() == (False, None)
    assert node.step()[1].state == ((11, -2), 0, 0, 0, 125)
    assert node.idle


@pytest.mark.parametrize("change", [
    dict(word=(2**21, 0)), dict(previous=(-2**21 - 1, 0)), dict(energy=-1),
    dict(previous_power=2**44), dict(real=2**52), dict(imag=-2**52 - 1),
    dict(sign=(2, 0)), dict(sign=(True, 0)), dict(phase=9), dict(token=2**16),
])
def test_reject_invalid_word_formats_before_accepting_a_ticket(change):
    values = vars(request(0, (1, 2, 3, 4, 1, -1))) | change
    with pytest.raises((ValueError, OverflowError)):
        FineSymbolRequest(**values)


def test_first_state_tokens_sink_types_and_accumulator_overflow_are_explicit():
    with pytest.raises(ValueError, match="First"):
        FineSymbolRequest(**(vars(request(0, (1, 2, 3, 4, 1, -1))) | dict(previous=None)))
    node = FineSymbolPipeline()
    with pytest.raises(ValueError, match="token"):
        node.step(request(1, (1, 2, 3, 4, 1, -1)))
    with pytest.raises(ValueError, match="typed"):
        node.step(1)
    with pytest.raises(ValueError, match="readiness"):
        node.step(output_ready=1)
    req = FineSymbolRequest(**(vars(request(0, (8, 8, 8, 8, 1, 1)))
                              | dict(energy=2**98 - 1)))
    node.step(req)
    with pytest.raises(OverflowError, match="format"):
        for _ in range(16):
            node.step()
    node.clear()
    node.next_token = 65535
    node.step(request(65535, (0, 0, 0, 0, 0, 0), first=True))
    with pytest.raises(ValueError, match="token"):
        node.step(request(0, (0, 0, 0, 0, 0, 0), first=True))


@pytest.mark.parametrize("peaks,minimum", [((2, 8), 0), ((0,), 0), ((2,), 19), ((), 0)])
def test_integrated_all_phase_samples_scores_and_selected_replay_fraction_exact(peaks, minimum):
    words, template = inputs(48), SyncSignTemplate(((1, 1), (-1, 1), (1, -1), (-1, -1)))
    assumptions = SplineScoreAssumptions(48, reuse_symbol_power=True, defer_all_future=True)
    arms = []
    for pipelined in (False, True):
        trace = []

        def sink(row):
            if row["event"] in ("sample", "score", "selected_sample"):
                trace.append({k: v for k, v in row.items() if k != "cycle"})

        result = SplineScoreFlow(assumptions, word_fabric=WordFabricAssumptions(2),
                                 gauss_difference=True, symbol_pipeline=pipelined).run(
                                     words, template, peaks, minimum_tick=minimum, sink=sink)
        arms.append((result, trace))
    (old, old_trace), (new, new_trace) = arms
    assert old_trace == new_trace and old["best"] == new["best"]
    assert new["best"] == oracle(words, template, peaks, minimum=minimum)[1]
    assert new["symbol_pipeline_idle"]
    for key in ("prepared_source_words", "prepared_second_words", "common_exponent",
                "scoring_reads", "preparation_cycles", "numerical_pass"):
        assert old[key] == new[key]
    if peaks:
        assert new["cycles"] < old["cycles"]


def test_experimental_requirements_cost_and_future_wait_abort_are_not_numeric_pass():
    with pytest.raises(ValueError, match="Symbol"):
        SplineScoreFlow(symbol_pipeline=True)
    with pytest.raises(ValueError, match="Symbol"):
        SplineScoreFlow(symbol_pipeline=1)
    a = SplineScoreAssumptions(32, reuse_symbol_power=True, defer_all_future=True)
    flow = SplineScoreFlow(a, word_fabric=WordFabricAssumptions(2), gauss_difference=True,
                           symbol_pipeline=True)
    cost = flow.estimate()
    assert cost["additional_symbol_pipeline_register_bits"] == 11651 <= 12288
    assert cost["total_scoring_DSP25x18_count"] == 17
    assert cost["fine_symbol_pipeline"]["dedicated_DSP25x18"] == 16
    assert cost["fine_symbol_pipeline"]["new_sample_read_ports"] == 0
    words, template = [(0, 0, 0)] * 32, SyncSignTemplate(((0, 1),) * 3)
    zero = flow.run(words, template, (2,), threshold=(0, 1))
    assert zero["best"]["zero_energy"] and zero["best"]["threshold_pass"]
    wait = flow.run(words, template, (2,), lookahead_ticks=32768)
    assert wait["geometry_wait"] and wait["numerical_pass"] is None
    assert wait["symbol_pipeline_statistics"]["accepted"] == 0
    limited = flow.run(words, template, (2,), max_cycles=300)
    assert limited["resource_drop"] and limited["numerical_pass"] is None


def test_capture_worker_reports_all_seventeen_score_DSP_and_same_owned_source():
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
    from nearlink_sdr.phy.tx_pipeline import TxConfig
    from nearlink_sdr.sim.capture_fine import CaptureFineWorker, pack_filter_word
    from nearlink_sdr.sim.capture_transport import CaptureRequest

    template = SyncSignTemplate.from_symbols(
        PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols())
    source = [pack_filter_word(*word) for word in inputs(256)]
    event = dict(geometry=CaptureRequest(256, 0, (2,), 8, 128),
                 window_samples=256, after_samples=256, path=0)
    arms = []
    for pipelined in (False, True):
        worker = CaptureFineWorker(
            template, 256, confirmation_required=True, reuse_symbol_power=True,
            defer_all_future=True, word_fabric=WordFabricAssumptions(2),
            gauss_difference=True, symbol_pipeline=pipelined,
        )
        arms.append(worker(event, source))
        assert worker.estimate()["scoring_DSP25x18_estimate"] == (17 if pipelined else 1)
    assert arms[0].disposition == arms[1].disposition
    assert arms[0].start_tick == arms[1].start_tick
    assert arms[1].cycles < arms[0].cycles
