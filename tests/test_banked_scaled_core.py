"""Continuous word arrivals and clocked scores/queries vs the existing bank."""

import math
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_maximum import RollingMaximumRankedBank
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.banked_scaled_core import BankedScaledCore
from nearlink_sdr.sim.scaled_native import DifferenceQueue


def reference(frame):
    cfg = TxConfig(
        frame_type=frame,
        symbol_rate_mhz=4,
        mcs_index=8,
        pilot_interval=0,
        ctrl_bits_len=28 if frame == 2 else 27,
        head_llid=None if frame == 2 else 0x351729,
    )
    return PSKReceiveProfile.ordinary(cfg).sync_symbols()


def words_for(ref, extra=160):
    length = (len(ref) - 1) * 4 + extra
    wave = np.zeros(length, complex)
    signal = np.repeat(ref, 4)
    wave[80 : 80 + len(signal)] = signal[: max(0, length - 80)]
    wave *= 217 * np.exp(1j * np.arange(length) * 0.0173)
    r, i = np.rint(wave.real).astype(int), np.rint(wave.imag).astype(int)
    e = np.zeros(length, int)
    e[-37:] = np.resize([-48, 32, 0, -17, 5], 37)
    return [tuple(map(int, v)) for v in zip(r, i, e, strict=True)]


@pytest.mark.parametrize("frame,lanes", [(2, 8), (2, 16), (3, 8), (3, 16), (4, 16)])
def test_continuous200M_scores_and_recent_queries_match_exact_reference(frame, lanes):
    ref = reference(frame)
    words = words_for(ref)
    origin = -40
    oracle = RollingMaximumRankedBank(ref, 4, {2: 256, 3: 376, 4: 632}[frame], origin=origin)
    expected_queries, observed = {}, []

    def observe(event):
        if event["kind"] != "point":
            return
        n = event["tag"]
        assert oracle.samples == n
        r, i, e = words[n]
        points = oracle.advance_words([r], [i], [e], inspect=True)
        if event["ready"]:
            assert len(points) == 1
            assert (event["numerator"], event["energy"]) == (
                points[0].numerator,
                points[0].energy,
            )
            assert event["rank"] == int(oracle._rank_scores[n % oracle.rank_history])
            assert event["hit"] == bool(oracle._hits[n % oracle.window_samples])
        if n % 37 == 0:
            values = []
            for j in range(oracle.k):
                source = n - 4 * (oracle.k - 1 - j)
                if source < 0:
                    values.append((0, 0, 0, 0))
                    continue
                a, b, ex = words[source]
                c, d, old = words[source - 4] if source >= 4 else (0, 0, 0)
                dr, di = a * c + b * d, b * c - a * d
                values.append((dr, di, dr * dr + di * di, ex + old))
            common = max((v[3] for v in values if v[2]), default=-96)
            real = imag = energy = 0
            for j, (r, i, p, e) in enumerate(values):
                shift = max(common - e, 0)
                rr, ii = round(Fraction(r, 1 << shift)), round(Fraction(i, 1 << shift))
                real += rr * int(oracle._coefficient_real[j]) + ii * int(
                    oracle._coefficient_imag[j]
                )
                imag += ii * int(oracle._coefficient_real[j]) - rr * int(
                    oracle._coefficient_imag[j]
                )
                energy += round(Fraction(p, 1 << (2 * shift)))
            assert (event["real"], event["imag"], event["energy"]) == (real, imag, energy)
        observed.append(n)

    core = BankedScaledCore(ref, lanes=lanes, origin=origin, sink=observe)
    sent, last_query_tag = 0, -1
    for _ in range(20000):
        if sent < len(words) and core.cycle >= math.ceil(sent * 12.5):
            assert core.offer(sent, *words[sent])
            sent += 1
        core.step()
        n = core.committed_count - 1
        if n >= core.k * 4 and n % 32 == 0 and n != last_query_tag:
            upper = origin + n - core.k * 4
            phase_origin = origin + n % 3
            expected = oracle.query_best(upper - 130, upper, phase_origin=phase_origin)
            request_id = core.next_query_id
            assert core.request_query(upper - 130, upper, phase_origin=phase_origin)
            expected_queries[request_id] = expected
            last_query_tag = n
        while core.candidates:
            actual = core.take_candidate()
            assert actual["reason"] is None
            assert (actual["enabled"], actual["count"], actual["chosen"]) == expected_queries.pop(
                actual["request_id"]
            )
        if (
            sent == len(words)
            and core.committed_count == sent
            and core.data_idle
            and not core.query_queue
            and core.query_active is None
        ):
            break
    assert not core.failure and sent == core.committed_count == len(words)
    assert observed == list(range(len(words))) and not expected_queries
    assert core.finish()
    assert core.stats["query_reads"] > 0
    assert core.stats["query_dropped"] == core.stats["query_expired"] == 0
    assert core.history.estimate()["per_bank_read_ports"] == 1
    cost = core.estimate()
    assert cost["DSP"] == 12 and cost["logical_module_research_cap_pass"]
    assert cost["whole_RX_16MSps"] is False and cost["Fmax"] is None


def test_registered_difference_RAM_front_keeps_every_credit_and_word_identity():
    queue = DifferenceQueue()
    from nearlink_sdr.phy.banked_words import DifferenceWord

    expected, observed = [], []
    next_tag = 0
    for cycle in range(1000):
        if cycle % 9 in (0, 1, 4) and queue:
            observed.append(queue.popleft())
        queue.advance()
        if next_tag < 137 and queue.can_write:
            item = next_tag, DifferenceWord(next_tag, -next_tag, 2 * next_tag * next_tag, 0)
            queue.append(item)
            expected.append(item)
            next_tag += 1
        assert len(queue) <= 64
        if observed == expected and next_tag == 137:
            break
    assert observed == expected and len(observed) == 137
    assert queue.stats["alias_holds"] == 0
    assert queue.estimate()["registered_front_bits"] == 156


@pytest.mark.parametrize("clock,lanes", [(100, 16), (200, 8)])
def test_insufficient_service_aborts_explicitly_without_compressing_input_tags(clock, lanes):
    core = BankedScaledCore(reference(4), lanes=lanes)
    sent = 0
    for _ in range(30000):
        if core.cycle >= math.ceil(sent * clock / 16):
            if not core.offer(sent, 17, -13, 0):
                core.input_overrun(sent)
                break
            sent += 1
        core.step()
    assert core.failure == "input-overrun"
    assert core.first_lost_tag == sent == core.input_count
    assert core.committed_count < core.input_count
    assert core.high_water["input"] == 16
    assert core.high_water["differences"] <= 64
    before = core.histogram.cycle
    core.step()
    assert core.histogram.cycle == before  # Aborted epoch cannot silently resume.
    assert not core.finish()
    with pytest.raises(ValueError, match="epoch"):
        core.offer(sent, 0, 0, 0)


def test_blocked_candidate_sink_drops_or_expires_queries_but_preserves_the_word_stream():
    core = BankedScaledCore(reference(2), lanes=8)
    sent, requested = 0, False
    for _ in range(25000):
        if sent < 1536 and core.cycle >= math.ceil(sent * 12.5):
            assert core.offer(sent, 0, 0, 0)
            sent += 1
        core.step()
        if core.committed_count >= 384 and not requested:
            upper = core.committed_count - 1 - core.k * 4
            offers = [core.request_query(upper - 130, upper) for _ in range(7)]
            assert sum(offers) == 2
            core.step()  # First query leaves the two-slot admission FIFO.
            assert core.request_query(upper - 130, upper)
            requested = True
        if core.committed_count == 1536 and core.data_idle:
            break
    assert sent == core.committed_count == 1536 and not core.failure
    assert core.stats["query_dropped"] == 5 and core.stats["query_output_held"] > 0
    # Add two more while the old result is held;their snapshot will also be
    # retained rather than retargeted to a new query range on release.
    upper = core.committed_count - 1 - core.k * 4
    assert core.request_query(upper - 130, upper)
    while core.query_active is not None or core.query_queue or core.candidates:
        while core.candidates:
            assert core.take_candidate()["chosen"] is None
        core.step()
    assert core.stats["query_expired"] >= 1
    assert core.finish()


def test_EOF_drain_requires_real_candidate_consumption_and_preserves_all_accepted_words():
    core = BankedScaledCore(reference(2))
    for n in range(191):
        while core.cycle < math.ceil(n * 12.5):
            core.step()
        assert core.offer(n, 23, 0, 0)
    while core.committed_count < 191:
        core.step()
    upper = core.committed_count - 1 - core.k * 4
    assert core.request_query(upper - 31, upper)
    with pytest.raises(ValueError, match="candidate"):
        core.finish()
    result = core.take_candidate()
    assert result["count"] == 32 and result["reason"] is None
    assert core.finish()
    assert core.input_count == core.committed_count == 191


def test_coordinate_guards_refuse_overflow_without_wrapping_or_faulting_normal_data():
    core = BankedScaledCore(reference(2), origin=(1 << 62) - 1)
    core.input_count = (1 << 62) + 1  # Diagnostic counter injection,not a source replay.
    with pytest.raises(OverflowError, match="coordinate"):
        core.offer(core.input_count, 0, 0, 0)
    assert not core.inputs
    query = BankedScaledCore(reference(2), origin=-(1 << 62))
    assert query.request_query((1 << 62) - 1, (1 << 62) - 1)
    for _ in range(9):
        query.step()
    result = query.take_candidate()
    assert result["reason"] == "query-range" and result["chosen"] is None
    assert not query.failure
    assert query.offer(0, 0, 0, 0)
    assert query.finish()


def test_public_histogram_credit_respects_registered_retirement_without_mutation():
    from nearlink_sdr.sim.maximum_flow import ExponentMaximumService

    histogram = ExponentMaximumService(31, counter_store="register2r2w")
    assert not histogram.can_issue
    histogram.step()
    before = dict(histogram.stats)
    assert histogram.can_issue and histogram.stats == before
    assert histogram.issue(0, (0, False), (0, False))
    for _ in range(4):
        histogram.step()
    assert histogram.ready and not histogram.can_issue
    histogram.take()
    assert not histogram.can_issue
    histogram.step()
    assert histogram.can_issue
