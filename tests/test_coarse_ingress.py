"""Finite ingress identities/queries/overload,not RF or complete RX acceptance."""

import copy
import math
from unittest.mock import patch

import numpy as np
import pytest
from scripts.model_scaled_maximum import configuration

from nearlink_sdr.phy.acquisition_maximum import RollingMaximumRankedBank
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.coarse_ingress import CoarseIngress
from tests.test_rrc_bridge import oracle as bridge_oracle


def profile(ft=2):
    return PSKReceiveProfile.ordinary(configuration(ft, 8, 0, 113))


def run(ft=2, *, normalizer="pipeline", count=704, block_until=0, partitions=()):
    p = profile(ft)
    values = [
        ((j * 137) % 131071 - 65535, (j * 197) % 65521 - 32760, -17 + (j // 97) % 4)
        for j in range(count)
    ]
    triples = [bridge_oracle(*v, 10)[0] for v in values]
    ref = RollingMaximumRankedBank(p.sync_symbols(), 4, {2: 256, 3: 376, 4: 632}[ft], origin=-40)
    points, expected, results = [], {}, []

    def observe(event):
        if event["kind"] != "point":
            return
        tag = event["tag"]
        assert tag == ref.samples
        rows = ref.advance_words(*(np.asarray([v]) for v in triples[tag]), inspect=True)
        if event["ready"]:
            assert (event["numerator"], event["energy"]) == (rows[0].numerator, rows[0].energy)
            assert event["rank"] == int(ref._rank_scores[tag % ref.rank_history])
            assert event["hit"] == bool(ref._hits[tag % ref.window_samples])
        points.append(dict(event))

    flow = CoarseIngress(
        p.sync_symbols(), lanes=16 if ft == 4 else 8, normalizer=normalizer, sink=observe
    )
    original = flow.core.request_query

    def query(lo, hi):
        qid = flow.core.next_query_id
        answer = ref.query_best(lo, hi)
        ok = original(lo, hi)
        if ok:
            expected[qid] = answer
        return ok

    flags_at = {math.ceil(j * 200 / 16) + 10: j for j in range(count)}
    words_at = {math.ceil(j * 200 / 16) + 60: j for j in range(count)}
    checkpoints = []
    with patch.object(flow.core, "request_query", query):
        for cycle in range(max(words_at) + max(8192, block_until)):
            if cycle in partitions:
                checkpoints.append((cycle, flow.core.input_count))
            if cycle >= block_until:
                while flow.core.candidates:
                    result = flow.take_candidate()
                    answer = expected.pop(result["request_id"])
                    if result["reason"] is None:
                        assert (result["enabled"], result["count"], result["chosen"]) == answer
                    else:
                        assert result["reason"] == "history-expired"
                    results.append(result)
            if cycle in flags_at:
                flow.push_flag(flags_at[cycle], True)
            if cycle in words_at:
                tag = words_at[cycle]
                assert flow.push_rrc(tag, pack_filter_word(*values[tag]))
            flow.step()
            if cycle > max(words_at) and flow.drained:
                break
    assert not expected and flow.drained
    closed = flow.finish()
    assert flow.closed and flow.core.committed_count == count
    assert len(points) == count and flow.join.stats["words_joined"] == count
    assert flow.join.stats["flags_accepted"] == count
    return flow, points, results, checkpoints, closed


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("mode", ["pipeline", "serial2"])
def test_independent_words_points_and_snapshot_queries(ft, mode):
    f, _, result, _, closed = run(ft, normalizer=mode)
    assert closed and result and f.stats["query_requests"] == f.stats["query_submitted"]
    assert f.stats["core_query_dropped"] == f.stats["pending_query_dropped"] == 0
    assert f.finish()
    c = f.estimate()
    assert c["own_register_bits_upper"] <= 1536 and c["new_DSP"] == c["new_sample_RAM"] == 0
    assert c["coarse"]["DSP"] == 12
    assert c["raw_detector_arithmetic_cycles"] is c["fine_packet_decoder_cycles"] is None
    assert c["actual_synthesis"] is c["whole_RX_16MSps"] is False


def test_call_group_boundaries_do_not_reset_state():
    a = run(normalizer="serial2")
    b = run(normalizer="serial2", partitions=(1, 17, 111, 2048, 4097, 8111))
    assert a[0].stats == b[0].stats and a[1:3] == b[1:3]
    assert len(b[3]) == 6


def test_blocked_candidate_counts_losses_without_stopping_words():
    f, _, _, _, closed = run(count=1024, block_until=16000, normalizer="serial2")
    assert not closed and not f.lossless and not f.finish()
    assert f.fault_reason == 0
    assert f.stats["core_query_dropped"] > 0 and f.core.stats["query_expired"] > 0
    assert f.core.committed_count == 1024


@pytest.mark.parametrize("word", [True, -1, 2**46, 1.5, "word"])
def test_malformed_word_rejected_before_event_consumption(word):
    f = CoarseIngress(profile().sync_symbols())
    old = copy.deepcopy(f.__dict__)
    with pytest.raises((ValueError, OverflowError, TypeError)):
        f.push_rrc(0, word)
    assert f.stats == old["stats"] and f.next_rrc_tag == 0 and f.last_rrc_cycle is None
    assert f.join.bridge.stats["accepted"] == 0


@pytest.mark.parametrize("exponent", [-97, 49])
def test_illegal_rrc_exponent_does_not_consume_a_source_tag(exponent):
    f = CoarseIngress(profile().sync_symbols())
    word = pack_filter_word(3, -1, exponent)
    with pytest.raises(ValueError):
        f.push_rrc(0, word)
    assert f.stats["rrc_events"] == f.next_rrc_tag == 0
    assert f.last_rrc_cycle is None and f.join.bridge.stats["accepted"] == 0


@pytest.mark.parametrize("port", ["flag", "rrc"])
def test_query_coordinate_headroom_is_checked_at_both_ports(port):
    f = CoarseIngress(profile().sync_symbols())
    with pytest.raises(ValueError):
        if port == "flag":
            f.push_flag(2**62, True)
        else:
            f.push_rrc(2**62, pack_filter_word(0, 0, 0))
    assert f.next_rrc_tag == f.stats["rrc_events"] == f.stats["epoch_faults"] == 0


def test_one_source_port_and_no_duplicate_identity():
    f = CoarseIngress(profile().sync_symbols())
    assert f.push_rrc(0, pack_filter_word(7, 2, -10))
    with pytest.raises(ValueError):
        f.push_rrc(1, pack_filter_word(7, 2, -10))
    f.step()
    with pytest.raises(ValueError):
        f.push_rrc(0, pack_filter_word(7, 2, -10))
    assert f.stats["rrc_events"] == 1


@pytest.mark.parametrize("kind", ["gap", "invalid", "credit"])
def test_epoch_fault_counts_all_later_source_pulses(kind):
    f = CoarseIngress(profile().sync_symbols())
    word = pack_filter_word(7, 2, -10)
    assert f.push_rrc(0, word)
    f.push_flag(0, True)
    f.step()
    tag = 2 if kind == "gap" else 1
    assert not f.push_rrc(tag, None if kind == "invalid" else word)
    assert f.fault_reason and not f.take_candidate()
    f.push_flag(1, False)
    f.step()
    assert not f.push_rrc(tag + 1, word)
    f.push_flag(2, False)
    f.step()
    assert f.stats["rrc_events"] == 3 and f.stats["rrc_discarded"] == 2
    assert f.stats["epoch_faults"] == 1 and f.join.stats["flags_discarded"] == 2
    assert f.drained and not f.finish() and f.closed
    assert not f.finish()
    with pytest.raises(RuntimeError):
        f.push_rrc(tag + 2, word)
    with pytest.raises(RuntimeError):
        f.push_flag(3, False)
    with pytest.raises(RuntimeError):
        f.step()


def test_missing_rrc_tail_and_missing_detector_flag_are_explicit():
    f = CoarseIngress(profile().sync_symbols())
    f.push_flag(0, True)
    for _ in range(700):
        f.step()
    assert not f.finish() and f.stats["rrc_missing_tags"] == 1
    g = CoarseIngress(profile().sync_symbols())
    g.push_rrc(0, pack_filter_word(9, -3, -8))
    for _ in range(700):
        g.step()
    assert g.join.bridge.ready and g.drained
    assert not g.finish() and g.join.fault_reason == g.join.reasons["EOF_flag_missing"]


def test_early_finish_preserves_work_for_later_drain():
    f = CoarseIngress(profile().sync_symbols())
    f.push_flag(0, True)
    f.push_rrc(0, pack_filter_word(3, 8, -4))
    before = dict(f.stats)
    with pytest.raises(RuntimeError):
        f.finish()
    assert not f.closed and before == f.stats
    for _ in range(800):
        f.step()
    assert f.drained and f.finish() and f.core.committed_count == 1


def test_abort_cancels_pending_candidate_and_never_rearms():
    f, _, _, _, _ = run()
    # A new independent epoch is needed;finished state never restarts.
    with pytest.raises(RuntimeError):
        f.push_flag(704, True)
    g = CoarseIngress(profile().sync_symbols())
    for cycle in range(6000):
        if cycle % 13 == 0:
            g.push_flag(cycle // 13, True)
            g.push_rrc(cycle // 13, pack_filter_word(11, -5, -7))
        g.step()
        if g.core.candidates and g.pending is not None:
            break
    assert g.core.candidates and g.pending is not None
    g.abort()
    assert not g.core.candidates and g.take_candidate() is None
    assert g.stats["candidate_canceled"] == g.stats["pending_query_canceled"] == 1
    assert not g.finish() and g.stats["epoch_faults"] == 1


def test_counter_exhaustion_never_wraps():
    f = CoarseIngress(profile().sync_symbols())
    f.stats["rrc_events"] = 2**64 - 1
    with pytest.raises(OverflowError):
        f.push_rrc(0, pack_filter_word(0, 0, 0))
    assert f.next_rrc_tag == 0 and f.last_rrc_cycle is None
    assert f.join.bridge.stats["accepted"] == 0
