"""Compiled serial normalization in the existing bounded coarse dataflow."""

import hashlib
import math
from dataclasses import asdict

import numpy as np
import pytest
from scripts.model_banked_scaled import timed
from scripts.model_scaled_maximum import configuration

from nearlink_sdr.sim.banked_scaled_core import BankedScaledCore
from tests.test_banked_scaled_core import reference, words_for


def run(ref, words, mode, lanes):
    points, answers, queries = [], [], []
    node = BankedScaledCore(
        ref,
        lanes=lanes,
        origin=-40,
        sink=lambda e: points.append(e) if e["kind"] == "point" else None,
        normalizer=mode,
    )
    sent, last = 0, -1
    for _ in range(50000):
        if sent < len(words) and node.cycle >= math.ceil(sent * 12.5):
            if not node.offer(sent, *words[sent]):
                node.input_overrun(sent)
                break
            sent += 1
        node.step()
        n = node.committed_count - 1
        if n >= node.k * 4 and n % 32 == 0 and n != last:
            upper = -40 + n - 4 * node.k
            queries.append((n, node.request_query(upper - 130, upper, phase_origin=-40 + n % 3)))
            last = n
        while node.candidates:
            result = node.take_candidate()
            answers.append(
                {
                    key: result[key]
                    for key in (
                        "request_id",
                        "snapshot",
                        "count",
                        "enabled",
                        "chosen",
                        "score",
                        "reason",
                    )
                }
            )
        if (
            sent == len(words)
            and node.data_idle
            and not node.query_queue
            and node.query_active is None
        ):
            break
    if not node.failure:
        assert node.finish()
    return node, points, answers, queries


@pytest.mark.parametrize("frame,lanes", [(2, 8), (3, 8), (4, 16)])
def test_serial2_keeps_exact_points_snapshots_and_queries_at200MHz(frame, lanes):
    ref = reference(frame)
    words = words_for(ref, extra=700)
    old, a, qa, ra = run(ref, words, "pipeline", lanes)
    new, b, qb, rb = run(ref, words, "serial2", lanes)
    assert old.failure is new.failure is None
    assert old.committed_count == new.committed_count == len(words)
    assert [{k: v for k, v in x.items() if k != "cycle"} for x in a] == [
        {k: v for k, v in x.items() if k != "cycle"} for x in b
    ]
    assert qa == qb and ra == rb and qa
    assert max(x["cycle"] - math.ceil(x["tag"] * 12.5) for x in b) <= max(
        x["cycle"] - math.ceil(x["tag"] * 12.5) for x in a
    )
    assert new.estimate()["normalization"]["research_register_cap_pass"]


def test_serial1_throughput_failure_counts_every_later_source_arrival_and_EOF(tmp_path):
    cfg = configuration(4, 8, 0, 71)
    path = tmp_path / "source.npz"
    np.savez_compressed(path, iq=0.17 * np.exp(0.071j * np.arange(2560)))
    config = asdict(cfg)
    config.pop("_mcs", None)
    row = dict(
        id=0,
        config=config,
        artifact=path.name,
        input_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    result = timed(tmp_path, row, 200, 16, normalizer="serial1", continue_aborted_source=True)
    assert result["failure"] == "input-overrun" and not result["word_stream_pass"]
    assert result["source_tags_presented"] == 2560
    assert result["accepted"] + result["source_epoch_discards"] == 2560
    assert result["source_epoch_discards"] > 1 and result["first_abort_cycle"] is not None
    assert result["all_planned_source_arrivals_processed"] and result["EOF_epoch_closed"]
    assert not result["automatic_rearm"] and not result["full_RX_16MSps"]
    ctrl = result["source_discard_controller"]
    assert ctrl["additional_counter_bits"] == 64
    assert ctrl["composed_logical_bits"] == result["cost"]["logical_state_bits_upper"] + 64
    assert ctrl["upstream_RRC_arrivals_and_rearm_cycles"] is None


@pytest.mark.parametrize("mode", [None, "serial3", 1])
def test_unvalidated_normalizer_options_are_not_enabled(mode):
    with pytest.raises(ValueError, match="study"):
        BankedScaledCore(reference(2), normalizer=mode)
