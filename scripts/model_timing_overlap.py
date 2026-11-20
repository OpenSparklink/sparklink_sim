"""Frozen power/TED schedule integration;expected exit1 for retained RX gaps."""

import argparse
import gzip
import json
from functools import partial
from pathlib import Path
from random import Random

import numpy as np
from audit_capture_hardware import sha, write
from model_header_words import source_identity
from model_live_head import FIELDS, GEOMETRY_FIELDS, TRACE, cases, save_event, suffix
from tests.test_head_word_replay import run as replay
from tests.timing_overlap_reference import node_vectors

from nearlink_sdr.common import polar
from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.data_joint_replay import DataJointFrameReplay
from nearlink_sdr.sim.power_ted_replay import PowerTEDFrameReplay
from nearlink_sdr.sim.timing_overlap_flow import ParallelTEDTimingWordService
from nearlink_sdr.sim.timing_word_flow import TimingWordService

POLICY = Path("tests/data/timing_overlap_policy.json")


def nodes(output, split):
    vectors = list(node_vectors(split))
    np.savez_compressed(
        output / "node-vectors.npz",
        current=np.asarray([w for w, _ in vectors], dtype=np.uint64),
        midpoint=np.asarray([0 if m is None else m for _, m in vectors], dtype=np.uint64),
        midpoint_valid=np.asarray([m is not None for _, m in vectors], bool),
    )
    rows, count = [], 0
    with gzip.open(output / "node-states.jsonl.gz", "wt") as log:
        for p, e in ((30, 24), (30, 32), (38, 32)):
            for divider in ("bounded-prefix", "radix4-prefix"):
                fmt = TimingWordFormat(p, e)
                services = [
                    cls(WordTimingState((1 << 53, -53), 57, 32, fmt), divider=divider)
                    for cls in (TimingWordService, ParallelTEDTimingWordService)
                ]
                reference = WordTimingState((1 << 53, -53), 57, 32, fmt)
                pairs = set()
                for k, (word, midpoint) in enumerate(vectors):
                    expected = reference.advance_words(word, midpoint)
                    latency = []
                    for s in services:
                        previous = s.state.snapshot()
                        assert s.issue(word, midpoint)
                        products = s.stats["native_products"]
                        for _ in range(200):
                            assert s.state.snapshot() == previous
                            s.step()
                            assert s.stats["native_products"] - products in (0, 4)
                            products = s.stats["native_products"]
                            if s.ready:
                                break
                        assert s.ready and s.result == expected
                        latency.append(s.ready_cycle - s.first_cycle)
                        assert s.take() == expected
                        s.step()
                    assert latency[0] - latency[1] == (8 if midpoint is not None else 0)
                    assert services[0].stats == services[1].stats
                    log.write(
                        json.dumps(
                            dict(
                                power=p,
                                error=e,
                                divider=divider,
                                index=k,
                                latency=latency,
                                state=expected,
                            )
                        )
                        + "\n"
                    )
                    pairs.add(tuple(latency))
                    count += 1
                old, new = [s.estimate() for s in services]
                assert new["state_register_bits_upper"] - old["state_register_bits_upper"] == 104
                rows.append(
                    dict(
                        power=p,
                        error=e,
                        divider=divider,
                        latency_pairs=sorted(pairs),
                        exact=True,
                        words=len(vectors),
                        cost=new,
                    )
                )
    return dict(
        cells=count, exact=True, rows=rows, transcript_SHA256=sha(output / "node-states.jsonl.gz")
    )


def fixed(output, policy):
    rows, resources = [], []
    from tests.test_selected_word import quantize_input

    for name, p, values, expected, selection, provenance in cases():
        words = quantize_input(np.r_[values, np.zeros(8000)])
        selection = selection | dict(sample_limit=len(words))
        np.savez_compressed(
            output / (name + "-input.npz"),
            words=words,
            payload=np.frombuffer(expected, np.uint8),
            **selection,
        )
        for layout in ("dense", "depth-parity"):
            before, old, _ = replay(
                p, words, selection, partial(DataJointFrameReplay, head_layout=layout)
            )
            save_event(output, name + "-" + layout + "-before", before)
            for label, pattern in [
                ("32", (32,)),
                ("fixed", (1, 7, 31)),
                ("random", tuple(Random(2026112017).choices(range(1, 33), k=23))),
            ]:
                after, new, barriers = replay(
                    p, words, selection, partial(PowerTEDFrameReplay, head_layout=layout), pattern
                )
                save_event(output, name + "-" + layout + "-" + label, after)
                assert all(
                    before["frame"].get(k) == after["frame"].get(k)
                    for k in FIELDS + GEOMETRY_FIELDS
                )
                assert all(np.array_equal(before[k], after[k]) for k in TRACE)
                assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
                assert (
                    old.consumer.normalization.snapshot() == new.consumer.normalization.snapshot()
                )
                assert new.history.end == old.history.end and new.history.head == old.history.head
                assert all(
                    new.history.words[k % 512] == words[k]
                    for k in range(new.history.head, new.history.end)
                )
                assert [x.mac_payload for x in new.take_mac_result().packets] == [expected]
                cost = new.estimate()
                delta = (
                    cost["timing_service"]["state_register_bits_upper"]
                    - old.estimate()["timing_service"]["state_register_bits_upper"]
                )
                assert delta == 104 <= policy["budgets"]["additional_register_bits_max"]
                pairs = new.stats["data_joint_completions"]
                assert 5 * pairs <= old.cycle - new.cycle <= 8 * pairs
                rows.append(
                    dict(
                        id=name,
                        layout=layout,
                        group=label,
                        provenance=provenance,
                        input_SHA256=sha(output / (name + "-input.npz")),
                        exact=True,
                        modeled_edges=[old.cycle, new.cycle],
                        pairs=pairs,
                        payload_bits=8 * len(expected),
                        bit_errors=0,
                        barriers=barriers,
                        register_increment_bits=delta,
                        cost=cost,
                        complete_latency_cycles=None,
                    )
                )
                write(output / "fixed-frames.json", rows)
        if name in ("legacy-ft3", "standard-ft3-crc32"):
            for cls in (DataJointFrameReplay, PowerTEDFrameReplay):
                sub = output / (name + "-" + cls.__name__)
                sub.mkdir()
                for layout in ("dense", "depth-parity"):
                    for clock in policy["clocks_Hz"]:
                        r = suffix(
                            sub,
                            p,
                            words,
                            selection,
                            input_id=name,
                            layout=layout,
                            clock=clock,
                            continue_data=True,
                            flow_class=cls,
                        )
                        assert (
                            r["source_capacity_words"]
                            == policy["budgets"]["history_capacity_words"]
                        )
                        resources.append(r | dict(service=cls.__name__))
                        write(output / "resource-prefixes.json", resources)
    return rows, resources


def run(output, *, freeze=None):
    if output.exists():
        raise ValueError("Use a fresh evidence directory")
    policy, source = json.loads(POLICY.read_text()), source_identity()
    split = "development" if freeze is None else "validation"
    if freeze is not None:
        assert json.loads(freeze.read_text()) == dict(policy=policy, source=source)
    output.mkdir(parents=True)
    write(
        output / "protocol.json",
        dict(
            policy=policy,
            source=source,
            split=split,
            new_RF_holdout=False,
            changed_loop_parameters=False,
        ),
    )
    node_result = nodes(output, split)
    assert node_result["cells"] == policy["known_node_cells"][split]
    rows, resources = fixed(output, policy) if freeze is None else ([], [])
    result = dict(
        stage=policy["stage"],
        split=split,
        nodes=node_result,
        fixed_comparisons=len(rows),
        fixed_success_losses=0,
        receiver_rate_gain=0,
        resources=len(resources),
        resource_exits=sum(r["frame"] is not None for r in resources),
        register_increment_bits=104,
        new_DSP=0,
        new_source_RAM=0,
        new_source_ports=0,
        actual_reference_rust=polar._HAS_RUST_ACCEL,
        no_Rust_word_model_implementation=True,
        new_RF_holdout=False,
        complete_latency_cycles=None,
        four_RX_gates_passed=False,
        default_changed=False,
        synthesis=False,
        expected_exit=1,
    )
    assert source_identity() == source
    write(output / "model.json", result)
    if freeze is None:
        write(output / "freeze.json", dict(policy=policy, source=source))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    run(args.output, freeze=args.freeze)
    raise SystemExit(1)
