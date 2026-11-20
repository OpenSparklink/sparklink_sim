"""Data timing/normalization joint ownership and finite service; expected exit1."""

import argparse
import json
from functools import partial
from pathlib import Path
from random import Random

import numpy as np
from audit_capture_hardware import sha, write
from model_header_words import source_identity
from model_live_head import FIELDS, GEOMETRY_FIELDS, TRACE, cases, save_event, suffix
from tests.test_data_joint_replay import at_data_issue
from tests.test_head_word_replay import run as replay
from tests.test_selected_rx import ordinary, selected_input
from tests.test_selected_word import quantize_input
from tests.test_sync_fit_word import packed

from nearlink_sdr.common import polar
from nearlink_sdr.sim.data_joint_replay import DataJointFrameReplay
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay

POLICY = Path("tests/data/data_joint_policy.json")


def node_words(rng, kind, index):
    r, i, mr, mi = [rng.randrange(-(1 << 21), 1 << 21) for _ in range(4)]
    e, me = -21, -21
    if kind == "zero-current":
        r = i = mr = mi = 0
    elif kind == "exponent-switch":
        e, me = [-128, 127, -23, 0][index % 4], [127, -128, 0, -23][index % 4]
    elif kind == "edge-signs":
        r, i = -(1 << 21), (1 << 21) - 1
        mr, mi = (r, i) if index & 1 else (i, r)
    return packed(r, i, e), packed(mr, mi, me)


def nodes(output, policy, split):
    rng, rows = Random(policy[split + "_seed"]), []
    for ft, mcs, pilot, kind in policy["node_" + split]:
        p = ordinary(ft, mcs, pilot)
        values, _, _ = selected_input(p, length=47)
        words = quantize_input(np.r_[values, np.zeros(1000)])
        flows = [at_data_issue(cls, ft=ft, mcs=mcs, pilot=pilot)
                 for cls in (HeadWordsFrameReplay, DataJointFrameReplay)]
        for k in range(policy["node_words_per_cell"]):
            vector = node_words(rng, kind, k)
            for flow in flows:
                while flow.mode != "timing_issue":
                    assert not flow.done and flow.pending_service is None
                    if flow.wants_replay_input:
                        pos = flow.history.end
                        assert flow.feed_replay_words(words[pos:pos + 32]) == 1
                    flow.step()
                flow.words = list(vector)  # Explicit synthetic node port,not an RF capture.
                old_count = flow.consumer._active["count"]
                while flow.consumer._active["count"] == old_count:
                    flow.step()
            a, b = flows
            assert a.consumer.word_timing.snapshot() == b.consumer.word_timing.snapshot()
            assert a.consumer.normalization.snapshot() == b.consumer.normalization.snapshot()
            rows.append(dict(frame_type=ft, mcs=mcs, pilot=pilot, kind=kind, vector=k,
                             current_word=vector[0], midpoint_word=vector[1],
                             timing=b.consumer.word_timing.snapshot(),
                             normalization=b.consumer.normalization.snapshot(),
                             exact=True, modeled_edges=[a.cycle, b.cycle],
                             not_RF_packet_acceptance=True))
            write(output / "nodes.json", rows)
        for flow in flows:
            assert flow.abort()["frame"]["header_crc"] is None
            assert not flow.take_mac_result().packets
    return rows


def fixed(output, policy):
    rows, resources = [], []
    for name, p, values, expected, selection, provenance in cases():
        words = quantize_input(np.r_[values, np.zeros(8000)])
        selection = selection | dict(sample_limit=len(words))
        np.savez_compressed(output / (name + "-input.npz"), words=words,
                            payload=np.frombuffer(expected, np.uint8), **selection)
        for layout in ("dense", "depth-parity"):
            before, old, _ = replay(p, words, selection,
                                   partial(HeadWordsFrameReplay, head_layout=layout))
            save_event(output, name + "-" + layout + "-before", before)
            for label, pattern in [("32", (32,)), ("fixed", (1, 7, 31)),
                                   ("random", tuple(Random(11201351).choices(range(1, 33), k=23)))]:
                records = []
                after, new, barriers = replay(p, words, selection, partial(
                    DataJointFrameReplay, head_layout=layout, service_observer=records.append
                ), pattern)
                save_event(output, name + "-" + layout + "-" + label, after)
                for key in FIELDS + GEOMETRY_FIELDS:
                    assert before["frame"].get(key) == after["frame"].get(key)
                for key in TRACE:
                    np.testing.assert_array_equal(before[key], after[key])
                assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
                assert (old.consumer.normalization.snapshot()
                        == new.consumer.normalization.snapshot())
                assert new.history.end == old.history.end and new.history.head == old.history.head
                for i in range(new.history.head, new.history.end):
                    assert new.history.words[i % 512] == words[i]
                assert [x.mac_payload for x in new.take_mac_result().packets] == [expected]
                old_cost, new_cost = old.estimate(), new.estimate()
                delta = (new_cost["new_controller_state_bits"]
                         - old_cost["new_controller_state_bits"])
                assert delta == 128 <= policy["budgets"]["new_controller_bits_max"]
                pairs = new.stats["data_joint_completions"]
                assert old.cycle - new.cycle == 24 * pairs
                write(output / (name + "-" + layout + "-" + label + "-services.json"), records)
                rows.append(dict(id=name, layout=layout, group=label, provenance=provenance,
                                 input_SHA256=sha(output / (name + "-input.npz")), exact=True,
                                 modeled_edges=[old.cycle, new.cycle], pairs=pairs,
                                 payload_bits=8 * len(expected), bit_errors=0, barriers=barriers,
                                 controller_increment_bits=delta, cost=new_cost,
                                 complete_latency_cycles=None))
                write(output / "fixed-frames.json", rows)
        if name in ("legacy-ft3", "standard-ft3-crc32"):
            for cls in (HeadWordsFrameReplay, DataJointFrameReplay):
                sub = output / (name + "-" + cls.__name__)
                sub.mkdir()
                for layout in ("dense", "depth-parity"):
                    for clock in policy["clocks_Hz"]:
                        row = suffix(sub, p, words, selection, input_id=name, layout=layout,
                                     clock=clock, continue_data=True, flow_class=cls)
                        assert row["source_capacity_words"] == 512
                        resources.append(row | dict(service=cls.__name__))
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
    write(output / "protocol.json", dict(policy=policy, source=source, split=split,
                                        RF_new_holdout=False, changed_loop_parameters=False))
    node_rows = nodes(output, policy, split)
    rows, resources = fixed(output, policy) if freeze is None else ([], [])
    result = dict(stage=policy["stage"], split=split, node_pairs=len(node_rows),
                  node_word_and_states_exact=True, fixed_frame_comparisons=len(rows),
                  fixed_success_losses=0, receiver_rate_gain=0,
                  resources=len(resources), resource_exits=sum(r["frame"] is not None
                                                             for r in resources),
                  controller_increment_bits=128, new_DSP=0, new_source_RAM=0, new_source_ports=0,
                  actual_reference_rust=polar._HAS_RUST_ACCEL,
                  no_Rust_joint_word_implementation=True, new_RF_holdout=False,
                  complete_latency_cycles=None, full_RX_gates_passed=False,
                  default_changed=False, synthesis=False, expected_exit=1)
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
