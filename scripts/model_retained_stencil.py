"""Frozen raw-stencil reuse audit;exit1 preserves unresolved full-RX resource gates."""

import argparse
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import write
from model_header_words import source_identity
from model_live_head import FIELDS, GEOMETRY_FIELDS, TRACE, cases, save_event, suffix
from tests.stencil_reference import address_vectors, prefix_oracle
from tests.test_head_word_replay import run as replay
from tests.test_selected_word import quantize_input

from nearlink_sdr.common import polar
from nearlink_sdr.sim.power_ted_replay import PowerTEDFrameReplay
from nearlink_sdr.sim.retained_stencil_replay import RetainedStencilFrameReplay, _overlap_words

POLICY = Path("tests/data/retained_stencil_policy.json")


def nodes(output, split):
    rows = []
    for previous, following in address_vectors(split):
        try:
            expected = prefix_oracle(previous, following)
        except OverflowError:
            try:
                _overlap_words(previous, following)
            except OverflowError:
                rows.append(dict(previous=hex(previous), following=hex(following),
                                 rejected="OverflowError", exact=True))
            else:
                raise AssertionError("Invalid four-word end was not rejected") from None
        else:
            count = _overlap_words(previous, following)
            assert count == expected
            # Opaque port values;SHA is the independent generator,not receiver work.
            def value(address):
                return int.from_bytes(hashlib.sha256(str(address).encode()).digest()[:6]) & (
                    (1 << 46) - 1)

            old = [value(previous + k) for k in range(4)]
            wanted = [value(following + k) for k in range(4)]
            assert old[4 - count:] == wanted[:count]
            rows.append(dict(previous=hex(previous), following=hex(following), count=count,
                             previous_words=[hex(x) for x in old], exact=True))
    write(output / "address-nodes.json", rows)
    return dict(cases=len(rows), exact=True,
                rejected=sum("rejected" in r for r in rows),
                word_or_numeric_budget_relaxed=False)


def fixed(output, policy):
    rows, resources = [], []
    for name, p, values, expected, selection, provenance in cases():
        words = quantize_input(np.r_[values, np.zeros(8000)])
        selection = selection | dict(sample_limit=len(words))
        np.savez_compressed(output / (name + "-input.npz"), words=words,
                            payload=np.frombuffer(expected, np.uint8), **selection)
        for layout in ("dense", "depth-parity"):
            before, old, _ = replay(p, words, selection, partial(
                PowerTEDFrameReplay, head_layout=layout))
            save_event(output, name + "-" + layout + "-before", before)
            for label, pattern in [("32", (32,)), ("fixed", (1, 7, 31)),
                                   ("mixed", (13, 2, 31, 1, 7, 19))]:
                after, new, barriers = replay(p, words, selection, partial(
                    RetainedStencilFrameReplay, head_layout=layout), pattern)
                save_event(output, name + "-" + layout + "-" + label, after)
                assert all(before["frame"].get(k) == after["frame"].get(k)
                           for k in FIELDS + GEOMETRY_FIELDS)
                assert all(np.array_equal(before[k], after[k]) for k in TRACE)
                assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
                assert (
                    old.consumer.normalization.snapshot() == new.consumer.normalization.snapshot()
                )
                assert old.history.end == new.history.end and old.history.head == new.history.head
                assert all(new.history.words[k % 512] == words[k]
                           for k in range(new.history.head, new.history.end))
                assert [x.mac_payload for x in new.take_mac_result().packets] == [expected]
                old_cost, cost = old.estimate(), new.estimate()
                delta = cost["new_controller_state_bits"] - old_cost["new_controller_state_bits"]
                assert delta == 64 <= policy["budgets"]["additional_controller_bits_max"]
                assert cost["borrowed_history"] == old_cost["borrowed_history"]
                assert cost["borrowed_interpolator"] == old_cost["borrowed_interpolator"]
                assert new.stats["raw_reads"] <= old.stats["raw_reads"]
                assert new.cycle <= old.cycle
                rows.append(dict(id=name, layout=layout, group=label, provenance=provenance,
                                 exact=True, bit_errors=0, payload_bits=8 * len(expected),
                                 modeled_edges=[old.cycle, new.cycle],
                                 raw_reads=[old.stats["raw_reads"], new.stats["raw_reads"]],
                                 tracked_symbols=new.stats["data_joint_completions"],
                                 reused_raw_words=new.stats["cross_symbol_words_reused"],
                                 source_words_SHA256=hashlib.sha256(np.asarray(
                                     words, dtype="<u8").tobytes()).hexdigest(),
                                 barriers=barriers, cost=cost, complete_latency_cycles=None))
                write(output / "fixed-frames.json", rows)
        if name in ("legacy-ft3", "standard-ft3-crc32"):
            for cls in (PowerTEDFrameReplay, RetainedStencilFrameReplay):
                sub = output / (name + "-" + cls.__name__)
                sub.mkdir()
                for layout in ("dense", "depth-parity"):
                    for clock in policy["clocks_Hz"]:
                        row = suffix(sub, p, words, selection, input_id=name, layout=layout,
                                     clock=clock, continue_data=True, flow_class=cls)
                        assert row["source_capacity_words"] == 512 and row["source_peak"] <= 512
                        assert row["exact_source_suffix"] and row["actual_epoch_valid"]
                        resources.append(row | dict(service=cls.__name__))
                        write(output / "resource-prefixes.json", resources)
    return rows, resources


def run(output, *, freeze=None):
    if output.exists():
        raise ValueError("Use a fresh evidence directory")
    policy, source = json.loads(POLICY.read_text()), source_identity()
    split = "development" if freeze is None else "validation"
    identity = dict(policy=policy, source=source)
    if freeze is not None:
        assert json.loads(freeze.read_text()) == identity
    output.mkdir(parents=True)
    write(output / "freeze.json", identity)
    write(output / "protocol.json", dict(identity=identity, split=split, new_RF_holdout=False,
                                        source_domain="selected46bit source port",
                                        actual_reference_rust=polar._HAS_RUST_ACCEL,
                                        no_Rust_stencil_implementation=True))
    arithmetic = nodes(output, split)
    assert arithmetic["cases"] == policy["node_" + split + "_cases"]
    rows, resources = fixed(output, policy) if split == "development" else ([], [])
    summary = dict(stage=policy["stage"], split=split, nodes=arithmetic,
                   fixed_frame_comparisons=len(rows), fixed_success_losses=0,
                   receiver_rate_gain=0, new_RF_holdout=False,
                   resources=len(resources), resource_exits=sum(r["frame"] is not None
                                                               for r in resources),
                   additional_controller_bits=64, new_DSP=0, new_source_RAM=0, new_source_ports=0,
                   component_structural_pass=True, complete_latency_cycles=None,
                   actual_reference_rust=polar._HAS_RUST_ACCEL,
                   no_Rust_stencil_implementation=True, four_RX_gates_passed=False,
                   default_changed=False, synthesis=False, expected_exit=1)
    write(output / "model.json", summary)
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    run(args.output, freeze=args.freeze)
    raise SystemExit(1)
