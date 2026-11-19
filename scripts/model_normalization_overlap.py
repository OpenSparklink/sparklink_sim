"""Exact finite normalization overlap, strict fields and bounded continuous source.

Exit1 preserves the four incomplete RX gates. No throughput claim for unmodeled
header/residual carrier/decoder. CRC/payload/truth only assess final decisions.
"""

import argparse
import gzip
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_normalization_words import DECISIONS, FMT, NORM, execute, periodic_prefixes, raw_trace
from model_selected_frame import TRACE_KEYS, metrics, normalized
from model_timing_handoff import fixed, fresh

from nearlink_sdr.phy.normalization_word import WordNormalizationState
from nearlink_sdr.sim.normalization_pipeline import PipelinedNormalizationService
from nearlink_sdr.sim.normalized_word_replay import NormalizedWordFrameReplay
from nearlink_sdr.sim.overlapped_normalization_replay import OverlappedNormalizationFrameReplay

POLICY = Path("tests/data/normalization_overlap_policy.json")


def source_vector():
    return {str(p): sha(p) for p in sorted({
        POLICY, Path("uv.lock"), *Path("src").rglob("*.py"), *Path("tests").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    })}


def pair(out, name, profile, words, selection, payload, policy, *, partitions):
    events, flows, traces, records = [], [], [], []
    for label, cls in [("serial", NormalizedWordFrameReplay),
                       ("overlap", OverlappedNormalizationFrameReplay)]:
        service = []
        event, flow, trace = execute(
            out, name + "-" + label, profile, words, selection, FMT, [32],
            flow_class=partial(cls, normalization_format=NORM, service_observer=service.append),
        )
        with gzip.open(out / (name + "-" + label + "-services.jsonl.gz"), "wt") as f:
            for item in service:
                f.write(json.dumps(normalized(item), separators=(",", ":")) + "\n")
        events.append(event)
        flows.append(flow)
        traces.append(trace)
        records.append(service)
    a, b = events
    assert normalized(a["frame"]) == normalized(b["frame"])
    assert all(normalized(a["frame"].get(k)) == normalized(b["frame"].get(k)) for k in DECISIONS)
    assert raw_trace(traces[0]) == raw_trace(traces[1])
    for key in TRACE_KEYS:
        np.testing.assert_array_equal(a[key], b[key])
    for key in ("word_timing_state", "word_normalization_state"):
        assert a[key] == b[key]
    assert [s["values"] for s in records[0]] == [s["values"] for s in records[1]]
    owner = WordNormalizationState(flows[1].consumer.initial_fit, NORM)
    for r in records[1]:
        assert r["taken_edge"] >= r["ready_edge"] == r["issued_edge"] + r["latency"]
        if r["kind"] == "normalization":
            assert r["values"] == owner.advance_word(r["values"]["raw_word"])
    assert owner.snapshot() == flows[1].consumer.normalization.snapshot()
    assert flows[0].stats["cache_reads"] == flows[1].stats["cache_reads"]
    if partitions:
        for label, pattern in [("fixed", [7, 31]),
                               ("random", np.random.default_rng(1122247).integers(1, 33, 23))]:
            other, flow, trace = execute(
                out, name + "-" + label, profile, words, selection, FMT, pattern,
                flow_class=partial(OverlappedNormalizationFrameReplay, normalization_format=NORM),
            )
            assert normalized(other["frame"]) == normalized(b["frame"])
            assert trace == traces[1]
            for key in TRACE_KEYS:
                np.testing.assert_array_equal(other[key], b[key])
            assert flow.consumer.normalization.snapshot() == owner.snapshot()
    cost = flows[1].estimate()
    node, budget = cost["normalization_service"], policy["node_budgets"]
    assert node["state_register_bits_upper"] + 192 <= budget["new_logic_state_bits_upper_max"]
    assert node["additional_DSP25x18"] <= budget["extra_DSP_max"]
    assert node["coefficient_ROM_RAMB18_estimate"] <= budget["extra_quarter_ROM_RAMB18_max"]
    assert cost["extra_source_or_cache_read_ports"] == cost["extra_original_sample_RAM_bits"] == 0
    before, after = [metrics(e, profile, expected=payload) for e in events]
    return dict(id=name, before=before, after=after,
                source_words_sha256=hashlib.sha256(
                    np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
                numerical=dict(numeric_pass=True, relative_RMS=0, relative_peak=0,
                               timing_trajectory_delta=0, exact_H1_words_states_services=True),
                success_loss=before["payload_exact"] and not after["payload_exact"],
                recovery=after["payload_exact"] and not before["payload_exact"],
                false_delivery=after["false_delivery"] and not before["false_delivery"],
                partitions_checked=partitions, cost=cost,
                normalization_word_count=owner.count,
                service_records=len(records[1]),
                actual_modeled_edges=[f.cycle for f in flows],
                modeled_edge_saving=flows[0].cycle - flows[1].cycle,
                actual_pipeline=b["normalization_pipeline"],
                complete_latency_cycles=None, full_receiver16MSps=False)


def run(args):
    policy, source = json.loads(POLICY.read_text()), source_vector()
    for split in ("development", "validation"):
        validate_profiles(policy[split]["profiles"], policy[split]["pid"])
    frozen = json.loads(args.freeze.read_text()) if args.freeze else None
    if frozen is not None:
        assert frozen["source"] == source and frozen["policy"] == policy
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", dict(source=source, policy=policy, split=args.split,
          upstream="whole-record differential maximum fixture;not bounded live capture",
          expectation_CRC_truth="postdecision only;not used in selection or scheduling",
          remaining_header_residual_carrier_LLR_decoder_cycles=None))
    rows = fresh(out, args.split, policy, True, pair_fn=pair)
    regressions, parents = (fixed(out, args.timing_root, args.capture_root, args.selected_root,
                                 policy, pair_fn=pair) if frozen is None else ([], {}))
    prefixes = {}
    if frozen is None:
        for label, cls in [("serial", NormalizedWordFrameReplay),
                           ("overlap", OverlappedNormalizationFrameReplay)]:
            path = out / ("continuous-" + label)
            path.mkdir()
            prefixes[label] = periodic_prefixes(path, flow_class=cls)
    reached = [r["comparison"] for r in rows + regressions if "comparison" in r]
    summary = dict(stage=policy["stage"], split=args.split,
                   before_full=sum(r["comparison"]["before"]["payload_exact"]
                                   for r in rows if "comparison" in r),
                   after_full=sum(r["comparison"]["after"]["payload_exact"]
                                  for r in rows if "comparison" in r),
                   fixed_before_full=sum(r["comparison"]["before"]["payload_exact"]
                                         for r in regressions if "comparison" in r),
                   fixed_after_full=sum(r["comparison"]["after"]["payload_exact"]
                                        for r in regressions if "comparison" in r),
                   success_losses=sum(r["success_loss"] for r in reached),
                   recoveries=sum(r["recovery"] for r in reached),
                   new_false_deliveries=sum(r["false_delivery"] for r in reached),
                   numeric_H1_values_exact=all(r["numerical"]["numeric_pass"] for r in reached),
                   negative_observation_samples=sum(r["input"].get("observation_samples", 0)
                                                    for r in rows),
                   node_cost=PipelinedNormalizationService.estimate(),
                   old_numeric_and_resource_failures_retained=True,
                   all_RX_gates=policy["receiver_gates"], default_changed=False,
                   periodic_prefixes=prefixes, parent_manifests=parents)
    write(out / "model.json", summary)
    write(out / "failure-associations.json", [r for r in rows + regressions
          if "comparison" not in r or not r["comparison"]["after"]["payload_exact"]])
    if frozen is None:
        write(out / "freeze.json", dict(source=source, policy=policy, parent_manifests=parents,
              adoption="explicit only;no post-held tuning;four whole_RX gates remain separate"))
    assert source_vector() == source
    write(out / "manifest.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                                if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=("development", "validation"), default="development")
    p.add_argument("--freeze", type=Path)
    p.add_argument("--timing-root", type=Path)
    p.add_argument("--capture-root", type=Path)
    p.add_argument("--selected-root", type=Path)
    args = p.parse_args()
    if (args.split == "validation") != (args.freeze is not None):
        p.error("Validation requires the existing development freeze")
    if args.freeze is None and any(v is None for v in
                                   (args.timing_root, args.capture_root, args.selected_root)):
        p.error("Development requires all three frozen parents")
    raise SystemExit(run(args))
