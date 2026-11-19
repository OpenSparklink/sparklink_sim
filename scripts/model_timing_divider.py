"""Exact registered quotient-prefix reduction; exit1 retains four open RX gates.

Only the divider schedule changes. The upstream whole-record capture fixture is
not a bounded live producer. Header/residual carrier/LLR/decoder cycles stay unknown.
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
from model_normalization_words import FMT, NORM, execute, periodic_prefixes, raw_trace
from model_selected_frame import TRACE_KEYS, metrics, normalized
from model_timing_handoff import fixed, fresh

from nearlink_sdr.sim.overlapped_normalization_replay import OverlappedNormalizationFrameReplay

POLICY = Path("tests/data/timing_divider_policy.json")


def source_vector():
    return {str(p): sha(p) for p in sorted({
        POLICY, Path("uv.lock"), *Path("src").rglob("*.py"), *Path("tests").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    })}


def pair(out, name, profile, words, selection, payload, policy, *, partitions):
    events, flows, traces, services = [], [], [], []
    for mode in ("full", "bounded-prefix"):
        observed = []
        event, flow, trace = execute(
            out, name + "-" + mode, profile, words, selection, FMT, [32],
            flow_class=partial(OverlappedNormalizationFrameReplay, normalization_format=NORM,
                               timing_divider=mode, service_observer=observed.append),
        )
        with gzip.open(out / (name + "-" + mode + "-services.jsonl.gz"), "wt") as f:
            for item in observed:
                f.write(json.dumps(normalized(item), separators=(",", ":")) + "\n")
        events.append(event)
        flows.append(flow)
        traces.append(trace)
        services.append(observed)
    a, b = events
    assert normalized(a["frame"]) == normalized(b["frame"])
    assert raw_trace(traces[0]) == raw_trace(traces[1])
    assert [r["values"] for r in services[0]] == [r["values"] for r in services[1]]
    for key in TRACE_KEYS:
        np.testing.assert_array_equal(a[key], b[key])
    for key in ("word_timing_state", "word_normalization_state"):
        assert a[key] == b[key]
    normal = flows[1].timing_service.normal
    checks = normal.stats["prefix_checks"] if normal else 0
    assert flows[0].cycle - flows[1].cycle == checks * 29  # 30 removed zero bits,1 guard.
    if partitions:
        for label, pattern in [("fixed", [7, 31]),
                               ("random", np.random.default_rng(1141039).integers(1, 33, 23))]:
            event, flow, trace = execute(
                out, name + "-" + label, profile, words, selection, FMT, pattern,
                flow_class=partial(OverlappedNormalizationFrameReplay, normalization_format=NORM,
                                   timing_divider="bounded-prefix"),
            )
            assert normalized(event["frame"]) == normalized(b["frame"])
            assert trace == traces[1] and flow.cycle == flows[1].cycle
            for key in TRACE_KEYS:
                np.testing.assert_array_equal(event[key], b[key])
    costs = [f.estimate()["timing_service"] for f in flows]
    budget = policy["divider_budgets"]
    assert costs[1]["state_register_bits_upper"] - costs[0]["state_register_bits_upper"] \
        <= budget["additional_state_bits_max"]
    assert costs[0]["DSP25x18"] == costs[1]["DSP25x18"] == 4
    assert costs[1]["extra_raw_RAM_bits"] == costs[1]["additional_source_RAM_ports"] == 0
    assert costs[1]["divider"]["extra_operand_mux_bits_upper"] \
        <= budget["additional_operand_mux_bits_upper_max"]
    before, after = [metrics(e, profile, expected=payload) for e in events]
    return dict(id=name, before=before, after=after,
                source_words_sha256=hashlib.sha256(
                    np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
                numerical=dict(numeric_pass=True, word_state_and_trajectory_delta=0),
                success_loss=before["payload_exact"] and not after["payload_exact"],
                recovery=after["payload_exact"] and not before["payload_exact"],
                false_delivery=after["false_delivery"] and not before["false_delivery"],
                partitions_checked=partitions, timing_cost_before=costs[0], timing_cost=costs[1],
                prefix_checks=checks, actual_modeled_edges=[f.cycle for f in flows],
                modeled_edge_saving=flows[0].cycle - flows[1].cycle,
                service_records=len(services[1]), full_receiver16MSps=False,
                complete_latency_cycles=None)


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
        for mode in ("full", "bounded-prefix"):
            path = out / ("continuous-" + mode)
            path.mkdir()
            prefixes[mode] = periodic_prefixes(path, flow_class=partial(
                OverlappedNormalizationFrameReplay, timing_divider=mode))
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
                   values_exact=all(r["numerical"]["numeric_pass"] for r in reached),
                   negative_observation_samples=sum(r["input"].get("observation_samples", 0)
                                                    for r in rows),
                   old_numeric_and_resource_failures_retained=True,
                   all_RX_gates=policy["receiver_gates"], default_changed=False,
                   periodic_prefixes=prefixes, parent_manifests=parents)
    write(out / "model.json", summary)
    write(out / "failure-associations.json", [r for r in rows + regressions
          if "comparison" not in r or not r["comparison"]["after"]["payload_exact"]])
    if frozen is None:
        write(out / "freeze.json", dict(source=source, policy=policy, parent_manifests=parents,
              adoption="explicit only;exact values;no post-held tuning;four RX gates open"))
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
        p.error("Validation requires existing development freeze")
    if args.freeze is None and any(v is None for v in
                                   (args.timing_root, args.capture_root, args.selected_root)):
        p.error("Development requires all three frozen parents")
    raise SystemExit(run(args))
