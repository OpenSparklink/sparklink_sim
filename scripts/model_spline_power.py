"""Exact power reuse: new independent node sets and frozen actual capture requests.

Expect exit1: bounded capture replay/whole-RX gates remain incomplete even when
the integer node experiment passes. No new RF rate or default promotion.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from model_capture_fine import GEOMETRY, image_sha, replay
from tests.test_spline_power import node_cases
from tests.test_spline_score import oracle

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker
from nearlink_sdr.sim.capture_transport import CaptureRequest
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate

FIXTURE = Path("tests/data/spline_power_regression.json")


class ObservedFlow(SplineScoreFlow):
    """Host-only borrowed trace;execute the same node exactly once."""

    def __init__(self, assumptions, trace, *, events=("sample", "score", "selected_sample")):
        super().__init__(assumptions)
        self.trace = trace
        self.events = frozenset(events)

    def run(self, *args, **kwargs):
        def sink(event):
            if event["event"] in self.events:
                self.trace.append({k: v for k, v in event.items() if k != "cycle"})

        return super().run(*args, sink=sink, **kwargs)


def canonical_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_node(out, name, words, template, peaks, kw, *, independent):
    """All candidate scores/samples,not just the selected result or final CRC."""
    results, traces = {}, {}
    for reuse in (False, True):
        events = []

        def sink(event):
            if event["event"] in ("sample", "score", "selected_sample"):
                events.append({k: v for k, v in event.items() if k != "cycle"})

        arm = "reused" if reuse else "baseline"
        result = SplineScoreFlow(SplineScoreAssumptions(len(words), reuse_symbol_power=reuse)).run(
            words, template, peaks, sink=sink, **kw
        )
        results[arm], traces[arm] = result, events
        write(out / f"{name}-{arm}.json", result)
    old, new = results.values()
    assert old["best"] == new["best"] and old["numerical_rejection"] == new["numerical_rejection"]
    if old["numerical_rejection"]:
        assert old == new
    else:
        assert old["prepared_source_words"] == new["prepared_source_words"]
        assert old["prepared_second_words"] == new["prepared_second_words"]
    assert traces["baseline"] == traces["reused"]
    assert old["scoring_reads"] == new["scoring_reads"]
    assert not old["resource_drop"] and not new["resource_drop"]
    if independent and not old["numerical_rejection"]:
        _, best, samples = oracle(
            words,
            template,
            peaks,
            kw.get("minimum_tick", 0),
            kw.get("lookahead_ticks", 0),
            kw.get("threshold", (17, 20)),
        )
        assert new["best"] == best
        for event in traces["reused"]:
            if event["event"] == "sample":
                assert event["word"] == samples[event["start_tick"]][event["symbol"]]
            elif event["event"] == "score":
                y = samples[event["start_tick"]]
                assert event["energy"] == sum(
                    (a * r + b * i) ** 2 + (b * r - a * i) ** 2
                    for (a, b), (r, i) in zip(y[1:], y[:-1], strict=True)
                )
    summary = dict(
        name=name,
        baseline_cycles=old["cycles"],
        reused_cycles=new["cycles"],
        baseline_tiles=old.get("operation_counts", {}).get("tile", 0),
        reused_tiles=new.get("operation_counts", {}).get("tile", 0),
        every_score_word_and_selection_exact=True,
        numerical_rejection=old["numerical_rejection"],
        independent_fraction_and_difference_square_checked=independent
        and not old["numerical_rejection"],
        reused_monitor_nodes=len(new.get("declared_node_bits", {})),
    )
    if old["best"] is not None:
        assert summary["reused_cycles"] < summary["baseline_cycles"]
    return summary, results


def run(args):
    out, capture = args.output, args.capture_root
    out.mkdir(parents=True, exist_ok=False)
    parent_manifest = json.loads((capture / "manifest.json").read_text())
    assert all(sha(capture / k) == v for k, v in parent_manifest.items())
    source = [
        Path(__file__),
        FIXTURE,
        Path("tests/test_spline_power.py"),
        Path("tests/test_spline_score.py"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        *Path("src").rglob("*.py"),
    ]
    write(
        out / "protocol.json",
        dict(
            stage="SYS-C2A",
            source={str(p): sha(p) for p in source},
            parent_input_manifest_sha256=sha(capture / "manifest.json"),
            parent_capture_protocol=json.loads((capture / "protocol.json").read_text()),
            incremental_state_study_cap_bits=444,
            monitor_format="original640bit reservation=80x8bit;66/71 active slots",
            new_RF_parameters=False,
            new_RF_held=False,
            runtime_CRC_or_truth_choice=False,
            default_changed=False,
            integer_gate="all scores/words/selection/guards exact",
            resource_gate="same bounded calendar;do not infer whole-RX from node cost",
        ),
    )
    node_results = {}
    for validation in (False, True):
        split = "validation" if validation else "development"
        node_results[split] = []
        for meta, words, template, peaks, kw in node_cases(validation):
            name = f"{split}-{meta['id']}"
            np.savez_compressed(
                out / (name + "-input.npz"),
                words=np.asarray(words, np.int64),
                template=np.asarray(template.words, np.int8),
            )
            row, _ = run_node(out, name, words, template, peaks, kw, independent=True)
            row.update(
                meta=meta, input_sha256=sha(out / (name + "-input.npz")), peaks=peaks, config=kw
            )
            node_results[split].append(row)
        write(out / (split + ".json"), node_results[split])
        if not validation:
            write(
                out / "postdevelopment-freeze.json",
                dict(
                    source={str(p): sha(p) for p in source},
                    structure="signed22norm44-product44;fullgrowth;444extra bits,one scoreDSP,no RAMport",
                    no_parameter_tuning=True,
                    new_node_validation_not_yet_generated=True,
                    adoption="exact all values,guards;lower nonempty cycles;resource boundaries retained",
                ),
            )
    fixture = json.loads(FIXTURE.read_text())
    regressions, schedules = [], []
    rows = json.loads((capture / "rows.json").read_text())
    for row in rows:
        name, meta = row["id"], row["input"]
        entries = [e for e in fixture["cases"] if e["stream"] == name]
        assert len(entries) == row["requests"]
        cfg = TxConfig(
            frame_type=meta["frame_type"],
            mcs_index=meta["mcs"],
            pid=meta["pid"],
            symbol_rate_mhz=4,
            ctrl_bits_len=28 if meta["frame_type"] == 2 else 27,
            head_llid=None if meta["frame_type"] == 2 else 0x234567,
        )
        template = SyncSignTemplate.from_symbols(PSKReceiveProfile.ordinary(cfg).sync_symbols())
        _, window, history = GEOMETRY[meta["frame_type"]]
        services = {False: [], True: []}
        requests = []
        with np.load(capture / (name + "-filtered.npz")) as ar:
            packed_source = ar["words"]
        for entry in entries:
            old = json.loads((capture / entry["file"]).read_text())
            assert entry["raw_original_archive_sha256"] == meta["input_sha256"]
            assert canonical_sha(old["microtrace"]) == entry["microtrace_sha256"]
            req = old["request"].copy()
            req["peaks"] = tuple(req["peaks"])
            request = CaptureRequest(**req)
            requests.append(request)
            words = packed_source[
                request.after_samples - window : request.after_samples, request.path
            ]
            assert image_sha(words) == old["source_sha256"] == entry["source_sha256"]
            current, traces = {}, {}
            for reuse in (False, True):
                result = []
                worker = CaptureFineWorker(
                    template,
                    window,
                    confirmation_required=True,
                    reuse_symbol_power=reuse,
                    sink=lambda e, r: result.append(r),
                )
                traces[reuse] = []
                worker.engine = ObservedFlow(worker.engine.a, traces[reuse])
                service = worker(
                    dict(
                        id=old["id"],
                        after_samples=request.after_samples,
                        path=request.path,
                        window_samples=window,
                        geometry=request,
                    ),
                    words,
                )
                current[reuse] = result[0]
                if not reuse:
                    assert canonical_sha(result[0]) == entry["microtrace_sha256"]
                services[reuse].append(dict(source_sha256=image_sha(words), service=vars(service)))
                write(
                    out
                    / (
                        entry["file"].replace(
                            ".json", "-reused.json" if reuse else "-baseline.json"
                        )
                    ),
                    dict(
                        request=req,
                        source_sha256=image_sha(words),
                        service=vars(service),
                        microtrace=result[0],
                        score_and_sample_trace=traces[reuse],
                    ),
                )
            a, b = current.values()
            assert traces[False] == traces[True]
            assert a["best"] == b["best"] and a["numerical_rejection"] == b["numerical_rejection"]
            assert a["prepared_source_words"] == b["prepared_source_words"]
            assert a["prepared_second_words"] == b["prepared_second_words"]
            assert a["scoring_reads"] == b["scoring_reads"]
            regressions.append(
                dict(
                    stream=name,
                    request=old["id"],
                    source_sha256=image_sha(words),
                    baseline_cycles=a["cycles"],
                    reused_cycles=b["cycles"],
                    baseline_tiles=a["operation_counts"]["tile"],
                    reused_tiles=b["operation_counts"]["tile"],
                    best_and_prepared_words_exact=True,
                    every_candidate_score_and_sample_exact=True,
                    disposition=services[True][-1]["service"]["disposition"],
                )
            )
        for reuse in (False, True):
            for clock in (100000000, 160000000, 200000000):
                s = replay(
                    out,
                    name + ("-reused" if reuse else "-baseline"),
                    packed_source,
                    requests,
                    services[reuse],
                    clock,
                    history,
                    window,
                )
                s.update(stream=name, reuse=reuse)
                schedules.append(s)
    cost = {}
    for ft, (_, window, _) in GEOMETRY.items():
        cost[ft] = {}
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=8,
            symbol_rate_mhz=4,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
        )
        template = SyncSignTemplate.from_symbols(PSKReceiveProfile.ordinary(cfg).sync_symbols())
        for reuse in (False, True):
            worker = CaptureFineWorker(
                template, window, confirmation_required=True, reuse_symbol_power=reuse
            )
            # Cost depends on compiled window,not the template sign values.
            cost[ft][str(reuse)] = worker.estimate()
    increments = [
        c["True"]["fine"]["additional_query_state_upper_bits"]
        - c["False"]["fine"]["additional_query_state_upper_bits"]
        for c in cost.values()
    ]
    total = dict(
        node_sets=node_results,
        regressions=regressions,
        schedules=schedules,
        costs=cost,
        monitor_interpretation="old40x16 reservation packs80x8bit;no undercharge;prior wrong analysis retained",
        initial444bit_gate_pass=all(0 <= n <= 444 for n in increments),
        actual_incremental_state_bits=increments,
        node_numerical_rejections=[
            dict(split=s, **r)
            for s, rows in node_results.items()
            for r in rows
            if r["numerical_rejection"]
        ],
        all_completed_node_values_exact=True,
        old_default_microtraces_exact=True,
        all_actual_candidate_scores_and_samples_exact=True,
        default_changed=False,
        new_RF_gain=None,
        whole_receiver16msps=False,
        original2percent16KiB_changed=False,
    )
    write(out / "model.json", total)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            dict(
                node_development=len(node_results["development"]),
                node_validation=len(node_results["validation"]),
                actual_windows=len(regressions),
                all_exact=True,
                whole_RX=False,
            ),
            indent=2,
        )
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
