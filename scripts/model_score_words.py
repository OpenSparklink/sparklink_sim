"""Real word arithmetic/phase concurrency;exit1 retains incomplete RX acceptance."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import GEOMETRY, image_sha, replay
from model_spline_power import canonical_sha

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker
from nearlink_sdr.sim.capture_transport import CaptureRequest
from nearlink_sdr.sim.spline_score import SplineScoreAssumptions, SplineScoreFlow, SyncSignTemplate
from nearlink_sdr.sim.word_arithmetic import WordFabricAssumptions

SCORE_EVENTS = frozenset(("sample", "score", "selected_sample"))
FIXTURE = Path("tests/data/spline_geometry_regression.json")


def node_cases(validation=False):
    """Node validation,not a new RF held set;vary words,geometry and templates."""
    specs = (
        [(640, 4, (3, 70), "chirp"), (480, 3, (31, 120), "burst"),
         (272, 2, (12,), "weak"), (632, 4, (113,), "zero"),
         (384, 3, (55,), "tone"), (256, 2, (), "random")]
        if validation else
        [(632, 4, (5,), "random"), (376, 3, (6, 80), "tone"),
         (256, 2, (8, 50), "burst"), (640, 4, (4, 90), "weak"),
         (256, 2, (126,), "random"), (40, 2, (1,), "zero")]
    )
    rng = np.random.default_rng(820333 if validation else 820117)
    for k, (n, ft, peaks, kind) in enumerate(specs):
        cfg = TxConfig(
            frame_type=ft, mcs_index=8 if ft != 3 else 12, symbol_rate_mhz=4,
            pid=4 if validation else 1, ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
        )
        tpl = SyncSignTemplate.from_symbols(PSKReceiveProfile.ordinary(cfg).sync_symbols())
        t = np.arange(n)
        signal = rng.normal(size=n) + 1j * rng.normal(size=n)
        signal *= .15 if kind == "weak" else 1
        if kind == "zero":
            signal[:] = 0
        elif kind == "tone":
            signal += 3 * np.exp(1j * .43 * t)
        elif kind == "chirp":
            signal += 2 * np.exp(1j * (.12 * t + .00071 * t * t))
        elif kind == "burst":
            signal[n // 3:n // 3 + 9] *= 7
        words = [(round(z.real * 4096), round(z.imag * 4096), -7) for z in signal]
        kw = dict(minimum_tick=0, lookahead_ticks=128)
        yield dict(id=k, n=n, frame_type=ft, kind=kind, validation=validation), words, tpl, peaks, kw


def compare(out, name, words, tpl, peaks, kw, configurations):
    traces, results = {}, {}
    for lanes in configurations:
        trace = []

        def sink(e):
            if e["event"] in SCORE_EVENTS:
                trace.append({k: v for k, v in e.items() if k != "cycle"})

        a = SplineScoreAssumptions(len(words), reuse_symbol_power=True, defer_all_future=True)
        f = SplineScoreFlow(a, word_fabric=None if lanes == 0 else WordFabricAssumptions(lanes))
        r = f.run(words, tpl, peaks, sink=sink, **kw)
        results[lanes], traces[lanes] = r, trace
        write(out / f"{name}-alu{lanes}.json", dict(microtrace=r, trace=trace, cost=f.estimate()))
    baseline = results[0]
    for lanes in configurations[1:]:
        r = results[lanes]
        assert not r["resource_drop"] and not baseline["resource_drop"]
        assert r["best"] == baseline["best"] and traces[lanes] == traces[0]
        assert r["numerical_pass"] == baseline["numerical_pass"]
        assert r["prepared_source_words"] == baseline["prepared_source_words"]
        assert r["prepared_second_words"] == baseline["prepared_second_words"]
        if not r["numerical_rejection"]:
            assert r["operation_counts"] == baseline["operation_counts"]
            assert r["observed_node_bits"] == baseline["observed_node_bits"]
            assert r["word_fabric_idle"] and r["word_context_peak"] <= 9
    return dict(
        name=name, cycles={str(k): v["cycles"] for k, v in results.items()},
        geometry_wait=baseline.get("geometry_wait", False),
        evaluated=baseline.get("scoring_evaluated", False),
        numerical_pass=baseline["numerical_pass"], exact_all_scored_words_and_scores=True,
        trace_sha256=canonical_sha(traces[0]),
    ), results


def run(args):
    out, capture = args.output, args.capture_root
    out.mkdir(parents=True, exist_ok=False)
    parent = json.loads((capture / "manifest.json").read_text())
    assert all(sha(capture / k) == v for k, v in parent.items())
    source = [Path(__file__), FIXTURE, Path("uv.lock"), Path("pyproject.toml"), *Path("src").rglob("*.py")]
    protocol = dict(
        stage="SYS-C2C", source={str(p): sha(p) for p in source},
        parent_capture_manifest_sha256=sha(capture / "manifest.json"),
        budgets=dict(fabric_bits=8192, additional_context_and_fabric_bits=24576),
        rule="choose2 ALU if >=30% reduction in sum of evaluated development cycles;4 only if >=10% further;else reject;1 diagnostic",
        scope="node exactness+actual frozen capture window/calendar;not RF gain or complete RX",
        old16kib_and2percent_failures_retained=True, new_RF_held=False,
    )
    write(out / "protocol.json", protocol)
    development = []
    for meta, words, tpl, peaks, kw in node_cases():
        np.savez_compressed(out / f"development-{meta['id']}-input.npz", words=words, template=tpl.words)
        row, _ = compare(out, "development-" + str(meta["id"]), words, tpl, peaks, kw, (0, 1, 2, 4))
        development.append(row | dict(meta=meta, peaks=peaks, config=kw))
    sums = {k: sum(r["cycles"][str(k)] for r in development if r["evaluated"]) for k in (0, 1, 2, 4)}
    selected = 4 if sums[4] <= .9 * sums[2] else 2
    accepted = sums[selected] <= .7 * sums[0]
    write(out / "postdevelopment-freeze.json", protocol | dict(
        development=development, evaluated_cycle_sums=sums, selected_ALU=selected,
        experimental_node_adopted=accepted, independent_node_validation_not_yet_generated=True,
    ))
    validation = []
    if accepted:
        for meta, words, tpl, peaks, kw in node_cases(True):
            np.savez_compressed(out / f"validation-{meta['id']}-input.npz", words=words, template=tpl.words)
            row, _ = compare(out, "validation-" + str(meta["id"]), words, tpl, peaks, kw, (0, selected))
            validation.append(row | dict(meta=meta, peaks=peaks, config=kw))
    regressions, schedules, costs = [], [], {}
    fixture = json.loads(FIXTURE.read_text())
    if accepted:
        for row in json.loads((capture / "rows.json").read_text()):
            name, meta = row["id"], row["input"]
            p = validate_profiles([(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"])[0]
            tpl = SyncSignTemplate.from_symbols(p.sync_symbols())
            _, window, history = GEOMETRY[meta["frame_type"]]
            with np.load(capture / (name + "-filtered.npz")) as ar:
                packed = ar["words"]
            requests, services = [], {0: [], selected: []}
            for entry in (e for e in fixture["cases"] if e["stream"] == name):
                old = json.loads((capture / entry["file"]).read_text())
                q = old["request"].copy()
                q["peaks"] = tuple(q["peaks"])
                request = CaptureRequest(**q)
                requests.append(request)
                owned = packed[request.after_samples-window:request.after_samples, request.path]
                assert image_sha(owned) == entry["source_sha256"]
                assert meta["input_sha256"] == entry["raw_original_archive_sha256"]
                actual, traces = {}, {}
                for lanes in (0, selected):
                    results = []
                    traces[lanes] = []
                    class RecordedFlow(SplineScoreFlow):
                        def run(self, *args, **kwargs):
                            def record(e):
                                if e["event"] in SCORE_EVENTS:
                                    traces[lanes].append({k: v for k, v in e.items() if k != "cycle"})
                            return super().run(*args, sink=record, **kwargs)
                    worker = CaptureFineWorker(tpl, window, confirmation_required=True,
                        reuse_symbol_power=True, defer_all_future=True,
                        word_fabric=None if lanes == 0 else WordFabricAssumptions(lanes),
                        sink=lambda e, r: results.append(r))
                    worker.engine = RecordedFlow(worker.engine.a, word_fabric=worker.engine.word_fabric)
                    event = dict(id=old["id"], after_samples=request.after_samples,
                        path=request.path, window_samples=window, geometry=request)
                    service = worker(event, owned)
                    services[lanes].append(dict(source_sha256=image_sha(owned), service=vars(service)))
                    actual[lanes] = results[0]
                    costs.setdefault(str(meta["frame_type"]), {})[str(lanes)] = worker.estimate()
                    write(out / (entry["file"][:-5] + f"-alu{lanes}.json"),
                        dict(request=q, source_sha256=image_sha(owned), service=vars(service), microtrace=results[0], trace=traces[lanes]))
                a, b = actual.values()
                assert a["best"] == b["best"] and a["numerical_pass"] == b["numerical_pass"]
                assert traces[0] == traces[selected]
                if not a["numerical_rejection"]:
                    assert a["operation_counts"] == b["operation_counts"]
                    assert a["observed_node_bits"] == b["observed_node_bits"]
                check = dict(name=entry["file"][:-5], cycles={str(k): v["cycles"] for k,v in actual.items()},
                    evaluated=a.get("scoring_evaluated",False), geometry_wait=a.get("geometry_wait",False),
                    numerical_pass=a["numerical_pass"], exact_all_scored_words_and_scores=True,
                    trace_sha256=canonical_sha(traces[0]))
                regressions.append(check | dict(stream=name, request=old["id"], source_sha256=image_sha(owned)))
            for lanes in (0, selected):
                for clock in (100000000, 160000000, 200000000):
                    s = replay(out, name + f"-alu{lanes}", packed, requests, services[lanes], clock, history, window)
                    schedules.append(s | dict(ALU=lanes, clock=clock, stream=name))
    model = dict(protocol=protocol, development=development, validation=validation,
        selected_ALU=selected, experimental_node_adopted=accepted, costs=costs,
        regressions=regressions, schedules=schedules,
        all_scored_values_exact=True, default_RX_changed=False, whole_receiver16msps=False,
        full_system_resources_accepted=False,
        note="No new RF acceptance;unscored geometry waits remain unknown;resource drops separate;cost is register reservation and cycle model,no synthesis")
    write(out / "model.json", model)
    write(out / "manifest.json", {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()})
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
