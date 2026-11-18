"""Post-preparation finite waiting guard;exit1 preserves the incomplete RX gate."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_capture_fine import GEOMETRY, image_sha, replay
from model_spline_power import ObservedFlow, canonical_sha
from tests.test_spline_geometry import enumerate_geometry

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.capture_fine import CaptureFineWorker
from nearlink_sdr.sim.capture_transport import CaptureRequest, CaptureTransport
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SyncSignTemplate,
)

FIXTURE = Path("tests/data/spline_geometry_regression.json")
SCORE_EVENTS = ("sample", "score", "selected_sample")
EVENTS = (*SCORE_EVENTS, "geometry_interval", "geometry_complete", "geometry_wait")


def node_cases(validation=False):
    """Independent node sets change geometry,template and signal construction."""
    specs = (
        [
            (640, 126, (126,), 1010, 128),
            (392, 62, (72, 180), 0, 128),
            (272, 32, (144,), 0, 128),
            (64, 8, (3, 11), 0, 250),
            (80, 4, (9,), 0, 468),
            (80, 4, (9,), 0, 469),
            (96, 126, (0,), 0, 0),
            (256, 32, (112,), 2048, 128),
        ]
        if validation
        else [
            (256, 32, (128,), 0, 128),
            (376, 62, (96,), 0, 128),
            (632, 126, (122,), 0, 128),
            (48, 4, (2, 7), 0, 256),
            (32, 4, (2,), 0, 140),
            (32, 4, (2,), 0, 141),
            (64, 4, (60,), 0, 0),
            (40, 4, (), 0, 0),
        ]
    )
    rng = np.random.default_rng(740219 if validation else 740117)
    for k, (n, count, peaks, minimum, lookahead) in enumerate(specs):
        if count in (32, 62, 126):
            ft = {32: 2, 62: 3, 126: 4}[count]
            cfg = TxConfig(
                frame_type=ft,
                mcs_index=8,
                symbol_rate_mhz=4,
                pid=3 if validation else 2,
                ctrl_bits_len=28 if ft == 2 else 27,
                head_llid=None if ft == 2 else 0x234567,
            )
            tpl = SyncSignTemplate.from_symbols(PSKReceiveProfile.ordinary(cfg).sync_symbols())
        else:
            tpl = SyncSignTemplate(
                tuple(((1, 0), (0, 1), (-1, 0), (0, -1))[j % 4] for j in range(count - 1))
            )
        r, i = rng.integers(-18000, 18001, (2, n))
        if k % 4 == 0:
            r[:] = 0
            i[:] = 0
        elif k % 4 == 1:
            r[:] = 11017
            i[:] = -3311
        elif k % 4 == 2:
            r[n // 3 : n // 3 + 7] *= 3
        exponent = np.full(n, -3 if validation else -7)
        if k == 3:
            exponent[n // 2 :] += 5
        words = [(int(a), int(b), int(e)) for a, b, e in zip(r, i, exponent, strict=True)]
        yield (
            dict(id=k, n=n, symbols=count, validation=validation),
            words,
            tpl,
            peaks,
            dict(minimum_tick=minimum, lookahead_ticks=lookahead),
        )


def compare(out, name, words, template, peaks, kw):
    results, traces = {}, {}
    for guard in (False, True):
        trace = []
        result = ObservedFlow(
            SplineScoreAssumptions(len(words), reuse_symbol_power=True, defer_all_future=guard),
            trace,
            events=EVENTS,
        ).run(words, template, peaks, **kw)
        arm = "guarded" if guard else "power"
        results[guard], traces[guard] = result, trace
        write(out / (name + "-" + arm + ".json"), dict(microtrace=result, trace=trace))
    old, new = results.values()
    ticks, ready = enumerate_geometry(
        peaks,
        len(words),
        len(template.words) + 1,
        kw.get("minimum_tick", 0),
        kw.get("lookahead_ticks", 0),
    )
    deferred = bool(ticks) and not ready
    if old["numerical_rejection"]:
        assert old == new and all(
            e["event"] not in ("geometry_complete", "geometry_wait") for e in traces[True]
        )
        relation = "preparation_rejection_unchanged"
    else:
        assert new["geometry_positions"] == len(ticks) and new["geometry_wait"] == deferred
        assert new["geometry_cycles"] <= 32
        assert old["preparation_arithmetic"] == new["preparation_arithmetic"]
        assert old["prepared_source_words"] == new["prepared_source_words"]
        assert old["prepared_second_words"] == new["prepared_second_words"]
        if deferred:
            assert old["best"]["disposition"] == "maximum_needs_future"
            assert new["best"] is None and new["numerical_pass"] is None
            assert new["scores_evaluated"] == 0 and not new["scoring_evaluated"]
            assert not any(e["event"] in SCORE_EVENTS for e in traces[True])
            assert new["scoring_reads"] == 0
            relation = "not_evaluated;all_positions_need_future"
        else:
            assert old["best"] == new["best"]
            assert traces[False] == [e for e in traces[True] if e["event"] in SCORE_EVENTS]
            assert new["cycles"] - old["cycles"] == new["geometry_cycles"]
            relation = (
                "all_scored_words_scores_and_selection_exact"
                if ticks
                else "no_valid_position;not_scored"
            )
    return (
        dict(
            name=name,
            power_cycles=old["cycles"],
            guarded_cycles=new["cycles"],
            valid_positions=len(ticks),
            ready_positions=len(ready),
            numerical_rejection=old["numerical_rejection"],
            geometry_wait=new.get("geometry_wait", False),
            geometry_cycles=new.get("geometry_cycles"),
            relation=relation,
        ),
        results,
        traces,
    )


def run(args):
    out, capture = args.output, args.capture_root
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((capture / "manifest.json").read_text())
    assert all(sha(capture / k) == v for k, v in manifest.items())
    source = [
        Path(__file__),
        FIXTURE,
        Path("scripts/model_spline_power.py"),
        Path("scripts/model_capture_fine.py"),
        Path("tests/test_spline_geometry.py"),
        Path("tests/data/spline_score_saturation.json"),
        Path("pyproject.toml"),
        Path("uv.lock"),
        *Path("src").rglob("*.py"),
    ]
    write(
        out / "protocol.json",
        dict(
            stage="SYS-C2B",
            source={str(p): sha(p) for p in source},
            node_increment_cap_bits=128,
            transport_statistics_increment_bits=64,
            guard_cycle_cap=32,
            parent_capture_protocol=json.loads((capture / "protocol.json").read_text()),
            parent_capture_manifest_sha256=sha(capture / "manifest.json"),
            default_changed=False,
            new_RF_held=False,
            resource_feedback_to_receiver=False,
        ),
    )
    node_results = {}
    for validation in (False, True):
        split = "validation" if validation else "development"
        node_results[split] = []
        for meta, words, tpl, peaks, kw in node_cases(validation):
            name = split + "-" + str(meta["id"])
            np.savez_compressed(
                out / (name + "-input.npz"),
                words=np.asarray(words, np.int64),
                template=np.asarray(tpl.words, np.int8),
            )
            row, _, _ = compare(out, name, words, tpl, peaks, kw)
            row.update(
                meta=meta, peaks=peaks, config=kw, input_sha256=sha(out / (name + "-input.npz"))
            )
            node_results[split].append(row)
        write(out / (split + ".json"), node_results[split])
        if not validation:
            write(
                out / "postdevelopment-freeze.json",
                dict(
                    source={str(p): sha(p) for p in source},
                    condition="nonempty valid set and all positions future;anyready executes old score;prepare before guard",
                    node_cap=128,
                    transport_cap=64,
                    guard_cycles_cap=32,
                    new_validation_not_yet_generated=True,
                    no_RF_tuning=True,
                ),
            )
    frozen = json.loads(Path("tests/data/spline_score_saturation.json").read_text())
    reject, _, _ = compare(
        out,
        "fixed-saturation",
        [tuple(w) for w in frozen["words"]],
        SyncSignTemplate(((1, 0),)),
        [1],
        dict(lookahead_ticks=32768),
    )
    assert reject["numerical_rejection"] and not reject["geometry_wait"]
    fixture = json.loads(FIXTURE.read_text())
    regressions = []
    schedules = []
    costs = {}
    for row in json.loads((capture / "rows.json").read_text()):
        name, meta = row["id"], row["input"]
        profile = validate_profiles(
            [(meta["frame_type"], meta["mcs"], meta["pilot"], meta["length"])], meta["pid"]
        )[0]
        tpl = SyncSignTemplate.from_symbols(profile.sync_symbols())
        _, window, history = GEOMETRY[meta["frame_type"]]
        with np.load(capture / (name + "-filtered.npz")) as a:
            packed = a["words"]
        requests = []
        services = {False: [], True: []}
        entries = [e for e in fixture["cases"] if e["stream"] == name]
        assert len(entries) == row["requests"]
        for entry in entries:
            oldrecord = json.loads((capture / entry["file"]).read_text())
            q = oldrecord["request"].copy()
            q["peaks"] = tuple(q["peaks"])
            request = CaptureRequest(**q)
            requests.append(request)
            owned = packed[request.after_samples - window : request.after_samples, request.path]
            assert (
                image_sha(owned) == entry["source_sha256"]
                and meta["input_sha256"] == entry["raw_original_archive_sha256"]
            )
            current = {}
            traces = {}
            for guard in (False, True):
                result = []
                trace = []
                worker = CaptureFineWorker(
                    tpl,
                    window,
                    confirmation_required=True,
                    reuse_symbol_power=True,
                    defer_all_future=guard,
                    sink=lambda e, r: result.append(r),
                )
                worker.engine = ObservedFlow(worker.engine.a, trace, events=EVENTS)
                event = dict(
                    id=oldrecord["id"],
                    after_samples=request.after_samples,
                    path=request.path,
                    window_samples=window,
                    geometry=request,
                )
                service = worker(event, owned)
                current[guard] = result[0]
                traces[guard] = trace
                costs.setdefault(meta["frame_type"], {})[str(guard)] = worker.estimate()
                services[guard].append(dict(source_sha256=image_sha(owned), service=vars(service)))
                write(
                    out / (entry["file"][:-5] + ("-guarded.json" if guard else "-power.json")),
                    dict(
                        source_sha256=image_sha(owned),
                        request=q,
                        service=vars(service),
                        microtrace=result[0],
                        trace=trace,
                    ),
                )
            a, b = current.values()
            assert canonical_sha(a) == entry["power_microtrace_sha256"]
            ticks, ready = enumerate_geometry(
                q["peaks"], window, len(tpl.words) + 1, q["minimum_tick"], q["lookahead_ticks"]
            )
            assert b["geometry_wait"] == (bool(ticks) and not ready) and b["geometry_cycles"] <= 32
            assert a["preparation_arithmetic"] == b["preparation_arithmetic"]
            assert (
                a["prepared_source_words"] == b["prepared_source_words"]
                and a["prepared_second_words"] == b["prepared_second_words"]
            )
            if b["geometry_wait"]:
                assert (
                    a["best"]["disposition"] == "maximum_needs_future"
                    and b["best"] is None
                    and b["numerical_pass"] is None
                )
                assert (
                    not any(e["event"] in SCORE_EVENTS for e in traces[True])
                    and b["scoring_reads"] == 0
                )
                relation = "not_evaluated;all_positions_need_future"
            else:
                assert a["best"] == b["best"] and traces[False] == [
                    e for e in traces[True] if e["event"] in SCORE_EVENTS
                ]
                relation = "all_scored_words_scores_and_selection_exact"
            regressions.append(
                dict(
                    stream=name,
                    request=oldrecord["id"],
                    source_sha256=image_sha(owned),
                    power_cycles=a["cycles"],
                    guarded_cycles=b["cycles"],
                    geometry_cycles=b["geometry_cycles"],
                    geometry_wait=b["geometry_wait"],
                    score_relation=relation,
                )
            )
        for guard in (False, True):
            for clock in (100000000, 160000000, 200000000):
                s = replay(
                    out,
                    name + ("-guarded" if guard else "-power"),
                    packed,
                    requests,
                    services[guard],
                    clock,
                    history,
                    window,
                )
                s.update(stream=name, guard=guard)
                schedules.append(s)
    increments = [
        v["True"]["fine"]["additional_query_state_upper_bits"]
        - v["False"]["fine"]["additional_query_state_upper_bits"]
        for v in costs.values()
    ]
    transport_cost = CaptureTransport().estimate(measured_worker=True)
    transport_increment = transport_cost["register_reservation_bits"]["worker_statistics"] - 10 * 64
    total = dict(
        node_sets=node_results,
        regressions=regressions,
        schedules=schedules,
        costs=costs,
        fixed_saturation=reject,
        node_increment_bits=increments,
        node_budget_pass=all(0 <= x <= 128 for x in increments),
        transport_cost=transport_cost,
        transport_statistics_increment_bits=transport_increment,
        transport_budget_pass=0 <= transport_increment <= 64,
        guard_cycle_budget_pass=all(r["geometry_cycles"] <= 32 for r in regressions),
        deferred_numeric_status="not_evaluated",
        default_changed=False,
        new_RF_gain=None,
        old2percent16KiB_changed=False,
        whole_receiver16msps=False,
        resource_feedback_to_receiver=False,
    )
    write(out / "model.json", total)
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            dict(
                windows=len(regressions),
                geometry_wait=sum(r["geometry_wait"] for r in regressions),
                power_cycles=sum(r["power_cycles"] for r in regressions),
                guarded_cycles=sum(r["guarded_cycles"] for r in regressions),
                node_increment=increments,
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
