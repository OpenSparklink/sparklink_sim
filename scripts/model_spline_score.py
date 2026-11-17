"""Freeze/replay finite full-word fine scores; whole-system gate returns1."""

import argparse
import gzip
import json
import math
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from audit_spline_fixed import compare_stream, independent_packet, receive, summarize
from tests.test_spline_score import oracle
from tests.test_strict_receive import iq_record

from nearlink_sdr.phy import sync_candidates
from nearlink_sdr.phy.spline_fixed import FixedUniformCubicEighth, SplineFixedFormat
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.phy.uniform_spline import _DIGITS
from nearlink_sdr.sim.capture_transport import CaptureTransport, CaptureTransportAssumptions
from nearlink_sdr.sim.spline_score import (
    SplineScoreAssumptions,
    SplineScoreFlow,
    SyncSignTemplate,
    spline_score_cost,
)

FMT = SplineFixedFormat(18, 14)


def packets(held=False):
    """New contents/profile/channel combinations; existing sets are not retuned."""
    rng = np.random.default_rng(680117 if held else 670117)
    configs = (
        [(4, 8, 8), (2, 12, 8), (3, 0, 0), (4, 6, 8), (2, 8, 0), (3, 12, 8), (4, 12, 0), (2, 6, 0)]
        if held
        else [
            (2, 8, 0),
            (3, 12, 8),
            (4, 0, 0),
            (2, 6, 8),
            (3, 8, 0),
            (4, 12, 8),
            (2, 12, 0),
            (4, 8, 0),
        ]
    )
    lengths = [37, 157, 337, 701] if held else [23, 83, 211, 503]
    for j, (ft, mcs, pilot) in enumerate(configs):
        cfg = TxConfig(
            frame_type=ft,
            mcs_index=mcs,
            pilot_interval=pilot,
            pid=5 if held else 2,
            whitening_seed=0x52,
            crc_seed=0x123456,
            ctrl_bits_len=28 if ft == 2 else 27,
            head_llid=None if ft == 2 else 0x234567,
            symbol_rate_mhz=4,
        )
        p = PSKReceiveProfile.ordinary(cfg)
        length = lengths[j % 4]
        content = (
            (np.arange(length) * 91 + j * 37) % 256
            if held and j % 2
            else np.resize([0, 128, 255, 17, 1], length)
            if j % 3 == 0
            else rng.integers(0, 256, length)
        )
        payload = np.asarray(content, np.uint8).tobytes()
        cfo, ppm = (
            ((-37000, 50) if held else (23000, -75))
            if j % 2
            else ((17000, -100) if held else (-43000, 25))
        )
        clean, _, truth, step = iq_record(
            independent_packet(p, payload), cfo=cfo, ppm=ppm, offset=0.417 if held else 0.763
        )
        prefix = 193 + j * 19 if held else 137 + j * 11
        n, sigma = len(clean) + prefix + 256, (1.6 if j < 4 else 2.2)
        quiet = sigma / np.sqrt(2) * (rng.normal(size=2048) + 1j * rng.normal(size=2048))
        x = sigma / np.sqrt(2) * (rng.normal(size=n) + 1j * rng.normal(size=n))
        kind = ("noise", "tone", "wrong_sync", "burst")[j % 4]
        k = np.arange(n)
        if kind == "tone":
            x += 3.7 * np.exp((0.079117 if held else -0.129311) * 2j * np.pi * k + 0.29j)
            if held:
                x += 0.7 * np.exp(-0.1731 * 2j * np.pi * k)
        if kind == "wrong_sync":
            other = PSKReceiveProfile.ordinary(
                TxConfig(
                    mcs_index=8, pid=7, whitening_seed=0x52, crc_seed=0x123456, symbol_rate_mhz=4
                )
            )
            wrong, _, _, _ = iq_record(
                independent_packet(other, b"new wrong-sync content 670117"),
                cfo=-11000,
                offset=0.213,
            )
            x += 0.06 * np.resize(wrong, n)
        if kind == "burst":
            for start in range(233, n, 1237 if held else 1669):
                count = min(7 if held else 9, n - start)
                x[start : start + count] += 28 * np.exp(1j * rng.uniform(-np.pi, np.pi, count))
        background = x.copy()
        amplitude = (0.16, 0.4, 0.65)[j % 3]
        x[prefix : prefix + len(clean)] += amplitude * clean
        if held and kind == "burst":
            x[n // 2 :] *= 0.6
            background[n // 2 :] *= 0.6
        meta = dict(
            id=("validation" if held else "dev") + f"-{j}",
            positive=True,
            frame_type=ft,
            mcs=mcs,
            pilot=pilot,
            pid=cfg.pid,
            length=length,
            kind=kind,
            amplitude=amplitude,
            cfo=cfo,
            ppm=ppm,
            truth=prefix + truth,
            step=step,
        )
        yield meta, p, x, quiet, payload
        if j < 4:
            yield (
                {**meta, "id": meta["id"] + "-negative", "positive": False},
                p,
                background,
                quiet,
                b"",
            )


def dataset(held, frozen=None):
    if frozen is None:
        cases = packets(held)
    else:
        cases = []
        prefix = "validation" if held else "dev"
        for path in sorted(frozen.glob(prefix + "-*-metadata.json")):
            meta = json.loads(path.read_text())
            if "word" in meta["id"]:
                continue
            cfg = TxConfig(
                frame_type=meta["frame_type"],
                mcs_index=meta["mcs"],
                pilot_interval=meta["pilot"],
                pid=meta["pid"],
                whitening_seed=0x52,
                crc_seed=0x123456,
                ctrl_bits_len=28 if meta["frame_type"] == 2 else 27,
                head_llid=None if meta["frame_type"] == 2 else 0x234567,
                symbol_rate_mhz=4,
            )
            profile = PSKReceiveProfile.ordinary(cfg)
            with np.load(frozen / (meta["id"] + "-input.npz")) as ar:
                cases.append((meta, profile, ar["iq"], ar["quiet"], ar["payload"].tobytes()))
    for meta, profile, x, quiet, payload in cases:
        # Reporting aliases required by the shared postselection BER/truth helper.
        # Never passed to capture,ranking,carrier,or the adopt decision.
        yield (
            {**meta, "sync_start": meta["truth"], "cfo_hz": meta["cfo"]},
            profile,
            x,
            quiet,
            payload,
        )


def node_cases(output, held=False, frozen=None):
    rng = np.random.default_rng(680119 if held else 670119)
    rows, arrays = [], {}
    for j in range(6):
        n = (47, 127, 255, 377, 631, 640)[j] if held else (32, 64, 128, 256, 376, 632)[j]
        r, i = rng.integers(-(2**18), 2**18, (2, n))
        e = rng.integers(-70, 70, n)
        if j == 0:
            r[:], i[:] = 0, 0
        if j == 1:
            e[n // 2 :] -= 40
        words = tuple((int(a), int(b), int(c)) for a, b, c in zip(r, i, e, strict=True))
        tpl = SyncSignTemplate(
            (((1, 1), (-1, 1), (1, -1), (-1, -1)) if j % 2 else ((0, 1), (0, -1))) * 3
        )
        ident = ("validation" if held else "dev") + f"-word-{j}"
        if frozen is not None:
            with np.load(frozen / (("validation" if held else "dev") + "-word-inputs.npz")) as old:
                words = tuple(tuple(map(int, w)) for w in old[ident])
        peaks = [2] if j % 2 else [2, 6]
        reference, best, samples = oracle(words, tpl, peaks)
        records = []
        for reuse in (True,) if held else (False, True):
            selected = []
            with (
                gzip.open(output / (ident + "-trace.jsonl.gz"), "wt")
                if j == 0 and reuse and not held
                else nullcontext()
            ) as handle:

                def observe(event):
                    if event["event"] == "selected_sample":
                        selected.append(event["word"])
                    if handle is not None:
                        handle.write(json.dumps(event, sort_keys=True) + "\n")

                result = SplineScoreFlow(SplineScoreAssumptions(max(256, n), reuse)).run(
                    words, tpl, peaks, sink=observe
                )
            assert result["best"] == best and not result["aborted"]
            assert result["prepared_second_words"] == reference.second_words
            assert selected == ([] if best is None else samples[best["start_tick"]])
            records.append(result)
        if not held:
            assert records[0]["best"] == records[1]["best"]
            assert records[0]["operation_counts"] == records[1]["operation_counts"]
            assert records[1]["scoring_reads"] < records[0]["scoring_reads"]
        ident = ("validation" if held else "dev") + f"-word-{j}"
        write(output / (ident + ".json"), records)
        arrays[ident] = np.asarray(words, np.int64)
        rows.append(
            dict(
                id=ident,
                exact=True,
                n=n,
                reads=[r["scoring_reads"] for r in records],
                cycles=[r["cycles"] for r in records],
            )
        )
    np.savez_compressed(output / (("validation" if held else "dev") + "-word-inputs.npz"), **arrays)
    return rows


class ReceiveAdapter:
    """Explicit numerical model bridge; reference runs afterwards for metrics only."""

    def __init__(self, folder):
        self.folder, self.rows, self.results = folder, [], []
        self.original = sync_candidates.search_candidate_grid

    def __call__(self, *args, **kwargs):
        if kwargs.get("interpolator") != "word-eighth":
            return self.original(*args, **kwargs)
        z, _, ref, _ = args
        coarse = kwargs.get("coarse_starts")
        if coarse is None:
            raise ValueError("This experiment requires actual bounded coarse-bank peaks")
        boundary = FixedUniformCubicEighth(z, FMT)
        chosen, scores = [], []

        def sink(event):
            if event["event"] == "selected_sample":
                chosen.append(event["word"])
            elif event["event"] == "score":
                scores.append({k: v for k, v in event.items() if k not in ("cycle", "event")})

        flow = SplineScoreFlow(SplineScoreAssumptions(max(256, len(z))))
        result = flow.run(
            boundary.source_words,
            SyncSignTemplate.from_symbols(ref),
            [int(p) for p in coarse],
            exponent=boundary.exponent,
            minimum_tick=round(kwargs.get("min_start", 0) * 8),
            lookahead_ticks=round(kwargs.get("lookahead", 0) * 8),
            sink=sink,
        )
        assert not result["aborted"]
        assert result["prepared_source_words"] == boundary.source_words
        assert result["prepared_second_words"] == boundary.second_words
        best = result["best"]
        # Collect a separate postselection shadow; never return it or retry its CRC.
        d = {}
        old = self.original(*args, **{**kwargs, "diagnostics": d})
        score = (
            (math.sqrt(best["numerator"] / best["denominator"]) if best["denominator"] else 0.0)
            if best
            else None
        )
        start = best["start_tick"] / 8 if best else None
        error = (
            abs(score - d["best_score"])
            if score is not None and d["best_score"] is not None
            else None
        )
        start_error = (
            abs(start - d["best_start"]) if start is not None and "best_start" in d else None
        )
        row = dict(
            index=len(self.rows),
            scores=scores,
            new=best,
            old_start=d.get("best_start"),
            old_score=d["best_score"],
            score_error=error,
            start_error=start_error,
            passed=(error is None or error <= 0.005)
            and (start_error is None or start_error <= 0.125),
            old_ready=old is not None,
            cycles=result["cycles"],
            scoring_reads=result["scoring_reads"],
        )
        self.rows.append(row)
        self.results.append(result)
        target = kwargs["diagnostics"]
        target.update(d)
        target.update(
            best_score=score, disposition=best["disposition"] if best else "no_legal_window"
        )
        if best:
            target["best_start"] = start
        if best is None or best["disposition"] != "ready":
            return None
        y = np.array(
            [
                complex(
                    math.ldexp(a, result["common_exponent"]),
                    math.ldexp(b, result["common_exponent"]),
                )
                for a, b in chosen
            ]
        )
        return dict(
            score=score,
            phase=start,
            offset=0,
            delta=math.atan2(best["correlation"][1], best["correlation"][0]),
            phase_symbols=y,
        )

    def save(self):
        write(self.folder / "scores.json", self.rows)
        write(self.folder / "flow-results.json", self.results)


def receive_case(output, meta, profile, x, quiet, payload, chunking="whole"):
    base_path = output / (meta["id"] + "-reference-" + chunking)
    base, arrays, _ = receive(
        base_path, meta, profile, x, quiet, payload, fmt=FMT, chunking=chunking
    )
    folder = output / (meta["id"] + "-score-" + chunking)
    adapter = ReceiveAdapter(folder)
    with patch.object(sync_candidates, "search_candidate_grid", adapter):
        row, values, _ = receive(
            folder, meta, profile, x, quiet, payload, fmt=FMT, chunking=chunking
        )
    adapter.save()
    comparison = compare_stream(base, row, arrays, values)
    comparison["exact_score_budget_pass"] = all(r["passed"] for r in adapter.rows)
    comparison["passed"] &= comparison["exact_score_budget_pass"]
    return base, row, comparison, adapter.results


def floor_boundary(output):
    # Diagnostic compatibility counterexample, not RF capability or algorithm tuning.
    records = []
    for exponent in (0, -120):
        words = [(100, 0, exponent)] * 32
        template = SyncSignTemplate(((1, 0),) * 3)
        result = SplineScoreFlow(SplineScoreAssumptions(32)).run(words, template, [2])
        boundary = FixedUniformCubicEighth.from_rrc_words(*zip(*words, strict=True), FMT)
        y = boundary(result["best"]["start_tick"] / 8 + 4 * np.arange(4))
        dy = y[1:] * y[:-1].conj()
        old = abs(np.sum(dy)) / max(np.sqrt(np.sum(abs(dy) ** 2) * 3), 1e-30)
        new = math.sqrt(result["best"]["numerator"] / result["best"]["denominator"])
        records.append(
            dict(
                input_exponent=exponent,
                old_float_score=float(old),
                exact_word_score=new,
                score_error=abs(new - float(old)),
                float_score_budget_pass=abs(new - float(old)) <= 0.005,
                meaning="representable raw-port scaling;not calibrated ADC precision or recoverable RF packet",
            )
        )
    write(output / "normalization-floor-failure.json", records)
    assert records[0]["float_score_budget_pass"] and not records[1]["float_score_budget_pass"]
    return records


def run(output, frozen=None, manifest_sha=None):
    output.mkdir(parents=True, exist_ok=False)
    if frozen is not None:
        assert manifest_sha is not None and sha(frozen / "manifest.json") == manifest_sha
        original_manifest = json.loads((frozen / "manifest.json").read_text())
        assert all(sha(frozen / k) == v for k, v in original_manifest.items())
        write(
            output / "frozen-input-provenance.json",
            dict(
                root=str(frozen),
                manifest_sha256=manifest_sha,
                status="first validation now fixed regression;no signal parameter/algorithm retuning",
                diagnostic_alias_correction="truth->sync_start,cfo->cfo_hz;originalinput and payload bytes unchanged",
            ),
        )
    sources = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            Path("scripts/audit_spline_fixed.py"),
            Path("tests/test_spline_score.py"),
            Path("tests/data/spline_fixed_format.json"),
            Path("tests/data/sync_carrier_tone_success.npz"),
            Path("uv.lock"),
            *Path("src").rglob("*.py"),
        )
    }
    write(
        output / "protocol.json",
        dict(
            source=sources,
            expected_default_rx_change=False,
            no_crc_or_truth_selection=True,
            score_limit=0.005,
            start_limit=0.125,
            center_limit=0.25,
            word_reference="independent Fraction spline weights/rational score",
            real_iq_stream_model_scope="numeric aligned18 boundary;phase_symbols bridge float,not clocked ADC/hardware coherent confirmation",
            adoption="reuse if exact word score and fewer reads;full-growth scorer experimental,resource and RF gates separate",
        ),
    )
    floor = floor_boundary(output)
    nodes = node_cases(output, frozen=frozen)
    all_rows, comparisons, flows, bases = [], [], [], []
    for meta, profile, x, quiet, payload in dataset(False, frozen):
        path = output / (meta["id"] + "-input.npz")
        np.savez_compressed(path, iq=x, quiet=quiet, payload=np.frombuffer(payload, np.uint8))
        meta["frozen_source_input_sha256"] = meta.get("input_sha256")
        meta["input_sha256"] = sha(path)
        write(output / (meta["id"] + "-metadata.json"), meta)
        base, row, comparison, result = receive_case(output, meta, profile, x, quiet, payload)
        bases.append(base)
        all_rows.append(row)
        comparisons.append(comparison)
        flows += result
    write(
        output / "development.json",
        dict(
            nodes=nodes,
            reference=summarize(bases),
            score=summarize(all_rows),
            comparisons=comparisons,
        ),
    )
    dev_pass = all(c["passed"] for c in comparisons)
    write(
        output / "selection.json",
        dict(
            frozen=True,
            reuse_stencil=True,
            format=vars(FMT),
            source=sources,
            signal_parameters_changed=False,
            development_pass=dev_pass,
            new_validation_allowed=dev_pass,
            scorer_default_promotion=False,
        ),
    )
    held_nodes, held_rows, held_comparisons, held_bases = [], [], [], []
    if dev_pass:
        held_nodes = node_cases(output, True, frozen=frozen)
        for meta, profile, x, quiet, payload in dataset(True, frozen):
            path = output / (meta["id"] + "-input.npz")
            np.savez_compressed(path, iq=x, quiet=quiet, payload=np.frombuffer(payload, np.uint8))
            meta["frozen_source_input_sha256"] = meta.get("input_sha256")
            meta["input_sha256"] = sha(path)
            write(output / (meta["id"] + "-metadata.json"), meta)
            for chunking in ("whole", "fixed", "random"):
                base, row, c, result = receive_case(
                    output, meta, profile, x, quiet, payload, chunking
                )
                held_bases.append(base)
                held_rows.append(row)
                held_comparisons.append(c)
                flows += result
    write(
        output / "validation.json",
        dict(
            nodes=held_nodes,
            reference=summarize(held_bases) if held_bases else None,
            score=summarize(held_rows) if held_rows else None,
            comparisons=held_comparisons,
        ),
    )
    # The viewed old success is a veto-only fixed regression.
    fixture = Path("tests/data/sync_carrier_tone_success.npz")
    frozen = json.loads(fixture.with_suffix(".json").read_text())
    assert sha(fixture) == frozen["derived_npz_sha256"]
    with np.load(fixture) as ar:
        x, quiet, payload = ar["iq"], ar["quiet"], ar["expected"].tobytes()
    profile = PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=6, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )
    meta = dict(id="old-success", positive=True, input_sha256=sha(fixture))
    old = []
    for chunking in ("whole", "fixed", "random"):
        _, row, c, result = receive_case(output, meta, profile, x, quiet, payload, chunking)
        old.append(c)
        assert row["correct"] == 1
        flows += result
    write(output / "old-success.json", old)
    # These are measured finite micro-operation services, not a full receiver.
    rng = np.random.default_rng(670121)
    resource_words = [(int(a), int(b), 0) for a, b in rng.integers(-100000, 100000, (256, 2))]
    np.save(output / "resource-input.npy", np.asarray(resource_words, np.int64))
    ft2_template = SyncSignTemplate.from_symbols(
        PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4)).sync_symbols()
    )
    ft2 = SplineScoreFlow(SplineScoreAssumptions(256)).run(resource_words, ft2_template, [64])
    naive = SplineScoreFlow(SplineScoreAssumptions(256, False)).run(
        resource_words, ft2_template, [64]
    )
    assert naive["best"] == ft2["best"] and naive["operation_counts"] == ft2["operation_counts"]
    write(output / "resource-shared-vs-naive.json", dict(reference=naive, shared=ft2))
    raw = np.random.default_rng(670121).integers(0, 2**46, (4096, 2), dtype=np.uint64)
    # Score/threshold schedule is value-independent for this fixed interior
    # candidate set; only the final selected phase changes replay length.
    replay_cycles = [
        32 * (2 if phase == 0 else 4 * sum(map(len, _DIGITS[phase])) + 6) for phase in range(8)
    ]
    chosen_phase = ft2["best"]["start_tick"] % 8
    service = ft2["cycles"] - replay_cycles[chosen_phase] + max(replay_cycles)
    two = SplineScoreFlow(SplineScoreAssumptions(256)).run(resource_words, ft2_template, [64, 68])
    two_bound = two["cycles"] - replay_cycles[two["best"]["start_tick"] % 8] + max(replay_cycles)
    write(
        output / "resource-worst-replay.json",
        dict(
            one_peak_measured=ft2["cycles"],
            one_peak_bound=service,
            two_peak_measured=two["cycles"],
            two_peak_bound=two_bound,
            all_phase_replay_cycles=replay_cycles,
            assumptions="N256,FT2sync32,interior separated peaks,read1,ALU32,DSP3,no output stall;no coherent/packet work",
            estimate_not_synthesis=True,
        ),
    )
    scenarios = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        result = CaptureTransport(
            CaptureTransportAssumptions(clock_hz=clock, fine_service_cycles=service)
        ).simulate(raw, [(j, (j // 32) % 2) for j in range(256, 4097, 32)])
        scenarios.append(dict(clock=clock, service_cycles=service, result=result))
    failures = [
        dict(
            id=r["id"],
            positive=r["positive"],
            captured=r["captured"],
            head_crc=r["head_crc"],
            data_crc=r["data_crc"],
            correct=r["correct"],
            bit_errors=r["bit_errors"],
            ber_bits=r["ber_bits"],
        )
        for r in all_rows + held_rows
        if r["chunking"] == "whole" and r["positive"] and not r["correct"]
    ]
    for row in failures:
        metadata = json.loads((output / (row["id"] + "-metadata.json")).read_text())
        row["input_sha256"] = metadata["input_sha256"]
        row["frozen_source_input_sha256"] = metadata.get("frozen_source_input_sha256")
        row["metadata"] = row["id"] + "-metadata.json"
        row["frames"] = row["id"] + "-score-whole/frames.json"
        row["classification"] = (
            "captured,headCRC good,dataCRC failed;no new recovery claim,attribution not theoretical impossibility"
        )
    write(output / "failures.json", failures)
    model = dict(
        node_word_pass=True,
        float_all_range_compatibility_pass=False,
        normalization_floor_failures=floor,
        development_pass=dev_pass,
        validation_evaluated=dev_pass,
        validation_pass=dev_pass and all(c["passed"] for c in held_comparisons),
        old_fixture_success_preserved=all(c["passed"] for c in old),
        development=summarize(all_rows),
        validation=summarize(held_rows) if held_rows else None,
        costs=[spline_score_cost(SplineScoreAssumptions(n)) for n in (256, 376, 640)],
        representative_flow=ft2,
        one_peak_service_upper_cycles=service,
        two_peak_service_upper_cycles=two_bound,
        scenarios=scenarios,
        failures=failures,
        resource_gate=False,
        whole_system_accepted=False,
        main_rx_default_changed=False,
        old_rrc2percent_and16KiB_failures_preserved=True,
        remaining="coherent confirmation/phase bridge,compact score formats,full capture feedback and packet/decoder/TX/MAC concurrency",
    )
    write(output / "model.json", model)
    write(
        output / "manifest.json",
        {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in model.items()
                if k not in ("representative_flow", "costs", "scenarios", "failures")
            },
            indent=2,
        )
    )
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-inputs", type=Path)
    parser.add_argument("--manifest-sha256")
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.frozen_inputs, args.manifest_sha256))
