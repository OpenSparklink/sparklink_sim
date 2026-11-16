"""Observe capture work without feedback, then replay optimistic bounded fine queues."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from audit_capture_hardware import profile, score, sha, write

from nearlink_sdr.phy.capture_capacity import CaptureCapacity, CaptureCapacityAssumptions
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.tone_notch import ToneNotchConfig


class RequestObserver(PSKStreamReceiver):
    """Caller-owned diagnostics only; no delayed/oracle result enters capture selection."""

    def __init__(self, *args, request_sink, **kwargs):
        self.request_sink = request_sink
        super().__init__(*args, **kwargs)

    def _search_path(self):
        before = self.capture_stats["full_searches"]
        future = self.capture_stats["maxima_needing_future"]
        positions = self.capture_stats["computed_scored_positions"]
        source = self.integer_screen_backend
        origin = self._origin + len(self._filtered) - self.search_samples
        super()._search_path()
        if self.capture_stats["full_searches"] != before:
            self.request_sink(
                dict(
                    after_samples=min(self._processed, self.input_samples),
                    processed_samples=self._processed,
                    virtual_eof=self._processed > self.input_samples,
                    window_start_raw=origin,
                    path=source,
                    candidate_sync_start=None if self._active is None else self._active["start"],
                    maxima_needing_future=self.capture_stats["maxima_needing_future"] - future,
                    computed_positions=self.capture_stats["computed_scored_positions"] - positions,
                )
            )


def generate_stress(out):
    """One prespecified development workload, no legal packet or held-out claim."""
    from tests.test_strict_receive import iq_record

    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(410117)
    symbols = np.tile(profile(8).sync_symbols(), 128)
    x, _, _, _ = iq_record(symbols, offset=0.619)
    x += 50 * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    q = rng.normal(size=2048) + 1j * rng.normal(size=2048)
    path = out / "repeated-sync-noise.npz"
    np.savez_compressed(path, iq=x, quiet=q)
    write(
        out / "frozen-inputs.json",
        dict(
            seed=410117,
            cases=[
                dict(
                    id="repeated-sync-noise",
                    path=str(path.resolve()),
                    sha256=sha(path),
                    samples=len(x),
                    mcs=8,
                    legal_packets=0,
                    kind="intentional known-sync workload with invalid/missing frame structure",
                )
            ],
            generator_sha256=sha(Path(__file__)),
            independent_pulse_reference_sha256=sha(Path("scripts/rx_timing_audit.py")),
            transmitter="independent analytic pulse, repeated known sync; no payload-derived selection",
        ),
    )


def trace_requests(path):
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            if not row["virtual_eof"]:
                yield row["after_samples"]


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    plan = json.loads((args.inputs / "frozen-inputs.json").read_text())
    sources = {str(p): sha(p) for p in Path("src").rglob("*.py")}
    assumptions = CaptureCapacityAssumptions()
    write(
        out / "frozen-plan.json",
        dict(
            input_plan=str(args.inputs / "frozen-inputs.json"),
            input_plan_sha256=sha(args.inputs / "frozen-inputs.json"),
            assumptions=asdict(assumptions),
            sources=sources,
            driver_sha256=sha(Path(__file__)),
            scope="trace replay and necessary resource bounds; not a closed-loop delayed receiver",
        ),
    )
    (out / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    rows = []
    for case in plan["cases"]:
        path = Path(case["path"])
        assert sha(path) == case["sha256"]
        with np.load(path) as data:
            x, q = data["iq"], data["quiet"]
        request_file = out / (case["id"] + "-requests.jsonl")
        with request_file.open("w") as request_log:

            def sink(r):
                request_log.write(json.dumps(r) + "\n")

            rx = RequestObserver(
                profile(case["mcs"]),
                q,
                request_sink=sink,
                quiet_batching=False,
                correlation_backend="scaled",
                tone_notch=ToneNotchConfig(),
            )
            events = []
            for start in range(0, len(x), 257):
                events += rx.feed(x[start : start + 257])
            events += rx.finish()
        result = (
            score(events, case)
            if "packets" in case
            else dict(
                captures=len(events),
                false_deliveries=sum(e["frame"]["status"] == "valid" for e in events),
                statuses=[e["frame"]["status"] for e in events],
            )
        )
        schedules = {}
        for fine_slots in (4, 18):
            model = CaptureCapacity(replace(assumptions, fine_multipliers=fine_slots))
            log_path = out / (case["id"] + f"-schedule-{fine_slots}.jsonl")
            with log_path.open("w") as log:

                def schedule_sink(row):
                    log.write(json.dumps(row) + "\n")

                schedules[str(fine_slots)] = model.simulate(
                    trace_requests(request_file), len(x), sink=schedule_sink
                )
        rows.append(
            dict(
                input=case,
                zero_latency_receiver=result,
                capture_stats=rx.capture_stats,
                schedules=schedules,
            )
        )
        write(out / "results.json", rows)
    model = CaptureCapacity(assumptions)
    write(out / "necessary-capacity.json", model.necessary_capacity())
    stress = {}
    for fine_slots in (4, 18):
        for history in (384, 640, 1024):
            a = replace(assumptions, fine_multipliers=fine_slots, history_samples=history)
            flow = CaptureCapacity(a)
            key = f"fine{fine_slots}-history{history}"
            stress[key] = dict(
                capacity=flow.necessary_capacity(),
                quiet=flow.simulate(iter([]), 1_000_000),
                all_quantum_requests=flow.simulate(range(256, 1_000_001, 32), 1_000_000),
                dense=flow.simulate(range(512, 1_000_001, 512), 1_000_000),
                blocked=flow.simulate(
                    range(256, 1_000_001, 32), 1_000_000, stalls=[(100000, 200000)]
                ),
            )
    write(out / "stress-scenarios.json", stress)
    write(
        out / "failures.json",
        dict(
            note="Queue/resource failures do not modify original floating receiver packet outcomes",
            rows=[
                r
                for r in rows
                if any(
                    s["dropped_requests"] or s["history_gaps"] or s["snapshot_history_gaps"]
                    for s in r["schedules"].values()
                )
            ],
        ),
    )
    capacity = model.necessary_capacity()
    acceptance = dict(
        worst_request_graph_fits_multiplier_budget=(
            capacity["fine_graph_all_multipliers_necessary_condition"]),
        full_receiver_16msps_verified=False,
        expected_exit_code=1,
        note="Resource failure retained; completion of diagnostics does not accept the architecture",
    )
    write(out / "acceptance.json", acceptance)
    assert sources == {str(p): sha(p) for p in Path("src").rglob("*.py")}
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()},
    )
    print(
        json.dumps(
            dict(
                inputs=len(rows),
                zero_latency_full_searches=sum(r["capture_stats"]["full_searches"] for r in rows),
                necessary_capacity=model.necessary_capacity(),
            ),
            indent=2,
        )
    )
    return not acceptance['worst_request_graph_fits_multiplier_budget']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-stress", action="store_true")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.generate_stress:
        generate_stress(args.output)
    elif args.inputs:
        raise SystemExit(run(args))
    else:
        parser.error("Need --inputs or --generate-stress")
