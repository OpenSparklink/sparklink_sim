"""Bounded causal tail/value relay;exit1 preserves full receiver/resource gaps."""

import argparse
import json
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from model_capture_fine import GEOMETRY, image_sha

from nearlink_sdr.sim.capture_tail import CaptureTailReplay
from nearlink_sdr.sim.capture_transport import (
    CaptureRequest,
    CaptureTransport,
    CaptureTransportAssumptions,
    FineService,
)


def verify_outputs(events, output, requests, source, clock, capacity):
    """Absolute-index oracle; no true packet boundary/length or CRC is consulted."""
    for event in events:
        ident = event["id"]
        request = requests[ident]
        words = [x for x in output if x["id"] == ident]
        if event["status"] == "completed" and event["disposition"] in ("ready", "confirm_pending"):
            end = min(len(source), event["cycle"] * 16000000 // clock + 1)
            assert event["tail"]["next_live_index"] == end
            begin = request.after_samples - event["window"]
            assert [x["index"] for x in words] == list(range(begin, end))
        for word in words:
            assert word["index"] < min(len(source), word["cycle"] * 16000000 // clock + 1)
            assert word["value"] == int(source[word["index"], request.path])
        if event["status"] == "tail_capacity_exceeded":
            lost = event["tail"]["overflow_index"]
            edge = (lost * clock + 15999999) // 16000000
            popped = sum(w["region"] == "tail" and w["cycle"] < edge for w in words)
            assert lost - request.after_samples - popped == capacity
            assert not event["tail"]["replay_done"]


class CausalArray:
    """Offline input storage is not a receiver buffer;only current row is exposed."""

    def __init__(self, words):
        self.words, self.seen = words, 0

    def __len__(self):
        return len(self.words)

    def __getitem__(self, index):
        assert type(index) is int and index == self.seen
        self.seen += 1
        return tuple(map(int, self.words[index]))


class ContinuedWords:
    """Word-interface continuation only,not a newly generated RF waveform.

    Frozen prefix is byte-identical. Suffix values are causal counter words;
    no new capture requests are invented,so the workload is optimistic.
    """

    def __init__(self, base, count):
        self.base, self.count = base, count

    def __len__(self):
        return self.count

    def __getitem__(self, key):
        if isinstance(key, tuple):
            index, path = key
            if index < len(self.base):
                return int(self.base[index, path])
            return ((index * 6364136223846793005) ^ (path * 0x9E3779B1)) & (2**46 - 1)
        return tuple(self[key, path] for path in (0, 1))


def simulate(out, name, source, requests, services, clock, window, history, *, blocked=()):
    emitted, events = [], []
    t = CaptureTailReplay(sink=emitted.append)
    wrapped = CausalArray(source)

    def worker(event, owned):
        entry = services[event["id"]]
        assert image_sha(owned) == entry["source_sha256"]
        return FineService(**entry["service"])

    def sink(event, words):
        event["window"] = window
        events.append(event)
        if words is not None:
            assert image_sha(words) == services[event["id"]]["source_sha256"]

    a = CaptureTransportAssumptions(clock_hz=clock, window=window, history=history)
    result = CaptureTransport(a).simulate(
        wrapped, requests, fine_worker=worker, tail_replay=t, sink=sink, stalls=blocked
    )
    assert wrapped.seen == result["input_samples"] == len(source)
    assert sum(
        result[k]
        for k in ("completed", "queue_drops", "history_drops", "startup_drops", "tail_drops")
    ) == len(requests)
    assert result["peak_owned"] <= 3 and not result["adc_backpressure"]
    verify_outputs(events, emitted, requests, source, clock, t.a.capacity)
    file = name + f"-{clock}.json"
    write(out / file, dict(result=result, events=events, assumptions=vars(a)))
    np.savez_compressed(
        out / (name + f"-{clock}-replayed.npz"),
        words=np.asarray(
            [[e["id"], e["index"], e["value"], e["cycle"]] for e in emitted], np.int64
        ),
    )
    return dict(
        file=file,
        sha256=sha(out / file),
        clock=clock,
        requests=len(requests),
        completed=result["completed"],
        queue_drops=result["queue_drops"],
        history_drops=result["history_drops"],
        tail_drops=result["tail_drops"],
        replay_done=sum(bool(e["tail"] and e["tail"]["replay_done"]) for e in events),
        replay_live=sum(
            bool(e["tail"] and e["tail"]["replay_done"])
            and e["tail"]["next_live_index"] < len(source)
            for e in events
        ),
        replay_EOF=sum(
            bool(e["tail"] and e["tail"]["replay_done"])
            and e["tail"]["next_live_index"] == len(source)
            for e in events
        ),
        replayed_words=len(emitted),
        max_latency=result["max_latency_cycles"],
        every_emitted_word_exact=True,
        source_current_only=True,
    )


def node_cases(validation=False):
    """Independent word/port nodes,not an RF held set or decoded legal packets."""
    specs = (
        [
            (376, 504, 7000, 65000, 160000000, 3),
            (256, 384, 3500, 1500, 100000000, 5),
            (632, 760, 6000, 500, 200000000, 4),
            (256, 384, 9000, 80000, 160000000, 1),
        ]
        if validation
        else [
            (256, 384, 2600, 600, 200000000, 2),
            (376, 504, 4000, 800, 200000000, 5),
            (632, 760, 8000, 40000, 200000000, 2),
            (256, 384, 7000, 70000, 100000000, 1),
        ]
    )
    rng = np.random.default_rng(830311 if validation else 830117)
    for k, (window, history, n, cycles, clock, count) in enumerate(specs):
        source = rng.integers(0, 2**46, (n, 2), dtype=np.uint64)
        if k % 2 == 0:
            source[n // 3 : n // 2] &= 31  # Large dynamic-range and boundary changes.
        else:
            source[::7] = 0
            source[n // 2 : n // 2 + 8] = 2**46 - 1
        requests = [CaptureRequest(window + j * 32, j % 2, (8,)) for j in range(count)]
        services = [
            dict(
                source_sha256=image_sha(source[q.after_samples - window : q.after_samples, q.path]),
                service=vars(FineService(cycles, "confirm_pending", 64)),
            )
            for q in requests
        ]
        yield k, source, requests, services, clock, window, history


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path("uv.lock"), *Path("src").rglob("*.py")]
    write(
        out / "protocol.json",
        dict(
            stage="SYS-C2D",
            source={str(p): sha(p) for p in sources},
            word_scope_only=True,
            source_interface=(
                "frozen46bit filtered words,including42 C1 filter-flush words per stream;"
                "not a new ADC/RRC schedule"
            ),
            no_RF_gain_claim=True,
            default_RX_changed=False,
            fixed_tail_words=4096,
            slots=3,
            consumer_II=4,
            read_latency=1,
            tail_BRAM_cap=24,
            combined_module_and_join_register_cap=4096,
            selection=(
                "fixed before implementation;no tuning on node validation or viewed RF inputs"
            ),
            baseline_raw_release="fine completion,unchanged at tail_replay=None",
        ),
    )
    nodes = {}
    for validation in (False, True):
        split = "validation" if validation else "development"
        nodes[split] = []
        for k, source, requests, services, clock, window, history in node_cases(validation):
            np.savez_compressed(out / f"{split}-{k}-input.npz", words=source)
            blocked = ((5000, 15000),) if k == 1 else ()
            row = simulate(
                out,
                split + str(k),
                source,
                requests,
                services,
                clock,
                window,
                history,
                blocked=blocked,
            )
            nodes[split].append(row)
        write(out / (split + ".json"), nodes[split])
        if not validation:
            write(
                out / "postdevelopment-freeze.json",
                dict(
                    source={str(p): sha(p) for p in sources},
                    capacity=4096,
                    owners=3,
                    consumer_II=4,
                    future_node_validation_not_generated=True,
                    RF_held_not_generated=True,
                    node_rule="exact words;explicit capacity drop;no tuning",
                ),
            )
    capture, score = args.capture_root, args.score_root
    for folder in (capture, score):
        manifest = json.loads((folder / "manifest.json").read_text())
        assert all(sha(folder / name) == digest for name, digest in manifest.items())
    rows, continuations = [], []
    fixture = json.loads(Path("tests/data/spline_geometry_regression.json").read_text())["cases"]
    for record in json.loads((capture / "rows.json").read_text()):
        name, meta = record["id"], record["input"]
        _, window, history = GEOMETRY[meta["frame_type"]]
        with np.load(capture / (name + "-filtered.npz")) as z:
            source = z["words"]
        requests, services = [], []
        for entry in (e for e in fixture if e["stream"] == name):
            parent = json.loads((score / (entry["file"][:-5] + "-alu2.json")).read_text())
            q = parent["request"].copy()
            q["peaks"] = tuple(q["peaks"])
            requests.append(CaptureRequest(**q))
            assert parent["source_sha256"] == entry["source_sha256"]
            assert meta["input_sha256"] == entry["raw_original_archive_sha256"]
            services.append(parent)
        for clock in (100000000, 160000000, 200000000):
            row = simulate(out, name, source, requests, services, clock, window, history)
            rows.append(row | dict(stream=name, frame_type=meta["frame_type"]))
        if requests:
            last = requests[-1].after_samples
            longest = max(q["service"]["cycles"] for q in services)
            cycles = (last * 200000000 + 15999999) // 16000000
            cycles += (len(requests) + 1) * (longest + 8 * (window + 4096) + 1024)
            count = max(len(source), cycles * 16000000 // 200000000 + 2)
            continued = ContinuedWords(source, count)
            control = simulate(
                out,
                name + "-continued",
                continued,
                requests,
                services,
                200000000,
                window,
                history,
            )
            assert control["replay_EOF"] == 0
            continuations.append(
                control
                | dict(
                    stream=name,
                    suffix_words=count - len(source),
                    original_filtered_words=len(source),
                    no_new_capture_requests=True,
                    actual_RF_waveform_claim=False,
                )
            )
    cost = CaptureTailReplay().estimate()
    assert cost["tail_bram36"] <= 24 and cost["register_bits"] <= 4096
    write(
        out / "model.json",
        dict(
            nodes=nodes,
            schedules=rows,
            continuation_controls=continuations,
            cost=cost,
            exact_word_node_pass=True,
            node_resource_cap_pass=True,
            target_system_accepted=False,
            target_tail_configuration_adopted=False,
            rejection_reason=(
                "uniform4096-per-owner still overflows actual FT3/4 requests;"
                "24BRAM and remaining RX/decoder/TX/MAC not a proven system fit"
            ),
            whole_receiver16msps=False,
            fixed_inputs_only_not_RF_held=True,
            default_RX_changed=False,
            coherent_head_timing_decoder_not_included=True,
            formal_packet_recovery_not_measured=True,
            original2percent16KiB_failures_preserved=True,
            capture_parent_manifest_sha256=sha(capture / "manifest.json"),
            score_parent_manifest_sha256=sha(score / "manifest.json"),
        ),
    )
    write(
        out / "manifest.json",
        {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()},
    )
    return 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture-root", type=Path, required=True)
    p.add_argument("--score-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(p.parse_args()))
