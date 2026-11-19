"""Actual finite normalization at the selected-frame port; four whole-RX gates stay open.

Exit1 is expected, including when the node passes its separately frozen numeric budget.
Independent110digit arithmetic and independent pulse/channel fixtures are diagnostics only.
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
from audit_capture_hardware import sha, write
from audit_carrier_fixed import validate_profiles
from model_selected_frame import metrics, normalized
from model_timing_handoff import fixed, fresh
from model_timing_words import execute as execute_port
from tests.normalization_reference import raw_word, relative_error

from nearlink_sdr.phy.normalization_word import NormalizationFormat, WordNormalizationState
from nearlink_sdr.phy.sync_fit_word import TURN, SyncFitFormat, compile_sync_phases
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.initial_sync_fit_flow import InitialSyncFitService
from nearlink_sdr.sim.normalization_word_flow import SymbolNormalizationService
from nearlink_sdr.sim.normalized_word_replay import NormalizedWordFrameReplay
from nearlink_sdr.sim.pipelined_timing_replay import PipelinedTimingFrameReplay
from nearlink_sdr.sim.receive_history import ReceiveHistoryPort

POLICY = Path("tests/data/normalization_word_policy.json")
FMT = TimingWordFormat(30, 32)
NORM = NormalizationFormat("linear")
DECISIONS = ("status", "header_crc", "data_crc", "length_ok", "mcs", "header", "payload")


def execute(*args, **options):
    # Host watchdog only: includes real added work in long coded packets.
    # Continuous ADC/resource deadlines remain separate and unchanged.
    return execute_port(*args, host_edges_per_symbol=256, **options)


def source_vector():
    return {str(p): sha(p) for p in sorted({
        POLICY, Path("uv.lock"), *Path("src").rglob("*.py"), *Path("tests").rglob("*.py"),
        *Path("scripts").glob("*.py"),
    })}


def arithmetic(out, split, policy):
    """New arithmetic configurations; nearest is a rejected control, never a packet fallback."""
    rng = np.random.default_rng(policy[f"arithmetic_{split}_seed"])
    samples, rows = [], []
    for index in range(384):
        gain = (int(rng.integers(1, 1 << 54)), int(rng.integers(-512, 512)))
        fit = dict(valid=True, phase_turn40=int(rng.integers(0, 1 << 40)),
                   frequency_turn40=int(rng.integers(-(1 << 41), 1 << 41)), gain=gain)
        word = raw_word(*map(int, rng.integers(-(1 << 21), 1 << 21, 2)),
                        int(rng.integers(-128, 128)))
        sample = dict(index=index, fit=fit, input_word=word, candidates={})
        for mode in ("nearest", "linear"):
            functional = WordNormalizationState(fit, NormalizationFormat(mode))
            actual = WordNormalizationState(fit, NormalizationFormat(mode))
            service = SymbolNormalizationService(actual)
            assert service.issue(word)
            while service.peek() is None:
                service.step()
            result = service.take()
            assert result == functional.advance_word(word)
            assert actual.snapshot() == functional.snapshot()
            evm = float(relative_error(word, result["normalized_word"], fit))
            sample["candidates"][mode] = dict(
                output_word=result["normalized_word"], relative_error=evm,
                state=actual.snapshot(), actual_cycles=service.cycle,
                service_statistics=service.stats,
            )
        samples.append(sample)
    write(out / "arithmetic-samples.json", samples)
    for mode in ("nearest", "linear"):
        e = np.array([s["candidates"][mode]["relative_error"] for s in samples])
        rms, peak = float(np.sqrt(np.mean(e * e))), float(max(e))
        b = policy["node_budgets"]
        rows.append(dict(mode=mode, samples=len(samples), independent_reference_digits=110,
                         relative_RMS=rms, relative_peak=peak,
                         numeric_pass=rms <= b["node_relative_RMS_max"]
                         and peak <= b["node_relative_peak_max"],
                         cost=SymbolNormalizationService.estimate(mode),
                         actual_values_and_states_exact=True))
    write(out / "arithmetic.json", rows)
    return rows


def raw_trace(rows):
    return [{k: r[k] for k in ("symbol", "role", "cell", "phase", "word")} for r in rows]


def compare(before, after, old_words, new_words, budgets):
    # Frame decisions/content are exact. Carrier residual diagnostics need not be identical.
    for key in DECISIONS:
        assert normalized(before["frame"].get(key)) == normalized(after["frame"].get(key)), key
    assert before["word_timing_state"] == after["word_timing_state"]
    assert raw_trace(old_words) == raw_trace(new_words)
    for key in ("symbol_centers_filtered", "timing_rate_ppm", "timing_detector_error"):
        np.testing.assert_array_equal(before[key], after[key])
    a, b = before["symbols"], after["symbols"]
    assert a.shape == b.shape
    if len(a) and np.any(a):
        rms = float(np.linalg.norm(a - b) / np.linalg.norm(a))
        peak = float(np.max(abs(a - b)) / np.max(abs(a)))
    else:
        assert np.array_equal(a, b)
        rms = peak = 0.0
    return dict(relative_RMS=rms, relative_peak=peak,
                numeric_pass=rms <= budgets["packet_relative_RMS_max"]
                and peak <= budgets["packet_relative_peak_max"],
                timing_trajectory_delta=0, raw_interpolants_and_decisions_exact=True)


def pair(out, name, profile, words, selection, payload, policy, *, partitions):
    before, _old, old_words = execute(out, name + "-float-normalization", profile, words,
                                     selection, FMT, [32], flow_class=PipelinedTimingFrameReplay)
    services = []
    after, flow, new_words = execute(
        out, name + "-word-normalization", profile, words, selection, FMT, [32],
        flow_class=partial(NormalizedWordFrameReplay, normalization_format=NORM,
                           service_observer=services.append),
    )
    numerical = compare(before, after, old_words, new_words, policy["node_budgets"])
    owner = WordNormalizationState(flow.consumer.initial_fit, NORM)
    records = [r for r in services if r["kind"] == "normalization"]
    for r in records:
        assert r["values"] == owner.advance_word(r["values"]["raw_word"])
        assert r["taken_edge"] >= r["ready_edge"] == r["issued_edge"] + r["latency"]
    assert owner.snapshot() == flow.consumer.normalization.snapshot()
    n = owner.count
    assert flow.stats["normalization_requests"] == flow.stats["normalization_completions"] == n
    assert flow.normalization_service.stats["native_products"] == 10 * n
    assert flow.normalization_service.stats["rom_reads"] == 4 * n
    assert not after["normalization_service"]["cancelled_inflight"]
    write(out / (name + "-actual-services.json"), services)
    np.savez_compressed(out / (name + "-normalized-words.npz"),
                        **{k: np.array([r["values"][k] for r in records], np.uint64)
                           for k in ("raw_word", "normalized_word", "phase_before", "phase")})
    if partitions:
        for label, pattern in [("fixed", [7, 31]),
                               ("random", np.random.default_rng(1100773).integers(1, 33, 23))]:
            other_services = []
            other, other_flow, obs = execute(
                out, name + "-" + label, profile, words, selection, FMT, pattern,
                flow_class=partial(NormalizedWordFrameReplay, normalization_format=NORM,
                                   service_observer=other_services.append),
            )
            assert normalized(other["frame"]) == normalized(after["frame"])
            for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm",
                        "timing_detector_error"):
                np.testing.assert_array_equal(other[key], after[key])
            assert obs == new_words and other_services == services
            assert other_flow.consumer.normalization.snapshot() == owner.snapshot()
    cost = flow.estimate()
    node, budget = cost["normalization_service"], policy["node_budgets"]
    assert node["state_register_bits_upper"] + 192 <= budget["new_logic_state_bits_upper_max"]
    assert node["additional_DSP25x18"] <= budget["extra_DSP_max"]
    assert node["coefficient_ROM_RAMB18_estimate"] <= budget["extra_quarter_ROM_RAMB18_max"]
    assert node["new_original_source_RAM_bits"] == node["new_source_or_cache_read_ports"] == 0
    a, b = [metrics(e, profile, expected=payload) for e in (before, after)]
    return dict(id=name, before=a, after=b, numerical=numerical,
                source_words_sha256=hashlib.sha256(
                    np.asarray(words, dtype="<u8").tobytes()).hexdigest(),
                success_loss=a["payload_exact"] and not b["payload_exact"],
                recovery=b["payload_exact"] and not a["payload_exact"],
                false_delivery=b["false_delivery"] and not a["false_delivery"],
                partitions_checked=partitions, service_trace_file=name + "-actual-services.json",
                normalization_latency_histogram=dict(Counter(r["latency"] for r in records)),
                normalization_word_count=n, actual_services=after["normalization_service"],
                cost=cost, complete_latency_cycles=None, full_receiver16MSps=False)


def periodic_prefixes(out):
    """Continuous ADC, actual normalization, no zero-cycle header/decoder resolution."""
    from tests.test_selected_rx import ordinary, selected_input
    from tests.test_selected_word import quantize_input

    rows = []
    for ft in (2, 3, 4):
        profile = ordinary(ft, 8, 0)
        iq, _, selection = selected_input(profile, length=31)
        words = quantize_input(np.r_[iq, np.zeros(8192)])
        selection["sample_limit"] = len(words)
        file = out / f"prefix-ft{ft}-input.npz"
        np.savez_compressed(file, words=words, selected_sync=selection["selected_sync"])
        cell, phase = divmod(selection["start_tick"], 8)
        origin = cell - 1
        for hz in (100_000_000, 160_000_000, 200_000_000):
            source = ReceiveHistoryPort(first_index=origin)
            init = InitialSyncFitService(source, compile_sync_phases(profile.sync_symbols()),
                                         round(selection["delta"] * TURN / math.tau), cell,
                                         phase * 8192, fmt=SyncFitFormat(30))
            pos, flow, stop, first_header = origin, None, None, None
            while stop is None:
                cycle = init.cycle if flow is None else flow.cycle
                due = math.ceil((pos - origin) * hz / 16_000_000)
                assert due >= cycle, "ADC arrival must not be suppressed"
                if due == cycle:
                    ok = source.write(pos, words[pos], cycle) if flow is None else (
                        flow.write_input(words[pos]))
                    pos += 1
                    if not ok:
                        stop = "initial_history_overflow" if flow is None else "history_overflow"
                        if flow is None:
                            init.abort(reason=stop)
                        break
                if flow is None:
                    init.step()
                    if init.ready:
                        flow = NormalizedWordFrameReplay(
                            profile, init, normalization_format=NORM, word_timing_format=FMT,
                            reuse_stencil=True, **selection,
                        )
                else:
                    flow.step()
                    if first_header is None and flow.pending_service is not None:
                        assert flow.pending_service["kind"] == "header"
                        first_header = cycle
                    if flow.done:
                        stop = flow.event["frame"]["status"]
                assert cycle < 200_000 and pos < len(words)
                assert source.live <= source.capacity == 512
            row = dict(frame_type=ft, clock_hz=hz, input_rate=16_000_000,
                       input_samples_arrived=pos-origin, source_sha256=sha(file),
                       first_header_request_edge=first_header, stop=stop, terminal_edge=cycle,
                       history_peak_live=source.stats["peak_live"], history_capacity=512,
                       capture_producer_not_modeled=True, full_receiver16MSps=False,
                       complete_latency_cycles=None, algorithm_misses=None,
                       resource_drop_contexts=1, raw_port_read_write_conflicts_modeled=True,
                       unmodeled_header_residual_carrier_LLR_decoder_cycles=None)
            if flow is not None:
                assert flow.stats["functional_resolutions"] == 0
                assert not flow.take_mac_result().packets
                row.update(normalized_symbols=flow.consumer.normalization.count,
                           normalizer_statistics=flow.normalization_service.stats,
                           blocked_service=flow.pending_service,
                           actual_services=flow.event.get("normalization_service"))
            rows.append(row)
    write(out / "periodic-prefixes.json", rows)
    return rows


def run(a):
    policy, source = json.loads(POLICY.read_text()), source_vector()
    for split in ("development", "validation"):
        validate_profiles(policy[split]["profiles"], policy[split]["pid"])
    frozen = json.loads(a.freeze.read_text()) if a.freeze else None
    if frozen is not None:
        assert frozen["source"] == source and frozen["policy"] == policy
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    write(out / "protocol.json", dict(source=source, policy=policy, split=a.split,
          upstream="whole-record differential max fixture;not live unknown-start capture",
          truth_payload_CRC="postdecision only;no candidate choice/retry",
          remaining_header_residual_carrier_LLR_decoder_cycles=None))
    numeric = arithmetic(out, a.split, policy)
    rows = fresh(out, a.split, policy, True, pair_fn=pair)
    regressions, parents = (fixed(out, a.timing_root, a.capture_root, a.selected_root, policy,
                                 pair_fn=pair) if frozen is None else ([], {}))
    prefixes = periodic_prefixes(out) if frozen is None else []
    reached = [r["comparison"] for r in rows + regressions if "comparison" in r]
    assert reached
    summary = dict(stage=policy["stage"], split=a.split, arithmetic=numeric,
                   before_full=sum(r["comparison"]["before"]["payload_exact"]
                                   for r in rows if "comparison" in r),
                   after_full=sum(r["comparison"]["after"]["payload_exact"]
                                  for r in rows if "comparison" in r),
                   fixed_before_full=sum(r["comparison"]["before"]["payload_exact"]
                                         for r in regressions if "comparison" in r),
                   fixed_after_full=sum(r["comparison"]["after"]["payload_exact"]
                                        for r in regressions if "comparison" in r),
                   fixed_requests=len(regressions), fixed_reached=len(regressions) and sum(
                       "comparison" in r for r in regressions),
                   success_losses=sum(r["success_loss"] for r in reached),
                   recoveries=sum(r["recovery"] for r in reached),
                   new_false_deliveries=sum(r["false_delivery"] for r in reached),
                   negative_observation_samples=sum(r["input"].get("observation_samples", 0)
                                                    for r in rows),
                   node_numeric_pass=numeric[1]["numeric_pass"] and all(
                       r["numerical"]["numeric_pass"] for r in reached),
                   old_RRC_interpolation16KiB_node4096_controller2048_failures_retained=True,
                   all_RX_gates=policy["receiver_gates"], default_changed=False,
                   parent_manifests=parents, periodic_prefixes=prefixes)
    write(out / "model.json", summary)
    write(out / "failure-associations.json", [r for r in rows + regressions
          if "comparison" not in r or not r["comparison"]["after"]["payload_exact"]
          or not r["comparison"]["numerical"]["numeric_pass"]])
    if frozen is None:
        write(out / "freeze.json", dict(source=source, policy=policy, parent_manifests=parents,
              adoption="explicit only;no post-held tuning;serial timing rejection remains"))
    assert source_vector() == source
    write(out / "manifest.json", {str(p.relative_to(out)): sha(p) for p in out.rglob("*")
                                if p.is_file()})
    print(json.dumps(summary, indent=2))
    return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--timing-root", type=Path)
    parser.add_argument("--capture-root", type=Path)
    parser.add_argument("--selected-root", type=Path)
    args = parser.parse_args()
    if (args.split == "validation") != (args.freeze is not None):
        parser.error("Validation requires an existing development freeze")
    raise SystemExit(run(args))
