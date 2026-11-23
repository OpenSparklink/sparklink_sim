"""Frozen RRC-only change on unknown-origin standard IQ→strict MAC.

The common coarse search,timing,carrier,notch and header options are fixed before
new inputs. Truth is used after each complete receive call,never to retry an arm.
This is functional integration,not a complete fixed-point or FPGA schedule gate.
"""

import argparse
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.signal import lfilter
from scripts.audit_rx_soft_capture import diagnostic_bits
from scripts.model_rrc_full14 import metrics, sha, sources, write
from scripts.model_scaled_maximum import configuration
from scripts.rx_timing_audit import pulse
from tests.captured_frame_reference import packet
from tests.full14_stream_reference import receive
from tests.maximum_reference import channel

from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.rrc_full14 import PairRRC14Flow, RRC14Memory

POLICY = Path(__file__).with_name("full14_stream_policy.json")
ARMS = ("reference", "word-full12-c18", "word-full14-c18")
CHUNKS = ("whole", "fixed", "random")
TRACE = ("symbols", "symbol_centers_raw", "timing_rate_ppm", "timing_detector_error")


def plain_config(cfg):
    return {k: v for k, v in asdict(cfg).items() if not k.startswith("_")}


def profile(spec):
    return replace(PSKReceiveProfile.ordinary(TxConfig(**spec["config"])),
                   head_backend="word-depth-parity", allowed_mcs=(8, 12))


def specifications(policy, split):
    held = split == "validation"
    rows = []
    for f, family in enumerate(policy["families"]):
        for r in range(policy[split+"_repeats"]):
            cfg = configuration((2, 3, 4)[(f+2*r+held) % 3], (8, 12)[(f+r+held) % 2],
                                (0, 4, 8, 16)[(f+3*r+2*held) % 4], 2411+f*31+r*7+held*301)
            rows.append(dict(id=f"{family}-{r}", family=family, family_index=f, repeat=r,
                             config=plain_config(cfg), positive=True))
    for f, family in enumerate(("noise", "tone", "wrong-sync", "burst", "zero")):
        cfg = configuration((2, 3, 4)[f % 3], (8, 12)[f % 2], (0, 4, 8, 16)[f % 4],
                            2411+f*31+held*301)
        rows.append(dict(id="negative-"+family, family=family, family_index=f,
                         repeat=f % policy[split+"_repeats"], config=plain_config(cfg),
                         positive=False))
    # Reject illegal recipes before creating any new input or output directory.
    for row in rows:
        profile(row)
    return rows


def payload(n, ident, seed, packet_index=0):
    if ident % 3 == 0:
        return bytes(((j*j*23+j*47+ident*19) ^ (j >> 3)) & 255 for j in range(n))
    if ident % 3 == 1:
        return bytes(((1 << ((5*j+ident) % 8)) ^ ((j//11)*31)) & 255 for j in range(n))
    rng = np.random.Generator(np.random.Philox(seed+ident*313+packet_index*719))
    return rng.integers(0, 256, n, np.uint8).tobytes()


def generate(policy, split, spec):
    p = profile(spec)
    held, f, r = split == "validation", spec["family_index"], spec["repeat"]
    seed = policy[split+"_seed"]+f*1013+r*79
    rng = np.random.Generator(np.random.Philox(seed))
    family, gain = spec["family"], policy[split+"_gains"][r]
    expected = []
    if spec["positive"]:
        n = policy[split+"_lengths"][r]
        data = [payload(n, f*17+r, seed)]
        wires = [packet(p, data[0])]
        counts = [len(wires[0])]
        starts = [0]
        if family == "dense":
            data.append(payload(n//2+17, f*17+r+43, seed, 1))
            gap = policy[split+"_dense_gaps_symbols"][r]
            starts.append(len(wires[0])+gap)
            wires += [np.zeros(gap), packet(p, data[1])]
            counts.append(len(wires[-1]))
        wire = np.concatenate(wires)
        offset, ppm, cfo = (policy[split+k][r] for k in ("_offsets", "_ppm", "_cfo_Hz"))
        x, quiet, legacy, step, noise = channel(wire, offset=offset, ppm=ppm, cfo=cfo,
            prefix=53+f*17+r*29+held*31, amplitude=gain, family=(
                "clean" if family in ("clean", "dense") else "noise"), seed=seed,
            snr_db=policy[split+"_SNR_dB"][r])
        origin = legacy-40*step
        for data_bytes, start, symbols in zip(data, starts, counts, strict=True):
            expected.append(dict(sync_start=origin+(start+{2: 40, 3: 48, 4: 64}[
                p.config.frame_type])*step, step=step, cfo_Hz=cfo, ppm=ppm,
                expected_hex=data_bytes.hex(), mcs=p.config.mcs_index,
                true_frame_end_sample=origin+(start+symbols)*step,
                wire_length=len(data_bytes)+(p.config.crc_len//8 if p.control_format == "B1" else 0)))
    else:
        gain = policy[split+"_gains"][(f+1) % policy[split+"_repeats"]]
        sigma = 2.5*gain
        x = sigma*(rng.normal(size=8192)+1j*rng.normal(size=8192))
        quiet = sigma*(rng.normal(size=2048)+1j*rng.normal(size=2048))
        if family == "zero":
            x[:] = 0
        noise = dict(noise_std_per_component=sigma, requested_SNR_dB=None)
    if family == "tone":
        x += 100*gain*policy[split+"_tone_factors"][r]*np.exp(
            2j*np.pi*policy[split+"_tone_cycles"][r]*np.arange(len(x)))
    elif family == "wrong-sync":
        cfg = replace(p.config, pid=(p.config.pid+1) % 6)
        wrong = replace(PSKReceiveProfile.ordinary(cfg), head_backend="word-depth-parity")
        bad_wire = packet(wrong, payload(103+r*11 if spec["positive"] else 103, f*31+r, seed))
        background, _, _, _, _ = channel(bad_wire, offset=.731, ppm=-25, cfo=-17000,
            prefix=0, amplitude=1.7*gain, family="clean", seed=seed+17)
        x += np.resize(background, len(x))
    elif family == "burst":
        if spec["positive"]:
            start = int(.43*len(x))
            x[start:start+7+2*r] += 100*gain*(17-13j)
        else:
            for start in range(97, len(x), 997):
                stop = min(start+31, len(x))
                x[start:stop] += 100*gain*np.exp(1j*rng.uniform(-np.pi, np.pi, stop-start))
    return x, quiet, spec | dict(seed=seed, expected=expected, noise=noise,
                                truth_used_for_decisions=False)


def evaluate(a, p, meta):
    # Entire arm is already terminal. Association and expected bytes never feed it.
    frames, matched, successes, bit_errors, compared_bits = [], set(), set(), 0, 0
    for event in a["events"]:
        frame = event["frame"]
        targets = meta["expected"]
        index = min(range(len(targets)), key=lambda k: abs(targets[k]["sync_start"]-
                    event["sync_start_raw"]), default=None)
        if index is not None and abs(targets[index]["sync_start"]-event["sync_start_raw"]) > 4:
            index = None
        correct, timing, errors, count = False, None, None, 0
        mcs_ok, expected_length_ok, carrier_error = False, False, None
        if index is not None:
            target = targets[index]
            matched.add(index)
            mcs_ok = frame.get("mcs") == target["mcs"]
            expected_length_ok = frame.get("header", {}).get("data_length") == target["wire_length"]
            carrier_error = event["cfo_hz"]/(1+target["ppm"]*1e-6)-target["cfo_Hz"]
            correct = (frame["status"] == "valid" and frame.get("payload") is not None and
                       frame["payload"].hex() == target["expected_hex"] and
                       frame.get("mcs") == target["mcs"] and frame.get("length_ok", False) and
                       frame["header"]["data_length"] == target["wire_length"])
            if correct:
                successes.add(index)
            error = event["symbol_centers_raw"]-(target["sync_start"]+
                    np.arange(len(event["symbols"]))*target["step"])
            if len(error):
                timing = dict(rms=float(np.sqrt(np.mean(error**2))),
                              head_mid_tail=[float(error[k]) for k in (0, len(error)//2, -1)])
                cycle = np.rint(error/target["step"]).astype(np.int64)
                stable = np.convolve((abs(error) <= .25).astype(int), np.ones(64, int), "valid")
                begin = len(p.sync_symbols())
                first = np.flatnonzero(stable[begin:] == 64)+begin
                window = int(first[0]) if len(first) else None
                timing.update(initial_truth_cycle_index=int(cycle[0]),
                    truth_cycle_index_changes=int(np.count_nonzero(np.diff(cycle))),
                    first64stable_truth_window_symbol=window,
                    postwindow_error_budget_violations=(None if window is None else
                        int(np.count_nonzero(abs(error[window+64:]) > .25))))
                rates = event.get("timing_rate_ppm", [])
                timing["median_last128_PI_state_error_ppm"] = (
                    float(np.median(rates[-128:]))-target["ppm"] if len(rates) else None)
            if frame.get("header_crc") and frame.get("mcs") == target["mcs"]:
                bits = diagnostic_bits(event, p)
                wanted = np.unpackbits(np.frombuffer(bytes.fromhex(target["expected_hex"]), np.uint8),
                                       bitorder=p.payload_bitorder)
                if bits is not None and len(bits) == len(wanted):
                    errors, count = int(np.count_nonzero(bits != wanted)), len(bits)
                    bit_errors, compared_bits = bit_errors+errors, compared_bits+count
        frames.append(dict(target=index, sync_start=event["sync_start_raw"], status=frame["status"],
            header_crc=bool(frame.get("header_crc")), data_crc=bool(frame.get("data_crc")),
            length_ok=bool(frame.get("length_ok")), mcs=frame.get("mcs"), correct=bool(correct),
            expected_mcs_ok=mcs_ok, expected_wire_length_ok=expected_length_ok,
            timing_truth_error=timing, bit_errors=errors, compared_bits=count,
            content_exact=None if errors is None else errors == 0,
            native_timing_diagnostics=event.get("timing_loop"),
            emitted_after_samples=event.get("emitted_after_samples"),
            latency_source_samples=(None if index is None or "emitted_after_samples" not in event
                or "true_frame_end_sample" not in targets[index] else
                event["emitted_after_samples"]-targets[index]["true_frame_end_sample"]),
            carrier_phase_truth_error=None,
            actual_carrier=frame.get("carrier"),
            CFO_after_payload_truth_error_Hz=(None if index is None or frame.get("carrier", {}).get(
                "residual_cycles_per_symbol") is None else (event["cfo_hz"]+frame["carrier"][
                    "residual_cycles_per_symbol"]*4e6)/(1+targets[index]["ppm"]*1e-6)-
                    targets[index]["cfo_Hz"]),
            CFO_nominal_Hz=event["cfo_hz"], CFO_truth_error_Hz=carrier_error, payload_hex=(
                None if frame.get("payload") is None else frame["payload"].hex())))
    return dict(known=len(meta["expected"]), captured=sorted(matched), correct=sorted(successes),
        observed_algorithm_misses=len(meta["expected"])-len(matched), clock_resource_packet_losses=None,
        header_crc=sum(r["header_crc"] for r in frames), data_crc=sum(r["data_crc"] for r in frames),
        length_ok=sum(r["length_ok"] for r in frames), frames=frames,
        expected_mcs_ok=sum(r["expected_mcs_ok"] for r in frames),
        expected_wire_length_ok=sum(r["expected_wire_length_ok"] for r in frames),
        content_exact=sum(r["content_exact"] is True for r in frames),
        false_candidates=sum(r["target"] is None for r in frames),
        false_MAC=sum(r["status"] == "valid" and not r["correct"] for r in frames),
        duplicate_MAC=sum(r["correct"] for r in frames)-len(successes),
        bit_errors=bit_errors, compared_bits=compared_bits,
        BER=bit_errors/compared_bits if compared_bits else None,
        delivered=[v.mac_payload.hex() for v in a["packets"]], before_EOF=a["before_EOF"])


def arm(p, x, quiet, backend, meta, output, policy):
    outcomes, traces = [], []
    for mode in CHUNKS:
        observed = []
        a = receive(p, x, quiet, backend, mode, rrc_sink=observed.append)
        result = evaluate(a, p, meta)
        rx = a["receiver"].phy
        words = tuple(np.concatenate([r[k] for r in observed]) for k in range(3)) if observed else ()
        arrays = {str(j)+"-"+k: e[k] for j, e in enumerate(a["events"]) for k in TRACE}
        arrays.update({"RRC_"+str(k): v for k, v in enumerate(words)})
        path = output/(meta["id"]+"-"+backend+"-"+mode+".npz")
        np.savez_compressed(path, **arrays)
        result.update(trace_SHA256=sha(path), stats=rx.capture_stats,
            per_million={k: v*1e6/len(x) for k, v in rx.capture_stats.items()},
            RRC_stats=None if not words else dict(rx._rrc_word_model.stats),
            RRC_numeric=None, history_limit=rx.history_limit, existing_search_paths=2,
            peak_buffered_samples=a["observed_peak_samples"],
            observed_peak_search_history_samples=rx.peak_history_samples,
            frame_symbol_capacity=rx.max_frame_symbols,
            frame_array_host_bytes=sum(getattr(rx, n).nbytes for n in
                                      ("_values", "_centers", "_rates", "_errors")),
            FIR_host_array_bytes=rx.fir_array_bytes,
            actual_timing_backend=rx.timing_backend, actual_integer_backend=rx.integer_screen_backend)
        if words:
            h = pulse(np.arange(-40, 41)/4)
            h /= np.linalg.norm(h)
            reference = lfilter(h, [1.0], np.r_[x, np.zeros(rx.future_samples)])
            y = np.ldexp(words[0].astype(float), words[2])+1j*np.ldexp(words[1].astype(float), words[2])
            assert len(y) == len(x)+rx.future_samples == len(x)+42
            result["RRC_numeric"] = metrics(y, reference, result["RRC_stats"],
                policy["generation_details"]["RRC_numeric"])
        traces.append(arrays)
        outcomes.append(result)
    # Same-mode chunk identity has no floating tolerance and no CRC-based retry.
    for j in (1, 2):
        assert set(traces[j]) == set(traces[0])
        assert all(traces[0][k].dtype == traces[j][k].dtype and
                   traces[0][k].tobytes() == traces[j][k].tobytes() for k in traces[0])
        for key in ("frames", "captured", "correct", "delivered", "before_EOF", "stats", "RRC_stats"):
            assert outcomes[j][key] == outcomes[0][key], (meta["id"], backend, key)
    return outcomes[0] | dict(chunk_exact=True, trace_files=[meta["id"]+"-"+backend+"-"+m+
                             ".npz" for m in CHUNKS])


def acceptance(rows, output, policy):
    failures = []
    for row in rows:
        candidate = row["arms"]["word-full14-c18"]
        for base in ("reference", "word-full12-c18"):
            original = row["arms"][base]
            if set(original["correct"])-set(candidate["correct"]):
                failures.append(dict(input=row["id"], reason="original_success_lost", baseline=base))
            if not row["positive"] and candidate["false_candidates"] > original["false_candidates"]:
                failures.append(dict(input=row["id"], reason="new_negative_candidate", baseline=base))
            for field in ("history_limit", "frame_symbol_capacity", "frame_array_host_bytes",
                          "existing_search_paths"):
                if candidate[field] != original[field]:
                    failures.append(dict(input=row["id"], reason="changed_existing_capacity",
                                         baseline=base, field=field))
            with np.load(output/original["trace_files"][0], allow_pickle=False) as old_trace, np.load(
                    output/candidate["trace_files"][0], allow_pickle=False) as new_trace:
                # Check every associated event pair;never hide a bad candidate
                # by selecting the one whose CRC or expected content succeeded.
                for oi, old in enumerate(original["frames"]):
                    if old["target"] is None:
                        continue
                    for ni, now in enumerate(candidate["frames"]):
                        if now["target"] != old["target"]:
                            continue
                        if abs(old["sync_start"]-now["sync_start"]) > policy[
                                "budgets"]["relative_sync_start_samples"]:
                            failures.append(dict(input=row["id"], reason="relative_start_budget",
                                                 baseline=base, events=[oi, ni]))
                        a, b = old_trace[str(oi)+"-symbol_centers_raw"], new_trace[
                            str(ni)+"-symbol_centers_raw"]
                        common = min(len(a), len(b))
                        if common and np.max(abs(a[:common]-b[:common])) > policy[
                                "budgets"]["relative_sampling_trajectory_samples"]:
                            failures.append(dict(input=row["id"], reason="relative_trajectory_budget",
                                                 baseline=base, events=[oi, ni]))
        if candidate["false_MAC"] or candidate["duplicate_MAC"]:
            failures.append(dict(input=row["id"], reason="false_or_duplicate_MAC"))
        if not candidate["RRC_numeric"]["passed"]:
            failures.append(dict(input=row["id"], reason="actual_RRC_numeric_budget"))
    write(output/"adoption-failures.json", failures)
    return not failures


def cycle_cost(samples):
    rows = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        a = PairRRCFlowAssumptions(clock_hz=clock)
        old, new = PairRRCFlow(a).simulate(samples+42), PairRRC14Flow(a).simulate(samples+42)
        assert old == new
        rows.append(dict(clock_Hz=clock, actual_source_samples=samples, virtual_EOF_samples=42,
                         total_FIR_requests=samples+42, result=new,
                         same_as_full12_dependency_calendar=True))
    return dict(RRC_flow=PairRRC14Flow().resource_estimate(), raw_memory=RRC14Memory().estimate(),
        scenarios=rows, existing_raw_and_notch_search_paths=2,
        notch_timing_carrier_head_payload_decoder_clock_service=None,
        actual_FIR_arithmetic_register_cycle_integration=False, full_RX16MSps=False,
        old16KiB_failure_retained=True, synthesis=False)


def run(out, split, development=None, regression=None):
    policy, pinned = json.loads(POLICY.read_text()), sources()
    if split == "validation":
        if development is None:
            raise ValueError("Need qualified development before new validation IQ")
        receipt = json.loads((development/"qualification.json").read_text())
        if not receipt["qualified"] or receipt["sources"] != pinned:
            raise ValueError("Unqualified or changed development source;no validation IQ generated")
        if regression is None:
            raise ValueError("Need qualified fixed regression before fresh validation IQ")
        fixed = json.loads(regression.read_text())
        if (not fixed["qualified"] or fixed["sources"] != pinned or fixed["streams"] != 88
                or fixed["frozen_metadata_SHA256"] != policy["fixed_regression_gate"][
                    "frozen_metadata_SHA256"]):
            raise ValueError("Unqualified or changed fixed regression;no validation IQ generated")
    specs = specifications(policy, split)
    out.mkdir(parents=True, exist_ok=False)
    write(out/"freeze-before-generation.json", dict(policy_SHA256=sha(POLICY), sources=pinned,
          split=split, specifications=specs, whole_receiver_accepted=False))
    rows, inputs = [], []
    for spec in specs:
        x, q, meta = generate(policy, split, spec)
        path = out/(spec["id"]+"-input.npz")
        np.savez_compressed(path, iq=x, quiet=q)
        meta.update(input_SHA256=sha(path), samples=len(x))
        inputs.append(meta)
        write(out/(spec["id"]+"-input.json"), meta)
        arms = {}
        for name in ARMS:
            try:
                arms[name] = arm(profile(spec), x, q, name, meta, out, policy)
            except Exception as exc:
                write(out/(spec["id"]+"-"+name+"-EXCEPTION.json"), dict(
                    input=meta, arm=name, exception_type=type(exc).__name__, message=str(exc),
                    traceback=traceback.format_exc(), qualified=False, swallowed_as_pass=False))
                raise
        row = dict(id=spec["id"], input_SHA256=sha(path), positive=spec["positive"], arms=arms)
        rows.append(row)
        write(out/(spec["id"]+".json"), row)
    write(out/"inputs.json", inputs)
    write(out/"results.json", rows)
    write(out/"cycle-cost.json", cycle_cost(inputs[0]["samples"]))
    passed = acceptance(rows, out, policy)
    assert sources() == pinned
    totals = {name: {k: sum(len(r["arms"][name][k]) for r in rows) for k in ("captured", "correct")}
              for name in ARMS}
    qualified = dict(qualified=bool(passed), sources=pinned, split=split, streams=len(rows),
        known_packets=sum(len(r["expected"]) for r in inputs), totals=totals,
        negative_samples=sum(r["samples"] for r in inputs if not r["positive"]),
        receiver_default_changed=False, all_four_RX_gates_passed=False)
    write(out/"qualification.json", qualified)
    write(out/"manifest.json", {f.name: sha(f) for f in out.iterdir() if f.is_file()})
    print(json.dumps({k: v for k, v in qualified.items() if k != "sources"}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--development", type=Path)
    parser.add_argument("--regression", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.split, args.development, args.regression))
