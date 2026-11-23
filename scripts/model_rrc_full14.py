"""One predeclared RRC input-format experiment;no live-RX or ADC-resolution claim.

Run development first. Validation requires its qualified,byte-identical source
receipt;an unqualified development set cannot unlock a fresh validation set.
"""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import lfilter
from scripts.model_scaled_maximum import configuration
from scripts.rx_timing_audit import pulse
from tests.captured_frame_reference import packet
from tests.maximum_reference import channel

from nearlink_sdr.phy.rrc_full14 import Full14MantissaRRC
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat, SymmetricRRC
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.rrc_full14 import PairRRC14Flow, RRC14Memory
from nearlink_sdr.sim.rrc_memory import RRCMemory

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path(__file__).with_name("rrc_full14_policy.json")
RANGE = ("exponent_underflows", "exponent_overflows", "input_saturations", "output_underflows",
         "output_overflows", "output_saturations", "intermediate_overflows")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sources():
    files = [ROOT/n for n in ["uv.lock", "pyproject.toml", ".github/workflows/ci.yml"]]
    files += [f for folder in ("src", "tests", "scripts") for f in (ROOT/folder).rglob("*")
              if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".pyc"]
    return {str(f.relative_to(ROOT)): sha(f) for f in sorted(files)}


def specs(policy, split):
    held = split == "validation"
    rows = []
    for f, family in enumerate(policy["families"]):
        for r in range(policy[split+"_repeats"]):
            length = policy[split+"_lengths"][r]
            ft, mcs = (2, 3, 4)[(f+2*r+held) % 3], (8, 12)[(f+r+held) % 2]
            pilots = (0, 4, 8, 16)[(f+3*r+2*held) % 4]
            index = 1739+f*19+r*3+held*200
            cfg = configuration(ft, mcs, pilots, index)
            PSKReceiveProfile.ordinary(cfg)  # Validate all configs before making IQ.
            rows.append(dict(id=f"{family}-{r}", family=family, family_index=f, repeat=r,
                             length=length, config={k: v for k, v in asdict(cfg).items()
                                                    if not k.startswith("_")}))
    return rows


def waveform(policy, split, spec):
    from nearlink_sdr.phy.tx_pipeline import TxConfig

    f, r, n = spec["family_index"], spec["repeat"], spec["length"]
    held, family = split == "validation", spec["family"]
    seed = policy[("validation_seed" if held else "development_seed")]+r*101+f*1009
    rng = np.random.Generator(np.random.SFC64(seed))
    if r % 3 == 0:
        payload = bytes((13*j*j+89*j+((j >> (1+r % 4)) ^ (0xA5+r))) & 255 for j in range(n))
    elif r % 3 == 1:
        payload = bytes(((1 << ((j+r) % 8)) ^ ((j // 7)*59)) & 255 for j in range(n))
    else:
        payload = rng.integers(0, 256, n, np.uint8).tobytes()
    if family == "adc12":
        codes = rng.integers(-2048, 2048, (2, 1024+17*n))
        codes[1, ::17] = np.resize([-1, 1], len(codes[1, ::17]))
        x = np.r_[np.zeros(37), (codes[0]+1j*codes[1])/2048, np.zeros(81)]
        truth = dict(kind="logical signed12 codes;no analog or timing calibration")
    else:
        profile = PSKReceiveProfile.ordinary(TxConfig(**spec["config"]))
        gains = policy[split+"_gains"]
        gain = gains[f]
        amplitude = 1 if family == "block-gain" else gain
        offset, ppm, cfo = (policy[split+key][r] for key in ("_offsets", "_ppm", "_cfo_Hz"))
        x, _, origin, step, meta = channel(packet(profile, payload), offset=offset, ppm=ppm,
            cfo=cfo, prefix=37+f*11+r*23+held*17, amplitude=amplitude,
            family="noise", seed=seed, snr_db=policy["recipe"]["noise_SNR_dB"][split][r])
        truth = dict(origin=origin, step=step, CFO_Hz=cfo, metadata=meta,
                     used_for_selection=False)
        if family == "burst":
            start = int(.43*len(x))
            x[start:start+7+2*r] += gain*(17-13j)
        elif family == "phase-step":
            x[int(.61*len(x)):] *= np.exp(1j*(.43+.17*r+.13*held))
        elif family == "block-gain":
            block = policy[split+"_block_lengths"][r]
            x *= np.roll(gains, r)[np.arange(len(x))//block % len(gains)]
        elif family == "tone-switch":
            freq = policy[split+"_tones_Hz"][r]
            k, mid = np.arange(len(x)), len(x)//2
            phase = np.where(k < mid, k*freq, mid*freq+(k-mid)*(-.61*freq))
            x += .7*gain*np.exp(2j*np.pi*phase/16e6)
    return x, dict(spec=spec, seed=seed, payload_SHA256=hashlib.sha256(payload).hexdigest(),
                   truth_diagnostic_only=truth, profile="ordinary;not WS63")


def relative(a, b):
    numerator, denominator = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return numerator/denominator if denominator else (0.0 if not numerator else float("inf"))


def metrics(output, reference, stats, budget):
    error = output-reference
    local = [relative(error[k:k+64], reference[k:k+64]) for k in range(0, len(error), 64)]
    peak = float(np.max(abs(reference)))
    peak_error = float(np.max(abs(error)))/peak if peak else relative(error, reference)
    measured = dict(global_normalized_rms=relative(error, reference),
                    global_relative_peak=peak_error, local64_relative_rms=max(local),
                    range_events=sum(stats[name] for name in RANGE))
    # JSON cannot misrepresent an undefined zero-energy error as a successful zero.
    passed = all(np.isfinite(v) and v <= budget[k] for k, v in measured.items())
    return dict(values={k: v if np.isfinite(v) else None for k, v in measured.items()},
                nonfinite_metrics=[k for k, v in measured.items() if not np.isfinite(v)],
                worst_local64_start=64*int(np.argmax(local)), stats=stats, passed=bool(passed))


def filtered(model, x, groups):
    parts, pos, count = [], 0, 0
    while pos < len(x):
        end = min(pos+groups[count % len(groups)], len(x))
        for j in range(pos, end, 32):
            parts.append(model.feed_output_words(x[j:min(j+32, end)]))
        pos, count = end, count+1
    # Capture the actual EOF words using the same finite integer output port.
    model.stats["zero_flush_samples"] += 80
    parts += [model.feed_output_words(np.zeros(n)) for n in (32, 32, 16)]
    model.seal()
    words = tuple(np.concatenate([p[k] for p in parts]) for k in range(3))
    y = np.ldexp(words[0].astype(float), words[2])+1j*np.ldexp(words[1].astype(float), words[2])
    return y, words, model.stats


def numeric(x, split, spec, policy, output):
    h = pulse(np.arange(-40, 41)/4)
    h /= np.linalg.norm(h)
    reference = lfilter(h, [1.0], np.r_[x, np.zeros(80)])
    floating = SymmetricRRC()
    structural = np.r_[floating.feed(x), floating.finish()]
    peak = float(np.max(abs(reference)))
    structure_error = float(np.max(abs(structural-reference)))/(peak or 1)
    rows, arrays = {}, dict(reference=reference, floating_structure=structural)
    rng = np.random.Generator(np.random.SFC64(811621+spec["family_index"]*137+spec["repeat"]))
    for name in ("full12", "full14"):
        make = Full14MantissaRRC if name == "full14" else lambda: FullMantissaRRC(FixedRRCFormat(12, 18))
        y, words, stats = filtered(make(), x, [len(x)])
        groups = [[policy[split+"_block_lengths"][spec["repeat"]]],
                  list(rng.integers(1, 198 if split == "validation" else 174, 97))]
        exact = []
        for sizes in groups:
            _, alternative, counts = filtered(make(), x, sizes)
            exact.append(all(np.array_equal(a, b) for a, b in zip(words, alternative, strict=True))
                         and counts == stats)
        row = metrics(y, reference, stats, policy["numerical_budget"])
        rows[name] = row | dict(chunk_exact=exact)
        arrays.update({name+"_"+str(k): v for k, v in enumerate(words)})
        arrays[name+"_output"] = y
    path = output/(spec["id"]+"-nodes.npz")
    np.savez_compressed(path, **arrays)
    return dict(id=spec["id"], samples=len(x), real_inputs=len(x), virtual_EOF=80, models=rows,
                floating_structure_error=structure_error,
                floating_structure_pass=structure_error <= policy["float_symmetric_max_abs_over_reference_peak"],
                nodes_SHA256=sha(path), capture=None, BER=None, MAC=None,
                packet_receiver_or_ADC_gain_claim=False)


def replay_ports(words, mode, assumptions, stalls=()):
    """Actual generated36-bit raw words;independent69-bit RAM identity producer."""
    memory = RRC14Memory(mode, assumptions.product_buffers)
    pending, now, checked = [], 0, 0

    def product(owner, j):
        return (owner*0x7165A7B8394C27ABCDEF+j*0x6A45B193704E4F1) % 2**69

    def observer(event):
        nonlocal pending, now, checked
        n = event["source_write"]
        if n is not None:
            now = n+1
        response = memory.tick(event["cycle"],
            source_write=None if n is None else (n, int(words[n])), raw_reads=event["raw_reads"],
            product_writes=[(slot, j, product(owner, j))
                            for slot, j, owner in event["product_writes"]],
            product_reads=[(slot, j) for slot, j, _ in event["product_reads"]])
        assert response["products"] == pending
        pending = [(slot, j, product(owner, j)) for slot, j, owner in event["product_reads"]]
        for index, value in zip(event["raw_reads"], response["raw"], strict=True):
            if index >= 0 and index < now-128:
                assert value is None
            elif value is not None:
                assert value == (0 if index < 0 else int(words[index]))
                checked += 1
            else:
                assert mode == "dual" and memory.stats["raw_conflict_banks"] > 0

    result = PairRRC14Flow(assumptions).simulate(len(words), downstream_stalls=stalls,
                                               access_sink=observer)
    assert not pending and not memory.pending and not memory.stats["product_hazards"]
    assert memory.stats["source_writes"] == len(words)
    assert memory.stats["product_reads"] == memory.stats["product_writes"]
    return dict(assumptions=asdict(assumptions), mode=mode, stalls=stalls,
                result=result, memory=memory.stats, checked_raw_words=checked)


def cost_and_calendar(words):

    old, new = PairRRCFlow().resource_estimate(), PairRRC14Flow().resource_estimate()
    before, after = RRCMemory().estimate(), RRC14Memory().estimate()
    cases = []
    for clock in (100_000_000, 160_000_000, 200_000_000):
        for mode in ("copies", "dual", "quad"):
            a = PairRRCFlowAssumptions(clock_hz=clock)
            cases.append(replay_ports(words, mode, a))
    for change, stalls in [(dict(product_buffers=2), ()), (dict(product_read_ports=1), ()),
                           (dict(consumer_ii=14), ()), (dict(arithmetic_ii=2), ()),
                           ({}, [(100, 4100)])]:
        a = replace(PairRRCFlowAssumptions(), **change)
        cases.append(replay_ports(words, "quad", a, stalls))
    healthy = next(c for c in cases if c["mode"] == "quad" and
                   c["assumptions"] == asdict(PairRRCFlowAssumptions()) and not c["stalls"])
    calendar_pass = (healthy["result"]["continuous_input_without_loss"] and
                     not healthy["memory"]["unserved_raw_reads"])
    return dict(flow=new, memory=after, old_flow=old, old_memory=before, scenarios=cases,
        extra_logical_state_bits=new["register_tap_logical_bits"]-old["register_tap_logical_bits"],
        extra_memory_LUT6=after["raw_memory_luts"]-before["raw_memory_luts"],
        extra_DSP=new["multiplier_units"]-old["multiplier_units"],
        extra_ports=0 if new["ports"] == old["ports"] else None,
        extra_FIFO_or_owned_slots=0, extra_32bit_leading_detectors=(
            new["leading_detectors_32bit"]-old["leading_detectors_32bit"]),
        II_increase=new["necessary_ii"]-old["necessary_ii"],
        raw_generated_words_replayed=len(words), calendar_pass=bool(calendar_pass),
        product_arithmetic_cycle_integrated=False,
        product_memory_oracle="independent synthetic69bit;not FIR producer",
        physical_ADC_to_word_cycles=None, full_RX_cycles=None, actual_synthesis=False)


def run(output, split, development=None):
    policy = json.loads(POLICY.read_text())
    pinned = sources()
    if split == "validation":
        if development is None:
            raise ValueError("Validation needs qualified development;no IQ generated")
        receipt = json.loads((development/"qualification.json").read_text())
        if not receipt["qualified"] or receipt["sources"] != pinned:
            raise ValueError("Unqualified or changed development source;no validation IQ generated")
    configurations = specs(policy, split)
    output.mkdir(parents=True, exist_ok=False)
    write(output/"freeze-before-generation.json", dict(policy_SHA256=sha(POLICY), sources=pinned,
          configurations=configurations, dependencies=dict(python=platform.python_version(),
          numpy=np.__version__, scipy=scipy.__version__), validation=split == "validation"))
    rows, identities, raw_words = [], [], None
    for spec in configurations:
        x, meta = waveform(policy, split, spec)
        path = output/(spec["id"]+"-input.npz")
        np.savez_compressed(path, iq=x)
        identities.append(meta | dict(input_SHA256=sha(path)))
        row = numeric(x, split, spec, policy, output)
        row["input_SHA256"] = sha(path)
        rows.append(row)
        write(output/(spec["id"]+".json"), row)
        if raw_words is None:
            f = Full14MantissaRRC()
            parts = [f._quantize(x[j:j+32]) for j in range(0, 512, 32)]
            r, i, e = [np.concatenate([p[k] for p in parts]) for k in range(3)]
            raw_words = (r & 16383) | ((i & 16383) << 14) | ((e & 255) << 28)
    costs = cost_and_calendar(raw_words)
    budget = policy["cost_budget"]
    cost_pass = all(costs[k.removesuffix("_max")] is not None and
                    costs[k.removesuffix("_max")] <= v for k, v in budget.items())
    passed = all(r["models"]["full14"]["passed"] and r["floating_structure_pass"] and
                 all(r["models"]["full14"]["chunk_exact"]) for r in rows)
    passed = passed and cost_pass and costs["calendar_pass"]
    assert sources() == pinned, "Source changed during experiment"
    write(output/"inputs.json", identities)
    write(output/"results.json", rows)
    write(output/"costs.json", costs)
    result = dict(qualified=bool(passed), sources=pinned, split=split, inputs=len(rows),
        candidate_passed=sum(r["models"]["full14"]["passed"] for r in rows),
        baseline_passed=sum(r["models"]["full12"]["passed"] for r in rows), cost_pass=cost_pass,
        new_receiver_or_ADC_gain=None, all_four_RX_gates_passed=False, default_changed=False)
    write(output/"qualification.json", result)
    write(output/"manifest.json", {f.name: sha(f) for f in output.iterdir() if f.is_file()})
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}))
    return 0 if passed else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "validation"], required=True)
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    raise SystemExit(run(args.output, args.split, args.development))
