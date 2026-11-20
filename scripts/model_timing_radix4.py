"""Exact two-bit timing division and finite suffix calendars; expected exit1.

New independent arithmetic validation is separate from viewed RF regressions.
No new RF threshold,loop gain,word format or reception-holdout claim.
"""

import argparse
import gzip
import json
from functools import partial
from pathlib import Path
from random import Random

import numpy as np
from audit_capture_hardware import sha, write
from model_header_words import source_identity
from model_live_head import FIELDS, GEOMETRY_FIELDS, TRACE, cases, save_event, suffix
from tests.test_head_word_replay import run as replay
from tests.test_timing_radix4 import ratio

from nearlink_sdr.common import polar
from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.head_word_replay import HeadWordsFrameReplay

POLICY = Path("tests/data/timing_radix4_policy.json")
MODES = ("bounded-prefix", "radix4-prefix")


def arithmetic(output, policy, split):
    """Independent integer divmod and Fraction RNE oracle; no CRC or payload."""
    from fractions import Fraction

    rng = Random(policy[split + "_seed"])
    rows, count = [], 0
    with gzip.open(output / "arithmetic.jsonl.gz", "wt") as log:
        for p, e in [(30, 24), (30, 32), (38, 32)]:
            fmt = TimingWordFormat(p, e)
            savings, digits = set(), [0, 0, 0, 0]
            divisors = [1, 2, 3, (1 << 29) - 1, 1 << 29, (1 << 49) - 1]
            divisors += [rng.randrange(1, 1 << 49)
                         for _ in range(policy["random_denominators_per_format"])]
            for d in divisors:
                maximum = min((1 << fmt.division_bits) - 1, d << (e + 1))
                values = [0, 1, maximum, maximum - 1, rng.randrange(maximum + 1)]
                if d % 2 == 0:
                    values += [d // 2, 3 * d // 2, 5 * d // 2]
                for n in values:
                    if not 0 <= n <= maximum:
                        continue
                    for negative in (False, True):
                        a, b = [ratio(fmt, n, d, negative, m) for m in MODES]
                        expected = round(Fraction(-n if negative else n, d))
                        exact = (a.job["error"] == b.job["error"] == expected
                                 and (a.quotient, a.remainder) == (b.quotient, b.remainder)
                                 == divmod(n, d))
                        record = dict(power=p, error=e, numerator=n, denominator=d,
                                      negative=negative, expected=expected,
                                      quotient=b.quotient, remainder=b.remainder,
                                      edges=[a.cycle, b.cycle], exact=exact)
                        log.write(json.dumps(record, separators=(",", ":")) + "\n")
                        if not exact:
                            write(output / "arithmetic-failure.json", record)
                            raise AssertionError("Independent ratio reference differs")
                        savings.add(a.cycle - b.cycle)
                        q = b.quotient
                        for _ in range((e + 2) // 2):
                            digits[q & 3] += 1
                            q >>= 2
                        count += 1
            assert savings == {(e + 2) // 2 - 1} and all(digits)
            old, new = a.estimate(), b.estimate()
            assert new["state_register_bits_upper"] - old["state_register_bits_upper"] == 51
            assert new["DSP25x18"] == old["DSP25x18"] == 4
            assert new["extra_raw_RAM_bits"] == new["additional_source_RAM_ports"] == 0
            assert policy["budgets"]["additional_register_bits_max"] >= 51
            rows.append(dict(power=p, error=e, exact=True, saved_edges=list(savings),
                             digit_counts=digits, before=old, after=new))
    return dict(independent_RNE_divmod_exact=True, cases=count, formats=rows,
                transcript_SHA256=sha(output / "arithmetic.jsonl.gz"))


def fixed_frames(output, policy):
    rows, resources = [], []
    for name, p, values, expected, selection, provenance in cases():
        from tests.test_selected_word import quantize_input

        words = quantize_input(np.r_[values, np.zeros(8000)])
        selection = selection | dict(sample_limit=len(words))
        np.savez_compressed(output / (name + "-input.npz"), words=words,
                            payload=np.frombuffer(expected, np.uint8), **selection)
        for layout in ("dense", "depth-parity"):
            before, old, _ = replay(p, words, selection,
                                   partial(HeadWordsFrameReplay, head_layout=layout))
            save_event(output, name + "-" + layout + "-before", before)
            for label, pattern in [("32", (32,)), ("fixed", (1, 7, 31)),
                                   ("random", tuple(Random(11201021).choices(range(1, 33), k=23)))]:
                after, new, barriers = replay(p, words, selection, partial(
                    HeadWordsFrameReplay, head_layout=layout, timing_divider=MODES[1]), pattern)
                save_event(output, name + "-" + layout + "-" + label, after)
                for key in FIELDS + GEOMETRY_FIELDS:
                    assert before["frame"].get(key) == after["frame"].get(key)
                for key in TRACE:
                    np.testing.assert_array_equal(before[key], after[key])
                assert old.consumer.word_timing.snapshot() == new.consumer.word_timing.snapshot()
                assert (old.consumer.normalization.snapshot()
                        == new.consumer.normalization.snapshot())
                checks = new.timing_service.normal.stats["prefix_checks"]
                assert old.cycle - new.cycle == 16 * checks
                assert [x.mac_payload for x in new.take_mac_result().packets] == [expected]
                assert new.history.end == old.history.end and new.history.head == old.history.head
                for i in range(new.history.head, new.history.end):
                    assert new.history.words[i % 512] == words[i]
                rows.append(dict(id=name, layout=layout, group=label, provenance=provenance,
                                 input_SHA256=sha(output / (name + "-input.npz")), exact=True,
                                 modeled_edges=[old.cycle, new.cycle], prefix_checks=checks,
                                 payload_bits=8 * len(expected), bit_errors=0, barriers=barriers,
                                 complete_latency_cycles=None))
                write(output / "fixed-frames.json", rows)
        if name in ("legacy-ft3", "standard-ft3-crc32"):
            for mode in MODES:
                sub = output / (name + "-" + mode)
                sub.mkdir()
                for layout in ("dense", "depth-parity"):
                    for clock in policy["clocks_Hz"]:
                        row = suffix(sub, p, words, selection, input_id=name, layout=layout,
                                     clock=clock, continue_data=True, timing_divider=mode)
                        assert row["source_capacity_words"] == 512
                        resources.append(row | dict(divider=mode))
                        write(output / "resource-prefixes.json", resources)
    return rows, resources


def run(output, *, freeze=None):
    if output.exists():
        raise ValueError("Use a fresh evidence directory")
    policy, source = json.loads(POLICY.read_text()), source_identity()
    split = "development" if freeze is None else "validation"
    if freeze is not None:
        frozen = json.loads(freeze.read_text())
        assert frozen == dict(source=source, policy=policy)
    output.mkdir(parents=True)
    write(output / "protocol.json", dict(source=source, policy=policy, split=split,
                                        new_RF_holdout=False))
    nodes = arithmetic(output, policy, split)
    rows, resources = fixed_frames(output, policy) if freeze is None else ([], [])
    result = dict(stage=policy["stage"], split=split, arithmetic=nodes,
                  fixed_frame_comparisons=len(rows), fixed_success_losses=0,
                  receiver_rate_gain=0, new_RF_holdout=False,
                  actual_reference_rust=polar._HAS_RUST_ACCEL,
                  no_Rust_timing_word_implementation=True,
                  complete_scope_resource_exits=sum(r["frame"] is not None for r in resources),
                  resource_calendars=len(resources), resources_bounded=True,
                  prospective_component_register_increment_bits=51,
                  full_RX_gates_passed=False, default_changed=False,
                  synthesized=False, expected_exit=1)
    assert source_identity() == source
    write(output / "model.json", result)
    if freeze is None:
        write(output / "freeze.json", dict(source=source, policy=policy))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    run(args.output, freeze=args.freeze)
    raise SystemExit(1)
