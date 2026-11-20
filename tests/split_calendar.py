"""Finite word-pair bench with a known node count;not an ADC or legal-frame scheduler.

The functional oracle is diagnostic only. Dispatch never reads it. All queues
and ownership live independently of that oracle and have fixed capacities.
"""

import math

import numpy as np

from nearlink_sdr.phy.timing_word import TimingWordFormat
from nearlink_sdr.sim.feedback_timing_word import FeedbackAgeTimingState
from nearlink_sdr.sim.split_timing_flow import (
    TimingDetectorService,
    TimingPIService,
    split_timing_states,
)


def run_node_calendar(period, age, blocked):
    fmt, hold, count = TimingWordFormat(30, 32), 32, 320
    capacity = age + 2
    rng = np.random.default_rng(2026112103)

    def vector():
        mask = (1 << 22) - 1
        return [(int(r) & mask) | ((int(i) & mask) << 22) | ((-18 & 255) << 44)
                for r, i in rng.integers(-(1 << 18), 1 << 18, (count, 2))]

    data, middle = vector(), vector()
    arrivals = [n * period + (0, -1, 0, 1)[n % 4] for n in range(count)]
    detector, pi = split_timing_states((1 << 50, -50), 4337, hold, fmt)
    ds, ps = TimingDetectorService(detector, divider="radix4-prefix"), TimingPIService(pi)
    reference = FeedbackAgeTimingState((1 << 50, -50), 4337, hold, fmt, error_age=age)
    golden = []  # Host-only checker storage;never part of the dispatch path.
    for n, word in enumerate(data):
        reference.advance_words(word, middle[n] if n >= hold else None)
        golden.append(reference.snapshot())
    det_messages, pi_messages, error_messages = {}, {}, {}
    peaks, accepted, joined, max_lead = [0, 0, 0], 0, 0, 0
    events, drop = [], None
    for cycle in range(count * period + 4096):
        assert ds.cycle == ps.cycle == cycle
        sink_block = blocked and hold * period + 1 <= cycle < hold * period + 641
        if not sink_block and joined in det_messages and joined in pi_messages:
            a, b = det_messages.pop(joined), pi_messages.pop(joined)
            g = golden[joined]
            state = g["timing"]
            assert a["count"] == b["count"] == state["count"] == joined + 1
            for k in ("power", "error_word", "error_clip_events", "power_floor_events",
                      "power_alignment_erased", "ted_alignment_erased"):
                assert a[k] == state[k]
            for k in ("previous_q32", "next_q32", "omega_q32", "rate_limit_events"):
                assert b[k] == state[k]
            assert a["previous_word"] == data[joined]
            assert b["applied_error_word"] == g["last_applied_error_word"]
            events.append(dict(cycle=cycle, kind="joined", index=joined, detector=a, PI=b))
            joined += 1
        if ds.ready and len(det_messages) < capacity:
            row = ds.result
            index = row["count"] - 1
            if index < hold or len(error_messages) < capacity:
                det_messages[index] = ds.take()
                if index >= hold:
                    error_messages[index] = cycle + 1, row["error_word"]
                events.append(dict(cycle=cycle, kind="detector_result", index=index))
        if ps.ready and len(pi_messages) < capacity:
            row = ps.take()
            index = row["count"] - 1
            pi_messages[index] = row
            events.append(dict(cycle=cycle, kind="PI_result", index=index))
        if accepted < count and cycle == arrivals[accepted]:
            if not ds.input_ready:
                drop = dict(cycle=cycle, index=accepted, reason="detector_busy_or_held_output")
                break
            n = accepted
            assert ds.issue(data[n], middle[n] if n >= hold else None)
            accepted += 1
            events.append(dict(cycle=cycle, kind="input", index=n))
        n = pi.count
        if ps.input_ready and n < count and n < accepted + age:
            message_index = n - age
            if n < hold + age:
                error = 0
            elif (message_index in error_messages
                  and error_messages[message_index][0] <= cycle):
                _, error = error_messages.pop(message_index)
            else:
                error = None
            if error is not None:
                assert ps.issue(error)
                events.append(dict(cycle=cycle, kind="PI_issue", index=n,
                                   source_error_index=None if n < hold + age else message_index,
                                   error=error))
        for k, messages in enumerate((det_messages, pi_messages, error_messages)):
            peaks[k] = max(peaks[k], len(messages))
            assert len(messages) <= capacity
        max_lead = max(max_lead, pi.count - detector.count)
        for bank, service in (("detector", ds), ("PI", ps)):
            if service.mode == "products" and service.age in (1, 3, 4):
                events.append(dict(cycle=cycle, kind="primitive", bank=bank, role=service.role,
                                   stage={1: "4DSP_capture", 3: "2x80bit_pair_add",
                                          4: "80bit_final_add"}[service.age]))
        if joined == count:
            break
        ds.step()
        ps.step()
    assert joined == count or drop is not None
    assert max_lead <= age + 1
    primitive = {}
    for e in events:
        if e["kind"] == "primitive":
            primitive.setdefault((e["cycle"], e["stage"]), []).append(e)
    conflicts = [dict(cycle=k[0], stage=k[1], owners=v)
                 for k, v in primitive.items() if len(v) > 1]
    # Conservative register implementation:includes every held snapshot,tag,
    # valid bit,pointer and controller counter;no queue is a free Python dict.
    queues = (capacity * (detector.persistent_bits() + pi.persistent_bits() + 100 + 2)
              + 16 + 12 * 64
              + 3 * (2 * math.ceil(math.log2(capacity)) + math.ceil(math.log2(capacity + 1))))
    upper = (ds.estimate()["state_register_bits_upper"]
             + ps.estimate()["state_register_bits_upper"] + queues)
    return dict(
        accepted=accepted, joined=joined, drop=drop, capacity_each=capacity,
        peak_det_PI_error_messages=peaks, max_PI_count_ahead_of_detector=max_lead,
        unused_tail_errors_discarded=len(error_messages),
        shared4DSP_or_tree_conflicts=conflicts, exclusive_two_bank_DSP=8,
        queue_and_controller_logical_bits_upper=queues,
        nodes_and_queues_logical_bits_upper=upper, old4096_node_budget_passed=upper <= 4096,
        actual_query_interpolation_source_handoff=False, header_boundary_gate_included=False,
        new_RF_holdout=False, whole_receiver16MSps=False, complete_latency_cycles=None,
        synthesis=False,
    ), events
