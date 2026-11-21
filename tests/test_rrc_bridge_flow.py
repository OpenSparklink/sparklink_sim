"""Registered values/credits and actual RRC emission clocks,including overload."""

from copy import deepcopy

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.rrc_bridge_flow import RRCScaledBridgeService
from tests.test_rrc_bridge import oracle


@pytest.mark.parametrize("bits", range(8, 15))
def test_real_three_registered_stages_hold_exact_words_and_one_credit(bits):
    service = RRCScaledBridgeService(ScaledBankFormat(bits), start_cycle=17)
    rows = [(0, 0, -96), (0, 0, 48), (-(2**18), 2**18 - 1, 48), (1, -1, -96)]
    rows += [
        (int(r), int(i), int(e))
        for r, i, e in np.random.default_rng(372717).integers(
            [-(2**18), -(2**18), -96], [2**18, 2**18, 49], size=(50, 3)
        )
    ]
    totals = dict(
        exponent_underflows=0, exponent_overflows=0, input_saturations=0, components_erased=0
    )
    for index, row in enumerate(rows):
        assert service.input_ready and service.issue(pack_filter_word(*row), index)
        assert not service.issue(0, index + 1) and service.take() is None
        start = service.cycle
        for mode in ("round", "clip", "ready"):
            service.step()
            assert service.mode == mode
            if not service.ready:
                assert service.take() is None
        assert service.cycle == start + 3
        values, events = oracle(*row, bits)
        expected = dict(tag=index, word=values, events=events)
        assert service.result == expected
        for _ in range(index % 4):
            service.step()
            assert service.result == expected and not service.input_ready
        assert service.take() == expected and service.take() is None
        assert service.result is None and not service.issue(0, index + 1)
        for key, value in events.items():
            totals[key] += value
        service.step()  # Registered credit release,no take/issue fallthrough.
    assert service.stats == dict(
        accepted=len(rows),
        completed=len(rows),
        retired=len(rows),
        held_edges=sum(j % 4 for j in range(len(rows))),
        aborted=0,
        **totals,
    )
    cost = service.estimate()
    assert cost["register_bits_upper"] == 842 <= 1024
    assert cost["DSP"] == cost["new_sample_RAM"] == 0
    assert cost["initiation_interval"] == 4
    assert cost["output_word_bits"] == 2 * bits + 7
    assert not cost["synthesis"] and not cost["old_fixed22_18DSP_cost_applicable"]


@pytest.mark.parametrize("steps", range(5))
def test_abort_has_no_stale_result_or_next_packet_receipt(steps):
    service = RRCScaledBridgeService()
    service.issue(pack_filter_word(3, -5, -17), 99)
    for _ in range(steps):
        service.step()
    service.abort()
    assert service.take() is None and service.result is None and not service.issue(0, 100)
    assert service.stats["aborted"] == 1
    service.abort()
    assert service.stats["aborted"] == 1
    with pytest.raises(RuntimeError, match="Aborted"):
        service.step()


def test_invalid_descriptors_and_counter_exhaustion_precede_state_mutation():
    for word, tag in [
        (2**46, 0),
        (True, 0),
        (0, True),
        (0, -1),
        (0, 2**63),
        (pack_filter_word(1, 0, 49), 0),
    ]:
        service = RRCScaledBridgeService()
        before = deepcopy(service.__dict__)
        with pytest.raises(ValueError):
            service.issue(word, tag)
        assert before == service.__dict__
    for options in (
        dict(fmt=True),
        dict(start_cycle=True),
        dict(start_cycle=-1),
        dict(start_cycle=2**63 - 1),
    ):
        with pytest.raises(ValueError):
            RRCScaledBridgeService(**options)
    service = RRCScaledBridgeService()
    service.stats["accepted"] = 2**64 - 1
    before = deepcopy(service.__dict__)
    with pytest.raises(OverflowError):
        service.issue(0, 0)
    assert service.__dict__ == before
    service = RRCScaledBridgeService()
    service.issue(0, 1)
    service.step()
    service.step()
    service.stats["completed"] = 2**64 - 1
    before = deepcopy(service.__dict__)
    with pytest.raises(OverflowError):
        service.step()
    assert service.__dict__ == before


def test_clock_exhaustion_does_not_wrap_or_change_a_held_receipt():
    service = RRCScaledBridgeService(start_cycle=2**63 - 4)
    service.issue(0, 2**63 - 1)
    for _ in range(3):
        service.step()
    before = deepcopy(service.__dict__)
    with pytest.raises(OverflowError):
        service.step()
    assert service.__dict__ == before and service.ready


@pytest.mark.parametrize("blocked", [False, True])
def test_actual_RRC_emission_words_do_not_arrive_early_or_backpressure_the_ADC(blocked):
    rng = np.random.default_rng(372718)
    x = rng.normal(size=128) + 1j * rng.normal(size=128)
    rrc = FullMantissaRRC(FixedRRCFormat(12, 18))
    groups = [rrc.feed_output_words(x[j : j + 32]) for j in range(0, len(x), 32)]
    words = [
        pack_filter_word(*row)
        for row in zip(*[np.concatenate([a[k] for a in groups]) for k in range(3)], strict=True)
    ]
    producer = PairRRCFlow().simulate(len(x), trace=True)
    assert producer["emitted"] == 128 and not producer["dropped"] and not producer["invalid"]
    arrivals = {e["cycle"]: e["sample"] for e in producer["trace"] if e["event"] == "output"}
    service = RRCScaledBridgeService()
    accepted, emitted, drops = [], [], []
    while service.cycle <= producer["cycles"] + 8:
        cycle = service.cycle
        if service.ready and not (blocked and 64 <= cycle < 200):
            result = service.take()
            values, events = oracle(*rrc_word_tuple(words[result["tag"]]), 10)
            assert result == dict(tag=result["tag"], word=values, events=events)
            emitted.append((result["tag"], cycle))
        if cycle in arrivals:
            tag = arrivals[cycle]
            (accepted if service.issue(words[tag], tag) else drops).append((tag, cycle))
        service.step()
    assert [tag for tag, _ in emitted] == [tag for tag, _ in accepted]
    assert len(accepted) + len(drops) == 128
    if not blocked:
        assert not drops
        assert [(tag, cycle + 3) for tag, cycle in accepted] == emitted
    else:
        assert drops and service.stats["held_edges"] > 0
        assert len({tag for tag, _ in emitted}) == len(emitted)


def rrc_word_tuple(word):
    # Explicit independent wire extraction for the cycle oracle input.
    fields = []
    for shift, bits in ((0, 19), (19, 19), (38, 8)):
        v = word >> shift & ((1 << bits) - 1)
        fields.append(v - (1 << bits) if v & (1 << (bits - 1)) else v)
    return fields
