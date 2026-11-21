"""Independent deque maxima,actual profile coefficients and unchanged score decisions."""

from collections import deque
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_bank import normalized_greater, sign_coefficients
from nearlink_sdr.phy.acquisition_maximum import (
    ExponentWindow,
    RollingMaximumRankedBank,
    RollingMaximumScaledBank,
)
from nearlink_sdr.phy.acquisition_scaled import ScaledBankFormat, ScaledDifferentialBank
from nearlink_sdr.phy.coarse_rank import RankedCoarseBank
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def reference(frame):
    cfg = TxConfig(
        frame_type=frame,
        mcs_index=8,
        ctrl_bits_len=28 if frame == 2 else 27,
        head_llid=None if frame == 2 else 0x351729,
    )
    return PSKReceiveProfile.ordinary(cfg).sync_symbols()


def events(terms, length=2500):
    rng = np.random.default_rng(381209 + terms)
    history = [deque([(0, False)] * terms) for _ in range(4)]
    for tag in range(length):
        phase = tag % 4
        # Exercise empty→strong→weak→empty,extreme bins and same-bin replacements.
        exponent = int(rng.integers(-96, 65)) if tag % 5 else [-96, -95, -81, 63, 64][tag % 25 // 5]
        incoming = exponent, bool((tag // (terms * 4)) % 4 != 0)
        outgoing = history[phase].popleft()
        history[phase].append(incoming)
        maximum = max((e for e, nz in history[phase] if nz), default=-96)
        yield tag, outgoing, incoming, maximum


@pytest.mark.parametrize("terms", [31, 61, 125])
def test_histogram_equals_independent_phase_deques_and_every_bin(terms):
    node = ExponentWindow(terms)
    for tag, outgoing, incoming, expected in events(terms):
        assert node.advance(tag, outgoing, incoming) == expected
        phase = tag % 4
        bitmap = sum((count > 0) << j for j, count in enumerate(node.counts[phase]))
        assert bitmap == node.bitmaps[phase]
        assert node.totals[phase] == sum(node.counts[phase])
        assert node.groups[phase] == sum(
            bool((bitmap >> (16 * j)) & 0xFFFF) << j for j in range(11)
        )
    assert node.stats["events"] == 2500
    assert node.estimate()["extra_IQ_history_bits"] == 0


@pytest.mark.parametrize("terms", [31, 61, 125])
def test_replacing_last_maximum_and_same_bin_keeps_exact_membership(terms):
    node = ExponentWindow(terms)
    assert node.advance(0, (0, False), (64, True)) == 64
    for tag in range(1, 4):
        node.advance(tag, (0, False), (-96, True))
    assert node.advance(4, (64, True), (-96, True)) == -96
    for tag in range(5, 8):
        node.advance(tag, (-96, True), (-96, True))
    assert node.advance(8, (-96, True), (0, False)) == -96
    assert node.groups[0] == node.bitmaps[0] == node.totals[0] == 0
    assert node.maximum(1) == -96 and node.totals[1] == 1


@pytest.mark.parametrize(
    "tag,outgoing,incoming",
    [
        (1, (0, False), (0, False)),
        (True, (0, False), (0, False)),
        (0, (-97, False), (0, False)),
        (0, (0, False), (65, True)),
        (0, (0, False), (0, 1)),
        (0, (0, True), (0, False)),
        (0, (0,), (0, False)),
        (0, (0, False), (0.5, True)),
    ],
)
def test_bad_descriptor_or_absent_outgoing_does_not_mutate(tag, outgoing, incoming):
    node = ExponentWindow(31)
    with pytest.raises(ValueError):
        node.advance(tag, outgoing, incoming)
    assert node.next_tag == sum(node.totals) == sum(node.stats.values()) == 0


def test_count_and_statistics_overflow_precedes_commit():
    node = ExponentWindow(31)
    node.totals[0] = node.counts[0][96] = 31
    node.bitmaps[0], node.groups[0] = 1 << 96, 1 << 6
    with pytest.raises(OverflowError):
        node.advance(0, (0, False), (0, True))
    assert node.counts[0][96] == 31 and node.next_tag == 0
    node = ExponentWindow(31)
    node.stats["entered"] = 2**64 - 1
    with pytest.raises(OverflowError):
        node.advance(0, (0, False), (0, True))
    assert node.counts[0][96] == 0 and node.next_tag == 0
    with pytest.raises(ValueError):
        node.maximum(4)


@pytest.mark.parametrize("frame,window", [(2, 256), (3, 376), (4, 632)])
@pytest.mark.parametrize("ranked", [False, True])
def test_actual_template_words_scores_gate_and_normalized_rank_are_identical(frame, window, ranked):
    ref = reference(frame)
    cr, ci = sign_coefficients(ref)
    assert bool(np.all(cr * cr + ci * ci == 1)) == (frame != 2)
    baseline = (RankedCoarseBank if ranked else ScaledDifferentialBank)(ref, 4, window)
    candidate = (RollingMaximumRankedBank if ranked else RollingMaximumScaledBank)(ref, 4, window)
    rng = np.random.default_rng(381271 + frame)
    words = [
        rng.integers(-512, 512, 1400),
        rng.integers(-512, 512, 1400),
        rng.integers(-48, 33, 1400),
    ]
    words[0][100:500], words[1][100:500] = 0, 0
    previous = None
    for start in range(0, 1400, 7):
        args = [a[start : start + 7] for a in words]
        old = baseline.advance_words(*args, inspect=True)
        new = candidate.advance_words(*args, inspect=True)
        assert old == new and baseline.stats == candidate.stats
        for p in new:
            if previous is not None:

                def score(point):
                    return (
                        Fraction(point.numerator, point.energy * point.coefficient_energy)
                        if point.energy
                        else Fraction(0)
                    )

                assert normalized_greater(p, previous) == (score(p) > score(previous))
            previous = p
        lo, hi = candidate.samples - 4 * candidate.k - 16, candidate.samples - 4 * candidate.k
        assert baseline.gate(lo, hi) == candidate.gate(lo, hi)
        if ranked:
            assert baseline.query_best(lo, hi) == candidate.query_best(lo, hi)
    assert candidate.logical_storage_bits > baseline.logical_storage_bits
    assert candidate.emulator_state_bytes > baseline.emulator_state_bytes
    assert candidate.worst_operations_per_sample()["other"]["exponent_max_terms"] == 0
    assert baseline.worst_operations_per_sample()["other"]["exponent_max_terms"] == len(ref) - 1


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(sps=2),
        dict(window_samples=300),
        dict(scaled_format=ScaledBankFormat(12)),
        dict(scaled_format="not a format"),
    ],
)
def test_experimental_geometry_and_format_guards(kwargs):
    args = dict(sps=4, window_samples=256) | kwargs
    with pytest.raises(ValueError):
        RollingMaximumScaledBank(reference(2), **args)


def test_new_bank_detects_foreign_history_and_preflights_bounded_statistics():
    bank = RollingMaximumScaledBank(reference(2), 4, 256)
    bank.maximum_state.stats["events"] = 2**64 - 2
    with pytest.raises(OverflowError):
        bank.advance_words([0, 0], [0, 0], [0, 0])
    assert bank.samples == bank.maximum_state.next_tag == 0
    bank.maximum_state.next_tag = 1
    with pytest.raises(ValueError):
        bank.advance_words([0], [0], [0])


def test_bad_input_is_atomic_and_mid_batch_history_fault_invalidates_epoch(monkeypatch):
    bank = RollingMaximumScaledBank(reference(2), 4, 256)
    with pytest.raises(ValueError):
        bank.advance_words([0], [0], [-49])
    assert bank.samples == bank.maximum_state.next_tag == 0
    advance = bank.maximum_state.advance

    def fault(tag, old, new):
        if tag == 1:
            raise ValueError("Injected borrowed-history fault")
        return advance(tag, old, new)

    monkeypatch.setattr(bank.maximum_state, "advance", fault)
    with pytest.raises(ValueError, match="Injected"):
        bank.advance_words([0, 0], [0, 0], [0, 0])
    assert bank.samples == 0 and bank.maximum_state.next_tag == 1
    with pytest.raises(ValueError, match="epoch"):
        bank.advance_words([0], [0], [0])


def test_stale_read_prepare_and_commit_cannot_reapply_a_window():
    node = ExponentWindow(31)
    request = node.request(0, (0, False), (0, True))
    read = node.read(request)
    prepared = node.prepare(read)
    with pytest.raises(ValueError, match="owned"):
        node.commit(prepared, owner=object())
    node.commit(prepared)
    with pytest.raises(ValueError, match="changed"):
        node.read(request)
    with pytest.raises(ValueError, match="changed"):
        node.prepare(read)
    with pytest.raises(ValueError, match="changed"):
        node.commit(prepared)
