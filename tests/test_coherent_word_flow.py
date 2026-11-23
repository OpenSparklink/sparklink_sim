"""Independent integer/trigonometric checks of actual finite coherent operations."""

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from nearlink_sdr.phy.sync_coherent import coherent_sync_confirmation
from nearlink_sdr.phy.sync_fit_word import compile_sync_phases
from nearlink_sdr.sim.coherent_word_flow import CoherentWordReceipt, CoherentWordService
from nearlink_sdr.sim.fine_selection import FineSelection
from tests.captured_frame_reference import fixture


def source(words, correlation=(2**40, 0), exponent=0, *, available=7):
    count = len(words)
    window = {32: 256, 62: 376, 126: 632}[count]
    numerator = sum(v * v for v in correlation)
    energy = max(1, (numerator + count - 2) // (count - 1))
    return FineSelection(0, window, window, available, 8, exponent, tuple(words),
                         tuple(correlation), numerator, energy, count - 1)


def run(selection, codes, *, sink=None):
    flow = CoherentWordService(selection, codes, sink=sink)
    while not flow.ready:
        flow.step()
    return flow, flow.take()


def integer_reference(selection, codes, step):
    """Independent quarter-ROM definition and wide Python/Fraction reference.

    No executable word-fabric helper or phase_rotation_words call is shared.
    Python wide arithmetic is allowed here because this is the test oracle.
    """
    cr, ci, energy = 0, 0, 0
    for k, ((r, i), code) in enumerate(zip(selection.words, codes, strict=True)):
        phase = ((k * step) + (code << 16)) % 2**20
        address, remainder = divmod(phase, 256)
        address += 2 * remainder > 256 or (2 * remainder == 256 and address & 1)
        angle = (address % 4096) * math.tau / 4096
        c, s = round(math.cos(angle) * 65536), round(-math.sin(angle) * 65536)
        cr += r * c - i * s
        ci += r * s + i * c
        energy += r * r + i * i
    numerator = cr * cr + ci * ci
    denominator = energy * len(codes) * 2**32
    return (cr, ci), energy, numerator, denominator


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("omega", [-math.pi, -0.73, 0.0, 0.319, math.pi - 1e-9])
def test_actual_profile_template_CFO_and_native_tiled_sums_against_independent_reference(ft, omega):
    profile, *_ = fixture(ft)
    codes = compile_sync_phases(profile.sync_symbols())
    rng = np.random.default_rng(410331 + ft)
    y = profile.sync_symbols() * np.exp(1j * (0.41 + omega * np.arange(len(codes))))
    y = y * 700003 + (rng.normal(size=len(y)) + 1j * rng.normal(size=len(y))) * 17001
    words = tuple((round(z.real), round(z.imag)) for z in y)
    c = (round(math.cos(omega) * 2**45), round(math.sin(omega) * 2**45))
    selected = source(words, c, exponent=-23)
    flow, receipt = run(selected, codes)
    ref = integer_reference(selected, codes, receipt.step_word20)
    assert (receipt.correlation, receipt.signal_energy, receipt.numerator, receipt.denominator) == ref
    angle = math.atan2(c[1], c[0])
    phase_error = math.remainder(receipt.angle_turn40 * math.tau / 2**40 - angle, math.tau)
    step_error = math.remainder(receipt.step_word20 * math.tau / 2**20 - angle, math.tau)
    assert abs(phase_error) < 1e-6 and abs(step_error) <= 3.1e-6
    floating = coherent_sync_confirmation(np.array(words)[:, 0] + 1j * np.array(words)[:, 1],
                                         profile.sync_symbols(), angle, 0.45)
    assert abs(receipt.quality - floating["quality"]) <= 0.005
    assert receipt.accepted is floating["accepted"] is True
    assert flow.stats["native_products"] == 8 * len(codes) + 12
    assert flow.stats["cache_reads"] == flow.stats["coefficient_outputs"] == len(codes)
    assert flow.stats["cordic_steps"] == 32
    assert 0 < flow.cycle - flow.started < flow.estimate()["finite_work_bound_edges"]
    assert flow.fabric.idle and flow.fabric.stats["issued"] == flow.fabric.stats["taken"]


@pytest.mark.parametrize("change,accepted", [(-1, True), (0, True), (1, False)])
def test_exact_squared_threshold_and_both_neighbor_sides_without_CRC_retry(change, accepted):
    # SumI=0,SumR=96,energy640:Q=96²/(32*640)=9/20 exactly.
    imaginary = [4, -4] * 8 + [3, -3] * 5 + [1, -1] * 3
    imaginary[0], imaginary[1] = 4 + change, -4 - change
    selected = source(tuple((3, i) for i in imaginary))
    _, receipt = run(selected, (0,) * 32)
    assert receipt.accepted is accepted
    q = Fraction(receipt.numerator, receipt.denominator)
    assert (q == Fraction(9, 20)) is (change == 0)
    assert (q >= Fraction(9, 20)) is accepted
    if change == 0:
        assert receipt.comparison == 0


@pytest.mark.parametrize("exponent", [-512, -31, 0, 37, 511])
def test_common_exponent_cancels_and_no_underflow_is_used_to_accept(exponent):
    selected = source(((12347, -47),) * 32, exponent=exponent)
    _, receipt = run(selected, (0,) * 32)
    assert receipt.quality == 1.0 and receipt.accepted
    assert (receipt.correlation, receipt.signal_energy, receipt.numerator, receipt.denominator) == (
        integer_reference(selected, (0,) * 32, receipt.step_word20)
    )
    if exponent == 511:
        with pytest.raises(OverflowError):
            receipt.reference_confirmation()


def test_zero_cache_energy_and_zero_upstream_differential_never_report_success():
    _, zero = run(source(((0, 0),) * 32), (0,) * 32)
    assert zero.signal_energy == zero.denominator == 0 and not zero.accepted
    assert zero.quality == 0 and zero.reference_confirmation()["zero_energy"]
    _, no_frequency = run(source(((1047, 0),) * 32, correlation=(0, 0)), (0,) * 32)
    assert no_frequency.quality == 1.0 and not no_frequency.accepted


@pytest.mark.parametrize("correlation", [(2**52 - 1, 0), (-(2**52), 0),
                                        (0, 2**52 - 1), (0, -(2**52)),
                                        (-(2**52), 1), (-(2**52), -1)])
def test_extreme_source_words_axes_and_branch_cut_do_not_overflow_or_change_alias(correlation):
    words = tuple((-(2**21) if k % 2 else 2**21 - 1, 2**21 - 1) for k in range(126))
    selected = source(words, correlation)
    _, receipt = run(selected, tuple(k % 16 for k in range(126)))
    expected = math.atan2(correlation[1], correlation[0])
    assert abs(math.remainder(receipt.angle_turn40 * math.tau / 2**40 - expected,
                              math.tau)) < 1e-6
    assert (receipt.correlation, receipt.signal_energy, receipt.numerator, receipt.denominator) == (
        integer_reference(selected, receipt.phase_codes, receipt.step_word20)
    )


def test_single_cache_port_registered_reads_held_output_and_descriptor_identity():
    selected = source(((79213, 1591),) * 32)
    events = []
    flow = CoherentWordService(selected, (0,) * 32, sink=events.append)
    assert flow.take() is None
    while not flow.ready:
        flow.step()
    ready = flow.result
    before = dict(flow.stats)
    for _ in range(7):
        flow.step()
    assert flow.result is ready and flow.stats["cache_reads"] == before["cache_reads"]
    assert flow.stats["output_hold_edges"] == 7
    reads = [r for r in events if r["event"] == "coherent_cache_read"]
    assert [r["index"] for r in reads] == list(range(32))
    assert len({r["cycle"] for r in reads}) == 32
    assert flow.take() is ready and ready.selection is selected and flow.take() is None
    with pytest.raises(RuntimeError, match="Retired"):
        flow.step()


def test_completed_receipt_survives_backpressure_longer_than_active_compute_budget():
    flow = CoherentWordService(source(((37, 5),) * 32), (0,) * 32)
    while not flow.ready:
        flow.step()
    result, products = flow.result, flow.stats["native_products"]
    for _ in range(32769):
        flow.step()
    assert flow.result is result and flow.stats["native_products"] == products
    assert flow.stats["output_hold_edges"] == 32769 and flow.take() is result


@pytest.mark.parametrize("cut", [0, 1, 33, 37, 150, 991])
def test_registered_abort_clears_native_work_and_prevents_stale_receipt(cut):
    events = []
    flow = CoherentWordService(source(((31963, -81),) * 32), (0,) * 32, sink=events.append)
    for _ in range(cut):
        flow.step()
    clock = flow.cycle
    flow.abort()
    assert flow.cycle == clock + 1 and flow.fabric.idle and flow.take() is None
    assert flow.result is None and flow.stats["cancelled"] == 1
    assert flow.stats["native_products"] == sum(
        r["event"] == "word_issue" and r["command"]["code"] == "tile" for r in events
    )
    with pytest.raises(RuntimeError, match="Retired"):
        flow.step()
    with pytest.raises(RuntimeError, match="Retired"):
        flow.abort()


def test_malformed_context_or_forged_receipt_does_not_pass_as_actual_confirmation():
    selected = source(((7, 0),) * 32)
    for codes in (None, (0,) * 31, (16,) * 32, (True,) * 32):
        with pytest.raises(ValueError):
            CoherentWordService(selected, codes)
    for clock in (True, 0, 2**63 - 1):
        with pytest.raises(ValueError):
            CoherentWordService(selected, (0,) * 32, initial_cycle=clock)
    _, receipt = run(selected, (0,) * 32)
    with pytest.raises(ValueError, match="Inconsistent"):
        replace(receipt, numerator=receipt.numerator + 1)
    with pytest.raises(ValueError, match="Inconsistent"):
        replace(receipt, comparison=-1)
    with pytest.raises(ValueError):
        CoherentWordReceipt(None, (), 0, 0, (0, 0), 0, 0, 0, 0, 1)


def test_concrete_state_native_widths_ports_and_unproven_system_gates_are_separate():
    cost = CoherentWordService.estimate()
    assert cost["register_bits_upper"] == cost["fabric"]["state_bits"] + sum(
        cost["register_fields"].values()
    )
    assert cost["register_bits_upper"] <= cost["register_research_cap_bits"] == 8192
    assert cost["DSP25x18"] == cost["ALU32"] == cost["cache_read_ports"] == 1
    assert cost["coefficient_RAM18"] == cost["coefficient_BRAM36_sites_upper"] == 1
    assert cost["template_ROM_bits"] == 512 and cost["coefficient_ROM_read_ports"] == 2
    assert cost["CORDIC"]["own_angle_ROM_bits"] == 1280
    assert cost["CORDIC"]["own_angle_LUT6_ROM_upper"] == 40
    assert cost["normalization"]["signed53_RNE_shifters"] == 2
    assert cost["register_fields"]["coefficient_address_ROM_pipeline"] == 94
    assert cost["new_IQ_RAM"] == cost["divide_sqrt_or_runtime_trig_in_acceptance"] == 0
    assert cost["full_RX_latency"] is cost["LUT_FF_or_achieved_Fmax"] is None
    assert not cost["full_fixed_RX"] and not cost["continuous16MSps"]
