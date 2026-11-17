"""Integer rounding, actual RRC word adapter, independent spline and overflow gates."""

import itertools
import json
from fractions import Fraction

import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from nearlink_sdr.phy import spline_fixed
from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.phy.spline_fixed import (
    FixedUniformCubicEighth,
    SplineFixedFormat,
    fixed_preparation_calendar,
    fixed_spline_cost,
)
from nearlink_sdr.phy.sync_candidates import search_candidate_grid
from tests.test_sync_refine import profile


def test_signed_round_to_nearest_even_uses_independent_fraction_reference():
    for n in range(-129, 130):
        for shift in range(9):
            assert spline_fixed._rne_shift(n, shift) == round(Fraction(n, 2**shift))
        for d in (1, 3, 7, 16, 29):
            assert spline_fixed._rne_ratio(n, d) == round(Fraction(n, d))
    assert spline_fixed._rne_shift(-3, -2) == -12


@pytest.mark.parametrize('ib,fb', itertools.product((18, 21, 24), (14, 16, 17)))
def test_equations_polynomial_and_interpolation_against_independent_reference(ib, fb):
    rng = np.random.default_rng(630117)
    for n in (4, 17, 64, 256, 640):
        t = np.linspace(0, 1, n)
        y = (t**3-2*t**2+.7*t+1)+1j*(.2*t**3+.3*t**2-t)
        y += .01*(rng.normal(size=n)+1j*rng.normal(size=n))
        node = FixedUniformCubicEighth(y, SplineFixedFormat(ib, fb))
        times = np.arange((n-1)*8+1)/8
        actual = np.concatenate([node(times[k:k+128]) for k in range(0, len(times), 128)])
        reference = CubicSpline(np.arange(n), y)(times)
        assert np.max(abs(actual-reference)) <= .0002*np.max(abs(y))
        assert node.stats['saturations'] == 0
        assert node.stats['products'] == 4*n-6
        assert isinstance(node.source_words, tuple) and isinstance(node.second_words, tuple)


def test_four_point_extremal_windows_do_not_wrap_or_erase_saturation():
    fmt = SplineFixedFormat(18, 14)
    for signs in itertools.product((-1, 1), repeat=4):
        x = np.asarray(signs)*1.9999
        node = FixedUniformCubicEighth(x, fmt)
        node(np.arange(25)/8)
        assert node.stats['saturations'] == 0
    assert node._bound(2**22, 'forced') == 2**21-1
    assert node._bound(-2**22, 'forced') == -2**21
    assert node.stats['saturations'] == 2
    assert node.stats['saturation_by_node'] == {'forced': 2}
    node.stats['roundings'] = 2**64-1
    with pytest.raises(OverflowError, match='counter'):
        node._round(3, 1)


def test_exact_quantized_rom_prefix_tail_and_single_dsp_operand_widths():
    for fb in (14, 16, 17):
        c, last = spline_fixed._tables(fb)
        for words in (c, last):
            prefix, tail = spline_fixed._prefix_tail(words)
            assert all(FixedUniformCubicEighth._lookup(prefix, tail, i) == v
                       for i, v in enumerate(words))
            assert all(0 <= v < 2**fb for v in words)
        cost = fixed_spline_cost(SplineFixedFormat(18, fb))
        assert cost['coefficient_words_before_compression'] == 1275
        assert cost['coefficient_words_after_compression'] <= 12
        assert cost['dsp25x18_tiles_per_work_coefficient_product'] == 1
        assert cost['terminal_whole_coefficient_is_not_operand']
    assert fixed_spline_cost(SplineFixedFormat(24, 17))['dsp25x18_tiles_per_work_coefficient_product'] == 2
    assert not cost['schedule_or_16msps_verified']
    selected = fixed_spline_cost(SplineFixedFormat(18, 14))
    assert selected['source_and_work_logical_bits'] == 20480
    assert (selected['source_tdp36_bram36'], selected['work_tdp36_bram36']) == (1, 2)
    assert selected['monitor_counter_count'] == 18 and selected['maximum_state_bits'] == 56
    assert selected['serial_preparation_cycles_scenario'] == 1544


def test_integer_adapter_actual_rrc_outputs_and_unequal_exponents_match_float_boundary():
    rng = np.random.default_rng(630118)
    x = rng.normal(size=256)+1j*rng.normal(size=256)
    fir = FullMantissaRRC(FixedRRCFormat(12, 18))
    blocks = [fir.feed_words(*fir._quantize(x[j:j+32])) for j in range(0, len(x), 32)]
    r, i, e = [np.concatenate([row[k] for row in blocks]) for k in range(3)]
    fmt = SplineFixedFormat(18, 14)
    for real, imag, exponents in [(r, i, e),
        ([2**18-1, -2**18, 1, -1], [1, -1, -1, 1], [127, 127, -128, -128]),
        ([2**18-1, -1, 1, -1], [1, -1, -1, 1], [127, 127, -128, -128])]:
        z = np.ldexp(np.asarray(real, float), exponents)+1j*np.ldexp(np.asarray(imag, float), exponents)
        direct = FixedUniformCubicEighth(z, fmt)
        words = FixedUniformCubicEighth.from_rrc_words(real, imag, exponents, fmt)
        assert direct.exponent == words.exponent
        assert direct.source_words == words.source_words and direct.second_words == words.second_words
        times = np.arange(min(128, 8*(len(real)-1)+1))/8
        assert np.array_equal(direct(times), words(times))
    assert words.stats['discarded_input_components'] > 0
    assert words.stats['common_exponent_adjustments'] == 1


def test_zero_constant_scalar_endpoint_empty_range_and_common_exponent():
    fmt = SplineFixedFormat(18, 14)
    zero = FixedUniformCubicEighth.from_rrc_words([0]*4, [0]*4, [-128, 0, 127, 1], fmt)
    assert zero.exponent == 0 and zero(.5) == 0 and zero([]).shape == (0,)
    json.dumps(FixedUniformCubicEighth([0, 1, 0, 1], fmt).stats)
    for power in (-220, -31, 0, 29, 220):
        value = complex(2.**power, -2.**power)
        node = FixedUniformCubicEighth(np.full(4, value), fmt)
        assert node(.125) == value and node(3) == value
        assert -512 <= node.exponent <= 511
    with pytest.raises(OverflowError, match='exponent'):
        node._set_exponent(512)
    with pytest.raises(OverflowError, match='exponent'):
        FixedUniformCubicEighth(np.full(4, 2.**-1000), fmt)


@pytest.mark.parametrize('fmt', [(17, 14), (18, 13), (True, 14), (18, True)])
def test_undeclared_formats_are_explicitly_rejected(fmt):
    with pytest.raises(ValueError):
        SplineFixedFormat(*fmt)


@pytest.mark.parametrize('values', [np.zeros(3), np.zeros(641), np.zeros((4, 2)),
                                     [1, 2, 3, np.nan]])
def test_nonfinite_or_unbounded_windows_are_rejected(values):
    with pytest.raises(ValueError):
        FixedUniformCubicEighth(values, SplineFixedFormat(18, 14))


def test_bad_adapter_words_and_formats_do_not_silently_cast_or_saturate():
    fmt = SplineFixedFormat(18, 14)
    cases = [([1]*3, [0]*3, [0]*3), ([0]*4, [0]*3, [0]*4),
             ([2**18]*4, [0]*4, [0]*4), ([0]*4, [0]*4, [128]*4),
             ([False]*4, [0]*4, [0]*4), ([.5]*4, [0]*4, [0]*4)]
    for args in cases:
        with pytest.raises(ValueError):
            FixedUniformCubicEighth.from_rrc_words(*args, fmt)
    with pytest.raises(ValueError):
        FixedUniformCubicEighth(np.zeros(4), None)
    with pytest.raises(OverflowError):
        FixedUniformCubicEighth(np.full(4, 1e100), fmt)
    for args in [(None, 256), (fmt, 3), (fmt, True)]:
        with pytest.raises(ValueError):
            fixed_spline_cost(*args)


@pytest.mark.parametrize('positions', [[-1], [4], [.1], [np.nan], [np.inf], [True],
                                      ['1'], [1+0j], np.ones((2, 2)), np.zeros(129)])
def test_invalid_queries_never_extrapolate_or_fetch_future(positions):
    with pytest.raises(ValueError):
        FixedUniformCubicEighth(np.zeros(4), SplineFixedFormat(18, 14))(positions)


def test_explicit_search_interface_keeps_score_future_gate_and_overflow_visible(monkeypatch):
    fmt = SplineFixedFormat(18, 14)
    rng = np.random.default_rng(630119)
    x = rng.normal(size=256)+1j*rng.normal(size=256)
    ref = profile().sync_symbols()
    a, b = {}, {}
    search_candidate_grid(x, 4, ref, 8, coarse_starts=[128], lookahead=16, diagnostics=a)
    search_candidate_grid(x, 4, ref, 8, coarse_starts=[128], lookahead=16, diagnostics=b,
                          interpolator='word-eighth', spline_format=fmt)
    assert a['disposition'] == b['disposition'] and a['best_start'] == b['best_start']
    assert abs(a['best_score']-b['best_score']) < .005
    assert b['interpolation_word_stats']['saturations'] == 0
    for options in [dict(interpolator='word-eighth'), dict(spline_format=fmt)]:
        with pytest.raises(ValueError):
            search_candidate_grid(x, 4, ref, 8, **options)
    original = FixedUniformCubicEighth._prepare
    def clipped(self, y):
        original(self, y)
        self.stats['saturations'] += 1
    monkeypatch.setattr(FixedUniformCubicEighth, '_prepare', clipped)
    with pytest.raises(OverflowError):
        search_candidate_grid(x, 4, ref, 8, coarse_starts=[128], diagnostics=b,
                              interpolator='word-eighth', spline_format=fmt)
    assert b['disposition'] == 'interpolation_overflow'
    monkeypatch.setattr(FixedUniformCubicEighth, '_prepare', original)
    actual_call = FixedUniformCubicEighth.__call__
    def clipped_query(self, positions):
        out = actual_call(self, positions)
        self.stats['saturations'] += 1
        return out
    monkeypatch.setattr(FixedUniformCubicEighth, '__call__', clipped_query)
    with pytest.raises(OverflowError):
        search_candidate_grid(x, 4, ref, 8, coarse_starts=[32], diagnostics=b,
                              interpolator='word-eighth', spline_format=fmt)
    assert b['disposition'] == 'interpolation_overflow'


@pytest.mark.parametrize('n', [4, 256, 632, 640])
def test_scale_scan_drain_align_and_inplace_work_calendar_has_no_uninitialized_reads(n):
    events = list(fixed_preparation_calendar(n))
    initialized_y, initialized_q, overwritten_q = set(), set(), set()
    rr = yr = yw = qr = qw = 0
    for phase, raw, source, write_y, read_q, write_q in events:
        if raw is not None:
            assert 0 <= raw < n
            rr += 1
        if write_y is not None:
            assert phase == 'align' and write_y not in initialized_y
            initialized_y.add(write_y)
            yw += 1
        assert len(source) <= 2 and all(j in initialized_y for j in source)
        yr += len(source)
        if read_q is not None:
            assert read_q in initialized_q and read_q not in overwritten_q
            qr += 1
        if write_q is not None:
            assert read_q != write_q
            initialized_q.add(write_q)
            qw += 1
            if phase in ('backward_write', 'last_second_write'):
                overwritten_q.add(write_q)
    assert len(events) == 6*n+8
    assert (rr, yr, yw, qr, qw) == (2*n, 4*n, n, 2*n-1, 2*n)
    assert initialized_y == overwritten_q == set(range(n))
    with pytest.raises(ValueError):
        list(fixed_preparation_calendar(3))
