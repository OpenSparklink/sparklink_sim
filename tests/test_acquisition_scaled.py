"""Independent integer node and dynamic-range references for exponent-tagged capture."""

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_bank import normalized_greater
from nearlink_sdr.phy.acquisition_scaled import (
    ScaledBankFormat,
    ScaledDifferentialBank,
    align_words,
)
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def reference():
    return PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols()


def rne(value, shift):
    q, rem = divmod(abs(value), 1 << shift)
    if 2 * rem > 1 << shift or (2 * rem == 1 << shift and q % 2):
        q += 1
    return -q if value < 0 else q


def test_alignment_rounds_signed_ties_without_float_and_discards_large_shifts():
    words = np.array([0, 1, -1, 3, -3, 5, -5, 2**58 + 3, -(2**58 + 3)], np.int64)
    for shift in [0, 1, 2, 30, 58, 62, 63, 128, 320]:
        expected = [rne(int(a), shift) for a in words]
        np.testing.assert_array_equal(align_words(words, np.full(len(words), shift)), expected)
    for words, shifts in [([1.0], [1]), ([1], [-1]), ([1, 2], [1]), ([2**59], [1])]:
        with pytest.raises(ValueError):
            align_words(words, shifts)


def test_nodes_equal_independent_integer_reference_and_chunked_word_port():
    rng = np.random.default_rng(390119)
    ref = reference()
    x = (rng.normal(size=700) + 1j * rng.normal(size=700)) * 10 ** rng.uniform(-8, 5, 700)
    whole, chunks = [ScaledDifferentialBank(ref, 4, 256) for _ in range(2)]
    real, imag, exp = whole.quantize(x)
    points, other = [], []
    for k in range(0, len(x), 32):
        points += whole.advance_words(
            real[k : k + 32], imag[k : k + 32], exp[k : k + 32], inspect=True
        )
    for k in range(0, len(x), 7):
        other += chunks.advance(x[k : k + 7], inspect=True)
    assert points == other
    cr = np.rint((ref[1:] * ref[:-1].conj()).real * np.sqrt(2)).astype(int)
    ci = np.rint((ref[1:] * ref[:-1].conj()).imag * np.sqrt(2)).astype(int)
    for p in points[::17]:
        indices = range(p.start, p.start + 128, 4)
        rs, ims, es = [[int(v[i]) for i in indices] for v in (real, imag, exp)]
        d = [
            (a * c + b * d, b * c - a * d, e + f)
            for a, b, c, d, e, f in zip(
                rs[1:], ims[1:], rs[:-1], ims[:-1], es[1:], es[:-1], strict=True
            )
        ]
        common = max((e for a, b, e in d if a or b), default=-96)
        ar = [rne(a, max(0, common - e)) for a, b, e in d]
        ai = [rne(b, max(0, common - e)) for a, b, e in d]
        energy = sum(rne(a * a + b * b, 2 * max(0, common - e)) for a, b, e in d)
        real_c = sum(a * int(c) + b * int(d) for a, b, c, d in zip(ar, ai, cr, ci, strict=True))
        imag_c = sum(b * int(c) - a * int(d) for a, b, c, d in zip(ar, ai, cr, ci, strict=True))
        assert p.numerator == real_c**2 + imag_c**2 and p.energy == energy
    assert whole.stats["intermediate_overflows"] == 0
    assert whole.logical_storage_bits < 16 * 1024 * 8
    assert whole.worst_operations_per_sample()["other"]["variable_rne_shifts"] == 95


def test_power_of_two_scale_invariance_noise_and_weak_sync_numeric_budget():
    rng = np.random.default_rng(390120)
    ref = reference()
    x = np.tile(ref, 5).repeat(4)
    x += 0.07 * (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x)))
    all_points = []
    for scale in [2.0**-25, 1.0, 2.0**17]:
        bank = ScaledDifferentialBank(ref, 4, 256, scaled_format=ScaledBankFormat(10))
        points = []
        for k in range(0, len(x), 32):
            points += bank.advance(scale * x[k : k + 32], inspect=True)
        all_points.append(points)
        assert not bank.stats["input_saturations"]
    assert all_points[0] == all_points[1] == all_points[2]
    eligible = 0
    dr = ref[1:] * ref[:-1].conj()
    best = all_points[0][0]
    for p in all_points[0]:
        y = x[p.start : p.start + 128 : 4]
        d = y[1:] * y[:-1].conj()
        score = abs(np.vdot(dr, d)) / np.sqrt(sum(abs(d) ** 2) * sum(abs(dr) ** 2))
        if score >= 0.55:
            eligible += 1
            assert abs(np.sqrt(p.score_squared) - score) <= 0.005
        if normalized_greater(p, best):
            best = p
    assert eligible >= 15 and best.score_squared > 0.9


def test_zero_range_overflow_and_invalid_ports_are_explicit():
    model = ScaledDifferentialBank(reference(), 4, 256)
    for _ in range(8):
        model.advance(np.zeros(32))
    assert model.stats["zero_energy"] == 132
    model.quantize([1e-30, 1e100])
    assert model.stats["exponent_underflows"] == 1
    assert model.stats["exponent_overflows"] == 1
    assert model.stats["input_saturations"] == 1
    for kwargs in [dict(mantissa_bits=16), dict(minimum_exponent=-64), dict(maximum_exponent=64),
                   dict(minimum_exponent=-48.)]:
        with pytest.raises(ValueError):
            ScaledBankFormat(**kwargs)
    with pytest.raises(ValueError):
        ScaledDifferentialBank(reference(), 4, 256, scaled_format="bad")
    for values in [[], [np.nan], np.zeros(33), np.zeros((2, 2))]:
        with pytest.raises(ValueError):
            model.advance(values)
    for words in [([1.0], [0], [0]), ([1], [0], [33]), ([2**14], [0], [0]), ([1, 2], [0], [0])]:
        with pytest.raises(ValueError):
            model.advance_words(*words)
    model.samples = 2**63 - 1
    with pytest.raises(OverflowError):
        model.advance([0])


def test_scaled_bank_preserves_observer_and_crc_at_low_absolute_amplitude(tmp_path):
    import runpy
    from pathlib import Path

    from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
    from nearlink_sdr.phy.tone_notch import ToneNotchConfig
    from tests.test_stream_receive import equal_events
    from tests.test_strict_receive import iq_record, transmitted

    scripts = Path(__file__).parents[1]/'scripts'
    module = runpy.run_path(scripts/'replay_acquisition_regression.py')
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, _ = transmitted(p, bad_data=True)
    x, q, _, _ = iq_record(wire, offset=12.277, ppm=100, cfo=30000)
    kwargs = dict(quiet_batching=False, correlation_backend='scaled',
                  tone_notch=ToneNotchConfig())
    baseline = PSKStreamReceiver(p, q, **kwargs)
    expected = baseline.feed(x)+baseline.finish()
    plain = PSKStreamReceiver(p, q*1e-6, **kwargs)
    same_input = plain.feed(x*1e-6)+plain.finish()
    low = module['DiagnosticReceiver'](p, q*1e-6, diagnostic_file=tmp_path/'low.jsonl', **kwargs)
    actual = []
    for k in range(0,len(x),257):
        actual += low.feed(x[k:k+257]*1e-6)
    actual += low.finish()
    # Same input and algorithm: exact observer/chunk equivalence. Changing the
    # absolute floating amplitude uses the previously frozen trajectory budget.
    equal_events(same_input, actual)
    for key in ('status', 'header_crc', 'data_crc', 'payload'):
        assert expected[0]['frame'][key] == actual[0]['frame'][key]
    assert abs(expected[0]['sync_start_raw']-actual[0]['sync_start_raw']) <= .125
    assert np.max(abs(expected[0]['symbol_centers_raw']-actual[0]['symbol_centers_raw'])) <= .25
    assert actual[0]['frame']['status'] == 'bad_data_crc'
    assert low.numeric['eligible_score_observations'] > 0
    assert low.numeric['budget_violations'] == 0
    assert low.correlation_bank.stats['input_saturations'] == 0
    assert low.tone_correlation_bank.format.input_bits == 10


def test_maximum_word_port_and_128_symbol_reference_keep_energy_exact():
    rng = np.random.default_rng(390121)
    reference = np.exp(.5j*np.pi*rng.integers(0,4,128))
    bank = ScaledDifferentialBank(reference,4,640,scaled_format=ScaledBankFormat(14))
    limit = 1 << 13
    # |x*conj(y)|^2 <= 4*limit^4; at most127 terms. The actual
    # uint64 sum bound is <2^61 even though the generic width estimate is66.
    assert 127*4*limit**4 < 2**64
    points = []
    for _ in range(20):
        points += bank.advance_words([-limit]*32,[-limit]*32,[32]*32,inspect=True)
    assert points[-1].energy == 127*4*limit**4
    assert bank.stats['intermediate_overflows'] == 0
