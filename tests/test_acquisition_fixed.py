"""Independent integer dot/energy reference, corner widths, rounding and stream semantics."""

import numpy as np
import pytest

from nearlink_sdr.phy.acquisition_bank import normalized_greater, sign_coefficients
from nearlink_sdr.phy.acquisition_fixed import FixedBankFormat, FixedDifferentialBank
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig


def ref():
    return PSKReceiveProfile.ordinary(TxConfig(mcs_index=8)).sync_symbols()


@pytest.mark.parametrize('width', [4, 10, 18, 26])
@pytest.mark.parametrize('chunk', [1, 7, 32])
def test_integer_bank_matches_unbounded_direct_reference_at_every_position(width, chunk):
    rng = np.random.default_rng(340120)
    bound = 1 << (width-1)
    r = rng.integers(-bound, bound, 191, dtype=np.int64)
    i = rng.integers(-bound, bound, 191, dtype=np.int64)
    r[:8], i[:8] = [-bound, bound-1]*4, [bound-1, -bound]*4
    x = (r+1j*i)*1024/bound
    bank = FixedDifferentialBank(ref(), 4, 256, format=FixedBankFormat(width))
    points = []
    for pos in range(0, len(x), chunk):
        points.extend(bank.advance(x[pos:pos+chunk], inspect=True))
    a, b = sign_coefficients(ref())
    for p in points:
        y = [complex(int(r[k]), int(i[k])) for k in range(p.start, p.start+128, 4)]
        # Python complex is exact for each ≤53-bit integer difference; sums and squares use ints.
        d = [y[j+1]*y[j].conjugate() for j in range(31)]
        cre = sum(int(v.real)*int(s)+int(v.imag)*int(t) for v, s, t in zip(d, a, b, strict=True))
        cim = sum(int(v.imag)*int(s)-int(v.real)*int(t) for v, s, t in zip(d, a, b, strict=True))
        energy = sum(int(v.real)**2+int(v.imag)**2 for v in d)
        assert p.numerator == cre**2+cim**2 and p.energy == energy
        left = p.numerator << 24
        right = bank.widths['threshold_integer']*p.energy
        assert bank.gate(p.start, p.start) == (energy > 0 and left >= right, 1)
    assert bank.stats['input_saturations'] == bank.stats['intermediate_overflows'] == 0
    assert bank.logical_storage_bits < 16*1024*8
    assert bank.emulator_state_bytes > bank.state_array_bytes
    assert bank.worst_operations_per_sample()['divide'] == 0


def test_round_nearest_even_saturation_zero_and_overflow_contracts():
    bank = FixedDifferentialBank(ref(), 4, 256, format=FixedBankFormat(10))
    # LSB is2; half ties round to the even input word in both directions.
    re, im = bank.quantize(np.array([1, -1, 3, -3, 1024, -1024, 1e200, -1e200])+0j)
    np.testing.assert_array_equal(re, [0, 0, 2, -2, 511, -512, 511, -512])
    np.testing.assert_array_equal(im, np.zeros(8, int))
    assert bank.stats['input_saturations'] == 3
    for _ in range(12):
        bank.advance(np.zeros(32, complex))
    assert bank.gate(200, 250) == (False, 51)
    assert bank.stats['intermediate_overflows'] == 0
    with pytest.raises(OverflowError, match='accumulator'):
        bank._checked(256, 8)
    assert bank.stats['intermediate_overflows'] == 1
    assert not normalized_greater(bank.advance(np.zeros(32), inspect=True)[0],
                                  bank.advance(np.zeros(32), inspect=True)[0])
    for values in [[np.nan], np.ones(33), [], np.zeros((2, 2))]:
        with pytest.raises(ValueError):
            bank.advance(values)
    with pytest.raises(ValueError):
        bank.quantize([np.inf])
    with pytest.raises(ValueError, match='power-of-two'):
        FixedDifferentialBank(ref(), 4, 256, input_full_scale=1000)
    with pytest.raises(ValueError, match='FixedBankFormat'):
        FixedDifferentialBank(ref(), 4, 256, format=False)
    with pytest.raises(ValueError, match='rounded to zero'):
        FixedDifferentialBank(ref(), 4, 256, threshold=1e-10)
    with pytest.raises(ValueError, match='16KiB'):
        FixedDifferentialBank(np.ones(128), 16, 2560, format=FixedBankFormat(24))
    for kwargs in [dict(input_bits=True), dict(input_bits=27), dict(threshold_fractional_bits=7)]:
        with pytest.raises(ValueError):
            FixedBankFormat(**kwargs)
    bank.samples = 2**63-1
    with pytest.raises(OverflowError, match='counter'):
        bank.advance([1])


def test_full_scale_extension_preserves_lsb_without_assuming_the_same_node_width():
    small = FixedDifferentialBank(ref(), 4, 256, input_full_scale=1024., format=FixedBankFormat(14))
    large = FixedDifferentialBank(ref(), 4, 256, input_full_scale=262144., format=FixedBankFormat(22))
    x = np.array([.0625, .1875, -.1875, 250.]), np.array([.25, .375, -.5, 70.])
    a = small.quantize(x[0]+1j*x[1])
    b = large.quantize(x[0]+1j*x[1])
    for p, q in zip(a, b, strict=True):
        np.testing.assert_array_equal(p, q)
    assert large.widths['input'] == small.widths['input']+8


def test_full_scale_dc_exact_threshold_and_wide_products_recover_after_zero_input():
    bank = FixedDifferentialBank(np.ones(32, complex), 4, 256, threshold=1.,
                                 format=FixedBankFormat(26))
    points = []
    for _ in range(5):
        points += bank.advance(np.full(32, -1024-1024j), inspect=True)
    assert points[-1].numerator > 2**64
    assert points[-1].score_squared == 1
    assert bank.gate(points[-1].start, points[-1].start) == (True, 1)
    for _ in range(6):
        points = bank.advance(np.zeros(32, complex), inspect=True)
    assert points[-1].energy == points[-1].numerator == 0
    assert bank.gate(points[-1].start, points[-1].start) == (False, 1)
    assert bank.stats['input_saturations'] == bank.stats['intermediate_overflows'] == 0


def test_fixed_gate_reaches_strict_receiver_without_changing_bad_crc_to_success():
    from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
    from tests.test_strict_receive import iq_record, transmitted

    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    for bad in (False, True):
        wire, payload = transmitted(profile, bad_data=bad)
        x, q, truth, _ = iq_record(wire, offset=12.277, ppm=100, cfo=30000)
        rx = PSKStreamReceiver(profile, q, correlation_backend='fixed', quiet_batching=False,
                               correlation_input_bits=18)
        events = []
        for pos in range(0, len(x), 513):
            events.extend(rx.feed(x[pos:pos+513]))
        events.extend(rx.finish())
        event, = events
        assert event['frame']['status'] == ('bad_data_crc' if bad else 'valid')
        assert event['frame']['payload'] == (None if bad else payload)
        assert abs(event['sync_start_raw']-truth) <= .125
        assert event['integer_screen_backend'] == 'fixed'
        assert rx.correlation_bank.stats['intermediate_overflows'] == 0
