"""Validate the audit oracle and legal wire format, without requiring stressed RX success."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'rx_timing_audit', Path(__file__).parents[1] / 'scripts/rx_timing_audit.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize('kind', ['QPSK', '8PSK'])
@pytest.mark.parametrize('length', [255, 511, 1023, 2047])
def test_legal_a3_lengths_and_exact_content(kind, length):
    symbols, bits = audit.packet(kind, length)
    row = audit.decode_ft2_a3_capture(symbols[40:], sync_word=audit.WORD,
                                    data_crc_seed=0x555555, pilot_interval=0)
    assert row['data_crc'] and row['header']['data_length'] == length
    assert row['payload'] == np.packbits(bits, bitorder='little').tobytes()


def test_analytic_pulse_symmetry_singularities_and_true_clock_sign():
    t = np.array([0, .625, 1, 2, 3])
    assert np.isfinite(audit.pulse(t)).all()
    np.testing.assert_allclose(audit.pulse(t), audit.pulse(-t))
    symbols = np.ones(120)
    _, start, step = audit.reference(symbols, .193, 100)
    assert step == pytest.approx(4.0004)
    assert start == pytest.approx(512.193 + 40 * 4.0004)
    # Independent linear identity: fixed RX steps accumulate -4*ppm samples/symbol.
    j = np.array([0, 1000, 5000])
    error = start + 4*j - (start + step*j)
    np.testing.assert_allclose(error, [0, -.4, -2], atol=1e-10)


def test_crc_content_are_independent_and_bad_payload_never_accepted():
    y, _ = audit.packet('8PSK', 255)
    frame = audit.decode_ft2_a3_capture(y[40:], sync_word=audit.WORD,
                                      data_crc_seed=0x555555, pilot_interval=0)
    frame['payload'] = b'\0' * 255
    scored = audit.score_symbols(y[40:], frame, '8PSK', 255)
    assert scored['ber'] == 0 and not scored['content_ok']
    with pytest.raises(ValueError):
        audit.packet('8PSK', 2048)
