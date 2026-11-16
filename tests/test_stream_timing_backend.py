"""Independent waveform, backend and arbitrary-block parity for packet clock recovery."""

import importlib.util

import numpy as np
import pytest

from nearlink_sdr.phy import stream_rx
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.timing import GardnerPIState, PSKTimingConfig
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_stream_receive import equal_events
from tests.test_strict_receive import iq_record, transmitted


def test_explicit_backend_selection_never_claims_missing_native(monkeypatch):
    p = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    q = np.ones(1024, complex)
    with pytest.raises(ValueError, match="timing_backend"):
        stream_rx.PSKStreamReceiver(p, q, timing_backend="other")
    monkeypatch.setattr(stream_rx, "_RustStreamTiming", None)
    with pytest.raises(RuntimeError, match="not installed"):
        stream_rx.PSKStreamReceiver(p, q, timing_backend="rust")
    assert stream_rx.PSKStreamReceiver(p, q).timing_backend == "python"


@pytest.mark.skipif(
    importlib.util.find_spec("nearlink_sdr_accel") is None,
    reason="optional native timing extension is not built",
)
@pytest.mark.parametrize("ft,mcs", [(2, 6), (2, 8), (2, 10), (2, 12), (3, 8), (4, 12)])
@pytest.mark.parametrize("ppm", [-100, 100])
def test_native_packet_state_matches_python_and_arbitrary_chunks(ft, mcs, ppm):
    assert stream_rx._RustStreamTiming is not None, (
        "Installed accelerator must contain stream timing"
    )
    cfg = TxConfig(
        frame_type=ft,
        mcs_index=mcs,
        symbol_rate_mhz=4,
        pilot_interval=8,
        ctrl_bits_len=28 if ft == 2 else 27,
        head_llid=0x123456 if ft >= 3 else None,
    )
    p = PSKReceiveProfile.ordinary(cfg)
    wire, expected = transmitted(p, length=255)
    x, q, _, _ = iq_record(wire, ppm=ppm, cfo=30000, offset=1.731)
    # Frozen pre-existing tolerances in test_stream_receive; payload/control exact.
    python_rx = stream_rx.PSKStreamReceiver(p, q, timing_backend="python")
    reference = python_rx.feed(x) + python_rx.finish()
    assert [e["frame"]["payload"] for e in reference] == [expected]
    for pattern in ([len(x)], [1], [37], np.random.default_rng(194016).integers(1, 513, 100)):
        rx = stream_rx.PSKStreamReceiver(p, q, timing_backend="rust")
        events, pos, i = [], 0, 0
        while pos < len(x):
            n = int(pattern[i % len(pattern)])
            events.extend(rx.feed(x[pos : pos + n]))
            pos += n
            i += 1
        events.extend(rx.finish())
        assert all(e["timing_backend"] == "rust" for e in events)
        equal_events(reference, events)
        assert rx.max_frame_symbols == python_rx.max_frame_symbols
        assert rx.peak_history_samples <= python_rx.peak_history_samples


@pytest.mark.skipif(
    importlib.util.find_spec("nearlink_sdr_accel") is None,
    reason="optional native timing extension is not built",
)
def test_native_ffi_bounds_and_noncontiguous_input_reject_without_panic():
    native = stream_rx._RustStreamTiming
    assert native is not None
    initial = dict(
        sps=4.0,
        start=8.0,
        previous=(1.0, 0.0),
        power=1.0,
        kp=0.01,
        ki=0.001,
        lower=3.99,
        upper=4.01,
        phase=0.0,
        slope=0.0,
        gain=1.0,
        hold=2,
    )
    for changed in (
        dict(power=0.0),
        dict(gain=0.0),
        dict(hold=0),
        dict(kp=10.0),
        dict(slope=float("nan")),
    ):
        with pytest.raises(ValueError, match="Invalid packet timing"):
            native(**(initial | changed))
    z = np.ones(64, complex)
    outputs = (np.empty(4, complex), np.empty(4), np.empty(4), np.empty(4))
    tracker = native(**initial)
    with pytest.raises(ValueError, match="output bound"):
        tracker.advance(z, 0.0, 40.0, 5, *outputs)
    with pytest.raises(TypeError, match="contiguous"):
        tracker.advance(z[::2], 0.0, 20.0, 4, *outputs)
    state = tracker.advance(z, 0.0, 20.0, 2, *outputs)
    assert state[0] == 2 and state[1] == 16.0
    np.testing.assert_array_equal(outputs[1][:2], [8.0, 12.0])
    with pytest.raises(ValueError, match="retained history"):
        tracker.advance(z, 100.0, 200.0, 4, *outputs)


@pytest.mark.skipif(
    importlib.util.find_spec("nearlink_sdr_accel") is None,
    reason="optional native timing extension is not built",
)
def test_native_scalar_arithmetic_preserves_numpy_power_and_division():
    """Independent random DSP states exercise rounding, TED clips and PI bounds."""
    rng = np.random.default_rng(195016)
    count = 3000
    z = rng.normal(size=4 * count + 64) + 1j * rng.normal(size=4 * count + 64)
    loop = GardnerPIState(4, 1.0, PSKTimingConfig())
    start, phase, slope, gain = 8.0, 0.1, 0.013, 1.23
    previous = stream_rx._cubic(z, 0.0, start)
    native = stream_rx._RustStreamTiming
    assert native is not None
    tracker = native(
        sps=4.0,
        start=start,
        previous=(previous.real, previous.imag),
        power=loop.power,
        kp=loop.kp,
        ki=loop.ki,
        lower=loop.lower,
        upper=loop.upper,
        phase=phase,
        slope=slope,
        gain=gain,
        hold=32,
    )
    actual = (np.empty(count, complex), np.empty(count), np.empty(count), np.empty(count))
    state = tracker.advance(z, 0.0, len(z) - 3.0, count, *actual)
    expected = tuple(np.empty_like(a) for a in actual)
    t, previous_t = start, start
    for i in range(count):
        current = stream_rx._cubic(z, 0.0, t)
        mid = stream_rx._cubic(z, 0.0, (previous_t + t) / 2) if i >= 32 else None
        step, error = loop.advance(current, previous, mid)
        expected[0][i] = current * np.exp(-1j * (phase + slope * i)) / gain
        expected[1][i], expected[2][i], expected[3][i] = t, (loop.omega / 4 - 1) * 1e6, error
        previous, previous_t, t = current, t, t + step
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)
    assert state[-2:] == (loop.rate_limit_events, loop.error_clip_events)
    assert loop.rate_limit_events > 0 and loop.error_clip_events > 0
