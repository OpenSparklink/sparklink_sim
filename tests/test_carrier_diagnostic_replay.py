"""BER diagnostics must replay the selected receiver, including short-payload bypass."""

import copy
import importlib
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


@pytest.fixture
def audit(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("audit_rx_soft_capture")


def profile():
    return PSKReceiveProfile.ordinary(
        TxConfig(mcs_index=8, symbol_rate_mhz=4, whitening_seed=0x52, crc_seed=0x123456)
    )


@pytest.mark.parametrize("bad", [False, True])
def test_short_raw_iq_actual_receiver_and_diagnostic_have_same_bits(audit, monkeypatch, bad):
    p = profile()
    wire, expected = transmitted(p, length=1, bad_data=bad)
    x, q, _, _ = iq_record(wire, cfo=13000, ppm=50)
    rx = MacStreamReceiver(p, q, quiet_batching=False, defer_quiet_fir=False)
    first, last = rx.feed(x), rx.finish()
    events = first.diagnostics["events"] + last.diagnostics["events"]
    assert len(events) == 1 and events[0]["frame"]["data_crc"] is (not bad)
    monkeypatch.setattr(
        audit,
        "refine_psk_carrier",
        lambda *a, **k: pytest.fail("short branch must use sync estimate"),
    )
    bits = audit.diagnostic_bits(events[0], p)
    assert len(bits) == 8
    if not bad:
        assert np.packbits(bits).tobytes() == expected


@pytest.mark.parametrize("method", ["coherent", "phase_regression"])
def test_replay_forwards_actual_carrier_method_without_crc_choice(audit, monkeypatch, method):
    p = profile()
    wire, expected = transmitted(p, length=37)
    symbols = wire[40:]
    frame = decode_psk_frame(symbols, p, carrier_method=method)
    original = audit.refine_psk_carrier
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs["method"])
        return original(*args, **kwargs)

    monkeypatch.setattr(audit, "refine_psk_carrier", spy)
    bits = audit.diagnostic_bits(dict(symbols=symbols, frame=frame), p)
    assert calls == [method] and np.packbits(bits).tobytes() == expected


def test_unknown_missing_or_inconsistent_metadata_cannot_claim_ber(audit):
    p = profile()
    wire, _ = transmitted(p, length=37)
    original = decode_psk_frame(wire[40:], p)
    for method in (None, "unknown", "sync estimate only; fewer than 32 payload wire symbols"):
        frame = copy.deepcopy(original)
        frame["carrier"]["method"] = method
        with pytest.raises(ValueError, match=r"carrier|Carrier"):
            audit.diagnostic_bits(dict(symbols=wire[40:], frame=frame), p)
    assert audit.diagnostic_bits(dict(frame=dict(status="bad_header_crc")), p) is None
