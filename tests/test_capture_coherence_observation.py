"""Actual coherent decisions survive later tail loss or CRC failure in the funnel."""

from dataclasses import replace

import numpy as np
import pytest
from scripts import model_retired_capture as model

from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.sim.capture_frame import CapturedFrameReplay
from tests.captured_frame_reference import delivered, fixture, packet, receive
from tests.maximum_reference import channel


def two_packets_with_late_tail():
    # Original M28 development case2;fixed regression,not a fresh held set.
    cfg = model.shared.shared.configuration(4, 8, 8, 17)
    cfg.pid = 4
    profile = replace(
        PSKReceiveProfile.ordinary(cfg), allowed_mcs=(8,), head_backend="word-depth-parity"
    )
    data = [model.payload(17, 4, False), model.payload(61, 5, False)]
    symbols = np.r_[packet(profile, data[0]), np.zeros(32, complex), packet(profile, data[1])]
    iq, quiet, *_ = channel(
        symbols,
        family="clean",
        offset=0.761,
        ppm=50,
        cfo=-93000,
        prefix=147,
        amplitude=0.181,
        seed=1288759,
    )
    return profile, np.r_[iq, np.zeros(1536, complex)], quiet, data


@pytest.mark.parametrize("retire", [False, True])
def test_coherent_capture_before_tail_expiry_is_not_lost_from_packet_funnel(retire):
    p, iq, quiet, data = two_packets_with_late_tail()
    before = receive(p, iq, quiet, retire_before_fine=retire)
    legacy = sum(e.get("sync_confirmation", {}).get("accepted") is True for e, _ in before[1])
    assert legacy + 1 == before[3].stats["coherent_accepted"]
    observed = []
    after = receive(p, iq, quiet, retire_before_fine=retire, coherence_observer=observed.append)
    assert before[0] == after[0] and before[2] == after[2]
    assert model.event_metadata(before[1]) == model.event_metadata(after[1])
    assert delivered(after[1]) == [data[0]]
    decisions = {r["id"]: r["accepted"] for r in observed}
    assert sum(decisions.values()) == after[3].stats["coherent_accepted"]
    late = next(e for e, _ in after[1] if e["frame"]["status"] == "tail_history_expired")
    assert "sync_confirmation" not in late and decisions[late["selection_id"]] is True
    assert late["frame"]["data_crc"] is None
    for (x, _), (y, _) in zip(before[1], after[1], strict=True):
        for key in ("symbols", "symbol_centers_filtered", "timing_rate_ppm"):
            if key in x:
                np.testing.assert_array_equal(x[key], y[key])
    assert all(not any(k in r for k in ("CRC", "payload", "truth")) for r in observed)


@pytest.mark.parametrize("kind", ["head", "data", "length"])
def test_coherence_is_observed_before_bad_CRC_or_length_without_changing_delivery(kind):
    p, iq, quiet, *_ = fixture(bad_head=kind == "head", bad_data=kind == "data")
    if kind == "length":
        p = replace(p, max_payload_bytes=16)
    observed = []
    result, frames, _, owner, _, _ = receive(p, iq, quiet, coherence_observer=observed.append)
    assert delivered(frames) == [] and result["frame_payloads"] == 0
    assert sum(r["accepted"] for r in observed) == owner.stats["coherent_accepted"] == 1
    status = {"head": "bad_header_crc", "data": "bad_data_crc", "length": "invalid_length"}[kind]
    event = next(e for e, _ in frames if e["frame"]["status"] == status)
    assert next(r for r in observed if r["id"] == event["selection_id"])["accepted"]


def test_rejected_confirmation_stays_rejected_even_with_a_truthy_observer(monkeypatch):
    original = CapturedFrameReplay.__init__

    def threshold_one(owner, *args, **kwargs):
        kwargs["coherent_threshold"] = 1.0  # Explicit negative test context.
        original(owner, *args, **kwargs)

    monkeypatch.setattr(CapturedFrameReplay, "__init__", threshold_one)
    p, iq, quiet, *_ = fixture()
    records = []

    def observer(row):
        records.append(row)
        return {"accepted": True}  # Must not become a receiver decision.

    _, frames, _, owner, _, _ = receive(p, iq, quiet, coherence_observer=observer)
    assert delivered(frames) == [] and records and not any(r["accepted"] for r in records)
    assert owner.stats["coherent_accepted"] == 0
    assert any(e["frame"]["status"] == "sync_confirmation_rejected" for e, _ in frames)


@pytest.mark.parametrize("observer", [True, 1, "observer"])
def test_invalid_coherence_observer_is_rejected_before_processing(observer):
    with pytest.raises(ValueError, match="coherence observer"):
        receive(None, None, None, coherence_observer=observer)
