"""Actual upper-fence consumer plus the existing registered queued gate."""

from functools import partial
from unittest.mock import patch

import numpy as np
import pytest

from nearlink_sdr.sim.legal_tail_ceiling import UpperTailCaptureReplay
from tests.captured_frame_reference import delivered, receive
from tests.test_queued_retirement import fixed as fixed


def run(fixed, enabled, **options):
    checks = []
    factory = partial(UpperTailCaptureReplay, payload_mode="functional")
    with patch("tests.captured_frame_reference.CapturedFrameReplay", factory):
        result = receive(*fixed[:3], retire_before_fine=True, retire_queued=enabled,
                         retirement_observer=checks.append, **options)
    return result, checks


@pytest.fixture(scope="module")
def paired(fixed):
    return run(fixed, False), run(fixed, True)


def test_real_words_prior_consumed_frontier_and_existing_successes(fixed, paired):
    (old, _), (new, checks) = paired
    assert delivered(old[1]) and set(delivered(old[1])) <= set(delivered(new[1]))
    assert all(v in fixed[3] for v in delivered(new[1]))
    assert any(v["event"] == "queued_check" for v in checks)
    assert new[0]["input_samples"] == old[0]["input_samples"] == len(fixed[1])
    assert new[0]["live_source_statistics"] == old[0]["live_source_statistics"]
    pending = None
    for v in checks:
        if v["event"] == "queued_check":
            assert pending is None and v["frame_active"] and v["copied_cycle"] < v["cycle"]
            assert v["exclude"] == (v["source_end"] <= v["predicate_frontier"])
            assert not any("crc" in k.lower() or "payload" in k.lower() for k in v)
            pending = v
        elif v["event"] in ("queued_kept", "queued_retired"):
            assert pending is not None and v["id"] == pending["id"]
            assert v["cycle"] == pending["cycle"] + 1
            assert v["exclude"] is pending["exclude"]
            pending = None
    assert pending is None and new[0]["peak_owned"] <= 3
    assert new[0]["peak_source_operations_per_primitive"] <= 2
    for event, mac in new[1]:
        assert event["initialized_capture_bridge"]["peak_private_words"] <= 1024
        if mac.packets:
            assert event["frame"]["header_crc"] is event["frame"]["data_crc"] is True
        assert event["consumed_source_end"] <= event["copied_source_end"]
    for before, _ in old[1]:
        matches = [e for e, _ in new[1] if e["selection_id"] == before["selection_id"]]
        if matches:
            for key, v in before.items():
                if isinstance(v, np.ndarray):
                    np.testing.assert_array_equal(v, matches[0][key])
    cost = new[0]["queued_retirement_gate_cost"]
    assert cost["register_bits"] == 66 and cost["combined_retirement_state_bits"] == 132
    assert cost["extra_IQ_RAM_DSP_or_sample_ports"] == 0
    assert new[0]["carrier_frame_and_decoder_cycles"] is None
    assert not new[0]["whole_system_accepted"]


def test_random_cross_block_partition_preserves_every_read_and_registered_check(fixed, paired):
    baseline, checks = paired[1]
    groups = tuple(np.random.default_rng(298211).integers(1, 259, 17))
    result, observed = run(fixed, True, groups=groups)
    assert result[0] == baseline[0] and result[2] == baseline[2] and observed == checks
    assert delivered(result[1]) == delivered(baseline[1])
