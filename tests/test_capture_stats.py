"""Capture funnel counts are observational; bad CRC is never a search selector."""

import numpy as np
import pytest

from nearlink_sdr.phy.psk_sync import search_psk_sync
from nearlink_sdr.phy.stream_rx import PSKStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


@pytest.mark.parametrize(
    "damage,head_ok,head_bad,data_ok,data_bad",
    [({}, 1, 0, 1, 0), ({"bad_head": True}, 0, 1, 0, 0), ({"bad_data": True}, 1, 0, 0, 1)],
)
def test_funnel_counts_each_head_and_data_once(damage, head_ok, head_bad, data_ok, data_bad):
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    wire, _ = transmitted(profile, **damage)
    x, q, _, _ = iq_record(wire)
    rx = PSKStreamReceiver(profile, q)
    events = rx.feed(x) + rx.finish()
    assert len(events) == 1
    s = rx.capture_stats
    assert s["sync_candidates"] == 1
    assert s["energy_triggers"] == rx.sync_searches > 0
    assert s["phase_hypotheses"] == 32 * rx.sync_searches
    assert s["header_crc_ok"] == head_ok and s["header_crc_failed"] == head_bad
    assert s["data_crc_ok"] == data_ok and s["data_crc_failed"] == data_bad


def test_silence_and_preparation_maximum_have_distinct_counters():
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8))
    rx = PSKStreamReceiver(profile, np.zeros(1024, complex))
    assert rx.feed(np.zeros(2048, complex)) + rx.finish() == []
    assert rx.capture_stats["energy_triggers"] == rx.sync_searches == 0
    ref = profile.sync_symbols()
    y = np.zeros(130, complex)
    y[2:130:4] = ref
    diagnostic = {}
    assert search_psk_sync(y, 4, ref, 1, lookahead=16, diagnostics=diagnostic) is None
    assert diagnostic["best_score"] == pytest.approx(1)
    assert diagnostic["disposition"] == "maximum_needs_future"
