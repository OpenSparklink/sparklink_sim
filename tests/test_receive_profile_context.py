"""Configuration acceptance must be finite and must not borrow a caller's list."""

from dataclasses import replace
from itertools import repeat

import pytest

from nearlink_sdr.phy.mac_interface import MacStreamReceiver
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig
from tests.test_strict_receive import iq_record, transmitted


def test_caller_list_mutation_cannot_reject_a_packet_already_owned_by_receiver():
    allowed = [8]
    profile = PSKReceiveProfile.ordinary(TxConfig(mcs_index=8, symbol_rate_mhz=4))
    profile = replace(profile, allowed_mcs=allowed)
    wire, payload = transmitted(profile)
    x, quiet, _, _ = iq_record(wire)
    receiver = MacStreamReceiver(profile, quiet)
    first = receiver.feed(x[:97])
    allowed.clear()
    second = receiver.feed(x[97:])
    last = receiver.finish()
    assert profile.allowed_mcs == (8,)
    assert [p.mac_payload for r in [first, second, last] for p in r.packets] == [payload]
    assert sum(e['frame']['header_crc'] is True for r in [first, second, last]
               for e in r.diagnostics['events']) == 1


@pytest.mark.parametrize("bad", [None, (True,), (13,), repeat(8)])
def test_invalid_or_unbounded_permitted_set_rejected_before_receive(bad):
    with pytest.raises(ValueError, match=r"finite|at most13"):
        replace(PSKReceiveProfile.ordinary(TxConfig()), allowed_mcs=bad)


def test_empty_set_is_still_an_explicit_reject_all_configuration():
    profile = replace(PSKReceiveProfile.ordinary(TxConfig()), allowed_mcs=[])
    assert profile.allowed_mcs == ()
