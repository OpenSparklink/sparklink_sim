"""Real CRC gate and legal frame geometry;no CRC-guided receiver retry."""

from dataclasses import replace
from itertools import repeat

import pytest

from nearlink_sdr.phy.psk_fields import demap_psk_field
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame
from nearlink_sdr.sim.header_word_flow import HeaderWordService
from tests.test_receive_scope import make_profile
from tests.test_strict_receive import transmitted


def hard_header(wire, profile):
    ft = profile.config.frame_type
    n = len(profile.sync_symbols())
    end = HeaderWordService(profile).geometry.header_end
    return [
        int(b)
        for b in demap_psk_field(
            wire[40 + n : 40 + end], "BPSK" if ft == 4 else "QPSK", n, 16 if ft == 2 else 4
        )
    ]


def drain(service):
    for _ in range(10000):
        if service.ready:
            return service.take()
        service.step()
    pytest.fail("Finite real-header gate did not drain")


@pytest.mark.parametrize(
    "ft,mcs,pilot", [(2, 0, 0), (2, 12, 8), (3, 8, 4), (3, 10, 16), (4, 0, 4), (4, 12, 0)]
)
@pytest.mark.parametrize("layout", ["dense", "depth-parity"])
def test_real_header_gate_matches_existing_strict_parser_and_is_reusable(ft, mcs, pilot, layout):
    profile = make_profile(ft, mcs, pilot)
    service = HeaderWordService(profile, layout=layout)
    for length in [0, 1, 37, 511]:
        wire, payload = transmitted(profile, length=length)
        expected = decode_psk_frame(wire[40:], profile)
        assert expected["status"] == "valid" and expected["payload"] == payload
        assert service.issue(hard_header(wire, profile))
        row = drain(service)
        assert row["status"] == "header_valid" and row["header_crc"] is True
        assert row["data_crc"] is None and row["payload"] is None
        for key in [
            "header",
            "length_ok",
            "lqi",
            "mcs",
            "mcs_source",
            "header_end_symbol",
            "data_start_symbol",
            "frame_end_symbol",
            "coded_bits",
            "data_symbols",
            "wire_data_symbols",
        ]:
            assert row[key] == expected[key]
    assert service.stats["completed"] == 4
    assert service.stats["CRC_input_bits"] == 4 * profile.config.ctrl_bits_len
    assert service.stats["CRC_parity_bits"] == 4 * service.width
    cost = service.cost()
    assert cost["capacity_candidate_met"] and cost["DSP"] == 0
    assert sum(cost["profile_context_allocation_bits"].values()) == 237
    assert cost["profile_context_suballocated_bits"] <= cost["extra_state_bits"][
        "profile_context_upper"]
    assert cost["demap_cycles"] is cost["payload_cycles"] is None
    assert not cost["full_RX_acceptance"] and not cost["synthesis"]


@pytest.mark.parametrize(
    "damage,status", [({"bad_head": True}, "bad_header_crc"), ({"empty": True}, "invalid_length")]
)
def test_failed_head_never_reads_untrusted_fields_or_starts_geometry(damage, status):
    profile = make_profile(2, 8)
    wire, _ = transmitted(profile, **damage)
    service = HeaderWordService(profile)
    service.issue(hard_header(wire, profile))
    row = drain(service)
    assert row["status"] == status and row["payload"] is None
    assert service.geometry.stats["accepted"] == 0
    if status == "bad_header_crc":
        assert service.values == {} and service.stats["field_bits"] == 0
        assert row["header_crc"] is False and row["length_ok"] is None
    else:
        assert row["header_crc"] is True and row["length_ok"] is False


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_wrong_seed_or_LLID_is_real_crc_failure(ft):
    profile = make_profile(ft, 8)
    wire, _ = transmitted(profile)
    bad = replace(
        profile,
        config=replace(profile.config, **({"head_crc_seed": 0} if ft == 2 else {"head_llid": 0})),
    )
    service = HeaderWordService(bad)
    service.issue(hard_header(wire, profile))
    row = drain(service)
    assert row["status"] == "bad_header_crc" and service.stats["field_bits"] == 0


def test_unsupported_MCS_and_valid_overlength_never_start_geometry():
    profile = make_profile(3, 8)
    wire, _ = transmitted(profile, mcs=15)
    service = HeaderWordService(profile)
    service.issue(hard_header(wire, profile))
    assert drain(service)["status"] == "unsupported_mcs"
    assert service.geometry.stats["accepted"] == 0
    limited = replace(profile, max_payload_bytes=12)
    wire, _ = transmitted(profile, length=13)
    other = HeaderWordService(limited)
    other.issue(hard_header(wire, profile))
    assert drain(other)["status"] == "invalid_length"


def test_input_configuration_ownership_hold_and_abort():
    profile = make_profile(2, 8)
    wire, _ = transmitted(profile)
    events = []
    service = HeaderWordService(profile, observer=events.append)
    profile.config.whitening_seed = 1  # Existing service owns its separate immutable snapshot.
    for bits in [[], [0] * 63, [True] * 64, repeat(0)]:
        with pytest.raises(ValueError):
            service.issue(bits)
    assert service.stats["accepted"] == 0
    # Use the profile copied by the service,not the deliberately mutated external one.
    bits = hard_header(wire, service.profile)
    assert service.issue(bits) and not service.issue(bits)
    service.step()
    with pytest.raises(RuntimeError):
        service.reset()
    service.abort()
    with pytest.raises(RuntimeError):
        service.issue(bits)
    service.reset()
    service.issue(bits)
    for _ in range(10000):
        if service.ready:
            break
        service.step()
    for _ in range(3):
        service.step()
    assert service.stats["output_stalls"] == 3
    assert service.take()["header_crc"] and service.take() is None
    assert any(e["event"] == "CRC_parity" for e in events)
    assert any(e["event"] == "geometry" for e in events)
    service.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        service.step()


@pytest.mark.parametrize("kind", ["missing", "factory", "unknown_backend"])
def test_compiled_profiles_are_explicit(kind):
    with pytest.raises(ValueError):
        if kind == "missing":
            HeaderWordService(None)
        elif kind == "factory":
            HeaderWordService(
                PSKReceiveProfile.ws63_ft2_a3(sync_word=0, data_crc_seed=0, pilot_interval=0)
            )
        else:
            replace(make_profile(2, 8), head_backend="unbounded")


def test_shared_field_layout_cannot_be_mistaken_for_CRC_validation():
    from nearlink_sdr.phy.control_info import control_field_layout

    assert sum(width for _, width in control_field_layout("A2")) == 20
    assert sum(width for _, width in control_field_layout("B1")) == 27
    with pytest.raises(ValueError):
        control_field_layout("A3")


def test_permitted_MCS_container_is_bounded_and_sealed_before_packet_ownership():
    allowed = [8]
    profile = replace(make_profile(2, 8), allowed_mcs=allowed)
    wire, _ = transmitted(profile)
    service = HeaderWordService(profile)
    assert service.issue(hard_header(wire, profile))
    allowed.clear()
    assert profile.allowed_mcs == service.profile.allowed_mcs == (8,)
    assert drain(service)["status"] == "header_valid"
    with pytest.raises(ValueError, match="at most13"):
        replace(profile, allowed_mcs=repeat(8))
    with pytest.raises(ValueError, match="at most13"):
        replace(profile, allowed_mcs=(True,))
    with pytest.raises(ValueError, match="finite"):
        replace(profile, allowed_mcs=None)
