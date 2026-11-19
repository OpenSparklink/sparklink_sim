"""Real scalar recurrence vs existing bounded frame geometry,no payload array."""

import pytest

from nearlink_sdr.phy.psk_fields import psk_field_layout
from nearlink_sdr.phy.rx_pipeline import payload_coded_length
from nearlink_sdr.phy.tx_pipeline import TxConfig
from nearlink_sdr.sim.header_geometry_flow import HeaderGeometryService


def drain(service):
    for _ in range(500):
        if service.ready:
            return service.take()
        service.step()
    pytest.fail("Finite geometry did not drain")


@pytest.mark.parametrize("ft", [2, 3, 4])
@pytest.mark.parametrize("crc", [24, 32])
@pytest.mark.parametrize("pilot", [0, 4, 8, 16])
def test_actual_integer_program_matches_all_MCS_and_boundaries(ft, crc, pilot):
    # Original prototype regression is preserved under its explicit B1 contract.
    service = HeaderGeometryService(ft, crc, pilot, length_contract=(
        "standard" if ft == 2 else "legacy-b1-payload-length"))
    for mcs in range(13):
        cfg = TxConfig(frame_type=ft, mcs_index=mcs, crc_len=crc)
        for length in [0, 1, 31, 37, 127, 511, 2047]:
            coded = payload_coded_length(length * 8, cfg)
            positions, wire = psk_field_layout(
                coded, {"BPSK": 1, "QPSK": 2, "8PSK": 3}[cfg.mod_str], pilot
            )
            assert service.issue(length, mcs)
            assert not service.issue(length, mcs)
            row = drain(service)
            assert row["coded_bits"] == coded and row["data_symbols"] == len(positions)
            assert row["wire_data_symbols"] == wire
            assert row["frame_end_symbol"] == row["data_start_symbol"] + wire
    assert service.stats["completed"] == 91
    assert service.stats["division_steps"] % 23 == 0
    assert service.stats["product_steps"] % 10 == 0
    assert len(service.program) <= 64 and len(service.registers) <= 24


@pytest.mark.parametrize("ft", [3, 4])
@pytest.mark.parametrize("crc", [24, 32])
@pytest.mark.parametrize("pilot", [0, 4, 8, 16])
def test_B1_wire_port_real_subtraction_zero_branch_and_all_MCS(ft, crc, pilot):
    events = []
    service = HeaderGeometryService(ft, crc, pilot, observer=events.append)
    for mcs in range(13):
        cfg = TxConfig(frame_type=ft, mcs_index=mcs, crc_len=crc)
        for raw in [0, crc // 8, crc // 8 + 1, 37, 127, 511, 2047]:
            events.clear()
            assert service.issue(raw, mcs)
            row = drain(service)
            information = max(0, raw - crc // 8)
            bits = payload_coded_length(information * 8, cfg) if raw else 0
            positions, wire = psk_field_layout(
                bits, {"BPSK": 1, "QPSK": 2, "8PSK": 3}[cfg.mod_str], pilot)
            assert row["payload_length_bytes"] == information
            assert row["data_crc_present"] is bool(raw)
            assert row["coded_bits"] == bits and row["data_symbols"] == len(positions)
            assert row["wire_data_symbols"] == wire
            assert row["frame_end_symbol"] == row["data_start_symbol"] + wire
            subtractions = [e for e in events if e.get("opcode") == "sub" and
                            e.get("pc") == 3]
            assert bool(subtractions) == bool(raw)
            if raw:
                assert subtractions[0]["a"] == raw and subtractions[0]["b"] == crc // 8
            assert len(service.registers) <= 24
    assert service.stats["completed"] == 91
    assert service.cost()["program_instructions"] <= 64


def test_host_contract_validation_cannot_supply_the_length_answer(monkeypatch):
    import nearlink_sdr.sim.header_geometry_flow as module
    service = HeaderGeometryService(3, 32, 8)
    monkeypatch.setattr(module, "control_data_length", lambda *a, **k: (999, True))
    assert service.issue(39, 8)
    row = drain(service)
    assert row["payload_length_bytes"] == 35 and row["coded_bits"] == 312
    assert row["frame_end_symbol"] == 397


@pytest.mark.parametrize("crc", [24, 32])
def test_invalid_B1_crc_width_lengths_rejected_without_state_admission(crc):
    service = HeaderGeometryService(4, crc, 8)
    for n in range(1, crc // 8):
        with pytest.raises(ValueError, match="shorter"):
            service.issue(n, 8)
    assert service.stats["accepted"] == 0 and service.mode == "idle"
    assert service.cycle == 0


def test_contract_is_compiled_and_never_inferred_from_outputs():
    with pytest.raises(ValueError, match="explicit"):
        HeaderGeometryService(2, 24, 8, length_contract="legacy-b1-payload-length")
    with pytest.raises(ValueError, match="explicit"):
        HeaderGeometryService(3, 24, 8, length_contract="auto")
    with pytest.raises(ValueError, match="Observer"):
        HeaderGeometryService(3, 24, 8, observer=3)
    standard = HeaderGeometryService(3, 24, 8)
    legacy = HeaderGeometryService(3, 24, 8, length_contract="legacy-b1-payload-length")
    standard.issue(39, 8)
    legacy.issue(39, 8)
    assert drain(standard)["frame_end_symbol"] == 397
    assert drain(legacy)["frame_end_symbol"] == 410


def test_scalar_capture_parameter_latch_and_each_ALU_write_are_causal():
    events = []
    service = HeaderGeometryService(3, 32, 8, observer=events.append)
    begin = service.cycle
    service.issue(39, 6)
    drain(service)
    assert events[0]["event"] == "accept" and events[0]["input_edge"] == begin
    row = next(e for e in events if e["event"] == "parameter_read")
    assert row["cycle"] == begin and row["latch_edge"] == begin + 1
    writes = [e for e in events if e["event"] == "register"]
    assert all(e["register_edge"] == e["cycle"] + 1 for e in writes)
    assert len({e["cycle"] for e in writes}) == len(writes)
    instructions = [e for e in events if e["event"] == "instruction"]
    assert all(e["cycle"] >= row["latch_edge"] for e in instructions)
    assert service.cost()["constant_data_read_ports"] == 1
    assert service.cost()["instruction_ROM_read_ports"] == 1


@pytest.mark.parametrize("args", [(1, 24, 8), (2, 12, 8), (3, 24, 1), (True, 24, 8)])
def test_unsupported_geometry_does_not_claim_support(args):
    with pytest.raises(ValueError):
        HeaderGeometryService(*args)


def test_output_hold_abort_invalid_inputs_and_registered_engine():
    events = []
    service = HeaderGeometryService(2, 24, 8, observer=events.append)
    assert service.take() is None
    service.abort()
    service.reset()
    for bad in [(-1, 8), (2048, 12), (1, 13), (1, True)]:
        with pytest.raises(ValueError):
            service.issue(*bad)
    service.issue(100, 5)
    service.step()
    with pytest.raises(RuntimeError):
        service.reset()
    service.abort()
    assert not service.issue(1, 8)
    service.reset()
    service.issue(1023, 3)
    for _ in range(500):
        if service.ready:
            break
        service.step()
    for _ in range(3):
        service.step()
    assert service.ready and service.stats["completed"] == 1
    assert service.stats["output_stalls"] == 3 and not service.input_ready
    assert sum(e["event"] == "engine" and e["opcode"] == "div" for e in events) % 23 == 0
    assert service.take()["coded_bits"] > 0
    assert service.input_ready and service.stats["input_stalls"] == 1
    service.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        service.step()
