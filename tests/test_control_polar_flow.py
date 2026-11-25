"""Value-carrying bank arbitration,backpressure and exact hard SSC regression."""

import json
import random
from pathlib import Path

import numpy as np
import pytest

from nearlink_sdr.common.polar import get_polar_decoder
from nearlink_sdr.phy.control_polar_word import ControlPolarFormat
from nearlink_sdr.sim.control_polar_flow import ControlPolarService, control_polar_cost

REFERENCE = json.loads((Path(__file__).parent / "data/control_polar_reference.json").read_text())


def drain(service):
    for _ in range(6000):
        if service.ready:
            return
        service.step()
    pytest.fail("Bounded control service did not finish")


@pytest.mark.parametrize("N,K,cycles", [(64, 40, (1662, 1404)), (256, 51, (5512, 4042))])
def test_banked_service_exact_standard_and_arbitrary_received_words(N, K, cycles):
    rng = random.Random(206113 + N)
    patterns = [([0] * N, None), ([1] * N, None), ([k % 2 for k in range(N)], None)]
    patterns += [(v["coded_bits"], v["information_bits"]) for v in REFERENCE["vectors"]
                 if v["N"] == N]
    patterns += [([rng.randrange(2) for _ in range(N)], None) for _ in range(12)]
    decoder = get_polar_decoder(N, K)
    for layout, expected_cycles in zip(("dense", "depth-parity"), cycles, strict=True):
        service = ControlPolarService(ControlPolarFormat(N, K), layout=layout)
        for bits, independent in patterns:
            begin = service.cycle
            assert service.issue(bits)
            drain(service)
            assert service.cycle - begin == expected_cycles
            actual = service.take()
            assert actual["crc_status"] == "not_checked"
            expected = decoder.decode(np.asarray([1 - 2 * b for b in bits]) * 10.)
            assert actual["information_bits"] == tuple(expected.tolist())
            if independent is not None:
                assert actual["information_bits"] == tuple(independent)
            assert not service.input_ready
            service.step()
        assert service.stats["completed"] == len(patterns)
        assert max(service.memory.peak.values()) <= 2
        assert service.memory.peak["ROM"] == 1
        assert all(slot is None for slot in service.slots)
        assert not service.memory.pending


def test_registered_reads_ALU_writeback_and_held_output():
    events = []
    service = ControlPolarService(ControlPolarFormat(64, 40), observer=events.append)
    assert service.take() is None
    assert service.issue([0] * 64)
    assert not service.issue([1] * 64)
    drain(service)
    for _ in range(9):
        service.step()
    assert service.stats["output_stalls"] == 9
    assert service.ready and service.stats["completed"] == 1
    assert service.take()["information_bits"] == (0,) * 40
    assert service.take() is None
    computes = [e for e in events if e["event"] == "compute"]
    assert computes
    assert all(e["registered_write_edge"] == e["cycle"] + 1 for e in computes)
    for edge in {e["cycle"] for e in computes}:
        assert sum(e["cycle"] == edge for e in computes) == 1
    first_read = next(e for e in events if e["event"] == "read")
    typ = next(e for e in computes if e["opcode"] == "type")
    assert typ["cycle"] == first_read["cycle"] + 1
    assert service.stats["input_stalls"] == 1


@pytest.mark.parametrize("steps", [0, 3, 100, 900])
def test_abort_reset_discards_all_pending_words(steps):
    service = ControlPolarService(ControlPolarFormat(64, 40), layout="depth-parity")
    service.abort()
    service.reset()
    service.issue([1] * 64)
    for _ in range(steps):
        service.step()
    with pytest.raises(RuntimeError, match="abort"):
        service.reset()
    service.abort()
    service.abort()
    assert not service.ready and not service.memory.pending
    assert all(v is None for v in service.slots + service.stack)
    with pytest.raises(RuntimeError, match="reset"):
        service.issue([0] * 64)
    service.step()
    service.reset()
    assert service.issue([0] * 64)
    drain(service)
    assert service.take()["information_bits"] == (0,) * 40
    assert service.stats["aborted"] == 1


@pytest.mark.parametrize("bad", [[], [0] * 63, [0] * 65, [True] * 64, [2] * 64,
                                 [0.] * 64])
def test_input_validation_before_ownership(bad):
    service = ControlPolarService(ControlPolarFormat(64, 40))
    with pytest.raises(ValueError):
        service.issue(bad)
    assert service.input_ready and service.stats["accepted"] == 0


def test_infinite_input_is_bounded_and_invalid_modes_rejected():
    from itertools import repeat

    service = ControlPolarService(ControlPolarFormat(64, 40))
    with pytest.raises(ValueError):
        service.issue(repeat(0))
    with pytest.raises(ValueError):
        ControlPolarService(None)
    with pytest.raises(ValueError):
        ControlPolarService(service.fmt, layout="infinite")
    with pytest.raises(ValueError):
        ControlPolarService(service.fmt, observer=17)


def test_physical_banks_collision_zero_read_and_explicit_counter_overflow():
    service = ControlPolarService(ControlPolarFormat(64, 40))
    mem = service.memory
    bank, address = mem.address("L", 0, 0)
    with pytest.raises(RuntimeError, match="before"):
        mem.read(0, 0, bank, address)
    for plane, depth, position in [("bad", 0, 0), ("L", -1, 0), ("L", 0, 64)]:
        with pytest.raises(ValueError):
            mem.address(plane, depth, position)
    with pytest.raises(ValueError):
        mem.available([("bad", 0)])
    with pytest.raises(RuntimeError, match="read only"):
        mem.write("ROM", 0, 0)
    mem.write(bank, address, 0)
    assert not mem.available([(bank, address)])
    mem.begin()
    assert mem.available([(bank, address)])
    mem.used[bank] = 2
    assert not mem.available([(bank, address)])
    service.stats["completed"] = 2**64 - 1
    with pytest.raises(OverflowError):
        service.issue([0] * 64)
    assert service.input_ready and service.stats["accepted"] == 0
    service.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        service.step()


@pytest.mark.parametrize("N,K", [(64, 40), (256, 51)])
def test_physical_padding_and_complete_incremental_state_are_charged(N, K):
    dense, parity = [control_polar_cost(ControlPolarFormat(N, K), layout=v)
                     for v in ("dense", "depth-parity")]
    for cost in (dense, parity):
        assert cost["logical_total_bits"] <= 65536
        assert cost["native_total_bits"] == cost["RAMB18_equivalents"] * 18432
        assert cost["state_components_bits"]["four_contexts_padded"] == 768
        assert cost["prospective_capacity_budget_met"]
        assert cost["DSP"] == 0 and cost["LUT"] is None and cost["FF"] is None
        assert cost["unmodeled_head_demap_descramble_CRC_fields_cycles"] is None
        assert not cost["full_RX_acceptance"] and not cost["actual_synthesis"]
    assert (dense["RAMB18_equivalents"], parity["RAMB18_equivalents"]) == (
        (3, 5) if N == 64 else (6, 6))


def test_compact_b1_reuses_dead_nodes_and_retains_both_sibling_betas():
    fmt = ControlPolarFormat(256, 51)
    service = ControlPolarService(fmt, layout="compact-parity")
    cost = control_polar_cost(fmt, layout="compact-parity")
    assert cost["data_bank_depths"] == {"L0": 341, "L1": 170, "B0": 682, "B1": 340}
    assert cost["logical_RAM_bits"] == 6132
    assert cost["logical_total_bits"] <= 16384
    assert cost["RAMB18_equivalents"] == 5
    assert cost["DSP"] == 0

    mem = service.memory
    assert mem.address("L", 2, 0) == mem.address("L", 2, 64)
    assert mem.address("B", 2, 0) != mem.address("B", 2, 64)
    assert mem.address("B", 2, 0) == mem.address("B", 2, 128)
    assert mem.address("B", 8, 1) != mem.information(1)
    with pytest.raises(ValueError, match="compact information"):
        mem.information(-1)
    with pytest.raises(ValueError, match="N256/K51"):
        ControlPolarService(ControlPolarFormat(64, 40), layout="compact-parity")


def test_compact_b1_exact_hard_words_after_abort_and_output_stall():
    service = ControlPolarService(ControlPolarFormat(256, 51), layout="compact-parity")
    assert service.issue([1] * 256)
    for _ in range(809):
        service.step()
    service.abort()
    service.reset()
    standard = [v for v in REFERENCE["vectors"] if v["N"] == 256]
    rng = random.Random(47117)
    cases = [v["coded_bits"] for v in standard]
    cases += [[rng.randrange(2) for _ in range(256)] for _ in range(12)]
    decoder = get_polar_decoder(256, 51)
    for bits in cases:
        started = service.cycle
        assert service.issue(bits)
        drain(service)
        assert service.cycle - started <= 4042 + 64
        for _ in range(3):
            service.step()
        result = service.take()
        assert result["information_bits"] == tuple(decoder.decode((1 - 2*np.array(bits)) * 10.))
        assert result["crc_status"] == "not_checked"
        service.step()
    assert all(v <= 2 for v in service.memory.peak.values())
