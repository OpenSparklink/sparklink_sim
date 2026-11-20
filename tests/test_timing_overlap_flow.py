"""Exact frozen node sequences,real bank launches and atomic result ownership."""

import pytest

from nearlink_sdr.phy.timing_word import TimingWordFormat, WordTimingState
from nearlink_sdr.sim.hold_timing_flow import PipelinedPacketTimingService
from nearlink_sdr.sim.timing_overlap_flow import ParallelTEDTimingWordService
from nearlink_sdr.sim.timing_word_flow import TimingWordService
from tests.test_sync_fit_word import packed
from tests.timing_overlap_reference import node_vectors


@pytest.mark.parametrize(
    "fmt", [TimingWordFormat(), TimingWordFormat(30, 32), TimingWordFormat(38, 32)]
)
@pytest.mark.parametrize("divider", ["bounded-prefix", "radix4-prefix"])
@pytest.mark.parametrize("split", ["development", "validation"])
def test_frozen_words_exact_to_serial_and_functional_reference_with_four_product_limit(
    fmt, divider, split
):
    services = [
        cls(WordTimingState((1 << 53, -53), 57, 32, fmt), divider=divider)
        for cls in (TimingWordService, ParallelTEDTimingWordService)
    ]
    reference = WordTimingState((1 << 53, -53), 57, 32, fmt)
    for k, (current, midpoint) in enumerate(node_vectors(split)):
        expected = reference.advance_words(current, midpoint)
        latency = []
        for service in services:
            before = service.state.snapshot()
            assert service.issue(current, midpoint)
            assert not service.issue(current, midpoint)
            previous_products = service.stats["native_products"]
            for _ in range(200):
                assert service.state.snapshot() == before
                service.step()
                products = service.stats["native_products"]
                assert products - previous_products in (0, 4)
                previous_products = products
                terms = getattr(service, "prepared_ted_terms", None)
                if terms is not None:
                    # The expanded Python terms represent one94bit physical
                    # encoding plus wires,not four independently mutable ports.
                    dr, di = terms[0][0], terms[2][0]
                    mr = terms[0][1] + (terms[1][1] << 17)
                    mi = terms[2][1] + (terms[3][1] << 17)
                    assert -(1 << 24) <= dr < 1 << 24
                    assert -(1 << 24) <= di < 1 << 24
                    assert -(1 << 21) <= mr < 1 << 21
                    assert -(1 << 21) <= mi < 1 << 21
                    assert terms == [
                        (dr, mr & 131071, 0),
                        (dr, mr >> 17, 17),
                        (di, mi & 131071, 0),
                        (di, mi >> 17, 17),
                    ]
                if service.ready:
                    break
            assert service.ready and service.result == expected
            latency.append(service.ready_cycle - service.first_cycle)
            for _ in range(k % 3):
                service.step()
                assert service.result == expected
            assert service.take() == expected and service.take() is None
            assert not service.input_ready
            service.step()
        assert latency[0] - latency[1] == (8 if midpoint is not None else 0)
        assert services[0].state.snapshot() == services[1].state.snapshot()
        assert services[0].stats == services[1].stats
    before, after = [s.estimate() for s in services]
    assert after["state_register_bits_upper"] - before["state_register_bits_upper"] == 104
    assert sum(after["overlap_increment_parts"].values()) == 104
    assert after["overlap_power_adder_must_not_be_credited_as_shared_with_product_tree"]
    assert not after["full_receiver16MSps"] and not after["synthesized"]


@pytest.mark.parametrize("edge", [0, 2, 5, 6, 9, 10, 11, 17, 29, 45])
def test_abort_at_overlapped_microstages_and_clock_exhaustion_never_commit(edge):
    state = WordTimingState((1 << 53, -53), 57, 32, TimingWordFormat(30, 32))
    for _ in range(32):
        state.advance_words(packed(1 << 20, 17, -20), None)
    service = ParallelTEDTimingWordService(state, divider="radix4-prefix")
    before = state.snapshot()
    assert service.issue(packed(1 << 19, 31, -20), packed(-1 << 20, 13, -20))
    for _ in range(edge):
        assert not service.ready
        service.step()
    service.abort()
    for _ in range(3):
        service.step()
    assert state.snapshot() == before and service.take() is None
    service.cycle = 2**64 - 1
    with pytest.raises(OverflowError):
        service.step()
    assert state.snapshot() == before


def test_compile_choice_does_not_change_existing_packet_service_default():
    assert PipelinedPacketTimingService.normal_service_type is TimingWordService
    with pytest.raises(ValueError):
        list(node_vectors("unfrozen"))
