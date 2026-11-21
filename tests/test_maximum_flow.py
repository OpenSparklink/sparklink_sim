"""Real histogram state,registered RAM reads,finite credits and explicit epoch loss."""

import pytest

from nearlink_sdr.sim.maximum_flow import ExponentMaximumService
from tests.test_acquisition_maximum import events


def armed(terms, **kwargs):
    node = ExponentMaximumService(terms, **kwargs)
    for _ in range(node.estimate()["initialization_edges"]):
        node.step()
    assert node.mode == "idle"
    return node


@pytest.mark.parametrize("store,edges", [("register2r2w", 1), ("serial1r1w", 645)])
def test_initialization_charges_each_counter_write_and_prevents_input(store, edges):
    node = ExponentMaximumService(125, counter_store=store)
    assert not node.issue(0, (0, False), (0, True))
    with pytest.raises(ValueError, match="unfinished"):
        node.finish()
    writes = reset_words = 0
    for j in range(edges):
        node.step()
        writes += node.last_ports[1]
        reset_words += node.last_reset_words
        assert node.mode == ("idle" if j == edges - 1 else "initializing")
    assert writes == (644 if store == "serial1r1w" else 0)
    assert reset_words == (644 if store == "register2r2w" else 0)
    assert node.issue(0, (0, False), (0, True))


@pytest.mark.parametrize("terms", [31, 61, 125])
@pytest.mark.parametrize("store", ["register2r2w", "serial1r1w"])
def test_clocked_values_match_independent_history_and_per_clock_ports(terms, store):
    node = armed(terms, counter_store=store)
    count_reads = count_writes = 0
    for tag, outgoing, incoming, expected in events(terms, 1100):
        assert node.issue(tag, outgoing, incoming)
        for j in range(node.latency):
            node.step()
            assert node.ready == (j == node.latency - 1)
            reads, writes = node.last_ports
            bound = 1 if store == "serial1r1w" else 2
            assert reads <= bound and writes <= bound
            count_reads += reads
            count_writes += writes
        for _ in range(tag % 3):
            node.step()
        answer = node.take()
        assert answer[0:2] == (tag, expected)
        assert not node.issue(tag + 1, (0, False), (0, False))
        node.step()
    assert node.window.stats["events"] == node.stats["completed"] == 1100
    assert count_reads == 2200 and count_writes > 0
    assert node.stats["held_edges"] == sum(tag % 3 for tag in range(1100))
    assert node.estimate()["research_cap_pass"]
    assert node.estimate()["full_scaled_bank_cycles"] is None
    assert node.estimate()["extra_IQ_history_bits"] == 0
    node.finish()
    with pytest.raises(ValueError, match="epoch"):
        node.issue(1100, (0, False), (0, False))


@pytest.mark.parametrize("store", ["register2r2w", "serial1r1w"])
def test_held_output_backpressure_and_same_edge_input_are_not_consumption(store):
    node = armed(31, counter_store=store)
    assert node.issue(0, (0, False), (-96, True))
    assert not node.issue(1, (0, False), (64, True))
    with pytest.raises(ValueError, match="unfinished"):
        node.finish()
    for _ in range(node.latency):
        node.step()
    for _ in range(31):
        assert not node.issue(1, (0, False), (64, True))
        node.step()
        assert node.window.next_tag == 1 and node.output[:2] == (0, -96)
    before = node.stats["accepted"]
    assert node.take()[:2] == (0, -96)
    assert not node.issue(1, (0, False), (64, True))
    assert node.stats["accepted"] == before
    node.step()
    assert node.issue(1, (0, False), (64, True))
    with pytest.raises(ValueError, match="owns"):
        node.window.advance(1, (0, False), (64, True))


def test_serial_read_results_are_registered_and_partial_writes_abort_epoch():
    node = armed(31, counter_store="serial1r1w")
    node.issue(0, (0, False), (64, True))
    node.step()
    assert node.read is None and node.window.next_tag == 0
    node.step()
    assert len(node.read) == 2 and node.prepared is None
    node.step()
    assert node.prepared is not None and node.window.counts[0][160] == 0
    node.step()
    assert node.window.counts[0][160] == 1 and node.window.groups[0] == 0
    node.abort("unpausable producer exhausted external FIFO")
    assert node.take() is None and not node.ready
    assert node.window.counts[0][160] == 1  # No free644-word memory clear.
    with pytest.raises(ValueError, match="epoch"):
        node.issue(1, (0, False), (0, False))


@pytest.mark.parametrize("store", ["register2r2w", "serial1r1w"])
def test_invalid_borrowed_membership_is_explicit_failure_before_any_count_write(store):
    node = armed(31, counter_store=store)
    node.issue(0, (64, True), (0, True))
    with pytest.raises(ValueError, match="absent"):
        for _ in range(node.latency):
            node.step()
    assert node.mode == "failed" and node.reason
    assert node.window.next_tag == sum(node.window.totals) == 0
    assert sum(sum(x) for x in node.window.counts) == 0


@pytest.mark.parametrize("store", ["register2r2w", "serial1r1w"])
def test_natural_clock_and_statistics_overflow_do_not_wrap(store):
    node = armed(31, counter_store=store)
    node.cycle = 2**63 - 2
    with pytest.raises(OverflowError, match="headroom"):
        node.issue(0, (0, False), (0, False))
    node.step()
    with pytest.raises(OverflowError, match="clock"):
        node.step()
    node = armed(31, counter_store=store)
    node.stats["accepted"] = 2**64 - 1
    with pytest.raises(OverflowError, match="statistics"):
        node.issue(0, (0, False), (0, False))
    assert node.mode == "idle" and node.window.next_tag == 0


def test_foreign_owner_and_tag_are_detected():
    node = armed(31)
    node.issue(0, (0, False), (0, True))
    node.window.next_tag = 1
    with pytest.raises(ValueError, match="outside"):
        node.step()
    node = armed(31)
    node.window._owner = object()
    with pytest.raises(ValueError, match="ownership"):
        node.step()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(terms=32),
        dict(terms=True),
        dict(terms=31, counter_store="unbounded"),
        dict(terms=31, start_cycle=True),
    ],
)
def test_compiled_configuration_guards(kwargs):
    with pytest.raises(ValueError):
        ExponentMaximumService(**kwargs)
