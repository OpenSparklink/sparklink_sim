"""Real36-bit RAM words,ports,finite RRC calendars and unchanged legacy costs."""

from dataclasses import replace

import pytest

from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.sim.rrc_full14 import PairRRC14Flow, RRC14Memory
from nearlink_sdr.sim.rrc_memory import RRCMemory


def replay(mode, assumptions, samples=1024, stalls=()):
    memory, pending, now = RRC14Memory(mode, assumptions.product_buffers), [], 0

    def raw(index):
        real, imag, exponent = (index*7193) % 16384 - 8192, (index*6173) % 16384 - 8192, index % 81 - 48
        return (real & 16383) | ((imag & 16383) << 14) | ((exponent & 255) << 28)

    def product(sample, j):
        # Legal31/31/7 field widths. This is a RAM identity oracle,not FIR arithmetic.
        return ((sample*0x43839FED7652ABCDEF+j*0x7938F4A1) % 2**69)

    def observe(event):
        nonlocal pending, now
        write = event["source_write"]
        if write is not None:
            now = write+1
        result = memory.tick(event["cycle"],
            source_write=None if write is None else (write, raw(write)),
            raw_reads=event["raw_reads"],
            product_writes=[(slot, j, product(owner, j)) for slot, j, owner in event["product_writes"]],
            product_reads=[(slot, j) for slot, j, _ in event["product_reads"]])
        assert result["products"] == pending
        pending = [(slot, j, product(owner, j)) for slot, j, owner in event["product_reads"]]
        for index, value in zip(event["raw_reads"], result["raw"], strict=True):
            if index >= 0 and index < now-128:
                assert value is None
            elif value is not None:
                assert value == (0 if index < 0 else raw(index))
            else:
                assert mode == "dual" and memory.stats["raw_conflict_banks"] > 0

    result = PairRRC14Flow(assumptions).simulate(samples, downstream_stalls=stalls, access_sink=observe)
    assert not pending and not memory.pending and not memory.stats["product_hazards"]
    assert memory.stats["source_writes"] == samples
    assert memory.stats["product_reads"] == memory.stats["product_writes"]
    return memory, result


def test_exact_cost_delta_is_input_width_not_free_extra_memory_or_DSP():
    old, new = PairRRCFlow().resource_estimate(), PairRRC14Flow().resource_estimate()
    assert old["register_tap_logical_bits"] == 24900
    assert new["register_tap_logical_bits"] == 25460
    assert new["format"] == dict(input_bits=14, coefficient_bits=18, output_bits=19,
                                 product_bits=31, accumulator_bits=37)
    for k in ["multiplier_units", "wide_units", "small_units", "leading_detectors_32bit", "ports",
              "necessary_ii", "fir_latency", "sum_latency", "merge_latency", "product_bank_entry_bits"]:
        assert new[k] == old[k]
    assert new["multiplier_units"] == 8 and new["necessary_ii"] == 12
    assert new["physical_ADC_to_word_cycles"] is None and not new["full_RX_16MSps"]
    a, b = RRCMemory().estimate(), RRC14Memory().estimate()
    assert b["raw_memory_luts"] - a["raw_memory_luts"] == 64
    assert b["raw_logical_bits"] - a["raw_logical_bits"] == 512
    assert b["raw_allocated_primitive_bits"] - a["raw_allocated_primitive_bits"] == 4096
    for k in ["raw_bram36", "product_memory_luts", "product_logical_padded_bits",
              "product_allocated_primitive_bits", "extra_control_and_monitoring_ff", "extra_dsp"]:
        assert a[k] == b[k]
    assert b["raw_routing_mux_bit_inputs_upper"] - a["raw_routing_mux_bit_inputs_upper"] == 512


def test_high_raw_bits_are_stored_without_relaxing_the_legacy32bit_port():
    word = 2**35 + 917
    with pytest.raises(ValueError):
        RRCMemory().tick(0, source_write=(0, word))
    m = RRC14Memory()
    assert m.tick(0, source_write=(0, word), raw_reads=[0])["raw"] == [word]
    assert m.tick(1, raw_reads=[0])["raw"] == [word]
    with pytest.raises(ValueError):
        m.tick(2, source_write=(1, 2**36))


@pytest.mark.parametrize("mode", ["copies", "dual", "quad"])
def test_actual_ram_read_write_calendar_preserves36bit_values_at200MHz(mode):
    memory, result = replay(mode, PairRRCFlowAssumptions())
    assert result["continuous_input_without_loss"] and not memory.stats["unserved_raw_reads"]
    assert memory.stats["peak_raw_reads_per_bank"] <= (8 if mode == "copies" else 2)


@pytest.mark.parametrize("changes", [dict(clock_hz=100_000_000), dict(clock_hz=160_000_000),
    dict(product_buffers=2), dict(product_read_ports=1), dict(arithmetic_ii=2), dict(consumer_ii=14)])
def test_overload_keeps_finite_history_and_explicit_drops_not_wrong_words(changes):
    assumptions = replace(PairRRCFlowAssumptions(), **changes)
    memory, result = replay("quad", assumptions)
    old = PairRRCFlow(assumptions).simulate(1024)
    for k in ["emitted", "dropped", "invalid", "ownership_errors", "max_latency"]:
        assert result[k] == old[k]
    assert result["dropped"] > 0 and not memory.stats["unserved_raw_reads"]


def test_blocked_output_and_dual_port_rejection_remain_separate_from_numerical_gates():
    memory, result = replay("quad", PairRRCFlowAssumptions(), stalls=[(100, 4100)])
    assert result["invalid"] > 0 and result["dropped"] > 0 and memory.stats["expired_reads"] > 0
    memory, _ = replay("dual", PairRRCFlowAssumptions(clock_hz=160_000_000))
    assert memory.stats["unserved_raw_reads"] > 0 and memory.stats["raw_conflict_banks"] > 0
