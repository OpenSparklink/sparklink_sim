"""Word identity, independent RRC calendar replay and physical-port counterexamples."""

from dataclasses import replace

import pytest

from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.sim.rrc_memory import RRCMemory


def replay(mode, assumptions, samples=1024, stalls=()):
    memory = RRCMemory(mode, assumptions.product_buffers)
    baseline = PairRRCFlow(assumptions).simulate(samples, downstream_stalls=stalls)
    pending, expired_jobs = [], set()
    now = 0

    def product(sample, j):
        return (sample*0x43839FED7652ABCDEF+j*0x7938F4A1) % 2**69

    def observe(event):
        nonlocal pending, now
        write = event["source_write"]
        if write is not None:
            now = write+1
        result = memory.tick(event["cycle"],
            source_write=None if write is None else (write, (write*0x199F57D1) % 2**32),
            raw_reads=event["raw_reads"],
            product_writes=[(slot, j, product(owner, j))
                            for slot, j, owner in event["product_writes"]],
            product_reads=[(slot, j) for slot, j, _ in event["product_reads"]])
        assert result["products"] == pending
        pending = [(slot, j, product(owner, j)) for slot, j, owner in event["product_reads"]]
        for index, owner, value in zip(event["raw_reads"], event["raw_samples"],
                                       result["raw"], strict=True):
            if index >= 0 and index < now-128:
                assert value is None
                expired_jobs.add(owner)
            elif value is not None:
                assert value == (0 if index < 0 else (index*0x199F57D1) % 2**32)
            else:
                assert mode == "dual" and memory.stats["raw_conflict_banks"] > 0
        return "an observer cannot change the RRC schedule"

    observed = PairRRCFlow(assumptions).simulate(samples, downstream_stalls=stalls,
                                                access_sink=observe)
    assert observed == baseline and not pending and not memory.pending
    assert memory.stats["product_hazards"] == 0
    assert memory.stats["product_reads"] == memory.stats["product_writes"]
    assert memory.stats["source_writes"] == samples
    assert len(expired_jobs) == baseline["invalid"]
    return memory, observed


@pytest.mark.parametrize("mode", ["copies", "dual", "quad"])
def test_actual_rrc_schedule_preserves_words_and_old_outcomes_at_target_clock(mode):
    memory, result = replay(mode, PairRRCFlowAssumptions())
    assert not memory.stats["raw_conflict_banks"]
    assert result["continuous_input_without_loss"]


@pytest.mark.parametrize("changes", [dict(clock_hz=100_000_000), dict(clock_hz=160_000_000),
    dict(arithmetic_ii=2), dict(coefficient_ports=3), dict(product_buffers=2),
    dict(product_read_ports=1), dict(consumer_ii=14)])
def test_independent_quad_read_addresses_preserve_overload_without_extra_port_losses(changes):
    memory, result = replay("quad", replace(PairRRCFlowAssumptions(), **changes))
    assert not memory.stats["unserved_raw_reads"]
    assert result["dropped"] > 0
    assert memory.stats["peak_raw_reads_per_bank"] <= 2


def test_shared_write_address_dual_candidate_is_rejected_under_accumulation():
    memory, _ = replay("dual", PairRRCFlowAssumptions(clock_hz=160_000_000))
    assert memory.stats["unserved_raw_reads"] > 0 and memory.stats["raw_conflict_banks"] > 0


def test_blocked_output_keeps_original_expiry_and_drops_but_never_wrong_word():
    memory, result = replay("quad", PairRRCFlowAssumptions(), stalls=[(100, 4100)])
    assert result["invalid"] > 0 and result["dropped"] > 0
    assert memory.stats["expired_reads"] > 0 and not memory.stats["unserved_raw_reads"]


def test_product_read_register_has_exact_one_cycle_latency_and_full69bit_range():
    m = RRCMemory()
    assert m.tick(0, product_writes=[(0, 0, 2**69-1)])["products"] == []
    assert m.tick(1, product_reads=[(0, 0)])["products"] == []
    assert m.tick(2)["products"] == [(0, 0, 2**69-1)]


@pytest.mark.parametrize("overlap", ["same_cycle", "capture_edge"])
def test_read_write_hazards_return_missing_response_instead_of_new_data(overlap):
    events = []
    m = RRCMemory(diagnostic_sink=events.append)
    m.tick(0, product_writes=[(0, 0, 9)])
    m.tick(1, product_reads=[(0, 0)],
           product_writes=[(0, 0, 10)] if overlap == "same_cycle" else [])
    r = m.tick(2, product_writes=[(0, 0, 11)] if overlap == "capture_edge" else [])
    assert r["products"] == [(0, 0, None)]
    assert events[0]["reason"] == "product_read_write_overlap"


def test_raw_quad_replicas_write_together_and_startup_is_explicit_zero():
    m = RRCMemory()
    r = m.tick(0, source_write=(0, 2**32-1), raw_reads=[-80, 0])
    assert r["raw"] == [0, 2**32-1]
    assert all(replica[0] == 2**32-1 for replica in m.raw[0])
    assert m.estimate()["raw_replicated_image_bits"] == 32768


def test_lutram_mapping_charges_replication_padding_pipeline_and_control():
    cost = RRCMemory().estimate()
    assert cost["raw_memory_luts"]+cost["product_memory_luts"] == 1064
    assert cost["raw_allocated_primitive_bits"] == 32768
    assert cost["product_allocated_primitive_bits"] == 35328
    assert cost["raw_ideal_slicem_sites"]+cost["product_ideal_slicem_sites"] == 268
    assert cost["additional_read_pipeline_ff"] == 0 and cost["product_read_pipeline_bits"] == 276
    assert cost["extra_control_and_monitoring_ff"] == 1280
    assert cost["address_control_lut_count"] is None and not cost["whole_rx_accepted"]


@pytest.mark.parametrize("options", [dict(raw_mode="bad"), dict(buffers=True), dict(buffers=1)])
def test_unmodeled_geometry_is_not_automatically_supported(options):
    with pytest.raises(ValueError):
        RRCMemory(**options)


@pytest.mark.parametrize("options", [dict(cycle=1), dict(cycle=True),
    dict(cycle=0, source_write=(1, 0)), dict(cycle=0, source_write=(0, -1)),
    dict(cycle=0, raw_reads=[0]), dict(cycle=0, raw_reads=[-81]),
    dict(cycle=0, raw_reads=[-1]*9), dict(cycle=0, product_writes=[(3, 0, 0)]),
    dict(cycle=0, product_writes=[(0, 41, 0)]), dict(cycle=0, product_writes=[(0, 0, 2**69)]),
    dict(cycle=0, product_writes=[(0, 0, 0)]*5),
    dict(cycle=0, product_writes=[(0, 0, 0), (1, 0, 0)]),
    dict(cycle=0, product_reads=[(0, 0), (1, 0)])])
def test_invalid_ticks_never_get_a_free_read_or_wrapped_word(options):
    with pytest.raises(ValueError):
        RRCMemory().tick(**options)


def test_noncallable_observer_fails_even_for_empty_source():
    with pytest.raises(ValueError):
        PairRRCFlow().simulate(0, access_sink=1)


def test_driver_retains_rejected_ports_and_input_identity(tmp_path, monkeypatch):
    import importlib
    import json
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    driver = importlib.import_module("model_rrc_memory")
    assert driver.run(tmp_path/"out", 512) == 1
    d = json.loads((tmp_path/"out/model.json").read_text())
    assert len(d["scenarios"]) == 23 and not d["whole_system_accepted"]
    assert any(not r["port_candidate_valid"] for r in d["scenarios"] if r["mode"] == "dual")
    assert all(r["port_candidate_valid"] for r in d["scenarios"] if r["mode"] == "quad")
    assert d["known_partial_with_lutram_cells_bits"] > d["frozen_original_budget_bits"]
    with pytest.raises(ValueError):
        driver.run(tmp_path/"bad", 4)
