"""Wrong target, absent rows, physical port rounding and false TDD sharing checks."""

import json
import os
from pathlib import Path

import pytest

from nearlink_sdr.sim.system_resources import (
    bram36_layout,
    calendar_conflicts,
    parse_z7020_utilization,
    system_headroom,
)

REPORT = """| Device       : xc7z020clg400-1
| Slice LUTs     | 18135 | 0 | 0 | 53200 | 34.09 |
| Slice Registers | 21711 | 0 | 0 | 106400 | 20.41 |
| Block RAM Tile | 73 | 0 | 0 | 140 | 52.14 |
| DSPs           | 29 | 0 | 0 | 220 | 13.18 |
"""


def test_physical_remaining_differs_from_unapproved_reserve_scenario():
    v = system_headroom(parse_z7020_utilization(REPORT))
    if directory := os.environ.get("SPARKLINK_TEST_DIAGNOSTICS"):
        path = Path(directory) / "system-resources.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(dict(scope="report parser fixture; not fresh synthesis",
                                        result=v), indent=2) + "\n")
    assert v["resources"]["bram36"]["physically_remaining"] == 67
    assert v["resources"]["bram36"]["proposed_incremental_ceiling"] == 32
    assert v["resources"]["dsp"]["physically_remaining"] == 191
    assert v["resources"]["dsp"]["proposed_incremental_ceiling"] == 136
    assert not v["reserve_approved"] and not v["whole_system_accepted"]
    assert not v["timing_200mhz_verified"]


@pytest.mark.parametrize("bad", [REPORT.replace("xc7z020clg400-1", "xc7z030sbg485-1"),
                                     REPORT.replace("53200", "99999"),
                                     REPORT.replace("| DSPs", "| Unknown"),
                                     REPORT + "| DSPs | 3 | 0 | 0 | 220 | 1.36 |\n",
                                     REPORT.replace("18135", "18135.5"),
                                     REPORT.replace("73 |", "73.3 |")])
def test_bad_report_cannot_become_free_resource(bad):
    with pytest.raises(ValueError):
        parse_z7020_utilization(bad)


def test_continuous_write_requires_read_copies_and_primitive_rounding():
    raw = bram36_layout(32, 128, simultaneous_reads=8)
    assert raw["bram36_tiles"] == 8
    assert raw["physical_bits"] == 8 * 36 * 1024
    assert raw["logical_bits"] == 4096
    assert raw["writes_broadcast_to_copies"] == 8
    product_bank = bram36_layout(69, 33)
    assert product_bank["bram36_tiles"] == 2
    assert 4 * product_bank["bram36_tiles"] + raw["bram36_tiles"] == 16


@pytest.mark.parametrize("values", [dict(width=0, depth=16), dict(width=16, depth=1.5),
                                        dict(width=16, depth=128, simultaneous_writes=2)])
def test_unsupported_memory_ports_are_not_assumed_free(values):
    with pytest.raises(ValueError):
        bram36_layout(**values)


def test_rx_decode_and_tx_prepare_overlap_despite_rf_tdd():
    # Explicit synthetic stress calendar, not invented protocol deadlines.
    jobs = [dict(job="previous RX decode", resource="coder", start=0, stop=100, units=1),
            dict(job="next TX prepare", resource="coder", start=80, stop=150, units=1)]
    conflicts = calendar_conflicts(jobs, dict(coder=1))
    assert conflicts == [dict(cycle=80, resource="coder", units=2, capacity=1,
                              jobs=["previous RX decode", "next TX prepare"])]
    jobs[1]["start"] = 100
    assert not calendar_conflicts(jobs, dict(coder=1))
    assert not calendar_conflicts(jobs, dict(coder=2))


@pytest.mark.parametrize("changes", [dict(start=-1), dict(stop=0), dict(units=True),
                                         dict(resource="unallocated")])
def test_invalid_schedule_does_not_receive_a_sharing_discount(changes):
    row = dict(job="RX", resource="coder", start=0, stop=100, units=1)
    row.update(changes)
    with pytest.raises(ValueError):
        calendar_conflicts([row], dict(coder=1))


def test_missing_capacity_cannot_pass_as_an_empty_schedule():
    with pytest.raises(ValueError):
        calendar_conflicts([], {})


def test_whole_function_inventory_keeps_unknown_cost_and_deadlines_explicit():
    path = Path(__file__).resolve().parents[1] / "docs/reference/system-target-ledger.json"
    inventory = json.loads(path.read_text())
    ids = {r["id"] for r in inventory["items"]}
    assert {"dma", "rx_front", "polar_rate_match", "tx_prepare", "tx_rrc_output",
            "realtime_mac", "security", "packet_queues", "rf_interface_cdc_axi"} <= ids
    assert len(ids) == len(inventory["items"]) and not inventory["qualification"]
    for row in inventory["items"]:
        assert set(row["resources"]) == {"lut", "ff", "dsp", "bram36"}
        assert not row["deadline_verified"] and not row["ports_verified"]
        assert row["overflow_policy"] and row["concurrency"] and row["placement"]
        assert all(v["lower"] is None and v["upper"] is None and v["basis"]
                   for v in row["resources"].values())
