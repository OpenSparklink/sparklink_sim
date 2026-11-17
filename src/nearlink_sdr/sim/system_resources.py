"""Read-only system planning helpers; estimates and calendars are not synthesis.

Physical capacity, a SHA-bound implemented baseline and proposed reserves stay
distinct. Incomplete module estimates/deadlines never count as zero or passing.
"""

import math
import re
from dataclasses import dataclass

__all__ = ["bram36_layout", "calendar_conflicts", "parse_z7020_utilization", "system_headroom"]


def parse_z7020_utilization(text):
    """Parse only the selected ANTSDR build target, not another vendor's E310."""
    if not re.search(r"\| Device\s+: xc7z020clg400-1\s*$", text, re.M):
        raise ValueError("Need the selected xc7z020clg400-1 utilization report")
    rows = {"lut": ("Slice LUTs", 53200), "ff": ("Slice Registers", 106400),
            "bram36": ("Block RAM Tile", 140), "dsp": ("DSPs", 220)}
    used, physical = {}, {}
    for key, (label, expected) in rows.items():
        found = re.findall(
            rf"^\| {label}\s+\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
            text, re.M,
        )
        if not found or any(int(total) != expected for _, total in found):
            raise ValueError("Missing resource row or mismatched physical capacity")
        values = {float(value) for value, _ in found}
        if len(values) != 1:
            raise ValueError("Conflicting duplicate report rows")
        value = values.pop()
        if not 0 <= value <= expected or value * 2 != int(value * 2):
            raise ValueError("Invalid resource count")
        if key != "bram36" and value != int(value):
            raise ValueError("Only half BRAM tiles may be fractional")
        used[key], physical[key] = value, expected
    return dict(used=used, physical=physical, target_part="xc7z020clg400-1")


def system_headroom(report):
    """Proposed25% chip reserve, not an approved budget or timing guarantee."""
    rows = {}
    for resource, total in report["physical"].items():
        occupied = report["used"][resource]
        reserve = math.ceil(total / 4)
        rows[resource] = dict(physical=total, baseline_used=occupied,
                              physically_remaining=total - occupied,
                              proposed_reserve=reserve,
                              proposed_incremental_ceiling=total - occupied - reserve)
    return dict(resources=rows, reserve_approved=False, system_budget_frozen=False,
                target_increment_fully_estimated=False, whole_system_accepted=False,
                timing_200mhz_verified=False)


def bram36_layout(width, depth, *, simultaneous_reads=1, simultaneous_writes=1):
    """Explicit1R1W TDP copies; every read copy receives each concurrent write.

    Uniform primitive modes only, no packing unrelated memories or assuming a
    second read port while a write consumes it. This is one realizable allocation
    estimate, not the minimum across banking/LUTRAM/asymmetric-port designs.
    """
    if (any(type(v) is not int or v < 1 for v in
            [width, depth, simultaneous_reads, simultaneous_writes])
            or simultaneous_writes != 1):
        raise ValueError("Need positive dimensions and explicit single-writer1R1W mapping")
    modes = [(1, 32768), (2, 16384), (4, 8192), (9, 4096), (18, 2048), (36, 1024)]
    tiles, w, d = min((math.ceil(width / w) * math.ceil(depth / d), w, d) for w, d in modes)
    total = tiles * simultaneous_reads
    return dict(logical_bits=width * depth, mode_width=w, mode_depth=d,
                bram36_tiles=total, physical_bits=total * 36 * 1024,
                read_copies=simultaneous_reads, writes_broadcast_to_copies=simultaneous_reads,
                primitive_read_ports_allocated=total, primitive_write_ports_allocated=total,
                evidence="uniform true-dual-port1R1W allocation estimate; not synthesis")


@dataclass(frozen=True)
class _Reservation:
    job: str
    resource: str
    start: int
    stop: int
    units: int


def calendar_conflicts(reservations, capacities):
    """Check explicit half-open cycle reservations; TDD earns no implicit credit.

    This checks a supplied schedule, not job service times, deadline correctness
    or queue dynamics. Actual protocol deadlines must come from the profile.
    """
    if not capacities or any(type(v) is not int or v < 1 for v in capacities.values()):
        raise ValueError("Need positive integer capacities")
    events = []
    for index, row in enumerate(reservations):
        r = _Reservation(**row)
        if (not r.job or r.resource not in capacities
                or any(type(v) is not int for v in [r.start, r.stop, r.units])
                or not 0 <= r.start < r.stop or r.units < 1):
            raise ValueError("Invalid interval or unallocated resource")
        events += [(r.start, 1, index, r), (r.stop, -1, index, r)]
    active = {resource: {} for resource in capacities}
    conflicts = []
    for cycle, action, index, r in sorted(events, key=lambda e: e[:3]):
        slots = active[r.resource]
        if action == -1:
            del slots[index]
        else:
            slots[index] = r
            used = sum(item.units for item in slots.values())
            if used > capacities[r.resource]:
                conflicts.append(dict(cycle=cycle, resource=r.resource, units=used,
                                      capacity=capacities[r.resource],
                                      jobs=[v.job for v in slots.values()]))
    return conflicts
