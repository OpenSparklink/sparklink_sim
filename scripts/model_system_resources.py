"""SHA-bound baseline, proposed headroom and explicit memory mapping; not acceptance."""

import argparse
import json
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.sim.system_resources import (
    bram36_layout,
    parse_z7020_utilization,
    system_headroom,
)


def run(args):
    index = json.loads(args.baseline_index.read_text())
    files = {}
    for row in index:
        path = Path(row["saved"])
        assert sha(path) == row["sha256"], "Frozen baseline file changed"
        files.setdefault(path.name, path)
    implementation = json.loads(files["implementation.json"].read_text())
    qualification = json.loads(files["20261112-rx-live.json"].read_text())["qualification"]
    assert sha(files["implementation.json"]) == qualification["implementation_sha256"]
    assert implementation["effective_inputs_sha256"] == qualification["qualification"][
        "effective_sha256"]
    resources = parse_z7020_utilization(files["utilization.rpt"].read_text())
    assert resources["used"] == {
        "lut": qualification["resources"]["lut"], "ff": qualification["resources"]["ff"],
        "dsp": qualification["resources"]["dsp"],
        "bram36": qualification["resources"]["bram_tiles"],
    }
    inventory = json.loads(Path("docs/reference/system-target-ledger.json").read_text())
    assert inventory["qualification"] is False
    args.output.mkdir(parents=True, exist_ok=False)
    model = system_headroom(resources)
    model.update(
        baseline=dict(input_index_sha256=sha(args.baseline_index),
                      bitstream_sha256=implementation["bitstream_sha256"],
                      effective_inputs_sha256=implementation["effective_inputs_sha256"],
                      utilization_sha256=sha(files["utilization.rpt"]),
                      timing_sha256=sha(files["timing_summary.rpt"]),
                      qualification_timing=qualification["timing"],
                      sys_clock_hz=100_000_000,
                      scope="qualified fixedFT1/1M baseline, not targetPSK/16MS/s integration"),
        rrc_clock_sensitivity=[dict(clock_hz=clock, **PairRRCFlow(
            PairRRCFlowAssumptions(clock_hz=clock)).resource_estimate())
            for clock in [100_000_000, 160_000_000, 200_000_000]],
        explicit_memory_layouts=dict(
            rrc_raw_8read_1write=bram36_layout(32, 128, simultaneous_reads=8),
            each_of_four_owned_product_banks=bram36_layout(69, 33),
            each_filtered_history_direct_8read_1write=bram36_layout(
                46, 384, simultaneous_reads=8),
            each_of_five_draft_packet_slots=bram36_layout(8, 2048),
            draft_pooled_five_slots_unproven_arbitration=bram36_layout(32, 2560),
        ),
        function_inventory=inventory,
        response_deadline_known=False,
        sharing_hw_schedule_proven=False,
        no_credit="baseline includes existing FT1/observation/control; removing or sharing it earns no savings without a new bound build/schedule",
    )
    write(args.output / "system.json", model)
    sources = [Path(__file__), Path("docs/reference/system-target-ledger.json"),
               Path("pyproject.toml"), Path("uv.lock"), *Path("src").rglob("*.py")]
    write(args.output / "sources.json", {str(p): sha(p) for p in sources})
    (args.output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "manifest.json", {p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    print(json.dumps(dict(resources=model["resources"],
                         whole_system_accepted=model["whole_system_accepted"]), indent=2))
    return 1  # Explicitly incomplete system qualification, not execution failure.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
