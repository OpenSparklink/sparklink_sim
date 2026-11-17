"""Record the unified RX partial allocation; incomplete/overload evidence exits1."""

import argparse
import json
from pathlib import Path

from audit_capture_hardware import sha, write

from nearlink_sdr.phy.rx_resource_budget import receiver_resource_ledger


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    result = receiver_resource_ledger()
    write(output / "resources.json", result)
    files = [*Path("src").rglob("*.py"), Path(__file__), Path("pyproject.toml"), Path("uv.lock")]
    write(output / "sources.json", {str(p): sha(p) for p in files})
    (output / "driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(output / "manifest.json", {p.name: sha(p) for p in output.iterdir() if p.is_file()})
    print(json.dumps(dict(dsp=result["dsp"], storage=result["storage"], wide=result["wide"],
                         whole_rx_acceptance=result["whole_rx_acceptance"]), indent=2))
    return int(not result["whole_rx_acceptance"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args().output))
