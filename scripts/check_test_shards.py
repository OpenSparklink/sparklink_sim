"""Check disjoint pytest JUnit shards and the unchanged 120-second suite budget."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("reports", type=Path, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seen, duplicates, rows = set(), [], []
    for path in args.reports:
        tree = ET.parse(path)
        suites = list(tree.iter("testsuite"))
        if len(suites) != 1:
            raise ValueError(f"Expected one pytest suite in {path}")
        data = suites[0].attrib
        cases = list(tree.iter("testcase"))
        for case in cases:
            identity = (case.attrib["classname"], case.attrib["name"])
            if identity in seen:
                duplicates.append(identity)
            seen.add(identity)
        rows.append(
            dict(
                report=str(path),
                tests=int(data["tests"]),
                cases=len(cases),
                failures=int(data["failures"]),
                errors=int(data["errors"]),
                skipped=int(data["skipped"]),
                seconds=float(data["time"]),
            )
        )
    passed = not duplicates and all(
        r["tests"] == r["cases"] and r["tests"] > 0 and r["failures"] == r["errors"] == 0
        and 0 <= r["seconds"] < 120
        for r in rows
    )
    result = dict(
        reports=rows,
        duplicate_cases=duplicates,
        unique_cases=len(seen),
        walltime_budget_seconds=120,
        passed=passed,
        walltimes_not_summed=True,
        receiver_acceptance=False,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
