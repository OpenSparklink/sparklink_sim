"""Explicit receiver experiment replacing only post-confirmation sync initialization.

The runtime hook is a development harness, not a second production receiver.
Q confirmation/timing and the default PSKStreamReceiver source stay unchanged.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from audit_capture_hardware import sha, write
from audit_sync_candidates import run as replay

from nearlink_sdr.phy import stream_rx
from nearlink_sdr.phy.sync_carrier import initialize_sync_carrier


def run(args):
    if (type(args.iterations) is not int or args.iterations not in (1, 2)
            or type(args.fixed_initializer_nco) is not bool):
        raise ValueError("Need explicit1/2 initializer steps and boolean NCO selection")
    if args.selection is not None:
        selected = json.loads(args.selection.read_text())
        viewed = getattr(args, "viewed_regression", False)
        if viewed:
            if (sha(args.selection) !=
                    "21e1b5673ba3bffae1ba5bbb1bef30f6fa8cf0f55e72dff67fafcbbb3cf8e052"
                    or sha(args.inputs / "inputs.json") !=
                    "b612659b09c3172f9363dc681b7f912435381f933f5d6084e989710e25be9270"):
                raise ValueError("Viewed regression is restricted to the original CFO3 input identity")
        elif selected.get("initializer_iterations") != args.iterations:
            raise ValueError("Held initialization must match the frozen step count")
    original = stream_rx.PSKStreamReceiver._search_path

    def search(self):
        before = self._active
        original(self)
        if before is not None or self._active is None:
            return
        a = self._active
        assert a["count"] == 0 and "sync_confirmation" in a
        times = a["start"]+np.arange(len(self.reference))*self.sps
        raw = np.array([stream_rx._cubic(self._filtered, self._origin, t) for t in times])
        fitted = initialize_sync_carrier(
            raw, self.reference, a["sync_confirmation"]["omega_rad_per_symbol"], args.iterations,
            fixed_nco=args.fixed_initializer_nco,
        )
        a["sync_confirmation"]["carrier_initialization"] = fitted
        if not fitted["initialized"]:
            self._active = None
            return
        a["slope"], a["phase"] = fitted["omega_radians_per_symbol"], fitted["phase_radians"]
        if self.timing_backend == "rust":
            loop = a["loop"]
            a["native"] = stream_rx._RustStreamTiming(
                sps=float(self.sps), start=a["start"],
                previous=(float(raw[0].real), float(raw[0].imag)), power=loop.power,
                kp=loop.kp, ki=loop.ki, lower=loop.lower, upper=loop.upper,
                phase=a["phase"], slope=a["slope"], gain=a["gain"], hold=len(self.reference),
            )

    with patch.object(stream_rx.PSKStreamReceiver, "_search_path", search):
        replay(SimpleNamespace(
            inputs=args.inputs, budget=args.budget, output=args.output, candidates=1,
            normalized_coarse=True, coherent_threshold=.45, fixed_nco=False,
            selection=args.selection, chunks=args.chunks,
        ))
    write(args.output / "initializer-plan.json", dict(
        driver_sha256=sha(Path(__file__)), iterations=args.iterations,
        fixed_initializer_nco=args.fixed_initializer_nco, fixed_confirmation_nco=False,
        viewed_regression=getattr(args, "viewed_regression", False),
        changed="post-confirmation initial carrier only;Q/timing unchanged",
        default_promote=False, whole_system_accepted=False,
        prototype_native_note="rebuild native state before first symbol;production integration rejected",
    ))
    (args.output / "initializer-driver.executed.py").write_bytes(Path(__file__).read_bytes())
    write(args.output / "initializer-manifest.json", {str(p.relative_to(args.output)): sha(p)
                                                    for p in args.output.rglob("*") if p.is_file()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "budget", "output"]:
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--iterations", type=int, choices=[1, 2], required=True)
    parser.add_argument("--fixed-initializer-nco", action="store_true")
    parser.add_argument("--viewed-regression", action="store_true")
    parser.add_argument("--chunks", choices=["whole", "fixed", "random"], default="whole")
    run(parser.parse_args())
