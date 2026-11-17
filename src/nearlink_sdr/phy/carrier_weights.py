"""Causal bounded Mth-power carrier weights, an explicit experimental node.

Known-sync gain normalization supplies unit-power startup. The estimator sees
only current/past symbol power, never expected data, CRC or packet truth.
Fixed power input is an unsigned41-bitQ16 node interface, not a full fixed RX.
"""

import operator

import numpy as np

__all__ = ["BoundedCarrierWeights", "bounded_carrier_weights"]


class BoundedCarrierWeights:
    """w=min(p,kappa*Pprev);P+=1/64*(min(p,2*Pprev)-Pprev), then clamp.

    Power starts at1, with range2^-16..2^16. Gamma2 bounds upward slew to
    65/64 per symbol; at zero input decay is63/64 until the lower bound.
    Each packet owns a new instance. Caller chunks do not reset state.
    """

    def __init__(self, kappa, *, fixed=False):
        if type(kappa) is not int or kappa not in (2, 4, 8) or type(fixed) is not bool:
            raise ValueError("Need kappa2/4/8 and explicit boolean fixed flag")
        self.kappa, self.fixed = kappa, fixed
        self.power = 65536 if fixed else 1.0
        self.stats = dict(samples=0, weights_clipped=0, update_inputs_clipped=0,
                          lower_clamps=0, upper_clamps=0, input_power_saturations=0,
                          input_power_erased=0,
                          telemetry_saturated=False)

    def _count(self, key, value=1):
        result = self.stats[key] + int(value)
        if result > 2**63 - 1:
            self.stats["telemetry_saturated"] = True
        self.stats[key] = min(result, 2**63 - 1)

    def _check_count(self, count):
        if not 1 <= count <= 32:
            raise ValueError("Need1..32 power samples per bounded call")
        if self.stats["samples"] + count >= 2**63:
            raise OverflowError("Carrier weight sample counter exhausted")

    def advance_power(self, power):
        """Floating power adapter; quantization/range events remain observable."""
        values = np.asarray(power, dtype=float)
        if values.ndim != 1 or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Need finite nonnegative one-dimensional power")
        self._check_count(len(values))
        if self.fixed:
            # Clamp before conversion; very large finite powers cannot wrap.
            limit = (2**41 - 1) / 65536
            self._count("input_power_saturations", np.count_nonzero(values > limit))
            words = np.rint(np.minimum(values, limit) * 65536).astype(np.uint64)
            self._count("input_power_erased", np.count_nonzero((values > 0) & (words == 0)))
            return self.advance_words(words).astype(float) / 65536
        weights = []
        for value in values:
            previous = self.power
            weight = min(value, self.kappa * previous)
            clipped = min(value, 2 * previous)
            self._count("weights_clipped", value > self.kappa * previous)
            self._count("update_inputs_clipped", value > 2 * previous)
            updated = previous + (clipped - previous) / 64
            self._count("lower_clamps", updated < 2**-16)
            self._count("upper_clamps", updated > 2**16)
            self.power = min(2**16, max(2**-16, updated))
            weights.append(weight)
        self.stats["samples"] += len(values)
        return np.asarray(weights)

    def advance_words(self, power):
        """Exact unsigned41-bitQ16 interface; state33/output36/update signed34."""
        if not self.fixed or np.ndim(power) != 1:
            raise ValueError("Integer interface requires the fixed node")
        try:
            values = list(map(operator.index, power))
        except TypeError as error:
            raise ValueError("Power words must be integers, not rounded floats") from error
        self._check_count(len(values))
        if any(not 0 <= value < 2**41 for value in values):
            raise ValueError("Power exceeds unsigned41-bit interface")
        weights = []
        for value in values:
            previous = self.power
            weight = min(value, self.kappa * previous)
            clipped = min(value, 2 * previous)
            self._count("weights_clipped", value > self.kappa * previous)
            self._count("update_inputs_clipped", value > 2 * previous)
            delta = clipped - previous
            # Arithmetic floor, guard/sticky and ties-to-even, including negative
            # deltas. Independent Fraction oracle lives in tests.
            floor, remainder = divmod(delta, 64)
            step = floor + int(remainder > 32 or (remainder == 32 and floor & 1))
            updated = previous + step
            self._count("lower_clamps", updated < 1)
            self._count("upper_clamps", updated > 2**32)
            self.power = min(2**32, max(1, updated))
            weights.append(weight)
        self.stats["samples"] += len(values)
        return np.asarray(weights, dtype=np.uint64)


def bounded_carrier_weights(power, kappa, *, fixed=False):
    """Caller-owned complete output; internal node has constant bounded state."""
    values = np.asarray(power)
    if values.ndim != 1 or not len(values):
        raise ValueError("Need nonempty one-dimensional power")
    node = BoundedCarrierWeights(kappa, fixed=fixed)
    output = np.empty(len(values), dtype=float)
    for start in range(0, len(values), 32):
        output[start:start + 32] = node.advance_power(values[start:start + 32])
    return output, dict(kappa=kappa, fixed=fixed, stats=dict(node.stats),
                        final_estimated_power=node.power / 65536 if fixed else node.power,
                        power_initial=1.0, update_alpha="1/64", update_cap=2,
                        power_floor=2**-16, power_ceiling=2**16)
