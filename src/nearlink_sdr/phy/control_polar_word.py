"""Compiled hard-control Polar ROMs (§6.9.1.4 and Appendix C).

N64/K40 A2 and N256/K51 B1 only. The B1 includes CRC24B, not CRC12.
This module does not check CRC or parse fields; formats remain explicit trials.
"""

from dataclasses import dataclass
from functools import lru_cache

from nearlink_sdr.common.polar import RELIABILITY_SEQ_1024

__all__ = ["ControlPolarFormat", "compile_control_rom"]


@dataclass(frozen=True)
class ControlPolarFormat:
    """Two compiled codeword sizes with ten-bit signed internal min-sum LLR."""

    N: int
    K: int

    def __post_init__(self):
        if (type(self.N) is not int or type(self.K) is not int
                or (self.N, self.K) not in ((64, 40), (256, 51))):
            raise ValueError("Need compiled control Polar N64/K40 or N256/K51")

    @property
    def n(self):
        return self.N.bit_length() - 1

    @property
    def llr_bits(self):
        return 10

    @property
    def stack_frames(self):
        return self.n + 1


def compile_control_rom(fmt):
    """Offline node-type/position ROM; runtime uses one registered read port.

    Node type0 is frozen,1 is rate-1,2 is partial. Information addresses follow
    the complete heap node array. No decoder,received bits or CRC enters compile.
    """
    if not isinstance(fmt, ControlPolarFormat):
        raise ValueError("Need an explicit compiled control format")
    return _compile(fmt)


@lru_cache(maxsize=2)
def _compile(fmt):
    nodes, information = ssc_geometry(fmt.N, fmt.K)
    return nodes + information


@lru_cache(maxsize=64)
def ssc_geometry(n_code, k_info):
    """Compile the node types and sorted information positions for finite SSC."""
    if (type(n_code) is not int or n_code not in (64, 128, 256, 512, 1024)
            or type(k_info) is not int or not 0 < k_info <= n_code):
        raise ValueError("Need a supported SSC power of two and information length")
    sequence = [value for value in RELIABILITY_SEQ_1024 if value < n_code]
    if len(sequence) != n_code or len(set(sequence)) != n_code:
        raise ValueError("Reliability subset is not a permutation")
    information = sorted(sequence[-k_info:])
    mask = [int(k in information) for k in range(n_code)]
    prefix = [0]
    for bit in mask:
        prefix.append(prefix[-1] + bit)
    nodes = [0] * (2 * n_code - 1)
    for depth in range(n_code.bit_length()):
        length = n_code >> depth
        for start in range(0, n_code, length):
            count = prefix[start + length] - prefix[start]
            index = (1 << depth) - 1 + start // length
            nodes[index] = 0 if count == 0 else 1 if count == length else 2
    return tuple(nodes), tuple(information)
