"""Exclusive 7-series BRAM half sites and existing control SSC (§6.9.1.4).

Same registered reads,arithmetic,port arbitration and codewords. Explicit native
backing replaces logical lists;no extra decoder,CRC answer retry or synthesized
resource/timing claim. B1 uses existing parity layout,A2 existing dense layout.
"""

from dataclasses import asdict, dataclass

from nearlink_sdr.phy.control_polar_word import ControlPolarFormat
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.sync_fit_word import signed, unsigned
from nearlink_sdr.sim.control_polar_flow import ControlPolarService, _Memory
from nearlink_sdr.sim.header_word_flow import HeaderWordService
from nearlink_sdr.sim.psk_hard_flow import HeaderFromWordsService
from nearlink_sdr.sim.split_sampling_replay import SplitSamplingFrameReplay

__all__ = ["PackedControlPolarService", "PackedHeadSamplingReplay", "control_bank_plan"]


@dataclass(frozen=True)
class _Bank:
    name: str
    logical_length: int
    logical_bits: int
    native_bits: int
    word_bits: int
    depth: int
    site: int
    half: int | None


def _plan(memory):
    full = [k for k in memory.cells if memory.native_bits[k] == 36864]
    half = [k for k in memory.cells if memory.native_bits[k] == 18432]
    if len(full) + len(half) != len(memory.cells):
        raise ValueError("Only actual18/36Kb banks have a compiled placement")
    result = []
    for name, cells in memory.cells.items():
        bits = 10 if name.startswith("L") else memory.fmt.n if name == "ROM" else 1
        width = 18 if bits > 9 else 9 if bits > 1 else 1
        native = memory.native_bits[name]
        depth = native // width if width != 1 else native // 9 * 8
        if len(cells) > depth:
            raise OverflowError("Logical plane exceeds its native width/depth allocation")
        site = full.index(name) if name in full else len(full) + half.index(name) // 2
        part = None if name in full else half.index(name) & 1
        result.append(_Bank(name, len(cells), bits, native, width, depth, site, part))
    occupied = set()
    for bank in result:
        halves = (0, 1) if bank.half is None else (bank.half,)
        for part in halves:
            if (bank.site, part) in occupied:
                raise RuntimeError("Two planes alias one physical RAM half")
            occupied.add((bank.site, part))
    return tuple(result)


def _estimate(banks):
    sites = max(b.site for b in banks) + 1
    native = sum(b.native_bits for b in banks)
    return dict(
        banks=[asdict(b) | dict(
            primitive="RAMB36E1" if b.half is None else "RAMB18E1",
            mode="TDP;ROM uses only one read port", port_word_address_bits=b.depth.bit_length()-1,
            ports_used_max=1 if b.name == "ROM" else 2,
            native_word_includes_parity=b.word_bits > 1,
            unused_primitive_capacity_bits=b.native_bits - b.depth * b.word_bits,
            payload_uses_parity=b.logical_bits > (b.word_bits // 9 * 8 if b.word_bits > 1 else 1),
            ECC_enabled=False,
        ) for b in banks],
        physical_sites=sites, configured_native_bits=native, reserved_site_bits=sites * 36864,
        unassigned_reserved_half_bits=sites * 36864 - native,
        independent_halves=True, aliased_or_shared_banks=False,
        RAM_read_latency_edges=1, RAM_output_extra_register_enabled=False,
        runtime_arbiter_or_context_switch_added=False,
        new_DSP=0, new_logical_state_bits=0,
        padding_rule="logical signed10 two's complement or unsigned bits;zero unused write bits",
        uninitialized_None="host write-before-read assertion;not reset RAM or valid-bit array",
        placement_basis="UG473v1.14 ch1 width/depth and two independent18Kb halves",
        LUT_FF_or_Fmax=None, synthesized=False, full_receiver16MSps=False,
    )


def control_bank_plan(fmt, *, layout="dense"):
    """Concrete capacity/port allocation;not an optimality or placement proof."""
    if not isinstance(fmt, ControlPolarFormat) or layout not in ("dense", "depth-parity"):
        raise ValueError("Need compiled control format and existing layout")
    return _estimate(_plan(_Memory(fmt, layout)))


class _Plane:
    """One native padded plane;no second logical array retained.

    Operational reads/writes still pass the original physical port controller.
    Direct access here is a host diagnostic,not another hardware port.
    """

    def __init__(self, bank, initial):
        self.bank = bank
        self.physical_words = [None] * bank.depth
        for address, value in enumerate(initial):
            if value is not None:
                self[address] = value

    def __len__(self):
        return self.bank.logical_length

    def _address(self, index):
        if type(index) is not int or not 0 <= index < len(self):
            raise IndexError("Native plane address is outside its compiled logical range")

    def __getitem__(self, index):
        self._address(index)
        value = self.physical_words[index]
        if value is None:
            return None
        b = self.bank
        value = unsigned(value, b.word_bits) & ((1 << b.logical_bits) - 1)
        if b.name.startswith("L") and value & (1 << (b.logical_bits - 1)):
            value -= 1 << b.logical_bits
        return value

    def __setitem__(self, index, value):
        self._address(index)
        b = self.bank
        value = signed(value, b.logical_bits) if b.name.startswith("L") else unsigned(
            value, b.logical_bits
        )
        self.physical_words[index] = unsigned(value & ((1 << b.logical_bits) - 1), b.word_bits)


class _PackedMemory(_Memory):
    def __init__(self, fmt, layout):
        super().__init__(fmt, layout)
        self.banks = _plan(self)
        old = self.cells  # Constructor-only source;released after native planes take ownership.
        self.cells = {b.name: _Plane(b, old[b.name]) for b in self.banks}


class PackedControlPolarService(ControlPolarService):
    """Unchanged SSC microprogram and actual TDP budgets,exclusive native backing."""

    def _make_memory(self, fmt, layout):
        return _PackedMemory(fmt, layout)


class _PackedHeader(HeaderWordService):
    def _make_codec(self, fmt, layout, observer):
        return PackedControlPolarService(fmt, layout=layout, observer=observer)

    def cost(self):
        result = super().cost()
        codec = result["codec"]
        plan = _estimate(self.codec.memory.banks)
        codec["original_separate_site_upper"] = codec["BRAM36_sites_upper"]
        codec["explicit_native_plan"] = plan
        codec["BRAM36_sites_upper"] = plan["physical_sites"]
        result["original_separate_site_upper"] = result["BRAM36_sites_upper"]
        result["BRAM36_sites_upper"] = plan["physical_sites"]
        result["physical_site_estimate_not_synthesis"] = True
        return result


class _PackedWords(HeaderFromWordsService):
    def _make_header(self, profile, layout, observer):
        return _PackedHeader(profile, layout=layout, observer=observer)


class PackedHeadSamplingReplay(SplitSamplingFrameReplay):
    """Explicit compiled A2-dense/B1-parity choice,never a CRC-dependent fallback.

    Same512 source and all sampling/loop parameters. No payload or producer cycle
    credit;new development/holdout must establish the remaining deadline margin.
    """

    def __init__(self, profile, prepared, **options):
        if not isinstance(profile, PSKReceiveProfile):
            raise ValueError("Need actual PSK profile before transferring ownership")
        layout = "dense" if profile.config.frame_type == 2 else "depth-parity"
        if options.pop("head_layout", layout) != layout:
            raise ValueError("Packed head experiment uses compiled A2-dense/B1-parity")
        super().__init__(profile, prepared, head_layout=layout, **options)

    def _make_head_owner(self, profile, layout):
        return _PackedWords(profile, layout=layout)
