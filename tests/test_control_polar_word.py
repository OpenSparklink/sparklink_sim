"""Compiled control ROMs checked against original-standard matrix vectors."""

import json
from pathlib import Path

import pytest

from nearlink_sdr.phy.control_polar_word import ControlPolarFormat, compile_control_rom

REFERENCE = json.loads((Path(__file__).parent / "data/control_polar_reference.json").read_text())


@pytest.mark.parametrize("N,K", [(64, 40), (256, 51)])
def test_compiled_rom_matches_independent_standard_positions(N, K):
    fmt = ControlPolarFormat(N, K)
    rom = compile_control_rom(fmt)
    assert len(rom) == 2 * N - 1 + K
    assert rom[:2 * N - 1].count(2) > 0
    for vector in REFERENCE["vectors"]:
        if vector["N"] == N:
            assert list(rom[2 * N - 1:]) == vector["information_positions"]
    assert fmt.llr_bits == 10
    assert fmt.stack_frames == fmt.n + 1
    assert compile_control_rom(fmt) is rom


@pytest.mark.parametrize("N,K", [(64, 39), (256, 39), (64, 48), (128, 40), (True, 40),
                                (64, 40.0), (256, -1)])
def test_not_an_arbitrary_user_format_interface(N, K):
    with pytest.raises(ValueError, match="compiled"):
        ControlPolarFormat(N, K)


@pytest.mark.parametrize("bad", [None, {}, [], (64, 40)])
def test_rom_requires_validated_format(bad):
    with pytest.raises(ValueError, match="explicit"):
        compile_control_rom(bad)
