"""Distinct A/B length meanings and finite configuration acceptance (§6.4)."""

import pytest

from nearlink_sdr.phy.control_length import control_data_length


@pytest.mark.parametrize("width", [24, 32])
def test_b1_includes_crc_and_zero_means_crc_absent(width):
    assert control_data_length(39, "B1", width) == (39 - width // 8, True)
    assert control_data_length(0, "B1", width) == (0, False)
    assert control_data_length(width // 8, "B1", width) == (0, True)
    assert control_data_length(2047, "B1", width) == (2047 - width // 8, True)
    for n in range(1, width // 8):
        with pytest.raises(ValueError, match="shorter"):
            control_data_length(n, "B1", width)


@pytest.mark.parametrize("kind", ["A2", "A3"])
@pytest.mark.parametrize("width", [24, 32])
def test_A_control_and_factory_length_do_not_subtract_crc(kind, width):
    assert control_data_length(39, kind, width) == (39, True)
    assert control_data_length(0, kind, width) == (0, True)


def test_explicit_legacy_contract_preserves_frozen_payload_only_inputs():
    assert control_data_length(39, "B1", 24, contract="legacy-b1-payload-length") == (39, True)
    assert control_data_length(0, "B1", 24, contract="legacy-b1-payload-length") == (0, True)


@pytest.mark.parametrize("args,kwargs", [
    ((True, "B1", 24), {}), ((-1, "B1", 24), {}), ((2048, "B1", 24), {}),
    ((3.0, "B1", 24), {}), ((3, "B2", 24), {}), ((3, "B1", 16), {}),
    ((3, "B1", 24), {"contract": "auto"}),
    ((3, "A2", 24), {"contract": "legacy-b1-payload-length"}),
])
def test_no_implicit_context_or_crc_based_contract_guess(args, kwargs):
    with pytest.raises(ValueError, match="explicit"):
        control_data_length(*args, **kwargs)
