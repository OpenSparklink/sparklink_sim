"""Actual captured source and strict frame boundary regressions."""

from dataclasses import replace

import numpy as np
import pytest

from tests.captured_frame_reference import delivered, fixture, receive


@pytest.mark.parametrize(
    "damage,status",
    [
        (dict(bad_head=True), "bad_header_crc"),
        (dict(bad_data=True), "bad_data_crc"),
    ],
)
def test_actual_bad_CRC_never_delivers_or_retries_overlapping_sync(damage, status):
    p, iq, quiet, _, _, _ = fixture(**damage)
    result, frames, _, _, _, _ = receive(p, iq, quiet)
    assert delivered(frames) == []
    assert [e["frame"]["status"] for e, _ in frames].count(status) == 1
    assert result["frame_payloads"] == 0
    assert all(e["frame"].get("payload") is None for e, _ in frames)


@pytest.mark.parametrize("wrong", ["seed", "length"])
def test_bad_context_and_illegal_trusted_length_have_separate_failure(wrong):
    p, iq, quiet, _, _, _ = fixture()
    if wrong == "seed":
        p = replace(p, config=replace(p.config, head_crc_seed=0x2D9))
        status = "bad_header_crc"
    else:
        p = replace(p, max_payload_bytes=16)
        status = "invalid_length"
    result, frames, _, _, _, _ = receive(p, iq, quiet)
    assert delivered(frames) == [] and result["frame_payloads"] == 0
    assert any(e["frame"]["status"] == status for e, _ in frames)
    if wrong == "length":
        row = next(e["frame"] for e, _ in frames if e["frame"]["status"] == status)
        assert row["header_crc"] and not row["length_ok"]


def test_EOF_truncation_never_flushes_missing_payload_samples():
    p, iq, quiet, _, symbols, step = fixture(size=91)
    end = int(512 + 0.283 + 37 + (symbols - 12) * step + 40)
    result, frames, reads, _, _, _ = receive(p, iq[:end], quiet)
    assert delivered(frames) == []
    assert any(e["frame"]["status"] == "truncated_data" for e, _ in frames)
    assert all(r["index"] < end for r in reads)
    assert result["input_samples"] == end


def test_pure_silence_has_no_deliveries_and_finite_source_accounting():
    p, _, quiet, _, _, _ = fixture()
    result, frames, reads, _, source, _ = receive(p, np.zeros(4096, complex), quiet)
    assert source.done and frames == reads == [] and result["frame_payloads"] == 0
    assert result["input_samples"] == 4096


def test_actual_truncated_header_keeps_CRC_unknown_and_has_no_payload():
    p, iq, quiet, _, _, _ = fixture()
    end = int(512 + 0.283 + 37 + 40 * 4 + 32 * 4 + 18 * 4 + 40)
    result, frames, reads, _, _, _ = receive(p, iq[:end], quiet)
    assert delivered(frames) == []
    row = next(e["frame"] for e, _ in frames if e["frame"]["status"] == "truncated_header")
    assert row["header_crc"] is None and row["data_crc"] is None and row["payload"] is None
    assert result["input_samples"] == end and all(r["index"] < end for r in reads)


@pytest.mark.parametrize("backend", ["reference", "word-depth-parity"])
def test_unsupported_actual_B1_MCS_never_reaches_payload_decode(backend):
    from tests.captured_frame_reference import independent_preamble
    from tests.maximum_reference import channel
    from tests.test_b1_wire_length import b1_header_wire

    p, *_ = fixture(3, pilots=8)
    p = replace(p, head_backend=backend)
    wire, _ = b1_header_wire(p, 34, mcs=15)
    preamble = independent_preamble(p)
    iq, quiet, *_ = channel(
        np.r_[preamble, wire, np.zeros(64)],
        offset=0.283,
        ppm=25,
        cfo=-33000,
        prefix=37,
        amplitude=0.14,
        family="clean",
        seed=108371,
        snr_db=None,
    )
    decoded = []
    _, frames, _, _, _, _ = receive(p, iq, quiet, decoder_observer=decoded.append)
    assert delivered(frames) == [] and decoded == []
    row = next(e["frame"] for e, _ in frames if e["frame"]["status"] == "unsupported_mcs")
    assert row["header_crc"] and row["payload"] is None


def test_diagnostic_bits_observer_returns_actual_failed_CRC_payload_without_retry():
    p, iq, quiet, payload, _, _ = fixture(bad_data=True)
    observed = []
    first = receive(p, iq, quiet)
    second = receive(p, iq, quiet, decoder_observer=observed.append)
    assert first[0] == second[0] and first[2] == second[2]
    assert delivered(second[1]) == [] and len(observed) == 1
    assert observed[0]["CRC"] is False and len(observed[0]["bits"]) == 8 * len(payload)
    assert (
        np.count_nonzero(observed[0]["bits"] != np.unpackbits(np.frombuffer(payload, np.uint8)))
        == 16
    )
