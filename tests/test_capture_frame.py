"""Actual captured source and strict frame boundary regressions."""

from dataclasses import replace

import numpy as np
import pytest

from tests.captured_frame_reference import delivered, fixture, receive


@pytest.mark.parametrize("ft", [2, 3, 4])
def test_actual_source_to_MAC_and_all_borrowed_prefix_shared_tail_words(ft):
    p, iq, quiet, payload, _, _ = fixture(ft)
    result, frames, reads, owner, source, _ = receive(p, iq, quiet)
    assert delivered(frames) == [payload]
    event = next(e for e, m in frames if m.packets)
    row = event["frame"]
    assert row["header_crc"] is row["data_crc"] is row["length_ok"] is True
    assert row["mcs"] == p.config.mcs_index and row["payload_length_bytes"] == len(payload)
    assert event["consumed_source_end"] < len(iq) - 300
    if ft == 2:
        assert event["observed_source_pulses_at_delivery"] < len(iq)
    else:
        # Current longer fine services drain after these short FT3/4 captures end.
        # Functional frame closure is not a continuous-input scheduling pass.
        assert event["observed_source_pulses_at_delivery"] == len(iq)
    assert source.done and result["input_samples"] == len(iq)
    assert result["frame_prefix_reads"] and result["frame_source_reads"]
    assert result["peak_source_operations_per_primitive"] <= 2
    assert result["no_private_full_tail_banks"] and result["strict_frame_path_integrated"]
    assert not result["complete_cycle_evidence_valid"] and result["complete_latency_cycles"] is None
    assert result["confirmation_head_carrier_decoder_service_cycles"] is None
    used = [r for r in reads if r["id"] == event["selection_id"]]
    assert [r["index"] for r in used] == list(
        range(event["source_window_origin"], event["consumed_source_end"])
    )
    assert not owner.active and owner.peak_reference_bytes < 524288
    assert all(
        e["frame"]["status"] == "source_already_consumed" for e, m in frames if not m.packets
    )
    assert result["frame_overlap_retired"] == len(frames) - 1


def test_available_frame_is_delivered_while_later_input_continues():
    p, iq, quiet, payload, _, _ = fixture(2)
    iq = np.r_[iq, np.zeros(4096, complex)]
    result, frames, _, _, source, _ = receive(p, iq, quiet)
    assert delivered(frames) == [payload] and source.done
    event = next(e for e, m in frames if m.packets)
    assert event["observed_source_pulses_at_delivery"] < len(iq) - 4096
    assert event["consumed_source_end"] < len(iq) - 4096
    assert result["input_samples"] == len(iq) and result["complete_latency_cycles"] is None


def test_arbitrary_host_chunks_preserve_packets_all_words_and_sampling_trajectory():
    p, iq, quiet, payload, _, _ = fixture(2, pilots=8, size=47)
    first = receive(p, iq, quiet)
    for groups in [(1, 7, 53, 127), tuple(np.random.default_rng(17439).integers(1, 140, 17))]:
        other = receive(p, iq, quiet, groups=groups)
        assert first[0] == other[0] and first[2] == other[2]
        assert delivered(first[1]) == delivered(other[1]) == [payload]
        a = next(e for e, m in first[1] if m.packets)
        b = next(e for e, m in other[1] if m.packets)
        assert a["frame"] == b["frame"]
        for key in ["symbols", "symbol_centers_filtered", "timing_rate_ppm"]:
            np.testing.assert_array_equal(a[key], b[key])


@pytest.mark.parametrize("ft,backend", [(2, "word-dense"), (3, "word-depth-parity")])
def test_actual_word_header_service_reuses_real_CRC_and_legal_frame_end(ft, backend):
    p, iq, quiet, payload, _, _ = fixture(ft)
    result, frames, _, _, _, _ = receive(replace(p, head_backend=backend), iq, quiet)
    assert delivered(frames) == [payload]
    row = next(e["frame"] for e, m in frames if m.packets)
    assert row["header_crc"] and row["data_crc"] and row["length_ok"]
    assert row["head_word_model"]["service_edges"] > 0
    assert not row["head_word_model"]["whole_RX_cycle_acceptance"]
    assert result["complete_latency_cycles"] is None


def test_actual_B1_header_MCS_controls_payload_instead_of_negotiated_reference():
    p, iq, quiet, payload, _, _ = fixture(3, mcs=12, pilots=8)
    p = replace(p, config=replace(p.config, mcs_index=8), allowed_mcs=(8, 12))
    _, frames, _, _, _, _ = receive(p, iq, quiet)
    assert delivered(frames) == [payload]
    row = next(e["frame"] for e, m in frames if m.packets)
    assert row["mcs"] == 12 and row["header"]["mcs"] == 12
