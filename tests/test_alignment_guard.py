"""Independent rational alignment and guard placement,including signed ties."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from nearlink_sdr.phy.interpolation_word import (
    LagrangeWordFormat,
    aligned_components,
    interpolate_filter_word,
    unpack_interpolated_word,
)
from nearlink_sdr.sim.capture_fine import pack_filter_word
from nearlink_sdr.sim.interpolation_flow import LagrangeInterpolationFlow
from tests.test_interpolation_word import exact_reference

FMT = LagrangeWordFormat(16, 2, 22, alignment_guard_first=True)


def test_retains_alignment_fraction_without_wider_state_or_output():
    triples = [(v, -v, e) for v, e in [(1, 0), (1, -1), (3, -2), (5, -3)]]
    common, old, _ = aligned_components(triples, LagrangeWordFormat(16, 2, 22))
    _, new, _ = aligned_components(triples, FMT)
    assert common == 0 and old == ((4, 0, 4, 4), (-4, 0, -4, -4))
    assert new == ((4, 2, 3, 2), (-4, -2, -3, -2))
    assert FMT.state_bits == 25 and FMT.output_bits == 22


@pytest.mark.parametrize("exponent", [-128, -17, 0, 126, 127])
def test_alignment_exact_fraction_ties_erasure_and_signed_width(exponent):
    for shift, value in product(
        (0, 1, 2, 3, 7, 18, 20, 21, 22, 255), (-262144, -3, -1, 1, 3, 262143)
    ):
        e = max(-128, exponent - shift)
        triples = [(1, -1, exponent), (value, -value - 1, e), (0, 0, 127), (0, 0, -128)]
        common, rows, erased = aligned_components(triples, FMT)
        assert common == exponent
        expected = tuple(
            tuple(round(Fraction(t[k]) * Fraction(2) ** (t[2] - common + 2)) for t in triples)
            for k in (0, 1)
        )
        assert rows == expected
        assert all(-(2**20) <= x < 2**20 for row in rows for x in row)
        assert erased == sum(
            t[k] != 0 and x == 0
            for k, row in enumerate(rows)
            for t, x in zip(triples, row, strict=True)
        )


def test_independent_lagrange_tighter_bound_extremes_and_pipeline_stalls():
    rng = np.random.default_rng(945119)
    queries = []
    for signs in product((-262144, 262143), repeat=4):
        for phase in (0, 1, 32768, 65535, 65536):
            queries.append((tuple(pack_filter_word(v, -v - 1, -16) for v in signs), phase))
    for k in range(96):
        exponents = [-128, 127, -17, 0] if k % 3 == 0 else rng.integers(-128, 128, 4)
        queries.append(
            (
                tuple(
                    pack_filter_word(int(r), int(i), int(e))
                    for r, i, e in zip(
                        rng.integers(-262144, 262144, 4),
                        rng.integers(-262144, 262144, 4),
                        exponents,
                        strict=True,
                    )
                ),
                int(rng.integers(0, 65537)),
            )
        )
    results = []
    for words, phase in queries:
        word, info = interpolate_filter_word(words, phase, FMT)
        assert not info["output_saturations"]
        r, i, e = unpack_interpolated_word(word, FMT)
        bound = Fraction(11, 8) * Fraction(2) ** (info["common_exponent"] - 2)
        bound += Fraction(1, 2) * Fraction(2) ** e
        exact = exact_reference(words, phase, FMT)
        assert (
            max(abs(Fraction(v) * Fraction(2) ** e - y) for v, y in zip((r, i), exact, strict=True))
            <= bound
        )
        results.append((word, info))
    flow = LagrangeInterpolationFlow(FMT)
    issued = received = 0
    for _ in range(3000):
        if not 45 <= flow.cycle < 311:
            row = flow.take()
            if row is not None:
                assert row["tag"] == received
                assert (row["word"], row["info"]) == results[received]
                received += 1
        if issued < len(queries) and flow.issue(*queries[issued], tag=issued):
            issued += 1
        flow.advance()
        assert flow.live <= 7
        if received == len(queries):
            break
    assert received == len(queries) and flow.stats["backpressure"]
    cost = flow.estimate()
    assert cost["alignment_RNE_shifters19"] == 0 and cost["alignment_RNE_shifters21"] == 8
    assert cost["pipeline_register_bound_bits"] == 2057 and cost["DSP25x18"] == 6
    assert cost["no_stall_latency"] == 32 and cost["no_stall_II"] == 8
    assert not cost["alignment_RNE_timing_verified"]


def test_guard_placement_is_explicit_compiled_candidate_not_width_interface():
    for bad in (1, None, "before"):
        with pytest.raises(ValueError):
            LagrangeWordFormat(16, 2, 22, alignment_guard_first=bad)
    for args in ((16, 0, 19), (16, 2, 19), (20, 2, 19)):
        with pytest.raises(ValueError):
            LagrangeWordFormat(*args, alignment_guard_first=True)
    assert not LagrangeWordFormat(16, 2, 22).alignment_guard_first


@pytest.mark.parametrize("ft,mcs,pilot,length", [(2, 12, 0, 3), (3, 6, 8, 139), (4, 0, 16, 419)])
def test_candidate_real_head_payload_and_cached_partitions(ft, mcs, pilot, length):
    from tests.test_selected_rx import equal, ordinary, selected_input
    from tests.test_selected_word import quantize_input, word_consume

    profile = ordinary(ft, mcs, pilot)
    source, payload, selection = selected_input(profile, length=length)
    words = quantize_input(source)
    event, _ = word_consume(profile, words, selection, FMT, [32])
    frame = event["frame"]
    assert frame["payload"] == payload
    assert frame["header_crc"] and frame["data_crc"] and frame["length_ok"]
    assert frame["mcs"] == mcs
    for chunks in ([1], [3, 29, 17], [7, 31, 5, 11]):
        other, rx = word_consume(profile, words, selection, FMT, chunks, reuse=True)
        equal(event, other)
        assert rx.initial_interpolants.diagnostics()["valid_words"] == 0


@pytest.mark.parametrize(
    "damage,status", [({"bad_head": True}, "bad_header_crc"), ({"bad_data": True}, "bad_data_crc")]
)
def test_guard_precision_does_not_bypass_real_CRC_or_deliver_suffix(damage, status):
    from tests.test_selected_rx import ordinary, selected_input
    from tests.test_selected_word import quantize_input, word_consume

    profile = ordinary()
    source, _, selection = selected_input(profile, **damage)
    event, _ = word_consume(profile, quantize_input(source), selection, FMT, [7, 31], reuse=True)
    assert event["frame"]["status"] == status and event["frame"]["payload"] is None
