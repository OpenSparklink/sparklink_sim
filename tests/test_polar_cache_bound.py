"""Bound cross-packet state without changing decoder outputs or the public lookup API."""
from collections import OrderedDict

import numpy as np

from nearlink_sdr.common.code_block_seg import segment_without_crc
from nearlink_sdr.common.polar import PolarEncoder


def test_legal_payload_geometries_do_not_grow_cache_past_declared_limit(monkeypatch):
    import nearlink_sdr.common.polar as polar
    monkeypatch.setattr(polar, '_decoder_cache', OrderedDict())
    pairs = set()
    for rate in polar.RATE_TABLE:
        for length in range(1, 2048):
            pairs.update((n, len(info)) for n, info in segment_without_crc(
                np.zeros(length*8+24, dtype=int), rate, crc_len=24))
    assert len(pairs) > polar.POLAR_DECODER_CACHE_LIMIT
    pairs = sorted(pairs)
    first = polar.get_polar_decoder(*pairs[0])
    for pair in pairs[1:]:
        polar.get_polar_decoder(*pair)
        assert len(polar._decoder_cache) <= polar.POLAR_DECODER_CACHE_LIMIT
    rebuilt = polar.get_polar_decoder(*pairs[0])
    assert rebuilt is not first
    assert polar.get_polar_decoder(*pairs[0]) is rebuilt
    data = np.random.default_rng(68016).integers(0, 2, first.K, dtype=np.int8)
    encoded = PolarEncoder(first.N, first.K).encode(data)
    llr = (1-2*encoded.astype(float))*np.linspace(.2, 3, first.N)
    np.testing.assert_array_equal(first.decode(llr), data)
    np.testing.assert_array_equal(rebuilt.decode(llr), data)
    # A recent lookup protects the decoder while another old entry is replaced.
    polar.get_polar_decoder(1024, 1)
    assert polar.get_polar_decoder(*pairs[0]) is rebuilt
