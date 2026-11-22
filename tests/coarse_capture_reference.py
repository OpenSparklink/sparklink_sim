"""Independent pulse stimuli;live controller receives only timed RRC/flag ports."""

import math

import numpy as np

from nearlink_sdr.phy.rrc_pair import FullMantissaRRC
from nearlink_sdr.phy.rrc_pair_flow import PairRRCFlow, PairRRCFlowAssumptions
from nearlink_sdr.phy.rrc_word import FixedRRCFormat
from nearlink_sdr.sim.capture_fine import pack_filter_word
from tests.maximum_reference import channel, packet
from tests.test_coarse_ingress import profile


def stimulus(frame=2, *, clock=200, count=1536, groups=(), flags_true=True):
    p = profile(frame)
    data = bytes(((j * 79) ^ (j >> 2) ^ 0xD3) & 255 for j in range(157))
    iq, _, _, _, _ = channel(packet(p, data), offset=.371, ppm=25, cfo=19000,
                             prefix=33, amplitude=.13, family="clean", seed=551103)
    iq = iq[:count]
    filt, chunks, start = FullMantissaRRC(FixedRRCFormat(12, 18)), [], 0
    for size in (*groups, len(iq)):
        end = min(start + size, len(iq))
        for j in range(start, end, 32):
            chunks.append(filt.feed_output_words(iq[j:min(j + 32, end)]))
        start = end
    words = [np.concatenate([v[k] for v in chunks]) for k in range(3)]
    packed = [pack_filter_word(*(int(v[j]) for v in words)) for j in range(len(iq))]
    producer_node = PairRRCFlow(PairRRCFlowAssumptions(clock_hz=clock * 1000000))
    producer = producer_node.simulate(len(iq), trace=True)
    releases = [(e["cycle"], e["sample"], None if e["invalid"] else packed[e["sample"]])
                for e in producer["trace"] if e["event"] == "output"]
    flags = [(math.ceil(j * clock / 16) + 10, j, flags_true) for j in range(len(iq))]
    return p, packed, releases, flags, producer, producer_node.resource_estimate()
