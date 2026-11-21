"""Independent field mapping and analytic pulse reference for M17 comparisons."""

import numpy as np

from nearlink_sdr.phy.control_info import ControlInfoA2, ControlInfoB1
from nearlink_sdr.phy.tx_pipeline import encode_head, encode_payload, head_crc_parameters
from tests.test_psk_field_layout import independent_wire
from tests.test_rx_timing_audit import audit


def packet(profile, payload, *, bad_head=False, bad_data=False):
    cfg = profile.config
    if profile.control_format == "B1":
        raw = ControlInfoB1(0, 0, 0, cfg.mcs_index, len(payload) + cfg.crc_len // 8, 0, 0).pack(
            cfg.head_llid
        )
    else:
        raw = ControlInfoA2(0, 0, 0, 0, 0, 0, 0, len(payload)).pack(
            head_crc_parameters(cfg)[2], lqi=cfg.mcs_index
        )
    if bad_head:
        raw[0] ^= 1
    control = encode_head(raw, cfg)
    data = np.unpackbits(np.frombuffer(payload, np.uint8), bitorder=profile.payload_bitorder)
    data = encode_payload(data, cfg)
    if bad_data:
        data = data.copy()
        data[:16] ^= 1
    sync = independent_wire(
        profile.sync_bits(), "QPSK" if cfg.frame_type == 2 else "BPSK", 40, 0, True
    )
    control = independent_wire(
        control,
        "BPSK" if cfg.frame_type == 4 else "QPSK",
        40 + len(sync),
        16 if cfg.frame_type == 2 else 4,
        cfg.pilot_interval == 0,
    )
    data = independent_wire(
        data, cfg.mod_str, 40 + len(sync) + len(control), cfg.pilot_interval, True
    )
    preamble = np.exp(1j * np.tile([np.pi / 4, 0], 20))
    return np.r_[preamble, sync, control, data]


def channel(symbols, *, offset, ppm, cfo, prefix, amplitude, family, seed, snr_db=None):
    signal, truth, step = audit.reference(symbols, offset, ppm)
    signal *= amplitude * np.exp(2j * np.pi * cfo * np.arange(len(signal)) / 16e6)
    x = np.r_[np.zeros(prefix, complex), signal, np.zeros(300)]
    rng = np.random.default_rng(seed)
    quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * 0.001
    first = int(np.floor(512 + offset))
    last = int(np.ceil(512 + offset + len(symbols) * step))
    clean_power = float(np.mean(abs(signal[first:last]) ** 2))
    metadata = dict(
        active_clean_power=clean_power,
        active_signal_interval=[first, last],
        requested_SNR_dB=snr_db,
        noise_std_per_component=0.0,
        quiet_std_per_component=0.001,
        original_IQ_modified=False,
    )
    if family == "noise":
        if snr_db is None or not np.isfinite(snr_db):
            raise ValueError("Noise experiment needs an explicit finite SNR")
        sigma = float(np.sqrt(clean_power / (2 * 10 ** (snr_db / 10))))
        x += (rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))) * sigma
        quiet = (rng.normal(size=1024) + 1j * rng.normal(size=1024)) * sigma
        metadata.update(noise_std_per_component=sigma, quiet_std_per_component=sigma)
    elif family == "tone":
        x += amplitude * 0.4 * np.exp(2j * np.pi * 213000 * np.arange(len(x)) / 16e6)
    elif family == "burst":
        first = len(x) // 2
        x[first : first + 7] += amplitude * (17 - 13j)
    elif family == "step":
        x[len(x) // 2 :] *= 0.1875
    return x, quiet, truth + prefix, step, metadata
