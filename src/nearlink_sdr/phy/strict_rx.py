"""Explicit PSK receive profiles and CRC-gated frame parsing (§6.3/6.4/6.9/6.10).

Profile configuration is supplied before acquisition, never selected by expected
payload or by retrying CRC. Factory settings are opt-in and have no sync/seed defaults.
"""
from dataclasses import dataclass, replace
from itertools import islice

import numpy as np

from nearlink_sdr.phy.control_info import ControlInfoA2, ControlInfoA3, ControlInfoB1
from nearlink_sdr.phy.control_length import control_data_length
from nearlink_sdr.phy.psk import PSKModulator, refine_psk_carrier
from nearlink_sdr.phy.psk_fields import demap_psk_field, psk_field_layout, soft_demap_psk_field
from nearlink_sdr.phy.sync_sequence import sync_signal_2, sync_signal_3, sync_signal_4
from nearlink_sdr.phy.tx_pipeline import TxConfig, head_crc_parameters

__all__ = ["PSKReceiveProfile", "decode_psk_frame"]


@dataclass(frozen=True)
class PSKReceiveProfile:
    """One known wire format, including negotiated link context.

    Ordinary A2/A3 has no MCS field: its MCS comes from the negotiated config.
    B1 carries MCS. Only an explicit factory profile interprets A3 LQI low bits
    as MCS. Byte order describes payload serialization, not control bit order.
    """

    config: TxConfig
    control_format: str = "A2"
    has_lqi: bool = True
    sync_word: int | None = None
    mcs_from_lqi: bool = False
    payload_bitorder: str = "big"
    max_payload_bytes: int = 2047
    allowed_mcs: tuple[int, ...] = tuple(range(13))
    name: str = "ordinary"
    soft_payload: bool = True
    noise_variance: float | None = None
    llr_method: str = "maxlog"
    length_contract: str = "standard"
    head_backend: str = "reference"

    def __post_init__(self):
        try:
            allowed = tuple(islice(self.allowed_mcs, 14))
        except TypeError as error:
            raise ValueError("Need a finite permitted MCS sequence") from error
        if len(allowed) > 13 or any(type(mcs) is not int or not 0 <= mcs <= 12 for mcs in allowed):
            raise ValueError("Permitted MCS must contain at most13 ordinary indices0..12")
        object.__setattr__(self, "allowed_mcs", allowed)
        cfg = replace(self.config)
        if (not isinstance(self.soft_payload, bool) or self.llr_method not in ("maxlog", "logmap")
                or (self.noise_variance is not None and (
                    not np.isscalar(self.noise_variance) or not np.isfinite(self.noise_variance)
                    or self.noise_variance <= 0))):
            raise ValueError("Need boolean soft_payload, positive noise_variance and LLR method")
        if cfg.frame_type not in (2, 3, 4):
            raise ValueError("Strict PSK profiles support FT2/3/4; GFSK requires another frontend")
        if self.control_format not in ("A2", "A3", "B1"):
            raise ValueError("Explicit A2, A3 or B1 control format is required")
        control_data_length(0, self.control_format, cfg.crc_len, contract=self.length_contract)
        if (cfg.frame_type == 2) != (self.control_format != "B1"):
            raise ValueError("Control format does not match frame type")
        expected = (27 if self.control_format == "B1" else
                    {"A2": 20, "A3": 28}[self.control_format] + 8 * self.has_lqi)
        if cfg.ctrl_bits_len != expected or (self.control_format == "B1" and self.has_lqi):
            raise ValueError("Control format/LQI does not match ctrl_bits_len")
        if (self.payload_bitorder not in ("big", "little") or cfg.crc_len not in (24, 32)
                or not isinstance(self.max_payload_bytes, int)
                or not 0 <= self.max_payload_bytes <= 2047):
            raise ValueError("Invalid payload byte order, CRC or length bound")
        if self.mcs_from_lqi and (self.control_format != "A3" or not self.has_lqi):
            raise ValueError("Factory MCS-from-LQI needs explicit A3 with LQI")
        legal_pilots = (0, 1, 4, 16) if self.mcs_from_lqi else (0, 4, 8, 16)
        if cfg.pilot_interval not in legal_pilots:
            raise ValueError("Illegal pilot interval for the selected profile")
        if self.sync_word is not None:
            width = {2: 64, 3: 62, 4: 126}[cfg.frame_type]
            if not isinstance(self.sync_word, int) or not 0 <= self.sync_word < 2**width:
                raise ValueError("Sync word outside wire width")
            if cfg.frame_type == 2:
                if cfg.head_crc_seed is not None and cfg.head_crc_seed != self.sync_word & 0xFFF:
                    raise ValueError("Header CRC context disagrees with explicit sync word")
                cfg = replace(cfg, head_crc_seed=self.sync_word & 0xFFF)
        if self.head_backend not in ("reference", "word-dense", "word-depth-parity"):
            raise ValueError("Unknown head backend")
        if self.head_backend != "reference" and (
                self.control_format not in ("A2", "B1") or self.mcs_from_lqi
                or (cfg.frame_type == 2 and not self.has_lqi)):
            raise ValueError("Word header is compiled only for ordinary A2+LQI and B1")
        head_crc_parameters(cfg)
        object.__setattr__(self, "config", cfg)

    @classmethod
    def ordinary(cls, cfg: TxConfig, *, control_format: str | None = None,
                 has_lqi: bool | None = None, max_payload_bytes: int = 2047,
                 length_contract: str = "standard"):
        """Known negotiated PHY parameters; never infer a control layout using CRC."""
        if control_format is None:
            control_format = "A2" if cfg.frame_type == 2 else "B1"
        if has_lqi is None:
            has_lqi = (cfg.ctrl_bits_len == {"A2": 28, "A3": 36}.get(control_format))
        return cls(cfg, control_format, has_lqi, max_payload_bytes=max_payload_bytes,
                   length_contract=length_contract)

    @classmethod
    def ws63_ft2_a3(cls, *, sync_word: int, data_crc_seed: int, pilot_interval: int,
                    sps: int = 4, symbol_rate_mhz: float = 4):
        """Explicit measured factory format, not the generic link defaults.

        Unknown encoded factory variants remain unsupported; no prefix stripping.
        """
        if not isinstance(data_crc_seed, int) or not 0 <= data_crc_seed < 2**24:
            raise ValueError("Factory data CRC seed outside 24-bit width")
        cfg = TxConfig(frame_type=2, mcs_index=8, whitening_seed=0,
                       crc_seed=data_crc_seed, crc_len=24, ctrl_bits_len=36,
                       pilot_interval=pilot_interval, sps=sps, symbol_rate_mhz=symbol_rate_mhz)
        return cls(cfg, "A3", True, sync_word=sync_word, mcs_from_lqi=True,
                   payload_bitorder="little", allowed_mcs=(6, 8, 10, 12), name="ws63-ft2-a3")

    def sync_bits(self):
        cfg = self.config
        if self.sync_word is not None:
            width = {2: 64, 3: 62, 4: 126}[cfg.frame_type]
            return np.array([(self.sync_word >> i) & 1 for i in range(width)])
        return {2: sync_signal_2, 3: sync_signal_3, 4: sync_signal_4}[cfg.frame_type](cfg.pid)

    def sync_symbols(self):
        kind = "QPSK" if self.config.frame_type == 2 else "BPSK"
        return PSKModulator(kind, sps=1).map_symbols(self.sync_bits())


def control_field_limits(profile):
    """Compile scalar comparisons from known configuration,never received answers."""
    standard_b1 = profile.control_format == "B1" and profile.length_contract == "standard"
    minimum = profile.config.crc_len // 8 if standard_b1 else 0
    maximum = min(2047, profile.max_payload_bytes + minimum)
    return minimum, maximum


def control_field_policy(profile, head, lqi, *, limits=None):
    """Shared scalar policy after a real CRC gate;this function does not check CRC.

    No payload allocation or choice among receiver answers. The finite controller
    models length/empty and membership comparisons as an explicit policy edge.
    """
    mcs = (head.mcs if profile.control_format == "B1" else
           (lqi & 15) if profile.mcs_from_lqi else profile.config.mcs_index)
    result = dict(mcs=mcs, mcs_source="header" if profile.control_format == "B1"
                  or profile.mcs_from_lqi else "negotiated_profile", length_ok=None, status=None)
    if mcs not in profile.allowed_mcs:
        result["status"] = "unsupported_mcs"
        return result
    minimum, maximum = control_field_limits(profile) if limits is None else limits
    length_ok = (0 <= head.data_length <= maximum
                 and not 0 < head.data_length < minimum
                 and not (getattr(head, "empty_packet", 0) and head.data_length != 0))
    result.update(length_ok=bool(length_ok), status=None if length_ok else "invalid_length")
    return result


def _reference_header(raw, profile, required, nsync, nwire):
    from nearlink_sdr.phy.rx_pipeline import decode_head, payload_coded_length

    cfg = profile.config
    row = {}
    decoded, ok = decode_head(raw, cfg)
    if not ok:
        row.update(status="bad_header_crc", header_crc=False)
        return row
    if profile.control_format == "B1":
        head = ControlInfoB1.unpack(decoded, cfg.head_llid)
        lqi = None
    else:
        parser = ControlInfoA2 if profile.control_format == "A2" else ControlInfoA3
        lqi, head = parser.unpack(decoded, head_crc_parameters(cfg)[2], has_lqi=profile.has_lqi)

    policy = control_field_policy(profile, head, lqi)
    mcs = policy["mcs"]
    row.update(header_crc=True, header=vars(head), lqi=lqi, mcs=mcs,
               mcs_source=policy["mcs_source"], header_end_symbol=required,
               length_ok=policy["length_ok"])
    if policy["status"] is not None:
        row["status"] = policy["status"]
        return row
    cfg = replace(cfg, mcs_index=mcs)
    information, present = control_data_length(head.data_length, profile.control_format,
                                              cfg.crc_len, contract=profile.length_contract)
    row.update(payload_length_bytes=information, data_crc_present=present,
               length_contract=profile.length_contract)
    nbits = payload_coded_length(information * 8, cfg) if present else 0
    bps = {"BPSK": 1, "QPSK": 2, "8PSK": 3}[cfg.mod_str]
    positions, nwire_data = psk_field_layout(nbits, bps, cfg.pilot_interval)
    start = nsync + nwire
    end = start + nwire_data
    row.update(data_start_symbol=start, frame_end_symbol=end, coded_bits=nbits,
               data_symbols=len(positions), wire_data_symbols=nwire_data)
    row["status"] = "header_valid"
    return row


def decode_psk_frame(symbols: np.ndarray, profile: PSKReceiveProfile, *,
                      carrier_method: str = "coherent", carrier_weight_kappa: int | None = None,
                      carrier_weight_fixed: bool = False) -> dict:
    """Parse one already-selected sync candidate; deliver only after both real CRCs.

    No phase, header format, seed, MCS or decoder retry is driven by CRC. Length
    and MCS are read only after header CRC. All rejected/truncated rows remain.
    """
    if carrier_method not in ("coherent", "phase_regression"):
        raise ValueError("Carrier method must be coherent or phase_regression")
    if carrier_weight_kappa is not None:
        from nearlink_sdr.phy.carrier_weights import BoundedCarrierWeights
        BoundedCarrierWeights(carrier_weight_kappa, fixed=carrier_weight_fixed)
        if carrier_method != "coherent":
            raise ValueError("Experimental weights require coherent carrier search")
    elif type(carrier_weight_fixed) is not bool or carrier_weight_fixed:
        raise ValueError("Fixed carrier weights require explicit kappa")
    y = np.asarray(symbols)
    if y.ndim != 1 or not np.isfinite(y).all():
        raise ValueError("Expected finite synchronized symbols")
    cfg = profile.config
    nsync = len(profile.sync_symbols())
    nhead = 64 if cfg.frame_type == 2 else 256
    kind = "BPSK" if cfg.frame_type == 4 else "QPSK"
    bps = 1 if kind == "BPSK" else 2
    interval = 16 if cfg.frame_type == 2 else 4
    positions, nwire = psk_field_layout(nhead, bps, interval, cfg.pilot_interval == 0)
    required = nsync + int(positions[-1]) + 1
    row = dict(status="truncated_header", header_crc=None, data_crc=None, payload=None,
               length_ok=None, profile=profile.name)
    if len(y) < required:
        return row
    raw = demap_psk_field(y[nsync:required], kind, nsync, interval)[:nhead]
    if profile.head_backend == "reference":
        header = _reference_header(raw, profile, required, nsync, nwire)
    else:
        from nearlink_sdr.sim.header_word_flow import HeaderWordService

        service = HeaderWordService(profile, layout=profile.head_backend.removeprefix("word-"))
        if not service.issue([int(bit) for bit in raw]):
            raise RuntimeError("New finite header context was not ready")
        for _ in range(10000):
            if service.ready:
                break
            service.step()
        if not service.ready:
            raise RuntimeError("Finite header did not meet its explicit host drain bound")
        header = service.take()
        header["head_word_model"] = dict(backend=profile.head_backend,
                                         service_edges=service.cycle,
                                         statistics=service.stats.copy(),
                                         geometry_statistics=service.geometry.stats.copy(),
                                         demap_service_cycles=None,
                                         whole_RX_cycle_acceptance=False)
    row.update(header)
    if row["status"] != "header_valid":
        return row
    return _decode_psk_payload(y, profile, row, carrier_method=carrier_method,
                               carrier_weight_kappa=carrier_weight_kappa,
                               carrier_weight_fixed=carrier_weight_fixed)


def _decode_psk_payload(y, profile, header, *, carrier_method="coherent",
                        carrier_weight_kappa=None, carrier_weight_fixed=False):
    """Private continuation of this active frame's real CRC-gated header.

    Used only with the owner's already validated header;not a public unchecked
    receive API. Preserve the original header receipt,including service counters.
    """
    from nearlink_sdr.phy.rx_pipeline import decode_payload

    if (header.get("header_crc") is not True or header.get("length_ok") is not True
            or header.get("status") not in ("header_valid", "truncated_data")):
        raise ValueError("Payload continuation requires this frame's valid header")
    row, cfg = dict(header), profile.config
    nsync = len(profile.sync_symbols())
    cfg = replace(cfg, mcs_index=row["mcs"])
    start, end, nbits = row["data_start_symbol"], row["frame_end_symbol"], row["coded_bits"]
    positions, _ = psk_field_layout(nbits, {"BPSK": 1, "QPSK": 2, "8PSK": 3}[cfg.mod_str],
                                    cfg.pilot_interval)
    if len(y) < end:
        row["status"] = "truncated_data"
        return row
    if not row["data_crc_present"]:
        row["status"] = "valid_empty_control"
        return row
    data = y[start:end]
    if len(data) >= 32:
        options = ({} if carrier_weight_kappa is None and not carrier_weight_fixed else
                   dict(carrier_weight_kappa=carrier_weight_kappa,
                        carrier_weight_fixed=carrier_weight_fixed))
        data, row["carrier"] = refine_psk_carrier(data, cfg.mod_str, start,
                                                  method=carrier_method, **options)
    else:
        row["carrier"] = dict(method="sync estimate only; fewer than 32 payload wire symbols")
    bits = demap_psk_field(data, cfg.mod_str, start, cfg.pilot_interval)[:nbits]
    llr = None
    if not cfg.is_uncoded:
        if profile.soft_payload:
            variance = profile.noise_variance
            source = "explicit normalized complex variance"
            if variance is None:
                # Known sync only. Includes residual ISI/timing/carrier error; not pure AWGN truth.
                variance = max(float(np.mean(abs(y[:nsync]-profile.sync_symbols())**2)), 1e-12)
                source = "known-sync residual estimate (includes model mismatch)"
            llr = soft_demap_psk_field(data, cfg.mod_str, start, variance,
                                       cfg.pilot_interval, method=profile.llr_method)[:nbits]
            row["soft_information"] = dict(method=profile.llr_method, variance=variance,
                                            variance_source=source, llr_bound=64.)
        else:
            row["soft_information"] = dict(method="legacy hard +/-10")
    checked = decode_payload(bits, cfg, row["payload_length_bytes"] * 8, llr=llr, details=True)
    pilot_mask = np.ones(len(data), dtype=bool)
    pilot_mask[positions] = False
    pilots = np.flatnonzero(pilot_mask)
    row["pilot_count"] = len(pilots)
    if len(pilots):
        from nearlink_sdr.phy.pilot import pilot_symbol
        reference = np.array([pilot_symbol(cfg.mod_str, start + int(i)) for i in pilots])
        row["pilot_phase_rms_rad"] = float(np.sqrt(np.mean(np.angle(
            data[pilots] * reference.conj()) ** 2)))
    else:
        row["pilot_phase_rms_rad"] = None
    status = "valid" if checked.data_crc else "bad_data_crc"
    if checked.code_block_crcs:
        row["code_block_crcs"] = checked.code_block_crcs
        if not all(checked.code_block_crcs):
            status = "bad_code_block_crc"
    row.update(status=status, data_crc=checked.data_crc,
               modulation=cfg.mod_str, code_rate=cfg.rate_str)
    if checked.passed:
        row["payload"] = np.packbits(checked.data_bits, bitorder=profile.payload_bitorder).tobytes()
    return row
