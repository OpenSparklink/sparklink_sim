"""物理层发射流水线 -- TXS-10002-2025 标准 6.10

将 MAC 层载荷经过 CRC → 码块分割 → Polar 编码 → 加扰 → 帧组装 →
调制, 输出可用于发射的基带 IQ 信号。
"""

from __future__ import annotations

__all__ = [
    "TxConfig",
    "attach_head_crc",
    "encode_head",
    "encode_payload",
    "head_crc_parameters",
    "tx_chain",
]


from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from nearlink_sdr.common.code_block_seg import (
    segment_b1_standard,
    segment_with_crc,
    segment_without_crc,
)
from nearlink_sdr.common.crc import (
    CRC12_POLY,
    CRC24A_POLY,
    CRC24B_POLY,
    CRC32_POLY,
    crc_attach,
)
from nearlink_sdr.common.mcs import Modulation, get_mcs
from nearlink_sdr.common.polar import PolarEncoder
from nearlink_sdr.common.scrambler import scramble_sequence
from nearlink_sdr.phy.control_info import polar_encode_control
from nearlink_sdr.phy.frame import FrameConfig, assemble_frame_bits, frame_to_symbols
from nearlink_sdr.phy.gfsk import GFSKModulator
from nearlink_sdr.phy.psk import PSKModulator

# Head 编码后固定码长
_HEAD_CODED_LEN = {1: 0, 2: 64, 3: 256, 4: 256}


def _head_len(frame_type: int) -> int:
    """返回头部编码后的码长。"""
    return _HEAD_CODED_LEN[frame_type]

# 调制类型字符串映射
_MOD_STR = {
    Modulation.BPSK: "BPSK",
    Modulation.QPSK: "QPSK",
    Modulation.PSK8: "8PSK",
}


@dataclass
class TxConfig:
    """发射参数配置。

    :ivar frame_type: 帧类型 (1-4)
    :ivar mcs_index: 编码调制索引 (0-12)
    :ivar pid: 24-bit PID, 用于生成同步字
    :ivar whitening_seed: 7-bit 加扰种子
    :ivar crc_seed: CRC 初始值
    :ivar crc_len: CRC 比特长度 (24 或 32)
    :ivar ctrl_bits_len: 控制信息有效比特数 (不含 CRC12)
    :ivar pilot_interval: 导频间隔 (4/8/16), 0 表示无导频
    :ivar sps: 每符号采样数 (脉冲成型用)
    :ivar symbol_rate_mhz: 符号速率 (MHz)
    :ivar b1_code_block_mode: FT3/4 编码载荷的标准分段或历史 1024 字合同
    """
    frame_type: int = 2
    mcs_index: int = 7
    pid: int = 0
    whitening_seed: int = 0
    crc_seed: int = 0
    crc_len: int = 24
    ctrl_bits_len: int = 28
    pilot_interval: int = 8
    sps: int = 4
    symbol_rate_mhz: float = 1.0
    head_crc_seed: int | None = None
    head_llid: int | None = None
    b1_code_block_mode: str = "legacy"
    _mcs: object = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if (self.b1_code_block_mode not in ("legacy", "standard")
                or (self.b1_code_block_mode == "standard" and self.frame_type not in (3, 4))):
            raise ValueError("Standard B1 code blocks require an FT3/4 context")
        self._mcs = get_mcs(self.mcs_index)

    @property
    def code_rate(self) -> Fraction:
        return self._mcs.code_rate

    @property
    def rate_str(self) -> str:
        return f"{self._mcs.code_rate.numerator}/{self._mcs.code_rate.denominator}"

    @property
    def modulation(self) -> Modulation:
        return self._mcs.modulation

    @property
    def mod_str(self) -> str:
        return _MOD_STR[self._mcs.modulation]

    @property
    def is_uncoded(self) -> bool:
        return self._mcs.code_rate == Fraction(1, 1)

    @property
    def crc_poly(self) -> int:
        return CRC32_POLY if self.crc_len == 32 else CRC24A_POLY



def head_crc_parameters(cfg: TxConfig) -> tuple[int, int, int, np.ndarray | None]:
    """Return actual §6.4 control CRC context, independent of payload CRC seed.

    A group defaults to the transmitted sync low 12 bits. An explicit seed
    describes a separately supplied sync context. B group requires link LLID;
    absence is a configuration error, never a successful unchecked header.
    """
    if cfg.frame_type in (1, 2):
        from nearlink_sdr.phy.sync_sequence import sync_signal_1, sync_signal_2

        sync = (sync_signal_1 if cfg.frame_type == 1 else sync_signal_2)(cfg.pid)
        seed = cfg.head_crc_seed
        if seed is None:
            seed = sum(int(b) << i for i, b in enumerate(sync[:12]))
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 4096:
            raise ValueError("A-group header CRC seed must be a 12-bit integer")
        return CRC12_POLY, 12, seed, None
    if cfg.frame_type not in (3, 4):
        raise ValueError("Unsupported frame type")
    if cfg.head_crc_seed is not None:
        raise ValueError("B-group header CRC uses fixed seed 0x555555")
    llid = cfg.head_llid
    if not isinstance(llid, int) or isinstance(llid, bool) or not 0 <= llid < 2**24:
        raise ValueError("B-group header CRC requires explicit 24-bit head_llid")
    return CRC24B_POLY, 24, 0x555555, np.array([(llid >> i) & 1 for i in range(24)])


def attach_head_crc(info_bits: np.ndarray, cfg: TxConfig) -> np.ndarray:
    """Attach the real control CRC to supplied information (also for ideal simulations)."""
    info = np.asarray(info_bits)
    if (info.ndim != 1 or len(info) != cfg.ctrl_bits_len
            or np.any((info != 0) & (info != 1))):
        raise ValueError("Control information must be binary and match ctrl_bits_len")
    poly, width, seed, mask = head_crc_parameters(cfg)
    return crc_attach(info, poly, width, seed=seed, mask=mask)


def encode_payload(data_bits: np.ndarray, cfg: TxConfig) -> np.ndarray:
    """载荷编码: CRC → [分段 → Polar] → 加扰。

    :param data_bits: 原始载荷比特 (不含 CRC)。
    :param cfg: 发射配置。
    :returns: 加扰后的载荷比特 (txPyLdW)。
    """
    # CRC
    with_crc = crc_attach(
        np.asarray(data_bits, dtype=int),
        cfg.crc_poly, cfg.crc_len, seed=cfg.crc_seed,
    )

    if cfg.is_uncoded:
        # rate 1/1: 不经过 Polar 编码
        coded = with_crc
    else:
        # 码块分割 + Polar 编码
        if cfg.b1_code_block_mode == "standard":
            segments = segment_b1_standard(with_crc, cfg.rate_str, seed=cfg.crc_seed)
        elif cfg.frame_type in (3, 4):
            segments = segment_with_crc(with_crc, cfg.rate_str)
        else:
            segments = segment_without_crc(with_crc, cfg.rate_str, crc_len=cfg.crc_len)
        coded_blocks: list[np.ndarray] = []
        for n_code, info in segments:
            enc = PolarEncoder(n_code, len(info))
            coded_blocks.append(enc.encode(np.asarray(info, dtype=np.int8)))
        coded = np.concatenate(coded_blocks)

    # 加扰 (offset = head coded len)
    offset = _head_len(cfg.frame_type)
    sc = scramble_sequence(offset + len(coded), cfg.whitening_seed)
    return np.asarray(coded, dtype=np.uint8) ^ sc[offset: offset + len(coded)]


def encode_head(ctrl_info_bits: np.ndarray, cfg: TxConfig) -> np.ndarray:
    """控制信息头部编码: [Polar(64, K)] → 加扰。

    :param ctrl_info_bits: 控制信息 + CRC12 比特 (已 pack 好的)。
    :param cfg: 发射配置。
    :returns: 加扰后的头部比特 (txHeadW)。
    """
    if cfg.frame_type == 1:
        # FT1: 不经过 Polar 编码
        coded = np.asarray(ctrl_info_bits, dtype=np.uint8)
    else:
        # FT2: Polar(64, K), FT3/4: Polar(256, K)
        coded = polar_encode_control(
            np.asarray(ctrl_info_bits, dtype=np.int8),
            n_coded=_head_len(cfg.frame_type),
        )

    # 加扰 (offset=0, head 在加扰序列起始位置)
    sc = scramble_sequence(len(coded), cfg.whitening_seed)
    return np.asarray(coded, dtype=np.uint8) ^ sc[:len(coded)]


def tx_chain(
    ctrl_info_bits: np.ndarray,
    data_bits: np.ndarray,
    cfg: TxConfig,
    *,
    data_present: bool = True,
) -> np.ndarray:
    """完整发射链路: 头部编码 + 载荷编码 → 帧组装 → 调制。

    :param ctrl_info_bits: 控制信息 + CRC12 比特。
    :param data_bits: 原始数据比特 (不含 CRC)。
    :param cfg: 发射配置。
    :param data_present: False仅用于经过校验的B1零长度控制帧，不附加载荷CRC。
    :returns: 基带 IQ 信号 (complex128), 长度 = 帧符号数 × sps。
    """
    if type(data_present) is not bool:
        raise ValueError("data_present must be boolean")
    if not data_present:
        from nearlink_sdr.phy.control_info import ControlInfoB1
        if (cfg.frame_type not in (3, 4) or len(data_bits) != 0
                or cfg.ctrl_bits_len != 27):
            raise ValueError("Only an empty B1 control frame can omit data and its CRC")
        if ControlInfoB1.unpack(np.asarray(ctrl_info_bits), cfg.head_llid).data_length != 0:
            raise ValueError("Omitting B1 data requires a real CRC-checked zero length header")
    # 编码；已知配置的理想API默认仍附加数据CRC，MAC适配层提供实际B1字段语义。
    head_w = encode_head(ctrl_info_bits, cfg)
    payload_w = encode_payload(data_bits, cfg) if data_present else np.empty(0, np.uint8)

    # 帧组装
    frame_cfg = FrameConfig(
        frame_type=cfg.frame_type,
        symbol_rate_mhz=cfg.symbol_rate_mhz,
        pilot_interval=cfg.pilot_interval,
        crc_len=cfg.crc_len,
        sync_m_index=cfg.pid,
        mod_type=cfg.mod_str,
    )
    fields = assemble_frame_bits(head_w, payload_w, frame_cfg)
    symbols = frame_to_symbols(fields, frame_cfg)

    # 调制 → IQ
    if cfg.frame_type == 1:
        mod = GFSKModulator(sps=cfg.sps)
        return mod.modulate(symbols)
    else:
        mod = PSKModulator(mod_type=cfg.mod_str, sps=cfg.sps)
        return _pulse_shape_symbols(symbols, mod)


def _pulse_shape_symbols(symbols: np.ndarray, mod: PSKModulator) -> np.ndarray:
    """对已包含导频的复数符号序列进行上采样和 RRC 脉冲成型。"""
    sps = mod.sps
    n = len(symbols)
    # 上采样
    upsampled = np.zeros(n * sps, dtype=complex)
    upsampled[::sps] = symbols
    # RRC 成型滤波
    return np.convolve(upsampled, mod._rrc, mode="same")
