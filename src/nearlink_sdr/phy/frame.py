"""帧结构组装与解析 -- TXS-10002-2025 标准 6.3.2-6.3.5 节。

支持帧类型 1-4，包含前导码、同步序列、控制信息、数据载荷和导频插入。
"""


__all__ = [
    "FrameConfig",
    "FrameFields",
    "assemble_frame_bits",
    "decode_ft1_a3_candidates",
    "decode_ft2_a3_capture",
    "frame_to_symbols",
    "symbols_to_data_bits",
]


from dataclasses import dataclass, field

import numpy as np

from nearlink_sdr.common.crc import CRC24A_POLY, crc_check
from nearlink_sdr.common.scrambler import descramble
from nearlink_sdr.phy.control_info import ControlInfoA3
from nearlink_sdr.phy.preamble import generate_preamble
from nearlink_sdr.phy.psk_fields import demap_psk_field, map_psk_field
from nearlink_sdr.phy.sync_sequence import (
    sync_signal_1,
    sync_signal_2,
    sync_signal_3,
    sync_signal_4,
)


@dataclass
class FrameConfig:
    """无线帧配置参数。"""
    frame_type: int  # 1, 2, 3, or 4
    symbol_rate_mhz: float = 1.0
    pilot_interval: int = 0  # 0=no pilot, 4/8/16
    crc_len: int = 24  # 24 or 32
    mod_type: str = ""  # auto-determined if empty
    sync_m_index: int | None = None  # m-sequence index for sync signal

    def __post_init__(self):
        if self.frame_type not in (1, 2, 3, 4):
            raise ValueError(f"frame_type must be 1-4, got {self.frame_type}")
        if not self.mod_type:
            self.mod_type = _default_mod_type(self.frame_type)


def _default_mod_type(frame_type: int) -> str:
    """各帧类型的默认调制方式。"""
    return {1: "GFSK", 2: "QPSK", 3: "QPSK", 4: "BPSK"}[frame_type]


def decode_ft1_a3_candidates(
    bits: np.ndarray, *, sync_word: int, data_crc_seed: int,
    whitening_seed: int | None,
) -> list[dict]:
    """解析已判决比特中的 FT1/A3 候选（6.3.2、6.4.1.4、6.10.1/4）。

    同步字、CRC种子及是否加扰必须由调用者显式给出，不从期待载荷挑选。
    ``None`` 明确表示不加扰（例如经实测确认的厂测配置），不是普通链路默认值。
    头CRC通过后才读取11位长度；数据CRC通过后才返回payload。
    错候选和截断候选保留诊断，继续搜索其后的同步，不消耗整个窗口。
    返回坐标均为输入比特偏移，end为exclusive；不声称完成RF或硬件时间同步。
    """
    bits = np.asarray(bits)
    if bits.ndim != 1 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("bits must be a one-dimensional binary array")
    if not 0 <= sync_word <= 0xFFFFFFFF or not 0 <= data_crc_seed <= 0xFFFFFF:
        raise ValueError("sync/CRC seed outside wire width")
    if whitening_seed is not None and not 0 <= whitening_seed <= 127:
        raise ValueError("whitening seed outside 7-bit width")
    sync = np.array([(sync_word >> i) & 1 for i in range(32)])
    found = []
    for offset in range(len(bits) - 31):
        if not np.array_equal(bits[offset:offset + 32], sync):
            continue
        start = offset + 32
        row = dict(sync_start=offset, sync_end=start, status="truncated_header",
                   header_crc=None, data_crc=None, payload=None)
        found.append(row)
        if len(bits) - start < 40:
            continue
        tail = bits[start:].astype(np.uint8, copy=True)
        if whitening_seed is not None:
            tail = descramble(tail, whitening_seed)
        try:
            _, head = ControlInfoA3.unpack(tail[:40], sync_word & 0xFFF)
        except ValueError:
            row.update(status="bad_header_crc", header_crc=False)
            continue
        row.update(header_crc=True, header=vars(head), header_end=start + 40)
        size = head.data_length * 8
        row.update(payload_start=start + 40, payload_end=start + 40 + size,
                   frame_end=start + 40 + size + 24)
        if len(tail) < 40 + size + 24:
            row["status"] = "truncated_data"
            continue
        valid = crc_check(tail[40:40 + size + 24], CRC24A_POLY, 24, data_crc_seed)
        row.update(status="valid" if valid else "bad_data_crc", data_crc=bool(valid))
        if valid:
            row["payload"] = np.packbits(tail[40:40 + size], bitorder="little").tobytes()
    return found


def decode_ft2_a3_capture(symbols: np.ndarray, *, sync_word: int,
                           data_crc_seed: int, pilot_interval: int,
                           carrier_method: str = "coherent") -> dict:
    """解析显式无加扰/CRC24A的FT2/A3采集，symbols从已知同步起点开始。

    控制QPSK固定16:1导频，Polar(64,48)，实际A3头CRC后才读长度和MCS。
    数据复用现有码块分割、Polar和CRC；不使用decode_head的简化成功标志。
    1:1导频仅供显式厂测配置，不能推广为普通业务默认配置。
    返回payload仅表示CRC合格；是否符合外部源的已知数据仍须调用者另行核对。
    """
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame

    profile = PSKReceiveProfile.ws63_ft2_a3(
        sync_word=sync_word, data_crc_seed=data_crc_seed, pilot_interval=pilot_interval)
    return decode_psk_frame(symbols, profile, carrier_method=carrier_method)


# ── 各帧类型参数 ──

# 各帧类型同步信号配置
_SYNC_CONFIG = {
    1: {"func": sync_signal_1, "bits": 32, "sync_mod": "GFSK"},
    2: {"func": sync_signal_2, "bits": 64, "sync_mod": "QPSK"},
    3: {"func": sync_signal_3, "bits": 62, "sync_mod": "BPSK"},
    4: {"func": sync_signal_4, "bits": 126, "sync_mod": "BPSK"},
}

# 各帧类型控制信息编码配置
_CTRL_CONFIG = {
    1: {"coded_bits": None, "ctrl_mod": "GFSK", "pilot_interval": 0},
    2: {"coded_bits": 64, "ctrl_mod": "QPSK", "pilot_interval": 16},
    3: {"coded_bits": 256, "ctrl_mod": "QPSK", "pilot_interval": 4},
    4: {"coded_bits": 256, "ctrl_mod": "BPSK", "pilot_interval": 4},
}


@dataclass
class FrameFields:
    """已组装帧各字段的比特/符号数组。"""
    preamble_bits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    sync_bits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    ctrl_bits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    data_bits: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int8))
    frame_type: int = 1


def assemble_frame_bits(
    ctrl_info_bits: np.ndarray,
    data_payload_bits: np.ndarray,
    config: FrameConfig,
) -> FrameFields:
    """在比特级别组装完整帧结构。

    生成调制前的比特级帧结构。

    :param ctrl_info_bits: 物理层控制信息比特。
    :param data_payload_bits: 数据 + 完整性保护 + CRC 比特。
    :param config: 帧配置参数。

    :returns: 包含所有帧字段的 FrameFields 对象。
    """
    fields = FrameFields(frame_type=config.frame_type)

    # Preamble
    fields.preamble_bits = generate_preamble(config.frame_type, config.symbol_rate_mhz)

    # Sync sequence
    sync_cfg = _SYNC_CONFIG[config.frame_type]
    if config.frame_type in (1, 2):
        fields.sync_bits = sync_cfg["func"](config.sync_m_index)
    else:
        # sync_signal_3/4 take m_seq_index, default 0
        idx = config.sync_m_index if config.sync_m_index is not None else 0
        fields.sync_bits = sync_cfg["func"](idx)

    # Control info
    fields.ctrl_bits = ctrl_info_bits

    # Data payload
    fields.data_bits = data_payload_bits

    return fields


def frame_to_symbols(
    fields: FrameFields,
    config: FrameConfig,
) -> np.ndarray:
    """将帧字段转换为含导频插入的调制符号流。

    帧类型 1（GFSK）返回拼接后的比特序列（非复数符号）。
    帧类型 2-4（PSK）返回含导频的复数符号流。

    :param fields: 已组装的帧字段。
    :param config: 帧配置参数。

    :returns: 类型 1：比特数组（送入 GFSK 调制器）。
        类型 2-4：基带复数符号数组。
    """
    if config.frame_type == 1:
        # 帧类型 1：简单比特拼接，无导频，无信道编码
        return np.concatenate([
            fields.preamble_bits,
            fields.sync_bits,
            fields.ctrl_bits,
            fields.data_bits,
        ]).astype(np.int8)

    # PSK 帧类型（2、3、4）
    segments = []
    symbol_count = 0

    # 前导码符号（相位交替 [π/4, 0]）
    preamble_phases = np.where(
        fields.preamble_bits == 0,
        np.pi / 4,
        0.0,
    )
    preamble_syms = np.exp(1j * preamble_phases)
    segments.append(preamble_syms)
    symbol_count += len(preamble_syms)

    sync_mod = _SYNC_CONFIG[config.frame_type]["sync_mod"]
    sync_syms = map_psk_field(fields.sync_bits, sync_mod, symbol_count)
    segments.append(sync_syms)
    symbol_count += len(sync_syms)

    ctrl_cfg = _CTRL_CONFIG[config.frame_type]
    ctrl_wire = map_psk_field(fields.ctrl_bits, ctrl_cfg["ctrl_mod"], symbol_count,
                              ctrl_cfg["pilot_interval"],
                              omit_last=not len(fields.data_bits) or config.pilot_interval == 0)
    segments.append(ctrl_wire)
    symbol_count += len(ctrl_wire)
    if len(fields.data_bits):
        segments.append(map_psk_field(fields.data_bits, config.mod_type, symbol_count,
                                       config.pilot_interval, omit_last=True))

    return np.concatenate(segments)


def symbols_to_data_bits(
    symbols: np.ndarray,
    config: FrameConfig,
    n_ctrl_coded_bits: int = 0,
    n_data_bits: int = 0,
    *, head_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """将接收符号流解析还原为控制位和数据位。

    简化解析器，假定已知帧参数。

    :param symbols: 接收到的复数符号流。
    :param config: 帧配置参数。
    :param n_ctrl_coded_bits: 编码控制位的数量。
    :param n_data_bits: 数据位数量（调制前）。

    :returns: (ctrl_bits, data_bits) — 解调后的比特数组。
    """
    if config.frame_type == 1:
        # 帧类型 1：比特级解析
        n_preamble = len(generate_preamble(1, config.symbol_rate_mhz))
        n_sync = 32
        offset = n_preamble + n_sync
        ctrl_bits = symbols[offset : offset + n_ctrl_coded_bits].real.astype(np.int8)
        if head_only:
            return ctrl_bits, np.empty(0, dtype=np.int8)
        data_start = offset + n_ctrl_coded_bits
        data_bits = symbols[data_start : data_start + n_data_bits].real.astype(np.int8)
        return ctrl_bits, data_bits

    # PSK 帧类型
    pos = 0

    # 跳过前导码
    n_preamble = len(generate_preamble(config.frame_type, config.symbol_rate_mhz))
    pos += n_preamble

    # 跳过同步序列
    sync_bits_len = _SYNC_CONFIG[config.frame_type]["bits"]
    sync_mod = _SYNC_CONFIG[config.frame_type]["sync_mod"]
    n_sync_syms = _bits_to_symbols_count(sync_bits_len, sync_mod)
    pos += n_sync_syms

    # 控制符号
    ctrl_cfg = _CTRL_CONFIG[config.frame_type]
    n_ctrl_syms = _bits_to_symbols_count(
        ctrl_cfg["coded_bits"] or n_ctrl_coded_bits,
        ctrl_cfg["ctrl_mod"],
    )
    interval = ctrl_cfg["pilot_interval"]
    n_ctrl_with_pilots = n_ctrl_syms
    if interval:
        n_ctrl_with_pilots += (n_ctrl_syms - int(config.pilot_interval == 0)) // interval
    ctrl_raw = symbols[pos:pos + n_ctrl_with_pilots]
    ctrl_bits = demap_psk_field(ctrl_raw, ctrl_cfg["ctrl_mod"], pos, interval)
    pos += n_ctrl_with_pilots
    if head_only:
        return ctrl_bits, np.empty(0, dtype=np.int8)
    data_bits = demap_psk_field(symbols[pos:], config.mod_type, pos, config.pilot_interval)

    if n_data_bits > 0:
        data_bits = data_bits[:n_data_bits]

    return ctrl_bits, data_bits


# ── 内部调制辅助函数 ──


def _modulate_bits(bits: np.ndarray, mod_type: str) -> np.ndarray:
    """简单比特-符号映射（无脉冲成形）。"""
    if mod_type == "BPSK":
        phases = np.where(bits == 0, np.pi / 2, -np.pi / 2)
        symbols = np.exp(1j * phases)
        # 奇数索引符号附加 -90° 旋转 (exp(-j*pi/2) = -j)
        symbols[1::2] *= -1j
        return symbols
    elif mod_type == "QPSK":
        b0 = bits[0::2].astype(np.intp)
        b1 = bits[1::2].astype(np.intp)
        idx = b0 * 2 + b1
        phase_deg = np.array([45.0, 135.0, -45.0, -135.0])
        symbols = np.exp(1j * np.deg2rad(phase_deg[idx]))
        symbols[1::2] *= np.exp(-1j * np.deg2rad(45.0))
        return symbols
    elif mod_type == "GFSK":
        return bits.astype(np.int8)
    elif mod_type == "8PSK":
        from nearlink_sdr.phy.psk import _8PSK_MAP, _8PSK_ROTATION

        remainder = len(bits) % 3
        if remainder:
            bits = np.concatenate([bits, np.zeros(3 - remainder, dtype=bits.dtype)])
        b0 = bits[0::3].astype(int)
        b1 = bits[1::3].astype(int)
        b2 = bits[2::3].astype(int)
        idx = (b0 << 2) | (b1 << 1) | b2
        map_arr = np.array([_8PSK_MAP[k] for k in range(8)])
        symbols = map_arr[idx].copy()
        symbols[1::2] *= _8PSK_ROTATION
        return symbols
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")


def _demodulate_symbols(symbols: np.ndarray, mod_type: str) -> np.ndarray:
    """简单符号-比特硬判决（无匹配滤波器）。"""
    if mod_type == "BPSK":
        syms = symbols.copy()
        syms[1::2] *= 1j  # 撤销 -j 旋转
        return np.where(syms.imag > 0, 0, 1).astype(np.int8)
    elif mod_type == "QPSK":
        syms = symbols.copy()
        syms[1::2] *= np.exp(1j * np.deg2rad(45.0))
        phase = np.rad2deg(np.angle(syms)) % 360
        bits = np.zeros(len(syms) * 2, dtype=np.int8)
        bits[0::2] = np.where(phase >= 180, 1, 0)
        bits[1::2] = np.where((phase >= 90) & (phase < 270), 1, 0)
        return bits
    elif mod_type == "GFSK":
        return symbols.astype(np.int8)
    elif mod_type == "8PSK":
        from nearlink_sdr.phy.psk import _8PSK_MAP, _8PSK_ROTATION

        syms = symbols.copy()
        syms[1::2] *= np.conj(_8PSK_ROTATION)
        phases = np.angle(syms)
        map_phases = np.array([np.angle(_8PSK_MAP[k]) for k in range(8)])
        diffs = np.abs(
            np.exp(1j * phases[:, None]) - np.exp(1j * map_phases[None, :])
        )
        idx = np.argmin(diffs, axis=1)
        bits = np.zeros(len(syms) * 3, dtype=np.int8)
        bits[0::3] = (idx >> 2) & 1
        bits[1::3] = (idx >> 1) & 1
        bits[2::3] = idx & 1
        return bits
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")


def _bits_to_symbols_count(n_bits: int, mod_type: str) -> int:
    """计算给定比特数对应的符号数。"""
    if mod_type == "QPSK":
        return n_bits // 2
    elif mod_type == "8PSK":
        return n_bits // 3
    return n_bits
