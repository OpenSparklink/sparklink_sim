"""物理层接收流水线 -- TXS-10002-2025 标准

完成 TX 发射链路的逆操作: IQ 信号 → 帧同步 → 匹配滤波/解调 →
解扰 → Polar 解码 → CRC 校验, 还原 MAC 层载荷。
"""

from __future__ import annotations

__all__ = [
    "PayloadChecks",
    "RxResult",
    "decode_head",
    "decode_payload",
    "frame_sync",
    "payload_coded_length",
    "receive_capture",
    "rx_chain",
]


from dataclasses import dataclass, field

import numpy as np

from nearlink_sdr.common.code_block_seg import b1_code_block_plan, segment_without_crc
from nearlink_sdr.common.crc import (
    CRC24B_POLY,
    crc_check,
)
from nearlink_sdr.common.mcs import Modulation
from nearlink_sdr.common.polar import get_polar_decoder
from nearlink_sdr.common.scrambler import scramble_sequence
from nearlink_sdr.phy.control_info import polar_decode_control
from nearlink_sdr.phy.frame import FrameConfig, symbols_to_data_bits
from nearlink_sdr.phy.gfsk import GFSKDemodulator
from nearlink_sdr.phy.psk import rrc_filter
from nearlink_sdr.phy.sync_sequence import (
    sync_signal_1,
    sync_signal_2,
    sync_signal_3,
    sync_signal_4,
)
from nearlink_sdr.phy.tx_pipeline import TxConfig, head_crc_parameters

# Head 编码后固定码长
_HEAD_CODED_LEN = {1: 0, 2: 64, 3: 256, 4: 256}

# 调制类型字符串映射
_MOD_STR = {
    Modulation.BPSK: "BPSK",
    Modulation.QPSK: "QPSK",
    Modulation.PSK8: "8PSK",
}


@dataclass
class RxResult:
    """接收结果；data_bits 仅交付头/数据 CRC 均正确的载荷。

    diagnostic_data_bits 可保留头 CRC 正确但数据 CRC 失败的译码比特，
    仅供 BER 诊断，不交给 MAC。
    """
    data_bits: np.ndarray
    ctrl_bits: np.ndarray
    crc_ok: bool
    head_crc_ok: bool
    diagnostic_data_bits: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int8))


@dataclass(frozen=True)
class PayloadChecks:
    data_bits: np.ndarray
    data_crc: bool
    code_block_crcs: tuple[bool, ...]

    @property
    def passed(self) -> bool:
        return self.data_crc and all(self.code_block_crcs)


def payload_coded_length(n_info_bits: int, cfg: TxConfig) -> int:
    """Exact payload codeword length from a verified length and negotiated PHY context."""
    if (not isinstance(n_info_bits, int) or isinstance(n_info_bits, bool)
            or not 0 <= n_info_bits <= 2047 * 8):
        raise ValueError("Payload information length must be an integer in [0,16376]")
    b_len = n_info_bits + cfg.crc_len
    if cfg.is_uncoded:
        return b_len
    if cfg.frame_type in (3, 4):
        plan = b1_code_block_plan(
            b_len, cfg.rate_str, standard=cfg.b1_code_block_mode == "standard",
        )
        return sum(n for block in plan for n, _ in block.codewords)
    segments = segment_without_crc(np.zeros(b_len, dtype=int), cfg.rate_str, crc_len=cfg.crc_len)
    return sum(n for n, _ in segments)


def decode_payload(scrambled_bits: np.ndarray, cfg: TxConfig,
                   n_info_bits: int, *, llr: np.ndarray | None = None,
                   details: bool = False) -> tuple[np.ndarray, bool] | PayloadChecks:
    """载荷解码: 解扰 → [Polar 解码] → CRC 校验。

    :param scrambled_bits: 加扰后的载荷比特 (txPyLdW)。
    :param cfg: 发射/接收配置。
    :param n_info_bits: 原始信息比特数 (data_bytes * 8, 不含 CRC)。
    :param llr: 可选加扰域真实符号 LLR（正值为0）；省略时保留理想仿真的硬判决接口。
    :param details: 分别返回整帧 CRC 及逐逻辑码块 CRC。
    :returns: 默认返回 (data_bits, crc_ok)，CRC 通过要求所有适用校验成功。
    """
    result = _decode_payload(scrambled_bits, cfg, n_info_bits, llr)
    return result if details else (result.data_bits, result.passed)


def _decode_payload(scrambled_bits, cfg, n_info_bits, llr):
    if (not isinstance(n_info_bits, int) or isinstance(n_info_bits, bool)
            or not 0 <= n_info_bits <= 2047 * 8):
        raise ValueError("Payload information length must be an integer in [0,16376]")
    scrambled_bits = np.asarray(scrambled_bits)
    if scrambled_bits.ndim != 1 or np.any((scrambled_bits != 0) & (scrambled_bits != 1)):
        raise ValueError("Payload must be a one-dimensional binary array")
    b_len = n_info_bits + cfg.crc_len
    expected = payload_coded_length(n_info_bits, cfg)
    if llr is not None:
        llr = np.asarray(llr, dtype=float)
        if llr.shape != scrambled_bits.shape or not np.isfinite(llr).all():
            raise ValueError("LLR must be finite and match the scrambled coded bits")
    if len(scrambled_bits) != expected:
        return PayloadChecks(np.empty(0, dtype=np.int8), False, ())
    # 解扰
    offset = _HEAD_CODED_LEN[cfg.frame_type]
    sc = scramble_sequence(offset + len(scrambled_bits), cfg.whitening_seed)
    coded = np.asarray(scrambled_bits, dtype=np.uint8) ^ sc[offset: offset + len(scrambled_bits)]
    soft = (None if llr is None else np.clip(llr, -64., 64.) *
            (1.0 - 2.0 * sc[offset:offset+len(scrambled_bits)].astype(float)))

    block_crcs = []
    if cfg.is_uncoded:
        data_with_crc = coded.astype(np.int8)
    else:
        b1 = cfg.frame_type in (3, 4)
        standard = cfg.b1_code_block_mode == "standard"
        plan = b1_code_block_plan(b_len, cfg.rate_str, standard=standard) if b1 else ()
        if plan:
            dimensions = (word for block in plan for word in block.codewords)
        else:
            segments = segment_without_crc(np.zeros(b_len, dtype=int), cfg.rate_str,
                                           crc_len=cfg.crc_len)
            dimensions = ((n, len(info)) for n, info in segments)
        pos = 0
        info_blocks: list[np.ndarray] = []
        for n_code, k in dimensions:
            block = coded[pos: pos + n_code]
            # Compatibility fallback is explicitly hard; strict coded RX supplies soft LLR.
            block_llr = ((1.0 - 2.0 * block.astype(np.float64)) * 10.0
                         if soft is None else soft[pos:pos+n_code])
            dec = get_polar_decoder(n_code, k)
            info_blocks.append(dec.decode(block_llr))
            pos += n_code

        if plan:
            # 合并物理子码字，再扔掉逻辑块前导填充并校验该块。
            recovered, index = [], 0
            seed = cfg.crc_seed if standard else 0
            for block in plan:
                end = index + len(block.codewords)
                raw = np.concatenate(info_blocks[index:end])[-(block.data_bits + block.crc_bits):]
                if block.crc_bits:
                    block_crcs.append(bool(crc_check(raw, CRC24B_POLY, block.crc_bits, seed=seed)))
                recovered.append(raw[:block.data_bits])
                index = end
            info_blocks = recovered
        data_with_crc = np.concatenate(info_blocks)

    # CRC 校验
    ok = crc_check(data_with_crc, cfg.crc_poly, cfg.crc_len, seed=cfg.crc_seed)
    data = data_with_crc[: -cfg.crc_len]
    return PayloadChecks(data, bool(ok), tuple(block_crcs))


def decode_head(scrambled_bits: np.ndarray, cfg: TxConfig) -> tuple[np.ndarray, bool]:
    """头部解码: 解扰 → [Polar 解码] → 返回控制信息比特。

    :param scrambled_bits: 加扰后的头部比特 (txHeadW)。
    :param cfg: 发射/接收配置。
    :returns: (ctrl_bits, crc12_ok) — 控制信息 + CRC12 比特、CRC12 校验结果。
    """
    poly, width, seed, mask = head_crc_parameters(cfg)
    n_coded = _HEAD_CODED_LEN[cfg.frame_type]
    k_info = cfg.ctrl_bits_len + width
    if (not isinstance(cfg.ctrl_bits_len, int) or cfg.ctrl_bits_len <= 0
            or (n_coded and k_info > n_coded)):
        raise ValueError("Invalid control information length")
    bits = np.asarray(scrambled_bits)
    if bits.ndim != 1 or np.any((bits != 0) & (bits != 1)):
        raise ValueError("Header must be a one-dimensional binary array")
    expected = n_coded or k_info
    if len(bits) != expected:
        return np.empty(0, dtype=np.int8), False
    sc = scramble_sequence(expected, cfg.whitening_seed)
    coded = bits.astype(np.uint8) ^ sc
    decoded = (coded.astype(np.int8) if cfg.frame_type == 1 else
               polar_decode_control(coded.astype(np.int_), k_info, n_coded=n_coded))
    return decoded, bool(crc_check(decoded, poly, width, seed=seed, mask=mask))


def frame_sync(iq_signal: np.ndarray, cfg: TxConfig) -> int:
    """帧同步: 同步序列互相关峰值搜索。

    :param iq_signal: 接收到的基带 IQ 信号 (已匹配滤波 + 下采样到 1 sps)。
    :param cfg: 配置 (需要 pid 生成同步序列)。
    :returns: 同步序列起始符号位置, -1 表示未检测到。
    """
    if cfg.frame_type == 1:
        # FT1: 32 比特同步序列
        sync_bits = sync_signal_1(cfg.pid)
        sync_ref = 2.0 * sync_bits.astype(np.float64) - 1.0
        signal = 2.0 * np.real(iq_signal).astype(np.float64) - 1.0
    elif cfg.frame_type == 2:
        # FT2: 64 比特同步序列 → QPSK 符号
        sync_bits = sync_signal_2(cfg.pid)
        from nearlink_sdr.phy.psk import PSKModulator
        mod = PSKModulator(mod_type="QPSK", sps=1)
        sync_ref = mod.map_symbols(sync_bits)
        signal = iq_signal
    elif cfg.frame_type == 3:
        # FT3: 62 比特同步序列 → BPSK 符号
        sync_bits = sync_signal_3(cfg.pid or 0)
        from nearlink_sdr.phy.psk import PSKModulator
        mod = PSKModulator(mod_type="BPSK", sps=1)
        sync_ref = mod.map_symbols(sync_bits)
        signal = iq_signal
    else:
        # FT4: 126 比特同步序列 → BPSK 符号
        sync_bits = sync_signal_4(cfg.pid or 0)
        from nearlink_sdr.phy.psk import PSKModulator
        mod = PSKModulator(mod_type="BPSK", sps=1)
        sync_ref = mod.map_symbols(sync_bits)
        signal = iq_signal

    n_sync = len(sync_ref)
    n_signal = len(signal)
    if n_signal < n_sync:
        return -1

    # 滑动互相关
    corr = np.zeros(n_signal - n_sync + 1)
    for i in range(len(corr)):
        segment = signal[i: i + n_sync]
        corr[i] = np.abs(np.sum(segment * np.conj(sync_ref)))

    # 找峰值
    peak_idx = int(np.argmax(corr))
    peak_val = corr[peak_idx]
    mean_val = np.mean(corr)

    if peak_val < 2.0 * mean_val:
        return -1

    return peak_idx


def rx_chain(
    iq_signal: np.ndarray,
    cfg: TxConfig,
    n_data_bytes: int,
) -> RxResult:
    """完整接收链路: IQ → 帧解析 → 解码 → CRC 校验。

    :param iq_signal: 基带 IQ 信号。
    :param cfg: 发射/接收参数配置 (与发射端相同)。
    :param n_data_bytes: 数据长度 (字节), 用于确定码块分割。
    :returns: 包含恢复数据、控制信息和校验结果的 RxResult。
    """
    if (not isinstance(n_data_bytes, int) or isinstance(n_data_bytes, bool)
            or not 0 <= n_data_bytes <= 2047):
        raise ValueError("Known payload length must be an integer in [0,2047]")
    head_crc_parameters(cfg)  # Fail explicitly on missing link context.
    # 匹配滤波 + 下采样 → 符号级
    if cfg.frame_type == 1:
        demod = GFSKDemodulator(sps=cfg.sps)
        symbols = demod.demodulate(iq_signal)
    else:
        # RRC 匹配滤波 + 下采样, 保持复数符号 (不做硬判决)
        h = rrc_filter(0.4, cfg.sps)
        filtered = np.convolve(iq_signal, h, mode="same")
        symbols = filtered[::cfg.sps]

    frame_cfg = FrameConfig(
        frame_type=cfg.frame_type,
        symbol_rate_mhz=cfg.symbol_rate_mhz,
        pilot_interval=cfg.pilot_interval,
        crc_len=cfg.crc_len,
        sync_m_index=cfg.pid,
        mod_type=cfg.mod_str,
    )

    head_coded = _HEAD_CODED_LEN[cfg.frame_type]
    n_ctrl_coded = head_coded if head_coded > 0 else cfg.ctrl_bits_len + 12
    ctrl_bits_demod, _ = symbols_to_data_bits(
        symbols, frame_cfg,
        n_ctrl_coded_bits=n_ctrl_coded,
        n_data_bits=0, head_only=True,
    )

    # 头部解码
    ctrl_decoded, head_ok = decode_head(ctrl_bits_demod, cfg)

    if not head_ok:
        return RxResult(np.empty(0, dtype=np.int8), ctrl_decoded, False, False)

    # 帧结构解析 → 提取头部比特和数据比特
    n_data_bits = n_data_bytes * 8
    total_coded_bits = payload_coded_length(n_data_bits, cfg)

    _, data_bits_demod = symbols_to_data_bits(
        symbols, frame_cfg, n_ctrl_coded_bits=n_ctrl_coded, n_data_bits=total_coded_bits,
    )
    if len(data_bits_demod) != total_coded_bits:
        return RxResult(np.empty(0, dtype=np.int8), ctrl_decoded, False, True)

    # 载荷解码
    data_decoded, crc_ok = decode_payload(data_bits_demod, cfg, n_data_bits)

    return RxResult(
        data_bits=data_decoded if crc_ok else np.empty(0, dtype=np.int8),
        ctrl_bits=ctrl_decoded,
        crc_ok=crc_ok,
        head_crc_ok=head_ok,
        diagnostic_data_bits=data_decoded,
    )


def receive_capture(iq_signal: np.ndarray, quiet: np.ndarray, profile, *,
                    timing_subdivisions: int = 8, timing_tracking: bool = True,
                    timing_config=None, energy_window_symbols: int = 1,
                    carrier_method: str = "coherent") -> dict:
    """Strict IQ → candidates/frames using unknown starts and verified header lengths.

    The separate quiet calibration and explicit profile are required context.
    Every energy segment and rejection remains in acquisition/frames. Packet
    timing state resets per acquired sync. This whole-record entry is offline;
    it does not claim bounded streaming or real-time hardware operation.
    """
    from nearlink_sdr.phy.psk import acquire_psk_capture
    from nearlink_sdr.phy.strict_rx import PSKReceiveProfile, decode_psk_frame

    if not isinstance(profile, PSKReceiveProfile):
        raise ValueError("Explicit PSKReceiveProfile required")
    cfg = profile.config
    symbol_hz = cfg.symbol_rate_mhz * 1e6
    acquisition = acquire_psk_capture(
        iq_signal, quiet, cfg.sps * symbol_hz, symbol_hz, profile.sync_symbols(),
        timing_subdivisions=timing_subdivisions, timing_tracking=timing_tracking,
        timing_config=timing_config, energy_window_symbols=energy_window_symbols)
    frames = []
    for index, burst in enumerate(acquisition["bursts"]):
        if "symbols" not in burst:
            continue
        row = decode_psk_frame(burst["symbols"], profile, carrier_method=carrier_method)
        row.update(burst_index=index, sync_start_raw=burst["sync_start_raw"])
        frames.append(row)
    return dict(profile=profile.name, acquisition=acquisition, frames=frames)
