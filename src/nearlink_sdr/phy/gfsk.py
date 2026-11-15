
__all__ = [
    "GFSK_BW_CONFIG",
    "GFSKDemodulator",
    "GFSKModulator",
    "analyze_gfsk_capture",
]


import numpy as np
from scipy.signal import firwin, lfilter

# TXS-10002-2025 6.2.1.1 GFSK调制
# BT = 0.5, 调制系数 h 在 0.45~0.55 之间

# 标准带宽与符号间隔对应关系。
#
# 注意: §6.2.1.1 与 §8.2.2.1 对 0.1~0.5 Msym/s 的"最小频偏"给出不同数值。
# §6.2.1.1 的 46.25 / 92.5 / 185 kHz 分别高于同一节 h∈[0.45, 0.55] 在该符号
# 速率下所能产生的上限 34.375 / 68.75 / 137.5 kHz, 与本节自身矛盾, 应为标准
# 排版行移; §8.2.2.1 与 h 范围自洽 (其 fd1 区间正好以 h=0.5 的标称频偏为中心)。
# 此处按 §8.2.2.1 取值, 可执行的验收限值见
# nearlink_sdr.phy.rf_compliance.GFSK_FREQ_DEV_SPECS。
GFSK_BW_CONFIG = {
    0.1:  {"symbol_period_us": 10,   "min_freq_dev_khz": 18.5},
    0.125: {"symbol_period_us": 8,   "min_freq_dev_khz": 23.125},
    0.25: {"symbol_period_us": 4,    "min_freq_dev_khz": 46.25},
    0.5:  {"symbol_period_us": 2,    "min_freq_dev_khz": 92.5},
    1.0:  {"symbol_period_us": 1,    "min_freq_dev_khz": 185},
    2.0:  {"symbol_period_us": 0.5,  "min_freq_dev_khz": 370},
    4.0:  {"symbol_period_us": 0.25, "min_freq_dev_khz": 740},
}


def _gaussian_filter(bt: float, span: int, sps: int) -> np.ndarray:
    """生成高斯滤波器的脉冲响应。

    ``BT`` 定义为高斯滤波器单边 -3 dB 带宽与符号周期之积, 因此

    .. math:: h(t) \\propto \\exp\\!\\left(-\\frac{t^2}{2\\sigma^2}\\right),
              \\quad \\frac{\\sigma}{T} = \\frac{\\sqrt{\\ln 2}}{2\\pi \\cdot BT}

    该约定与 MathWorks ``gaussdesign`` 一致, 也与
    ``sparklink-hw/scripts/generate/gfsk_tables.py`` 的系数生成使用同一定义。

    :param bt: 带宽时间积 ``BT``
    :param span: 单侧截断跨度 (符号数)
    :param sps: 每符号采样数

    :returns: 归一化到单位直流增益的冲激响应
    """
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    h = np.exp(-0.5 * (t / sigma) ** 2)
    return h / np.sum(h)


class GFSKModulator:
    """TXS-10002-2025 6.2.1.1 GFSK调制器。

    比特1表示正频偏，比特0表示负频偏。
    """

    def __init__(self, sps: int = 8, mod_index: float = 0.5,
                 bt: float = 0.5, gauss_span: int = 3):
        if not 0.45 <= mod_index <= 0.55:
            raise ValueError(
                f"mod_index={mod_index} 超出标准范围 [0.45, 0.55] (§6.2.1.1)"
            )
        self.sps = sps
        self.mod_index = mod_index
        self.bt = bt
        self.gauss_span = gauss_span
        self._gauss_filter = _gaussian_filter(bt, gauss_span, sps)

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """GFSK调制。

        :param bits: 输入比特序列, shape (N,), 值为 0/1

        :returns: 复基带IQ信号, shape (N*sps,)
        """
        # NRZ映射: 0 -> -1, 1 -> +1
        nrz = 2.0 * bits.astype(float) - 1.0

        # 上采样：使用矩形脉冲（repeat），而非冲激
        upsampled = np.repeat(nrz, self.sps)

        # 高斯滤波平滑频率轨迹
        filtered = np.convolve(upsampled, self._gauss_filter, mode='same')

        # 频率积分得到瞬时相位
        # 每个符号的总相位变化应为 h*pi
        freq_deviation = self.mod_index * np.pi / self.sps
        phase = np.cumsum(filtered) * freq_deviation

        # 生成复基带信号
        signal = np.exp(1j * phase)
        return signal


class GFSKDemodulator:
    """GFSK非相干解调器（基于频率鉴别器）。"""

    def __init__(self, sps: int = 8):
        self.sps = sps

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """GFSK解调。

        :param signal: 复基带IQ信号

        :returns: 解调后的比特序列, 值为 0/1
        """
        # 频率鉴别：取相邻采样点的相位差
        phase_diff = np.angle(signal[1:] * np.conj(signal[:-1]))

        # 按符号累积频率 (包含末尾不足一个完整符号的部分)
        n_symbols = -(-len(phase_diff) // self.sps)  # ceiling division
        bits = np.zeros(n_symbols, dtype=int)
        for i in range(n_symbols):
            start = i * self.sps
            end = min(start + self.sps, len(phase_diff))
            segment = phase_diff[start:end]
            bits[i] = 1 if np.sum(segment) > 0 else 0

        return bits


def analyze_gfsk_capture(signal: np.ndarray, quiet: np.ndarray,
                         sample_hz: float, symbol_hz: float) -> dict:
    """Acquire energy bursts and call the existing GFSK discriminator.

    This front end does not infer factory frame fields, whitening seeds or CRC.
    Timing maximizes the integrated-frequency eye, without expected payload or
    sync correlation. Every energy segment is reported, including short/partial
    segments. Returned indexes refer to the original, unchanged input.

    :param signal: Complete complex IQ acquisition, no pre-alignment required.
    :param quiet: Associated quiet controls in the same native amplitude units.
    :param sample_hz: Actual acquisition sample rate.
    :param symbol_hz: Requested symbol rate, explicitly supplied, not assumed 8 SPS.
    :returns: Acquisition parameters and burst diagnostics; bits are not business data.
    """
    x = np.asarray(signal)
    noise = np.asarray(quiet)
    if (x.ndim != 1 or noise.ndim != 1 or len(x) < 65 or len(noise) < 65
            or not np.isfinite(x).all() or not np.isfinite(noise).all()):
        raise ValueError("Expected finite one-dimensional IQ and quiet controls")
    if not np.isfinite([sample_hz, symbol_hz]).all() or min(sample_hz, symbol_hz) <= 0:
        raise ValueError("Sample and symbol rates must be finite and positive")
    ratio = sample_hz / symbol_hz
    sps = round(ratio)
    if sps < 4 or abs(ratio - sps) > 1e-9:
        raise ValueError("Capture front end currently requires integer SPS >= 4")
    box = np.ones(sps) / sps
    quiet_power = np.convolve(abs(noise) ** 2, box, mode="same")
    threshold = 4 * float(np.quantile(quiet_power, 0.999))
    energy = np.convolve(abs(x) ** 2, box, mode="same") > threshold
    starts = np.flatnonzero(np.diff(energy.astype(int), prepend=0) == 1)
    ends = np.flatnonzero(np.diff(energy.astype(int), append=0) == -1) + 1
    # Fixed, documented acquisition filter, not a payload-dependent correction.
    taps = firwin(65, symbol_hz, fs=sample_hz)
    delay = (len(taps) - 1) // 2
    filtered = lfilter(taps, [1], x)
    bursts = []
    for a, b in zip(starts, ends, strict=True):
        burst = dict(start_raw=int(a), end_raw_exclusive=int(b),
                     edge_truncated=bool(a == 0 or b == len(x)))
        bursts.append(burst)
        if b - a < 64 * sps:
            burst["status"] = "short_energy_segment"
            continue
        lo, hi = int(a + 8 * sps), int(b - 8 * sps)
        frequency = (np.angle(filtered[lo + 1:hi] * np.conj(filtered[lo:hi - 1]))
                     * sample_hz / (2 * np.pi))
        centers = np.quantile(frequency, [0.2, 0.8])
        if np.ptp(centers) < symbol_hz * 0.001:
            burst["status"] = "insufficient_frequency_states"
            continue
        for _ in range(20):
            labels = abs(frequency[:, None] - centers[None, :]).argmin(axis=1)
            if not all(np.any(labels == k) for k in range(2)):
                break
            centers = np.array([frequency[labels == k].mean() for k in range(2)])
        else:
            cfo = float(centers.mean())
            corrected = filtered * np.exp(-2j * np.pi * cfo * np.arange(len(x)) / sample_hz)
            eye = []
            for phase in range(sps):
                begin = lo + phase
                disc = np.angle(corrected[begin + 1:hi] * np.conj(corrected[begin:hi - 1]))
                n = len(disc) // sps
                eye.append(float(np.mean(abs(disc[:n * sps].reshape(n, sps).sum(axis=1)))))
            phase = int(np.argmax(eye))
            start = lo + phase
            start -= ((start - max(int(a) - delay, 0)) // sps) * sps
            stop = min(int(b) + 4 * sps, len(x))
            bits = GFSKDemodulator(sps=sps).demodulate(corrected[start:stop])
            transition_energy = np.diff(frequency) ** 2
            transition_energy -= transition_energy.mean()
            nfft = 8 * 2 ** int(np.ceil(np.log2(len(transition_energy))))
            spectrum = abs(np.fft.rfft(transition_energy * np.hanning(len(transition_energy)),
                                      n=nfft))
            grid = np.fft.rfftfreq(nfft, 1 / sample_hz)
            band = np.flatnonzero((grid >= symbol_hz * 0.9) & (grid <= symbol_hz * 1.1))
            peak = int(band[np.argmax(spectrum[band])])
            burst.update(status="demodulated_unparsed", bits=bits,
                         cfo_hz=cfo, frequency_centers_hz=centers.tolist(),
                         cluster_half_separation_hz=float(np.ptp(centers) / 2),
                         timing_eye=eye, timing_phase=phase,
                         bit0_filtered_index=int(start), bit0_raw_reference=int(start - delay),
                         bit_sample_step=sps,
                         spectral_clock_hz=float(grid[peak]),
                         clock_line_contrast_db=float(20 * np.log10(
                             max(spectrum[peak], 1e-30) / max(np.median(spectrum[band]), 1e-30))),
                         clock_observation_s=len(transition_energy) / sample_hz,
                         clock_fft_grid_hz=sample_hz / nfft,
                         clock_caveat="finite-burst transition-energy peak near request; not calibrated SRO",
                         index_precision="one sample timing grid; FIR delay removed; no air timestamp")
            continue
        burst["status"] = "insufficient_frequency_states"
    return dict(sps=sps, sample_hz=sample_hz, requested_symbol_hz=symbol_hz,
                threshold_codes2=threshold, threshold_rule="4 * quiet 99.9% symbol-power quantile",
                filter_taps=taps, filter_group_delay_samples=delay,
                timing_method="maximum mean absolute integrated discriminator; no payload oracle",
                crc="unknown: factory framing not specified", bursts=bursts)
