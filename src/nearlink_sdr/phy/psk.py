
__all__ = [
    "BPSK_MAP",
    "BPSK_NOROT_MAP",
    "BPSK_ROTATION",
    "QPSK_MAP",
    "QPSK_ROTATION",
    "SLE_RRC_BETA",
    "PSKDemodulator",
    "PSKModulator",
    "acquire_psk_capture",
    "refine_psk_carrier",
    "rrc_filter",
]


from functools import lru_cache

import numpy as np

from nearlink_sdr.phy.timing import PSKTimingConfig, track_psk_timing

# TXS-10002-2025 6.2.1.2 PSK调制
# 滚降系数 beta = 0.4

SLE_RRC_BETA = 0.4


@lru_cache(maxsize=8)
def rrc_filter(beta: float, sps: int, span: int = 10) -> np.ndarray:
    """生成平方根升余弦脉冲成型滤波器。

    :param beta: 滚降系数
    :param sps: 每符号采样数
    :param span: 单侧符号跨度

    :returns: 滤波器系数, 归一化使得匹配滤波后采样点增益为1
    """
    N = 2 * span * sps + 1
    t = np.arange(-(N - 1) / 2, (N - 1) / 2 + 1) / sps

    h = np.zeros(len(t))
    for i, ti in enumerate(t):
        if abs(ti) < 1e-10:
            h[i] = 1.0 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den

    # 归一化
    h = h / np.sqrt(np.sum(h ** 2))
    return h


# 标准中的PSK星座映射表

# BPSK: 0->90°, 1->-90°, 偶数符号额外顺时针旋转90°
BPSK_MAP = {
    0: np.exp(1j * np.pi / 2),
    1: np.exp(-1j * np.pi / 2),
}
BPSK_ROTATION = np.exp(-1j * np.pi / 2)  # 偶数位顺时针90°

# QPSK: 00->45°, 01->135°, 11->-135°, 10->-45°
QPSK_MAP = {
    0b00: np.exp(1j * np.pi / 4),
    0b01: np.exp(1j * 3 * np.pi / 4),
    0b11: np.exp(-1j * 3 * np.pi / 4),
    0b10: np.exp(-1j * np.pi / 4),
}
QPSK_ROTATION = np.exp(-1j * np.pi / 4)  # 偶数位顺时针45°

# 8PSK: 000->22.5°, 001->67.5°, 011->112.5°, 010->157.5°,
#        110->-157.5°, 111->-112.5°, 101->-67.5°, 100->-22.5°
_8PSK_MAP = {
    0b000: np.exp(1j * np.deg2rad(22.5)),
    0b001: np.exp(1j * np.deg2rad(67.5)),
    0b011: np.exp(1j * np.deg2rad(112.5)),
    0b010: np.exp(1j * np.deg2rad(157.5)),
    0b110: np.exp(-1j * np.deg2rad(157.5)),
    0b111: np.exp(-1j * np.deg2rad(112.5)),
    0b101: np.exp(-1j * np.deg2rad(67.5)),
    0b100: np.exp(-1j * np.deg2rad(22.5)),
}
_8PSK_ROTATION = np.exp(-1j * np.deg2rad(22.5))  # 偶数位顺时针22.5°

# 无相位旋转BPSK: 0->0°, 1->180°
BPSK_NOROT_MAP = {
    0: np.exp(1j * 0),
    1: np.exp(1j * np.pi),
}


class PSKModulator:
    """TXS-10002-2025 6.2.1.2 PSK调制器。

    支持 BPSK, QPSK, 8PSK 及无相位旋转BPSK。
    """

    def __init__(self, mod_type: str = "QPSK", sps: int = 4,
                 beta: float = SLE_RRC_BETA, rrc_span: int = 10):
        self.mod_type = mod_type.upper()
        self.sps = sps
        self.beta = beta
        self.rrc_span = rrc_span
        self._rrc = rrc_filter(beta, sps, rrc_span)

        if self.mod_type == "BPSK":
            self._map = BPSK_MAP
            self._bits_per_symbol = 1
            self._rotation = BPSK_ROTATION
        elif self.mod_type == "QPSK":
            self._map = QPSK_MAP
            self._bits_per_symbol = 2
            self._rotation = QPSK_ROTATION
        elif self.mod_type == "8PSK":
            self._map = _8PSK_MAP
            self._bits_per_symbol = 3
            self._rotation = _8PSK_ROTATION
        elif self.mod_type == "BPSK_NOROT":
            self._map = BPSK_NOROT_MAP
            self._bits_per_symbol = 1
            self._rotation = None
        else:
            raise ValueError(f"不支持的调制方式: {mod_type}")

    @property
    def bits_per_symbol(self) -> int:
        return self._bits_per_symbol

    def map_symbols(self, bits: np.ndarray, start_symbol_index: int = 0) -> np.ndarray:
        """比特到星座符号的映射（含相位旋转）。"""
        bps = self._bits_per_symbol
        n_pad = (-len(bits)) % bps
        if n_pad:
            bits = np.concatenate([bits, np.zeros(n_pad, dtype=int)])

        n_symbols = len(bits) // bps
        symbols = np.zeros(n_symbols, dtype=complex)

        for i in range(n_symbols):
            idx = 0
            for b in range(bps):
                idx = (idx << 1) | int(bits[i * bps + b])
            symbols[i] = self._map[idx]
            # 偶数符号位额外旋转（标准中第一个符号处于奇数位，索引0为奇数位）
            if self._rotation is not None and ((i + start_symbol_index) % 2 == 1):
                symbols[i] *= self._rotation

        return symbols

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """PSK调制并进行RRC脉冲成型。

        :returns: 复基带IQ信号
        """
        symbols = self.map_symbols(bits)

        # 上采样
        upsampled = np.zeros(len(symbols) * self.sps, dtype=complex)
        upsampled[::self.sps] = symbols

        # RRC脉冲成型
        signal = np.convolve(upsampled, self._rrc, mode='same')
        return signal


class PSKDemodulator:
    """PSK解调器（匹配滤波+最近邻判决）。"""

    def __init__(self, mod_type: str = "QPSK", sps: int = 4,
                 beta: float = SLE_RRC_BETA, rrc_span: int = 10):
        self.mod_type = mod_type.upper()
        self.sps = sps
        self._rrc = rrc_filter(beta, sps, rrc_span)

        if self.mod_type == "BPSK":
            self._map = BPSK_MAP
            self._bits_per_symbol = 1
            self._rotation = BPSK_ROTATION
        elif self.mod_type == "QPSK":
            self._map = QPSK_MAP
            self._bits_per_symbol = 2
            self._rotation = QPSK_ROTATION
        elif self.mod_type == "8PSK":
            self._map = _8PSK_MAP
            self._bits_per_symbol = 3
            self._rotation = _8PSK_ROTATION
        elif self.mod_type == "BPSK_NOROT":
            self._map = BPSK_NOROT_MAP
            self._bits_per_symbol = 1
            self._rotation = None
        else:
            raise ValueError(f"不支持的调制方式: {mod_type}")

        # 构建星座参考
        self._constellation_keys = sorted(self._map.keys())
        self._constellation_vals = np.array([self._map[k] for k in self._constellation_keys])

    def demodulate(self, signal: np.ndarray) -> np.ndarray:
        """PSK解调。

        :param signal: 复基带IQ信号

        :returns: 解调后的比特序列
        """
        # 匹配滤波
        filtered = np.convolve(signal, self._rrc, mode='same')

        # 降采样（取峰值点）
        symbols = filtered[::self.sps]

        return self.demap_symbols(symbols)

    def demap_symbols(self, symbols: np.ndarray, start_symbol_index: int = 0) -> np.ndarray:
        """判决已滤波符号；奇偶参考帧起点，不能在字段/导频处重新计数。"""
        # 反旋转 + 最近邻判决
        bps = self._bits_per_symbol
        bits = []
        for i, sym in enumerate(symbols):
            # 反相位旋转
            if self._rotation is not None and ((i + start_symbol_index) % 2 == 1):
                sym = sym * np.conj(self._rotation)

            # 最近邻判决
            distances = np.abs(sym - self._constellation_vals)
            idx = self._constellation_keys[np.argmin(distances)]

            # 解码比特
            bits.extend((idx >> b) & 1 for b in range(bps - 1, -1, -1))

        return np.array(bits, dtype=int)


def acquire_psk_capture(signal: np.ndarray, quiet: np.ndarray, sample_hz: float,
                        symbol_hz: float, sync_symbols: np.ndarray, *,
                        timing_subdivisions: int = 1, timing_tracking: bool = False,
                        timing_config: PSKTimingConfig | None = None) -> dict:
    """从完整IQ能量段按已知同步获取符号；不使用期待载荷或CRC选择同步。

    beta=.4/10-symbol RRC复用原滤波器；居中离线滤波使用未来样本，非硬件时间。
    每SPS相位的差分归一相关消除常量相位/CFO；选最大相关，固定门限.85。
    由同步估计CFO/相位/增益，保留短段、边界段及未同步段；返回的symbols未通过CRC。
    timing_tracking=True时同步之后启用NDA Gardner/PI持续跟踪；实际采样坐标单独返回。
    """
    from scipy.interpolate import CubicSpline
    from scipy.signal import correlate, fftconvolve

    if timing_config is None:
        timing_config = PSKTimingConfig()
    if not isinstance(timing_tracking, bool) or not isinstance(timing_config, PSKTimingConfig):
        raise ValueError("Need boolean timing_tracking and PSKTimingConfig")
    x, q, ref = np.asarray(signal), np.asarray(quiet), np.asarray(sync_symbols)
    if (any(a.ndim != 1 or not np.isfinite(a).all() for a in (x, q, ref))
            or min(len(x), len(q)) < 65 or len(ref) < 16
            or not np.isfinite([sample_hz, symbol_hz]).all()
            or min(sample_hz, symbol_hz) <= 0):
        raise ValueError("Expected finite 1D IQ, quiet, known sync and positive rates")
    ratio = sample_hz / symbol_hz
    sps = round(ratio)
    if sps < 4 or abs(ratio - sps) > 1e-9:
        raise ValueError("PSK acquisition requires integer SPS >=4")
    if (not isinstance(timing_subdivisions, int) or isinstance(timing_subdivisions, bool)
            or not 1 <= timing_subdivisions <= 32):
        raise ValueError("timing_subdivisions must be an integer in1..32")
    box = np.ones(sps) / sps
    threshold = 4 * float(np.quantile(np.convolve(abs(q) ** 2, box, "same"), .999))
    energy = np.convolve(abs(x) ** 2, box, "same") > threshold
    starts = np.flatnonzero(np.diff(energy.astype(int), prepend=0) == 1)
    ends = np.flatnonzero(np.diff(energy.astype(int), append=0) == -1) + 1
    dr = ref[1:] * ref[:-1].conj()
    bursts = []
    for a, b in zip(starts, ends, strict=True):
        row = dict(start_raw=int(a), end_raw_exclusive=int(b),
                   edge_truncated=bool(a == 0 or b == len(x)), status="short_energy_segment")
        bursts.append(row)
        if b - a < 64 * sps:
            continue
        lo, hi = max(0, int(a) - 32 * sps), min(len(x), int(b) + 32 * sps)
        z = fftconvolve(x[lo:hi], rrc_filter(.4, sps), mode="same")
        interpolator = CubicSpline(np.arange(len(z)), z) if timing_subdivisions > 1 else None

        def sample(phase):
            if interpolator is None:
                return z[int(phase)::sps]
            return interpolator(np.arange(phase, len(z) - 1, sps))

        best = None
        for tick in range(sps * timing_subdivisions):
            phase = tick / timing_subdivisions if interpolator is not None else tick
            y = sample(phase)
            dy = y[1:] * y[:-1].conj()
            corr = correlate(dy, dr, mode="valid")
            norm = np.sqrt(np.convolve(abs(dy) ** 2, np.ones(len(dr)), "valid")
                           * np.sum(abs(dr) ** 2))
            scores = abs(corr) / np.maximum(norm, 1e-30)
            offset = int(scores.argmax())
            score = float(scores[offset])
            if best is None or score > best[0]:
                best = score, phase, offset, float(np.angle(corr[offset]))
        score, phase, offset, delta = best
        row.update(status="no_sync", sync_score=score)
        if score < .85:
            continue
        raw_start = lo + phase + offset * sps
        # Return only symbols whose center is present in the original recording.
        stop = min(int(b) + 4 * sps, len(x))
        if timing_tracking:
            tracked = track_psk_timing(z, sps, raw_start - lo, min(stop - lo, len(z) - 1),
                                      hold_symbols=len(ref), config=timing_config)
            y = tracked["symbols"]
            row.update(symbol_centers_raw=tracked["centers"] + lo,
                       timing_rate_ppm=tracked["rate_ppm"],
                       timing_detector_error=tracked["detector_error"],
                       timing_loop=tracked["diagnostics"])
        else:
            y = sample(phase)[offset:]
            centers = raw_start + np.arange(len(y)) * sps
            y = y[centers < stop]
        t = np.arange(len(y))
        v = y * np.exp(-1j * delta * t)
        slope, intercept = np.polyfit(np.arange(len(ref)),
                                      np.unwrap(np.angle(v[:len(ref)] * ref.conj())), 1)
        v *= np.exp(-1j * (intercept + slope * t))
        gain = float(np.mean(abs(v[:len(ref)])))
        v /= max(gain, 1e-30)
        row.update(status="synchronized_unparsed", symbols=v,
                   cfo_hz=(delta + slope) * symbol_hz / (2 * np.pi),
                   gain_codes=gain, timing_phase=phase, sync_start_raw=raw_start,
                   sync_end_raw_exclusive=raw_start + len(ref) * sps,
                   symbol_sample_step=None if timing_tracking else sps,
                   sync_evm=float(np.sqrt(np.mean(abs(v[:len(ref)] - ref) ** 2))))
    return dict(sample_hz=sample_hz, requested_symbol_hz=symbol_hz, sps=sps,
                threshold_codes2=threshold, sync_threshold=.85, beta=.4, rrc_span=10,
                timing_method="maximum normalized differential known-sync correlation; no payload",
                timing_subdivisions=timing_subdivisions, timing_tracking=timing_tracking,
                interpolation="none" if timing_subdivisions == 1 else "cubic after RRC; same ADC samples/rate",
                index_semantics="original file symbol-center sample; centered offline RRC; no hardware timestamp",
                bursts=bursts)


def refine_psk_carrier(symbols: np.ndarray, mod_type: str,
                       start_symbol_index: int) -> tuple[np.ndarray, dict]:
    """按星座M次方盲细化残余载波；不使用期待载荷、CRC或导频内容。

    先去帧参考的奇偶旋转；FFT选择归一频率±.1内的峰，再对残余M次方相位线性拟合。
    需要前置已知同步消除星座相位歧义；不搜索π/M旋转以迎合CRC。
    """
    y = np.asarray(symbols)
    if y.ndim != 1 or len(y) < 32 or not np.isfinite(y).all():
        raise ValueError("Need at least32 finite synchronized symbols")
    demod = PSKDemodulator(mod_type, sps=1)
    m = 2 ** demod._bits_per_symbol
    if demod._rotation is None:
        raise ValueError("Only alternating-rotation PSK is supported")
    i = np.arange(len(y))
    base = y.copy()
    base[(i + start_symbol_index) % 2 == 1] *= np.conj(demod._rotation)
    power = -(base / np.maximum(abs(base), 1e-12)) ** m * abs(base) ** 2
    nfft = 16 * 2 ** int(np.ceil(np.log2(len(y))))
    grid = np.fft.fftfreq(nfft)
    spectrum = abs(np.fft.fft(power * np.hanning(len(power)), n=nfft))
    band = np.flatnonzero(abs(grid) < .1)
    freq = float(grid[band[np.argmax(spectrum[band])]])
    slope, phase = np.polyfit(i, np.unwrap(np.angle(power * np.exp(-2j * np.pi * freq * i))), 1)
    phase = (phase + np.pi) % (2 * np.pi) - np.pi
    corrected = y * np.exp(-1j * (2 * np.pi * freq * i + slope * i + phase) / m)
    return corrected, dict(residual_cycles_per_symbol=(freq + slope / (2 * np.pi)) / m,
                           residual_phase_radians=phase / m,
                           method="blind de-rotated Mth-power FFT+phase regression; no expected bits")
