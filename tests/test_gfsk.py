import numpy as np
import pytest

from nearlink_sdr.phy.gfsk import (
    GFSK_BW_CONFIG,
    GFSKDemodulator,
    GFSKModulator,
    _gaussian_filter,
)
from nearlink_sdr.phy.rf_compliance import GFSK_FREQ_DEV_SPECS


class TestGFSK:
    def test_modulator_output_length(self):
        mod = GFSKModulator(sps=8)
        bits = np.array([1, 0, 1, 1, 0], dtype=int)
        signal = mod.modulate(bits)
        assert len(signal) == len(bits) * 8

    def test_modulator_output_is_complex(self):
        mod = GFSKModulator(sps=8)
        bits = np.array([0, 1, 0], dtype=int)
        signal = mod.modulate(bits)
        assert np.iscomplexobj(signal)

    def test_modulator_constant_envelope(self):
        """GFSK是恒包络调制，信号幅度应恒为1"""
        mod = GFSKModulator(sps=8)
        bits = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=int)
        signal = mod.modulate(bits)
        np.testing.assert_allclose(np.abs(signal), 1.0, atol=1e-10)

    def test_roundtrip_no_noise(self):
        """无噪声环境下GFSK调制解调的逆运算"""
        mod = GFSKModulator(sps=8, mod_index=0.5)
        demod = GFSKDemodulator(sps=8)
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, 100)
        signal = mod.modulate(bits)
        rx_bits = demod.demodulate(signal)
        # 允许首尾少量比特由于高斯滤波器截断导致误差
        valid = min(len(bits), len(rx_bits))
        margin = 3  # 高斯滤波器拖尾
        errors = np.sum(bits[margin:valid-margin] != rx_bits[margin:valid-margin])
        assert errors == 0, f"无噪声下解调错误 {errors} 比特"

    def test_roundtrip_high_snr(self):
        """高信噪比下GFSK误码率应极低"""
        mod = GFSKModulator(sps=8, mod_index=0.5)
        demod = GFSKDemodulator(sps=8)
        rng = np.random.default_rng(7)
        bits = rng.integers(0, 2, 500)
        signal = mod.modulate(bits)
        # 添加少量噪声 (SNR ~20dB)
        noise_power = 0.01
        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(signal)) +
                                              1j * rng.standard_normal(len(signal)))
        rx_signal = signal + noise
        rx_bits = demod.demodulate(rx_signal)
        valid = min(len(bits), len(rx_bits))
        margin = 3
        ber = np.sum(bits[margin:valid-margin] != rx_bits[margin:valid-margin]) / (valid - 2*margin)
        assert ber < 0.01, f"高信噪比下BER={ber}过高"

    def test_mod_index_range(self):
        """标准要求调制系数在0.45~0.55之间"""
        for h in [0.45, 0.50, 0.55]:
            mod = GFSKModulator(sps=8, mod_index=h)
            bits = np.array([1, 0, 1, 0], dtype=int)
            signal = mod.modulate(bits)
            assert len(signal) == 32

    def test_all_zeros(self):
        mod = GFSKModulator(sps=8)
        demod = GFSKDemodulator(sps=8)
        bits = np.zeros(20, dtype=int)
        signal = mod.modulate(bits)
        rx_bits = demod.demodulate(signal)
        valid = min(len(bits), len(rx_bits))
        margin = 3
        np.testing.assert_array_equal(rx_bits[margin:valid-margin], bits[margin:valid-margin])

    def test_all_ones(self):
        mod = GFSKModulator(sps=8)
        demod = GFSKDemodulator(sps=8)
        bits = np.ones(20, dtype=int)
        signal = mod.modulate(bits)
        rx_bits = demod.demodulate(signal)
        valid = min(len(bits), len(rx_bits))
        margin = 3
        np.testing.assert_array_equal(rx_bits[margin:valid-margin], bits[margin:valid-margin])


class TestGaussianFilter:
    """高斯滤波器带宽定义 (TXS-10002-2025 6.2.1.1: BT 为单边 -3 dB 带宽)。"""

    SPS = 8
    SPAN = 8
    DB3 = -3.010299956639812  # 10 * log10(0.5)

    @classmethod
    def _grid(cls):
        return np.arange(-cls.SPAN * cls.SPS / 2, cls.SPAN * cls.SPS / 2 + 1) / cls.SPS

    @classmethod
    def _response_db(cls, bt, freqs):
        taps = _gaussian_filter(bt, cls.SPAN, cls.SPS)
        t = cls._grid()
        response = np.abs(
            np.sum(taps[None, :] * np.exp(-2j * np.pi * freqs[:, None] * t[None, :]), axis=1)
        )
        return 20.0 * np.log10(response / response[0])

    @classmethod
    def _three_db_bandwidth(cls, bt):
        freqs = np.linspace(0.0, 2.0 * bt, 4001)
        db = cls._response_db(bt, freqs)
        below = np.flatnonzero(db <= cls.DB3)
        assert below.size, f"BT={bt}: 扫描范围内未衰减到 -3 dB"
        i = int(below[0])
        return float(np.interp(cls.DB3, [db[i], db[i - 1]], [freqs[i], freqs[i - 1]]))

    @pytest.mark.parametrize("bt", [0.3, 0.4, 0.5, 0.7])
    def test_three_db_bandwidth_equals_bt(self, bt):
        """单边 -3 dB 带宽必须等于 BT 本身。

        回归: 冲激响应曾多乘一个因子 2, 使实际带宽为标称 BT 的 √2 倍
        (BT=0.5 时实测 0.706/T)。闭环解调与 BER 测试对该偏差不敏感,
        只有频谱测量能发现。
        """
        measured = self._three_db_bandwidth(bt)
        assert measured == pytest.approx(bt, rel=0.01), (
            f"BT={bt}: 实测单边 -3 dB 带宽 {measured:.4f}/T, 应为 {bt}/T"
        )

    @pytest.mark.parametrize("bt", [0.3, 0.5, 0.7])
    def test_impulse_response_matches_closed_form(self, bt):
        """抽头必须等于标准闭式解 exp(-t^2 / (2 sigma^2))。"""
        t = self._grid()
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
        expected = np.exp(-0.5 * (t / sigma) ** 2)
        expected /= expected.sum()
        np.testing.assert_allclose(
            _gaussian_filter(bt, self.SPAN, self.SPS), expected, rtol=0, atol=1e-15
        )

    def test_symmetric_unit_dc_gain(self):
        taps = _gaussian_filter(0.5, 3, 8)
        np.testing.assert_allclose(taps, taps[::-1], atol=1e-15)
        assert taps.sum() == pytest.approx(1.0, abs=1e-12)
        assert np.all(taps > 0)

    def test_waveform_matches_independent_reference(self):
        """整条调制链与按标准公式独立实现的参考逐点一致。"""
        rng = np.random.default_rng(2026)
        bits = rng.integers(0, 2, 512)
        actual = GFSKModulator(sps=8, mod_index=0.5, bt=0.5).modulate(bits)

        sps, bt, mod_index = 8, 0.5, 0.5
        sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
        taps = np.exp(-0.5 * (np.arange(-12, 13) / sps / sigma) ** 2)
        taps /= taps.sum()
        # 用 full 卷积再居中裁剪, 与实现的 mode='same' 结构不同
        frequency = np.convolve(np.repeat(2.0 * bits - 1.0, sps), taps, mode="full")
        start = (len(taps) - 1) // 2
        frequency = frequency[start : start + len(bits) * sps]
        expected = np.exp(1j * np.cumsum(frequency) * (mod_index * np.pi / sps))

        assert len(actual) == len(expected)
        np.testing.assert_allclose(actual, expected, atol=1e-12)


class TestGFSKSpecTables:
    """GFSK 频偏限值必须与标准 §8.2.2.1 一致, 且与调制系数范围自洽。"""

    RATES = (0.1, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0)

    def test_bandwidth_config_matches_freq_dev_specs(self):
        """两张限值表必须给出同一组最小频偏。

        回归: GFSK_BW_CONFIG 曾原样抄录标准 §6.2.1.1 的表格, 而该表对
        0.125/0.25/0.5 Msym/s 的数值与 §8.2.2.1 及本节 h 范围矛盾。
        """
        assert set(GFSK_BW_CONFIG) == set(GFSK_FREQ_DEV_SPECS)
        for rate, spec in GFSK_FREQ_DEV_SPECS.items():
            cfg = GFSK_BW_CONFIG[rate]
            assert cfg["symbol_period_us"] == pytest.approx(spec.symbol_interval_us)
            assert cfg["min_freq_dev_khz"] == pytest.approx(spec.min_deviation_khz)

    @pytest.mark.parametrize("rate", RATES)
    def test_min_deviation_reachable_within_modulation_index(self, rate):
        """最小频偏要求不得高于 h∈[0.45, 0.55] 在该速率下能产生的上限。"""
        spec = GFSK_FREQ_DEV_SPECS[rate]
        symbol_period_s = spec.symbol_interval_us * 1e-6
        max_deviation_khz = 0.55 / (2 * symbol_period_s) / 1e3
        assert spec.min_deviation_khz <= max_deviation_khz

    @pytest.mark.parametrize("rate", RATES)
    def test_fd1_range_contains_nominal_half_deviation(self, rate):
        """fd1 区间必须包含 h=0.5 的标称频偏 h/(2T)。"""
        spec = GFSK_FREQ_DEV_SPECS[rate]
        nominal_khz = 0.5 / (2 * spec.symbol_interval_us * 1e-6) / 1e3
        assert spec.fd1_min_khz < nominal_khz < spec.fd1_max_khz
