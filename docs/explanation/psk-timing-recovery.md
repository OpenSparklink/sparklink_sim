# PSK 包内采样时钟恢复

本实现针对 TXS-10002-2025 §6.2.1.2 的 beta=0.4 PSK 波形，在模型的
`acquire_psk_capture` 已知同步入口提供可选的持续定时跟踪。
这是一种接收机实现选择，不是标准指定算法或射频一致性结论。

## 捕获与跟踪

原有整数/分数网格仍仅按已知同步的归一差分相关选择包起点。
`timing_tracking=True` 时，保持整个同步窗口的采样坐标不动，避免改变
头部载波估计，然后对后续每个符号更新时钟相位及采样间隔。
控制头、数据和导频沿同一符号序号推进，不重置交替旋转的奇偶计数。

跟踪误差为 `Re[(previous-current)*conj(midpoint)] / running_power`，
不依赖调制标签、期待载荷、CRC 或已知数据判决。二阶 PI 环路的比例项调相位，
积分项调每符号采样间隔。三次样条的系数只计算一次，反馈中用标量 Horner 求值，
避免逐符号反复调用 SciPy。平均功率归一化使用 64 符号时间常数。

开发参考的 QPSK/8PSK S 曲线在 ±0.25 采样处约为 ∓0.075，
因此检测增益固定为约1.2/符号。开发集仅比较归一带宽
0.0015/0.003/0.006，以静态、±100 ppm 的真值误差平衡选择0.003，
未以CRC或载荷选择相位、符号或环路输出。

## 使用与诊断

```python
from nearlink_sdr.phy.psk import acquire_psk_capture
from nearlink_sdr.phy.timing import PSKTimingConfig

result = acquire_psk_capture(
    iq, quiet, sample_hz, symbol_hz, sync_symbols,
    timing_subdivisions=8,
    timing_tracking=True,
    timing_config=PSKTimingConfig(bandwidth=0.003, max_rate_ppm=500),
)
```

默认 `timing_tracking=False` 保留原固定相位抽样。跟踪打开时，
每个成功获取的能量段新增与 `symbols` 等长的 NumPy 数组：

- `symbol_centers_raw`：原始输入文件中的实际符号中心坐标。
- `timing_rate_ppm`：PI积分器的速率状态，正值表示每符号包含更多输入样本。
- `timing_detector_error`：归一化并限制到[-2,2]的Gardner误差。

`symbol_sample_step=None` 明确表示变间隔，不能再用起点加固定SPS换算坐标。
`timing_loop` 包含增益、带宽、末128符号速率中位数及速率/误差限幅次数。
需要保存JSON时，应先将上述数组写入独立NPZ，再保存元数据。

积分速率默认限制为±500 ppm，误差限幅用于避免异常单点导致非单调采样。
这些是模型保护参数，不是标准合格限值。限幅不是锁定判据，
`lock_status=not_classified`；不能把速率估计或TED残差当成真值误差。
精确误差、稳定时间及半符号跨越只能在独立真值仿真中评估。

## 适用边界

环路在每包重置。当前居中RRC和样条预计算仍是离线处理，尚不是分块流式接收器，
不构成实时FPGA实现。短包可能结束于速率尚未稳定之前。
能量分段、载波细化、CRC、长度规则保持不变；噪声造成的能量段切碎
仍可能导致截断，不通过扩大包边界掩盖这些失败。
同步窗内SRO保持未校正，长同步或更大偏差需要单独验证。

参考算法结构见 [MathWorks Symbol Synchronizer](https://www.mathworks.com/help/comm/ref/symbolsynchronizer.html)。
独立发送波形与验证约束见 [离线边界实验](../how-to/rx-timing-audit.md)。

## 噪声下的能量分段

对于已验证的4Msymbol/s厂测profile，可额外指定 `energy_window_symbols=8`，
先按8个符号长度平均功率，再从同样处理的静默估计0.999分位数并乘4作门限。
同步门限0.85、CRC、长度和内容规则保持不变。
这可以抑制PSK包络凹陷和噪声导致的短暂门限穿越，不通过期待长度补齐输入。

默认值1保留旧分段。8符号窗口会降低时间分辨率，可能合并间隔很短的能量段；
不能未经验证推广到任意帧间隔、突发时序或接近检测门限的弱信号。
返回 `energy_window_symbols` 与 `energy_window_samples` 说明实际窗口。
原始IQ和文件边界保持不变，仍会保留真实录制截断。
