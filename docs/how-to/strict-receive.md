# 从未知起点的 PSK IQ 接收 MAC 载荷

使用严格接收入口时，提供已知的链路配置和独立静默校准 IQ，**不提供期待载荷或数据长度**。
普通 FT2 的 A2/A3 没有数据 MCS 字段，MCS 来自协商配置；FT3/4 的 B1 从已校验头部读取 MCS。
头 CRC、合法长度、支持的 MCS 和数据 CRC 均通过后才进入 MAC `packets`。

```python
from nearlink_sdr.phy.mac_interface import iq_to_mac_capture
from nearlink_sdr.phy.strict_rx import PSKReceiveProfile
from nearlink_sdr.phy.tx_pipeline import TxConfig

cfg = TxConfig(frame_type=2, mcs_index=8, pid=0x879012,
               ctrl_bits_len=28, whitening_seed=0x52, crc_seed=0x123456,
               pilot_interval=8, sps=4, symbol_rate_mhz=4)
profile = PSKReceiveProfile.ordinary(cfg)  # A2: 8 bit LQI + 20 bit information
received = iq_to_mac_capture(iq, quiet_iq, profile,
                              timing_subdivisions=8, timing_tracking=True,
                              energy_window_symbols=8)
for packet in received.packets:
    consume(packet.mac_payload)
for candidate in received.diagnostics["frames"]:
    record(candidate["status"], candidate["header_crc"], candidate["data_crc"])
```

上例 `iq`、`quiet_iq` 是未按包边界裁切的一维有限复数数组。采样率须匹配配置：
`sps * symbol_rate_mhz * 1e6`；当前捕获前端要求整数 SPS≥4。
`energy_window_symbols` 是显式算法参数，不能把某一厂测环境的选择当作普遍最佳值。
所有能量段（含短段、未同步）保存在 `diagnostics["acquisition"]["bursts"]`；
完成同步的帧即使 CRC 失败也保留在 `frames`，而不是从报告中删除。

对普通 A3，显式传入 `control_format="A3"` 并设置与有无 LQI 一致的 28/36 信息位。
对 B1，配置 `frame_type=3/4`、`ctrl_bits_len=27`、显式 24 位 `head_llid`；
缺少 LLID 报错，不猜测上下文。

WS63 厂测应单独构造 profile，下面三个值必须来自已确认的记录：

```python
profile = PSKReceiveProfile.ws63_ft2_a3(
    sync_word=confirmed_sync_word,
    data_crc_seed=confirmed_data_crc_seed,
    pilot_interval=confirmed_pilot_interval,
    sps=4, symbol_rate_mhz=4,
)
```

这个 profile 显式约定无加扰、CRC24A、A3＋LQI、LQI 低四位作为厂测 MCS、载荷 LSB-first。
普通路径的载荷字节是 MSB-first，不把厂测常量设为通用默认值。
厂测只支持已验证的 MCS6/8/10/12；未知编码前缀继续未知，不剥除后重新定义成功。
已有 `decode_ft2_a3_capture` 兼容入口调用同一个严格核心，不再保留另一套解析实现。

`receive_capture` 位于 `phy.rx_pipeline`，提供相同的 PHY 结果；
旧 `rx_chain(iq,cfg,n_data_bytes)` 和 `iq_to_mac(iq,cfg,n_mac_bytes)` 保留给理想对齐仿真，
它们现在也执行真实头 CRC，但仍要求已知长度和起点，不能拿来证明捕获能力。

## 当前限制

这是整段离线接口，居中 RRC 和插值使用未来样本，每个捕获到的包重置定时状态。
目前每个能量段选择一个同步峰；没有静默隔开的多包可能漏掉其他同步，后续流式阶段必须覆盖。
可分离能量段的多包、坏头后下一包、静默和输入结束截断已有测试。
该接口尚不是有界流式接收，不缓存整个文件的流式能力也尚未完成。
严格 PSK profile 支持 FT2/3/4；FT1/GFSK 仍使用独立获取/解析组件和理想接口，未接入本入口。
离线结果不代表实时 FPGA 或完整 SLE 验收。

## 复核

```sh
PYTHONPATH=src UV_CACHE_DIR=/tmp/rx-uv uv run --frozen --no-sync python \
  scripts/rx_strict_integration_audit.py --output /tmp/strict-integration-new
UV_CACHE_DIR=/tmp/rx-uv uv run --frozen --no-sync pytest tests/test_strict_receive.py -q
```

audit 使用独立角度映射及连续时间脉冲参考（不使用接收插值），在执行前冻结输入/源码哈希。
需要仓库中的测试参考文件和开发依赖。失败 IQ、参数、采样轨迹、旧/新入口结果全部落盘。
真实 IQ 的冻结回放命令及结果见[接收链修复记录](../reference/rx-chain-hardening.md)。
