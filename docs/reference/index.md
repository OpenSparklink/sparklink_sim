# 技术参考

本节列出项目全部公开模块的接口文档, 内容从源代码注释自动生成。

## 通用模块 (`common`)

| 模块 | 功能 |
|------|------|
| {py:mod}`nearlink_sdr.common.crc` | CRC 校验 |
| {py:mod}`nearlink_sdr.common.m_sequence` | m 序列生成 |
| {py:mod}`nearlink_sdr.common.bch` | BCH 编码 |
| {py:mod}`nearlink_sdr.common.polar` | Polar 编解码 |
| {py:mod}`nearlink_sdr.common.code_block_seg` | 码块分割 |
| {py:mod}`nearlink_sdr.common.scrambler` | 信道比特加扰 |
| {py:mod}`nearlink_sdr.common.mcs` | MCS 与速率匹配 |
| {py:mod}`nearlink_sdr.common.prbs` | PRBS 伪随机序列 |

## 物理层模块 (`phy`)

| 模块 | 功能 |
|------|------|
| {py:mod}`nearlink_sdr.phy.gfsk` | GFSK 调制/解调 |
| {py:mod}`nearlink_sdr.phy.psk` | PSK 调制/解调 |
| {py:mod}`nearlink_sdr.phy.preamble` | 前导码 |
| {py:mod}`nearlink_sdr.phy.sync_sequence` | 同步信号 |
| {py:mod}`nearlink_sdr.phy.pilot` | 导频 |
| {py:mod}`nearlink_sdr.phy.frame` | 帧结构 |
| {py:mod}`nearlink_sdr.phy.control_info` | 控制信息 |
| {py:mod}`nearlink_sdr.phy.tx_pipeline` | TX 发射流水线 |
| {py:mod}`nearlink_sdr.phy.rx_pipeline` | RX 接收流水线 |
| {py:mod}`nearlink_sdr.phy.channel` | 信道模型 |
| {py:mod}`nearlink_sdr.phy.equalizer` | 均衡器 |
| {py:mod}`nearlink_sdr.phy.freq_hopping` | 跳频 |
| {py:mod}`nearlink_sdr.phy.mac_interface` | MAC-PHY 适配层 |
| {py:mod}`nearlink_sdr.phy.measurement` | 位置信息测量信号 |
| {py:mod}`nearlink_sdr.phy.measurement_frame` | 测量帧 |
| {py:mod}`nearlink_sdr.phy.measurement_tx` | 窄带/UWB 测量链路参数与调度 |
| {py:mod}`nearlink_sdr.phy.uwb_pulse` | UWB 脉冲波形与芯片调制 |
| {py:mod}`nearlink_sdr.phy.multitone` | 多音信号生成 |
| {py:mod}`nearlink_sdr.phy.rf_compliance` | 射频合规参数校验 |
| {py:mod}`nearlink_sdr.phy.data_link` | 异步/同步数据链路传输规程 |
| {py:mod}`nearlink_sdr.phy.usrp` | USRP 接口 |

## MAC 层模块 (`mac`)

| 模块 | 功能 |
|------|------|
| {py:mod}`nearlink_sdr.mac.frame` | 控制面/数据面/复用帧 |
| {py:mod}`nearlink_sdr.mac.broadcast` | 广播帧 |
| {py:mod}`nearlink_sdr.mac.signaling` | 信令注册与编解码 |
| {py:mod}`nearlink_sdr.mac.link_control` | 链路控制信令 |
| {py:mod}`nearlink_sdr.mac.power_control` | 功率控制 |
| {py:mod}`nearlink_sdr.mac.link_manager` | 链路管理状态机 |
| {py:mod}`nearlink_sdr.mac.access` | 接入流程 |
| {py:mod}`nearlink_sdr.mac.scheduler` | 时序调度器 |
| {py:mod}`nearlink_sdr.mac.security` | 配对信令 |
| {py:mod}`nearlink_sdr.mac.crypto` | 加密与密钥派生 |
| {py:mod}`nearlink_sdr.mac.security_manager` | 安全流程集成 |
| {py:mod}`nearlink_sdr.mac.qos` | QoS 服务质量管理 |
| {py:mod}`nearlink_sdr.mac.smf` | 系统管理帧编解码 |
| {py:mod}`nearlink_sdr.mac.smf_scheduler` | SMF 发送调度 |
| {py:mod}`nearlink_sdr.mac.uwb_measurement_security` | UWB 脉冲测量安全 |

## 仿真模块 (`sim`)

| 模块 | 功能 |
|------|------|
| {py:mod}`nearlink_sdr.sim.link_sim` | 链路仿真 (Phase 1-15) |
| {py:mod}`nearlink_sdr.sim.usrp_sim` | USRP 环回仿真引擎 |

## 节点实体

| 模块 | 功能 |
|------|------|
| {py:mod}`nearlink_sdr.node` | SLE 节点实体 (统一收发接口) |

## 接收链验证

[分阶段接收链修复与证据](rx-chain-hardening.md)记录接口语义、失败及复现方法。

## 完整 API 文档

```{toctree}
:maxdepth: 2

rx-chain-hardening
ci-test-partitions
rx-capture-front
rx-acquisition-hardware
rx-interference-capture
sle-transceiver-model
rx-scope
rx-rrc-model
rx-rrc-stream-results
rx-resource-budget
system-feasibility
carrier-bounded-weights
carrier-prefix
carrier-loop
carrier-fixed
carrier-parallel
carrier-bounded-results
bounded-sync-candidates
sync-coherent-confirmation
rx-failure-controls
sync-carrier-initialization
capture-transport
capture-fine-integration
spline-power
spline-geometry
score-word-arithmetic
capture-tail
selected-frame
sliding-history
interpolation-word
initial-interpolant-reuse
alignment-guard-order
known-sync-word-fit
selected-symbol-handoff
finite-timing-words
timing-service-handoff
known-sync-pipeline
finite-normalization-words
normalization-overlap
timing-divider-prefix
control-polar-word
b1-length-contract
head-length-words
head-word-gate
hard-header-words
live-head-replay
timing-radix4
data-joint-service
power-ted-overlap
retained-raw-stencil
feedback-error-age
feedback-age-sampling
split-timing-owners
split-sampling-integration
packed-head-retention
initialized-frame-input
coverage-isolation
rrc-memory
spline-preparation
spline-fixed
spline-flow
spline-score
standard-sources
psk-timing-tracking-results
psk-noise-recovery-results
/apidocs/nearlink_sdr/nearlink_sdr
/apidocs/nearlink_sdr/nearlink_sdr.phy.control_length
/apidocs/nearlink_sdr/nearlink_sdr.sim.header_geometry_flow
/apidocs/nearlink_sdr/nearlink_sdr.sim.header_word_flow
/apidocs/nearlink_sdr/nearlink_sdr.phy.psk_hard_word
/apidocs/nearlink_sdr/nearlink_sdr.sim.psk_hard_flow
/apidocs/nearlink_sdr/nearlink_sdr.phy.word_head_rx
/apidocs/nearlink_sdr/nearlink_sdr.sim.head_word_replay
```
