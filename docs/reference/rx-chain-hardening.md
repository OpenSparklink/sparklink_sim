# 接收链修复分阶段记录

最终目标：让原始 IQ 经可审查的接收链到达 MAC，校验语义真实、流式状态有界、
软信息和性能收益可复现。失败保留为证据，不预设全通过。

| 阶段 | 子任务及完成标准 | 状态 |
|---|---|---|
| 1 | 真实 A/B 头 CRC、上下文、截断/非法长度、失败停止交付；基线复现及全量回归 | 实现并验证 |
| 2 | 捕获/同步/跟踪/严格帧解析接入 MAC；普通与 WS63 profile 分离，未知起点和长度反例 | 待完成 |
| 3 | 有界增量滤波/插值/定时/帧状态；任意分块、多包、结束截断与整段轨迹对照 | 待完成 |
| 4 | 按实际星座/旋转/导频生成可靠度 LLR；编码开发/保留集与 Polar 对照 | 待完成 |
| 5 | 真实路径 profile、吞吐/延迟/内存；独立记录 Python/Rust，并验证优化等价性 | 待完成 |
| 6 | CI 触发、依赖跳过、加速一致性；剩余低 SNR 的真值诊断及独立重复样本 | 待完成 |

每阶段更新本文及 CHANGELOG，运行针对性反例、ruff、全量测试，再单独提交推送。
只涉及离线模型、测试、文档和 CI；不操作设备、固件、RTL 或 Release。

## 阶段 1：控制 CRC 语义（§6.4）

基线 `e2f7b54` 的 `decode_head()` 对合法头、坏头和错误种子均返回 True。
使用同一个有导频、无编码的合法 8 字节载荷时，坏头和错误种子也会交付完整载荷。
现改为 A 组 CRC12（默认实际同步序列低 12 位，允许显式上下文种子），
B 组 CRC24B（固定 0x555555，异或显式 24 位 LLID）。数据 CRC 种子独立。
B 组缺失 LLID 抛出 ValueError；没有“未校验但成功”的状态。

| 固定种子 63016，数据导频 8 | 基线头/数据 CRC | 修复头/数据 CRC | 修复交付 |
|---|---|---|---|
| 合法头 | True / True | True / True | 64 bit |
| 头信息位损坏，重新 Polar 编码 | True / True | False / False | 空 |
| 用错误头 CRC 种子生成 | True / True | False / False | 空 |

头码长必须准确，短头和多余头位均拒绝；非二进制输入和非法配置报错。
`rx_chain(iq,cfg,n_data_bytes)` 仍是已知配置、理想对齐接口，长度必须为 [0,2047] 整数。
先只解头；坏头不进行载荷解映射、码块分割或译码。成功后才按已知长度处理载荷；
载荷截断或数据 CRC 失败不交付。CRC 失败但可计算的译码比特只在
`diagnostic_data_bits` 中保留，不能当成 MAC 数据。
两种 pipeline 仿真的 BER 按可译码比特统计，另报 erasure_rate；没有可比较比特则为 NaN。

A2 正例此前漏传可选 LQI，发送 32 位却按 40 位译码；现在配置 28 个信息位时生成
8 位 LQI＋20 位 A2。B1 MAC 配置要求 27 信息位及显式 head_llid。
随机“信息＋随机 CRC”的 FT3/4 正例改为随机信息附加真实 CRC，原无效输入另有拒收测试；
期待载荷和标准向量不变。USRPLoopbackSim 默认头也附加真实 CRC。
节点接收使用已知对端地址对应的模型 PID；显式 `receive(...,rx_config=...)` 可用于自身理想环回，
不尝试不同种子或以 CRC 选择上下文。

另一个已有缺陷被保留：数据无导频时，旧理想解析器错误消耗最后一个控制导频，
本次固定输入的合法载荷仍失败，旧接口现拒收，不能记作 CRC 语义修复后的成功。
阶段 2 将统一字段边界，禁止用放宽断言掩盖该问题。

## 复现与证据

```sh
PYTHONPATH=src UV_CACHE_DIR=/tmp/rx-uv uv run --frozen --no-sync python \
  scripts/rx_chain_crc_audit.py --output /tmp/header-after.json
UV_CACHE_DIR=/tmp/rx-uv uv run --frozen --no-sync pytest \
  tests/test_head_crc_semantics.py tests/test_rx_pipeline.py tests/test_ft_pipeline.py -q
```

相同 audit 可用 `git archive e2f7b54` 的源码作为 PYTHONPATH 回放基线；脚本无期待载荷驱动的选择。
Python 3.14.8，NumPy 2.4.3，SciPy 1.17.1，纯 Python Polar，无 Rust 扩展。
冻结种子、逐输入 IQ 哈希、四个接收源文件哈希、before/after JSON、基线 NPZ、全量测试报告保存在
本轮交付目录 `rx-chain-hardening`。本地证据根目录：
`/home/sanchuanhehe/.codex/visualizations/2026/11/15/01a1dfe8-f1ae-7233-a63e-c31ffad2d428/`。

此前低 SNR 边界以 `psk-noise-recovery-results.json` 为准：原 9 个恢复，新增 5 个仍失败。
不能用更换共享数据集的索引来复写冻结输入。
