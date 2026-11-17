# 目标窄带SLE整机资源与并发账本（SYS-A）

整机规划现在与RX算法同时推进；RX验收门前仍不展开大规模TX实现。
当前只完成一个可审查的资源/并发下界闭环，**目标整机资源、截止时间和16MS/s尚未验收**。
18DSP/16KiB是既有捕获模块约束，继续有效，不是芯片总容量或整机预算。

## 器件、基础构建及余量

核对的是MicroPhase ANTSDR原E310工程，不借用Ettus E310或E310v2资源/引脚。
制造商[ANTSDR E310手册](https://antsdr-docs.microphase.cn/en/latest/device_and_usage_manual/ANTSDR_E_Series_Module/ANTSDR_E310_Reference_Manual/AntsdrE310_Reference_Manual.html)
说明Zynq-7020；具体工程和匹配布线报告使用`xc7z020clg400-1`。
速度等级是已核对的工程/报告目标，未在本轮重新检查芯片丝印或设备，不能冒充新板卡身份测量。
[AMD DS190 v1.11.1表1](https://docs.amd.com/api/khub/documents/juMnxca71Tf2gfjmNyjM8A/content?Ft-Calling-App=ft/turnkey-portal&Ft-Calling-App-Version=5.1.65&download=true&locationValue=viewer)
与Vivado报告的物理容量一致。

选择已有限定验收的`rx-observe-20261112`基础构建，按原`source-before-build`和
`20261112-rx-live.json`绑定；不是把当前未构建RTL当作已综合结果。
原始实现JSON SHA为`41e4ae3490118b4eb76555f41cc2a18111d1c6bdf9b85c3d6e2bb34ed8112c8a`，
有效输入SHA为`2d03f4884369716700a505080e503155971e34e4e2413c091b647ac31b27f2ba`，
位流SHA为`f13ef89561893d6e906fd6ef75e7a6222d941e5cb7cb7f52e77a25ffd3527130`。
本轮只读报告，不运行Vivado、不重建或加载位流。

| 资源 | 物理容量 | 该基础构建占用 | 物理剩余 | 候选25%整片预留 | 候选新增上限 |
|---|---:|---:|---:|---:|---:|
| LUT | 53200 | 18135 | 35065 | 13300 | 21765 |
| FF | 106400 | 21711 | 84689 | 26600 | 58089 |
| DSP48E1 | 220 | 29 | 191 | 55 | 136 |
| BRAM36 | 140 | 73 | 67 | 35 | 32 |

25%是**待论证建议，未批准、未冻结为系统验收预算**；不是时序保证。
新增上限尚未扣除目标PSK/Polar、TX、实时MAC、DMA及新缓存，不能视为RX独占资源。
基础构建含ADI接口、FT1/GFSK、Radio/Event控制、CDC/AXI及IQ观测缓存；
冻结源明确未实例化旧DMA/旧PHY/旧TDD/ARQ/AES，不能套用旧AGENTS的总资源表。
73个BRAM包含观测设施，但移除/复用的节省未经新构建和生命周期验证，不先计收益。
每功能LUT/FF归属未有可靠分层报告，不把基础构建总量重复加到每个模块。

该基础工程SYS为100MHz，匹配布线WNS+0.044ns、TNS0、WHS+0.035ns；
这是当时限定FT1构建，不证明新PSK整机能运行200MHz。
更早`radio-ready-final-20261111`报告有WNS−0.414ns，不能因目录名有final便当合格时序。
当前RRC孤立II12需要至少192MHz才能接16MS/s；100/160MHz必要条件失败，200MHz仅孤立条件通过。
完整200MHz新时钟树、跨域、端口及布局布线尚未存在。

## FPGA与ARM分工及范围

机器可读清单为[system-target-ledger.json](system-target-ledger.json)，每功能包含放置、并发、
位宽/端口、II/延迟缺口及LUT/FF/DSP/BRAM区间。无可靠估算的上下界为null，
明确表示未知，不是0或已获资源。新的可用系统预算必须在相应验收前冻结。

| 功能 | 规划位置 | 决策依据及缺口 |
|---|---|---|
| 数字接口、CDC、连续滤波/检测/同步/插值/环路 | FPGA | 逐采样、无法容忍Linux抖动；接口/位宽/流水需闭合 |
| 头部CRC/解析、载荷CRC/加扰、必要BCH/速率匹配 | FPGA期限路径 | 实际profile使用范围逐项审计，不能按模块存在推断整链 |
| Polar译码/编码 | 优先评估FPGA；ARM须先证明最坏服务 | 现有硬件Nmax512/LLR16仅独立参考，目标MCS/codeblock RAM/周期尚未映射 |
| TX组帧、映射、成型、量化、确定输出 | FPGA；ARM提供已准备数据 | 现在预留资源，不提前大规模实现；发包后欠载明确中止，禁止悄悄插零 |
| 精确TDD、跳频触发、即时响应、ARQ执行、时间戳 | FPGA实时MAC | ARM只给策略/上下文；响应间隔、RF转向及完整期限尚未冻结 |
| AES-CCM包处理 | 按实际安全profile有条件配置 | ARM管理密钥/策略；位置按认证/回复最坏截止时间决定，不按平均CPU耗时 |
| MMIO/SPI/DMA/缓冲代次/恢复所有权 | ARM独占内核+FPGA数据/执行 | DMA带宽/QoS、描述符、完成预留必须显式计费 |
| 建链、协商、配对流程、一般管理/业务策略 | ARM | 不承担无法容忍软件抖动的逐包实时动作 |
| 调试/IQ观测/统计 | 有界FPGA+ARM收集，可选 | 单独收费、不可反压实时路径；满时丢诊断并计数 |

首要目标为单射频窄带4Msym/s QPSK/8PSK、16MS/s，FT2/3/4的严格模型覆盖见`rx-scope`。
FT1/GFSK独立验收：既有硬件1M固定profile证据不替代本模型完整16MS/s连续MAC接口。
UWB、多音、未知厂测编码前缀及未定义多连接并发在本版范围外，不虚构支持。

## 最坏并发和共享条件

```{mermaid}
flowchart LR
  ADC[连续ADC与下一包捕获] --> RXT[当前包尾处理]
  PREV[上一包Polar译码] --> AUTH[CRC与可选认证]
  RXT --> AUTH
  AUTH --> RESP[实时回复决策与截止时间]
  PREP[下一包TX准备] --> ENC[编码与可选CCM]
  ENC --> PREFILL[有界TX预填充]
  RESP --> PREFILL
  PREFILL --> DAC[定时首采样与连续DAC]
  DMA[DMA与包缓存仲裁] --> PREV
  DMA --> ENC
  DMA --> PREFILL
```

RF的RX/TX互斥不消除上述处理重叠。上一包译码可能覆盖下一包捕获；RX尾部/认证仍在运行时
必须准备下一包TX；编码、CCM与DMA可能同时请求包RAM。密集包、最长包、持续候选及下游阻塞
不得由空闲平均负载替代。共享候选须计上下文、仲裁、RAM端口复制、清空/重装和切换时间。

已经验证的是算法层的一次RRC输出分送原始/单音支路、银行差分/能量及重叠窗复用。
RX/TX乘法器、Polar编/译码、双向CCM、RAM/DMA的**整机硬件共享尚无已证明收益**。
`calendar_conflicts()`验证显式半开周期区间及资源数，不因TDD自动免除重叠；
测试中的前包译码/后包准备重叠反例是合成压力日历，不冒充实际协议期限或完整数据流模型。

既有实时MAC草案的`response_interval_ticks=null`，完整目标响应期限仍未知。
必须满足`RX尾部+译码/CRC/认证+决策+准备+RF转向+TX释放+余量≤实际profile响应间隔`；
当前不能把未知项填0后宣布回复可行。标准条款、RF转换及端口服务最坏界将由后续SYS-B补齐。

## 物理存储及工作量敏感性

[AMD RAMB36E1](https://docs.amd.com/r/2025.2-English/ug953-vivado-7series-libraries/RAMB36E1?contentId=Nm2aVbtVbH5UQNZ3PFqKqw)
的TDP模式以1/2/4/9/18/36位×对应深度计费；36Kb包含奇偶位，可用宽度依模式而异。
`bram36_layout()`明确一端口读、一端口持续写，多读用副本广播写；不把1R1W算成2R1W。
它是一个具体映射估算，不是所有银行化/LUTRAM/不对称端口方案的最优下界。

| 直接映射候选 | 逻辑内容 | BRAM36 |
|---|---|---:|
| RRC原始128×32bit历史，8读/1写副本 | 4096bit，不含复制 | 8 |
| 四个拥有乘积银行，每个33×69bit，1读/1写 | 每银行2277bit | 8（4×2） |
| 每个384×46bit滤波历史，8读/1写副本 | 17664bit | 16；双路32 |
| 草案五个独立2048字节包槽 | 10KiB | 5 |
| 同五槽合并32位池的容量估算 | 10KiB | 3；仲裁/并发未证，不能先记共享收益 |

仅上述RRC及双滤波历史的直接副本映射便需48BRAM，超过25%预留候选下的32个新增额度，
还未含其余功能。这拒绝的是该直接映射与该候选预留的组合，不等于芯片不可能容纳目标系统。
合理下一步是证明银行化/时间复用/LUTRAM和生命周期，或有证据地调整系统架构；
不能把16KiB逻辑状态直接除36Kb当物理BRAM资源。

32位复IQ在16MS/s每方向最低64MB/s；同时捕获与输出准备可要求128MB/s原始搬运，
尚未含描述符、对齐、重复读写、FFT/译码和DDR竞争。现有基础构建没有目标DMA，
这只是带宽需求，不是DDR服务能力证明。长下游停顿必须有限准入/丢弃，ADC不能暂停。
规定业务负载需保证不丢；持续干扰过载允许有界运行并明确丢弃，算法漏检和调度丢弃分别报告。

当前全相位细搜索仍是最有据可查的计算瓶颈，见[模块账本](rx-resource-budget.md)。
整机物理DSP余量不自动放宽18DSP模块约束。下一算法实验先只改载波幅度权重，
既有FFT/相位/解码不变，并把新增状态/乘法/端口纳入本清单；随后才比较有限候选及相干确认。

## 复现及后续验收

```sh
PYTHONPATH=src python scripts/model_system_resources.py \
  --baseline-index /home/sanchuanhehe/.codex/visualizations/2026/11/15/01a1dfe8-f1ae-7233-a63e-c31ffad2d428/system-feasibility/baseline-extended-source-index.json \
  --output /tmp/sle-system-resources
```

输出目录须不存在，预期退出1（系统验收不完整），不是将规划完成冒充整机通过。
输出包含物理/基础/候选余量、位宽/端口分配、100/160/200MHz孤立敏感性、完整功能缺口及源码SHA。
证据根`system-feasibility`保存原件索引/哈希和执行快照；基础报告是实际旧构建，
新增算法只有估算/孤立周期证据，没有目标整机的新综合、布线或设备证据。

SYS-B继续收敛实际并发/截止时间、译码/加密/编码服务及存储端口；
SYS-C将RX、译码、TX准备/输出、DMA/控制接入有界周期模型。新系统预算在对应验收前定义冻结。
RX数值2.348%>2%、local4起点失败、全部低SNR/CRC/截断及资源失败保留，不因规划扩展而取消。

```{toctree}
:hidden:

/apidocs/nearlink_sdr/nearlink_sdr.sim.system_resources
```
