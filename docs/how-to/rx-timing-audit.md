# 离线接收定时边界验证

本实验对照 TXS-10002-2025 §6.2.1.2、§6.3.3、§6.4 的模型，
使用经真实 IQ 交叉确认的 WS63 厂测 FT2/A3 profile，不能推广为其他链路配置。
主接收机保持基线实现。本工具不控制设备、固件、RTL或发布。

在独立 worktree 根目录执行，输出目录必须不存在：

```sh
PYTHONPATH="$PWD/src" MPLCONFIGDIR=/tmp/rx-mpl \
/home/sanchuanhehe/Documents/opensparklink/sparklink-sim/.venv/bin/python \
 scripts/rx_timing_audit.py \
 --dataset /home/sanchuanhehe/Documents/opensparklink/sparklink-hw/datasets/sle-rf/ws63/20261115-rx2 \
 --output output/rx-timing-new --stage full
```

首次可用 `--stage minimal` 先做静态偏移、原始 IQ 网格对照和静默检查。
退出码 1 表示存在实验失败（包括截断），2 或 traceback 表示执行出错；
不要以退出码 0 为调参目标。

固定 4 Msymbol/s、16 MS/s、255 字节 PRBS9，无数据导频、编码、加扰。
控制头仍有 Polar(64,48) 和 16:1 控制导频。
同步字 `0x6a28c4bd9d8eb6c5` 按低位先行；头 CRC12 种子 `0x6c5`，
数据 CRC24A 多项式 `0x00065b`、种子 `0x555555`。
QPSK MCS8，8PSK MCS12；数据起点为同步后第65个符号。
参考载荷始终使用全部 PRBS9，不忽略前缀。

`frozen-plan.json` 在开发/保留集处理之前固定全部参数、输入身份和实现身份。
开发集为 rx-00/seed11015，保留集为 rx-01/02/seed22015；没有自动调参。
静态偏移含一个采样周期内的网格点和非网格点；后续单独改变 SRO、长度、
CFO、复噪声和幅度，最后只增加一个组合条件。
SRO 的 0、±10、±25、±50、±100 ppm 是实验点，不是标准限值。

仿真 TX 直接计算连续 RRC 闭式脉冲（span16），不调用 RX 的 RRC 系数函数或
CubicSpline；以已知坐标采样。CRC/头编码复用模型，独立性限于波形和时间基准，
不是独立完整 SLE 发射机。正 ppm 表示每个发送符号占用更多 ADC 样本。
仿真 511/1023/2047 字节仅验证 A3 的11位长度及模型表示范围，
未证实 WS63 厂测或具体调度配置允许这些包长。静态和SRO精度依赖此波形模型。
真实 IQ 追加 ±100 ppm 使用独立的五阶 B 样条重采样，保存派生文件；
其插值误差与原未知 SRO 均存在，不将注入值称作总偏差。

`results.jsonl` 逐输入/网格记录全部能量段、同步分数、CRC、长度、内容、
可计算的 BER 和包头/中/尾误差及耗时。BER 在相位选择完成后按已知无编码布局计算，
包含坏 CRC 帧；与解码器头长度不一致时只可作为固定参考布局的诊断，不能用于验收。
`evidence/*.npz` 保存每臂同步符号和仿真误差轨迹；`sim-*.npz` 保存发送采样输入。
`failures.json` 关联所有失败到源文件/哈希、能量段索引和派生符号，原始 IQ 保持原位。
完整采集中的边界截断单独报告，不能将有限采集中的能量段统计称作空口 PER。

搜索步长不等于定时精度。真实 IQ 无定时真值，误差为 null；不从判决残差估算真值。
接收机没有定时环锁定状态，因此收敛时间、失锁计数和周期滑移为 null，
仿真另报真实误差跨过半符号判决边界的次数，不能等同于环路周期滑移。
32符号同步窗口为8 us，但居中滤波与整包载波细化使用未来样本，不能称作实时收敛。

耗时包括获取和解析，分别记录；不包含波形生成、压缩和写盘。
吞吐包含 IQ Msamples/s 和单包名义载荷 Mbit/s，后者不适合含多包的真实整文件。
`process_peak_rss_KiB` 是 Linux 进程累计 RSS 高水位（含生成、存档等），不是每臂增量。
图表与 JSON 为同一批数据；`manifest.json` 覆盖输出字节身份。

## 比较持续定时跟踪

在同一 worktree 中使用新输出目录：

```sh
UV_CACHE_DIR=/tmp/rx-uv MPLCONFIGDIR=/tmp/rx-mpl PYTHONPATH="$PWD/src" \
 uv run --frozen --no-sync python scripts/rx_timing_tracking_compare.py \
 --dataset /home/sanchuanhehe/Documents/opensparklink/sparklink-hw/datasets/sle-rf/ws63/20261115-rx2 \
 --output output/psk-tracking-new
```

此命令使用当前环境；新检出先用 `uv sync --frozen` 安装锁定依赖。
两臂均固定1/8初始相位搜索，只改变持续跟踪开关，不重选同步或CRC候选。
原开发/保留集之外，预先固定 seed33016、偏移0.317采样、50kHz CFO、30dB SNR，
以255/1023/2047字节与0/±100 ppm构成18个新增保留输入。
开发参数在这些复核之前冻结。

`results.jsonl` 每行保留同一输入的 fixed/tracked 两臂及同段恢复/退化索引。
`regressions.json` 记录原成功段退化，`recoveries.json` 记录恢复，
`failures.json` 同时保留两臂所有失败。全部生成IQ、附加SRO派生IQ、
实际符号采样坐标、积分器速率和TED误差轨迹均单独保存。

真值误差使用实际采样坐标减发送参考坐标，不以固定SPS外推代替跟踪轨迹。
`settling_symbol_0p05_samples` 表示误差从该符号起一直处于±0.05采样内，
且剩余完整包至少128符号；它不是接收机锁定声明，也不表示速率估计已经收敛。
真实IQ没有这个真值诊断。截断帧仅评价实际存在的符号，并标记误差轨迹未覆盖整包。

算法及API见 [PSK包内采样时钟恢复](../explanation/psk-timing-recovery.md)。

## 复核噪声分段与载波修复

使用上一轮冻结输出作为输入源，原九个失败及190个原仿真输入保持原样。
本次三臂依次为1符号能量窗+旧载波、8符号窗+旧载波、8符号窗+相干载波；
每臂都启用包内持续跟踪。方法在读取保留集前固定，不按CRC重试或选择算法。

```sh
UV_CACHE_DIR=/tmp/rx-uv MPLCONFIGDIR=/tmp/rx-mpl PYTHONPATH="$PWD/src" \
 uv run --frozen --no-sync python scripts/psk_noise_recovery_audit.py \
 --baseline /home/sanchuanhehe/.codex/visualizations/2026/11/15/01a1dfe8-f1ae-7233-a63e-c31ffad2d428/rx-timing-tracking/final \
 --dataset /home/sanchuanhehe/Documents/opensparklink/sparklink-hw/datasets/sle-rf/ws63/20261115-rx2 \
 --quiet-dataset /home/sanchuanhehe/Documents/opensparklink/sparklink-hw/build/evidence/20261116-ws63-release-r2/ws63-rx2-20261115-r2 \
 --output output/noise-recovery-new
```

`--baseline` 必须是上一轮包含生成IQ、派生IQ及manifest的完整冻结目录。
正样本和配对静默按该目录的旧元数据和哈希读取，不随主数据集索引的更新而换样。
当前共享数据集正在替换部分profile，因此扩展阴性测试显式使用独立完整r2快照；
其144份静默的元数据及逐文件哈希另行冻结，不将新profile正样本混入旧对照。
可用别处的完整静默数据集替换 `--quiet-dataset`，但须保留同样的元数据格式。
程序拒绝缺失/变更的输入，不静默跳过。

`target-nine.json` 关联原九项的全部三臂结果，`failures.json` 同时保留旧失败、
中间失败及新保留集失败；`regressions.json` 检查已成功包的退化。
原成功包在选择完成后按同步坐标±2采样匹配，不按载荷内容配对。
新增保留集固定seed44016/55016、8/10/12/15dB、CFO30kHz和SRO100ppm，
不得看到结果后调参再称为保留集。存在剩余失败时预期退出码1。
