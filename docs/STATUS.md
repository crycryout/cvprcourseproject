# 状态 v2：M0–M5完成

更新：2026-10-10（澳门）。**300/300正式serving配置、12/12完整4000图像AP评估、6/6 Torch profile及Nsight Systems/Compute实测均完成；9页英文报告和可公开证据已生成，发布校验通过。**

报告：[PDF](../results/report.pdf) · [Markdown](REPORT.md)。统计：[summary.csv](../results/summary.csv) · [证据索引](../results/README.md) · [delivery.json](../results/delivery.json)。

## 最终验收

| 阶段 | 状态 | 实际证据 |
|---|---|---|
| M0 模型/数据/质量 | 完成 | 固定DETR revision、官方5000图像与1000/4000 split；仅calibration选择统一FP32，BF16 b2未满足0.1 pp稳定性门槛 |
| M1 baseline | 完成 | 完整H800 PCIe、114 SM、85,017,493,504 bytes；E1串行E2E p95=32.409165 ms，CPU workers统一为8 |
| M2 graph/pipeline | 完成 | Graph/compile各1000请求验证；Graph最大误差0，compile logits/boxes最大差0.003231/0.000351；warmup后allocated增长0 |
| M3 调度/冻结 | 完成 | 252/252校准；共同lambda_ref=167.555394 rps，F0/A0 b4 wait0、C0 b8 wait0、D0 b8 tau1；冻结先于held-out推理 |
| M4/M5 实验与报告 | 完成 | 252主实验+12 EDF控制+36编译基线；AP 12/12通过；6组独立profile、native kernel计数器、阶段/占用统计、图表及英文报告 |

主轨为完整H800 GPU0（MIG Disabled），以实际CUDA设备属性及物理UUID hash核对，硬件身份见[environment_timing_v2.json](../results/environment_timing_v2.json)。最终检查未发现活动实验队列或GPU计算进程；项目未修改MIG、驱动、时钟或其他用户任务。

## 结果与范围

同精度eager b1参考AP=36.311754，12个backend/bucket各独立评估全部4000图像，最大绝对变化0.000112个百分点。线上失败/超时不改变离线AP图像集合。

**主要比较为负结果：D0平均按时goodput=19.580请求/秒，F0=33.955，配对差−14.375请求/秒（−42.3%）。** D0实际平均batch占用1.027、F0为2.832；报告讨论可能机制，未推断唯一因果。D0/C0差−0.448请求/秒仅作描述。A0只覆盖12个预定高负载控制，不能与36组主策略直接比较跨网格均值。

最低共同到达率50.267请求/秒高于串行E1容量33.663，因此该网格没有未饱和E1点。E0/E1/G0的零按时goodput是过载网格结果，不是未饱和时延结论。正式耗时不含profiler；独立Torch profile测得P0高负载拷贝时间与kernel重叠65.45%，不是E2E墙钟占比。

Nsight Systems成功采集Graph node粒度的实际kernel表；Nsight Compute成功采集首要累计时间kernel的一次真实calibration b1实例，10次replay passes、554个记录指标。计数器可访问，GPU时钟未改变。数据与完整二进制trace留在本地，公开命令、哈希及小型统计见[nsight_v2.json](../results/nsight_v2.json)。

## 冻结、失败与成本

冻结身份：`2ba3f2e355d8ee1c23b310df9206e71a9b3629384d6b7bc2e3938863014f9d9e`。12个核心runtime源文件SHA保持冻结值；本轮修改限于分析、报告、Nsight采集/解析及证据验证，没有重新调参或选择最快rerun。

4次正式GPU干扰失败保留并列于[delivery.json](../results/delivery.json)，不纳入300个有效比较。旧MIG校准及失败、BF16回退、初始compile容差失败均保留并计费；旧硬件scope不混入完整卡统计。

截至分析导出，累计12.473090/30 GPU预留小时，其中12.223090为metered leases，0.250为初始未计量smoke保守预留。存储约16.77/30 GB（含数据、模型和完整profile）。公开结果只含小型证据、检测示例、图表和报告，预算记录见[resource_ledger.json](../results/resource_ledger.json)。

## 本轮恢复与历史事件

| 事件 | 原因与处理 |
|---|---|
| 旧硬件scope错误 | 数字CUDA device 0曾解析到GPU1的MIG分区；113个完成校准不进入完整卡统计，原始179个attempt及752个文件按SHA归档，随后按实际UUID重测 |
| 2026-10-09 GPU干扰 | 另一张GPU出现计算进程时释放本项目GPU并保留失败；不停止其他任务 |
| 2026-10-10恢复 | 检查到旧队列不存在、GPU空闲后复用213个校验完成配置；旧队列退出原因缺少日志，不推测；正式300组于UTC 13:41:46完成 |
| Nsight Systems表缺失 | 默认Graph-level capture没有单kernel表；保留原报告后显式node tracing重采成功 |
| Nsight Compute解析错误 | 实际导入为wide CSV；扩展解析器后离线复用已成功捕获的报告，没有再次选择性能样本 |

历史硬件事件公开摘要见[hardware_scope_incident_v2.json](../results/hardware_scope_incident_v2.json)，原始请求CSV、失败日志及完整profile均在被Git忽略的本地`artifacts/`。

## 已通过的检查与复现

10项CPU协议/硬件边界测试通过。`python scripts/verify_results.py`校验300组唯一配置、36组共同trace、全部offered分母、12组AP及CSV SHA、6组profile、native Nsight硬件/冻结身份和12个runtime源文件SHA。正式请求阶段统计还独立复核原始CSV计数、按时判定及p50/p95/p99。

```bash
source .venv/bin/activate
python scripts/verify_results.py
```

完整重做使用新checkout/空artifacts并执行`CVPR_GPU=0 bash scripts/reproduce.sh`。已有本地epoch恢复前先检查活动进程，入口为`CVPR_GPU=0 bash scripts/resume_experiments.sh`；已完成结果按manifest和SHA复用，已有freeze拒绝覆盖。具体环境与范围见[README](../README.md)和[H800_HANDOFF.md](H800_HANDOFF.md)。
