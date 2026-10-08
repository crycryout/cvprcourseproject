# 状态 v2

更新日期：2026-10-09（澳门）。当前执行DETR目标检测serving；主轨已在held-out之前统一为FP32。旧AP-ToMe仅保留Git历史。

| 阶段 | 状态 | 证据 |
|---|---|---|
| v2研究设计/交接 | 已编写并通过本地配置/工具检查 | 当前docs/config/skill |
| M0 模型/数据/质量 | 已完成 | 官方COCO与固定模型revision；输入/类别/坐标验证；FP32 4桶calibration AP变化≤0.000003 pp |
| M1 baseline/profile | FP32 pilot已完成；profile待采集 | E0/E1各1000真实请求；E1 E2E p95=32.824750 ms |
| M2 graph/pipeline | 输出/生命周期验收通过；overlap待profile | 1000真实请求stress；所有bucket/部分batch/延迟CPU消费 |
| M3 调度/冻结 | 正在校准，未冻结 | service表、开放环与恢复机制已运行；R0/F0/C0/D0搜索进行中 |
| M4 实验 | 未开始 | 无AP/serving实测 |
| M5 报告 | 实现证据生成器，完整报告待实测 | 英文Markdown/PDF、曲线、消融与时间线生成代码已实现 |

当前任务：`calibrate`，由`run_when_idle.py`在空闲窗口自动恢复。最近预算约0.70 GPU-hour，其中0.25为未计量smoke的保守预留；准确实时值以`artifacts/cost_ledger.jsonl`和active lease为准。短暂其他GPU活动会中止当前配置并保留失败，不结束其他任务。项目存储约10.5 GB。

## 历史执行记录（以顶部当前状态为准）

## 服务器执行进展（2026-10-08）

已建立项目独立Python 3.10环境（PyTorch 2.14.0/CUDA 13.0、Transformers 4.46.3）；主实验设备为完整H800 PCIe。另一张卡是MIG 2g.20gb，仅做兼容性smoke，不能把该smoke称为完整H800性能实验。

模型固定revision `70120ba84d68ca1211e007c4fb61d0cd5424be54`；三个下载文件SHA-256已落盘。官方processor固定pad/mask API已验证。完整DETR forward在MIG smoke上capture成功，尚未完成真实图像graph一致性验收。

已实现src package与所有计划CLI：数据、模型/quality、slot/event池、开放环harness、FIFO/EDF/feasibility调度、校准/冻结、矩阵恢复、profile和分析。5项CPU协议测试通过；这些是逻辑测试，不是性能结果。COCO通过官方S3分块下载，按Range、ETag和ZIP CRC校验；下载尚在进行。

初始smoke在lease meter启用前执行，预算保守计入0.25 GPU-hour预留，明确不是实测耗时。之后的GPU工作由lease ledger记录实际进程占用墙钟时间。GPU 0另有任务，正式计时等待共享任务结束；不停止其他任务。

当前gate：M0进行中；M1–M5仍待真实证据。下一步：完成COCO下载/固定split，运行calibration上的FP32/BF16质量与输入契约验证。

每次推进更新实际run、证据、active process、累计成本、blocking issue与下一步。旧v1本地工具测试不等于v2 runtime实测，旧artifact不得混入统计。

## 交付验证

新版matrix实际生成252主run与12消融，旧协议、closed-loop、仅completed分母、选择性过期丢弃和训练设置均被拒绝。Skill结构及Markdown链接校验通过。preflight默认无GPU计算，显式smoke在当前无CUDA环境准确报不可用。这些不是H800性能证据。

M0验收（UTC 2026-10-08，澳门2026-10-09）：COCO固定split SHA-256 `3ede0df1a6eb2bfc8a49c99084f2fdcac1fcb6b9fa3685176f635ba36502b9a2`。1000 calibration图像FP32 AP 36.083574、BF16 AP 35.955744；差0.127831个百分点，小于0.3门槛，主轨统一BF16。尚未读取held-out推理结果。真实检测框图已生成，许可/归属信息待发布前核对。

M1/M2阶段证据：`artifacts/pilot_E0_v2.json`、`pilot_E1_v2.json`、`stress_v2.json`、`service_table_v2.json`。Graph dispatch-to-result p95（b1/2/4/8）=5.208/8.088/15.100/24.515 ms；这是dispatch服务时间，不能称为完整请求E2E。下一步：20分钟上限的Inductor兼容/质量尝试，再进行相同预算的R0/F0校准。

冻结前质量修正：BF16 eager b2 calibration AP=35.844414，相对b1变化−0.111329个百分点，超过0.1 runtime门槛。所有BF16 pilot/compile/stress/service证据保留在本地precision_attempts；主轨在calibration阶段统一回退FP32，正在重验全部bucket及重标定。held-out推理尚未开始。第一次1000-request内存观察到32 MiB惰性workspace；重测记录显示首100请求以后allocated增长0 byte。

FP32稳定性验收：calibration b1/b2/b4/b8 AP=36.083574/36.083573/36.083571/36.083571，最大变化0.000003个百分点。主轨FP32已经在held-out之前确定。正在重跑同精度E0/E1、Graph压力测试和服务表。

FP32编译基线验收通过：4个calibration bucket最大AP变化0.000006个百分点；1000请求stress最大logit差0.003231、normalized box差0.000351（冻结的绝对界0.01/0.000521）。初始严格FP32容差失败已保留；最终编译基线将列入全部质量/serving比较。当前正在执行R0/F0初始搜索和最终rate复核，再校准C0/D0。主实验待freeze后开始。

校准前13个配置已完成；修正D0剩余时间估计后重新校准，旧记录按source hash保留，不混用。未观察到CUDA forward开始的排队批次保持完整R_b；第一次观察后才扣除elapsed，避免把排队时间当成执行时间。6项CPU协议测试覆盖此回归。FP32 E1 E2E p95=32.824750 ms，服务表b1/2/4/8=10.632/17.055/30.173/56.955 ms。尚未开始held-out推理。
