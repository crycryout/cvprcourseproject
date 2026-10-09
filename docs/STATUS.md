# 状态 v2

更新日期：2026-10-09（澳门）。完整GPU0已恢复，M0–M3的新设备验证、252组校准与冻结完成；M4正在进行留出集精度评估，正式serving结果与报告尚未完成。

| 阶段 | 当前状态 | 真实证据与剩余工作 |
|---|---|---|
| M0 模型/数据/质量 | 新scope验证完成 | 官方5000图像、1000/4000固定split、不可变模型revision与SHA；仅calibration确定FP32，BF16 b2不满足0.1 pp稳定性门槛 |
| M1 baseline | 完整卡重测完成 | 实际CUDA设备为H800 PCIe、114 SM、85,017,493,504 bytes；E1 E2E p95=32.409165 ms，统一8个CPU workers |
| M2 graph/pipeline | 新scope验证完成 | Graph与compile各1000请求，覆盖两slot及逆序logits/boxes；Graph最大误差0，compile logits/boxes最大差0.003231/0.000351；warmup后显存allocated增长0 |
| M3 调度/冻结 | 252/252，已冻结 | R0/F0各96、C0 48、D0 12；共同lambda_ref=167.555394 rps，F0/A0 b4 wait0、C0 b8 wait0、D0 b8 tau1；freeze先于留出集推理 |
| M4 实验 | 留出集AP进行中 | 12个backend/bucket各遍历4000图像；eager b1已完成；300个正式serving配置随后执行，尚无正式serving结论 |
| M5 报告 | 生成器完成，待真实结果 | 曲线、消融、质量表、profile与英文Markdown/PDF代码齐备；暂无报告PDF |

## 当前运行与预算

恢复队列已自动推进至`evaluate-quality`。整机没有其他GPU计算任务，GPU0为完整卡、MIG Disabled；所有新阶段以相同物理UUID hash核对实际CUDA设备。冻结hash：`2ba3f2e355d8ee1c23b310df9206e71a9b3629384d6b7bc2e3938863014f9d9e`。held-out评估开始后不再修改参数。

当前已入账4.961/30 GPU-hour（含旧失败/错scope工作及0.25小时保守预留），存储11.57/30 GB；正在运行的阶段结束后追加实际成本。300个正式配置的10秒预热+60秒窗口合计至少5小时50分钟，另含drain和初始化。完成后还需真实profile、Nsight、英文报告、证据校验及GitHub发布。

10项CPU协议/硬件边界测试通过；本轮GPU数值一致性也已通过，二者分别记录。旧MIG结果不进入本轮统计。

## 已解决的硬件阻塞与恢复记录

恢复前，GPU0处于MIG Enabled，但没有GPU/compute instance、没有计算进程。绑定其物理UUID时CUDA返回`No CUDA GPUs are available`。数字`CUDA_VISIBLE_DEVICES=0`实际枚举GPU1上的`H800 MIG 2g.20gb`（30 SM、21,072,183,296 bytes），不是nvidia-smi物理GPU0。校准manifest证明此前113个当前时钟版本成功配置均在MIG上运行，因此不进入完整卡比较。旧Full smoke与部分baseline可以保留为历史，不能与这些MIG校准组合成正式结论。

已把179个旧校准attempt及752个关联文件按SHA-256保存在本地`artifacts/hardware_scope_attempts`，事件记录为`artifacts/hardware_scope_incident_v2.json`。原始成本照计，当前2.367/30 GPU-hour，其中0.25为未计量smoke保守预留；存储10.60/30 GB。未修改MIG、驱动或其他用户任务。

当前账号无免密码sudo。此前请管理员在确认GPU0无实例、无计算进程后执行以下命令；目前已观察到GPU0恢复，项目没有执行GPU设置变更：

```bash
sudo nvidia-smi -i 0 -mig 0
```

依据：[NVIDIA MIG说明](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html)指出没有GPU/compute instance时不能运行CUDA。只针对GPU0；GPU1已有任务不在修改范围内。

恢复入口：激活项目环境后运行`CVPR_GPU=0 bash scripts/resume_experiments.sh`。入口绑定物理UUID，等待MIG Disabled及整机连续60秒无其他GPU进程；依次重验模型、pilot、编译与校准，冻结后进行12×4000图像质量评估及300个正式配置（若compile无效为8×4000与264配置）。不会自动修改GPU设置。已有freeze不会重新调参。

本轮已重新通过GPU数值验证并冻结，正式性能比较尚待执行。

## 历史执行记录（已归档，不是当前验收）

以下为硬件身份修复前的记录，部分阶段使用完整卡、部分校准使用MIG；不得将其合并为当前正式结果。

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

冻结前计时复核：CPU消费/解码后重新读取决策时钟，避免数毫秒旧now影响等待/可行性判断；CPU结果若在drain cap之后才完成，标记为cap时unfinished并保留实际时间。观察时长也在cap截断。8项CPU协议测试通过。已保存的旧时钟版本校准记录不复用；新版本重新搜索。尚未开始held-out推理。
