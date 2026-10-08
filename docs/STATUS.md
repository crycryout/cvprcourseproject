# 状态 v2

更新日期：2026-10-09（澳门）。实现已完成；当前实验因完整H800不可用而阻塞，尚未完成正式结果或报告。

| 阶段 | 当前状态 | 真实证据与剩余工作 |
|---|---|---|
| M0 模型/数据/质量 | 数据、固定权重和旧质量检查已保存；新scope待验 | 官方5000图像、1000/4000固定split、不可变模型revision与SHA；新设备身份检查需在完整卡恢复后通过 |
| M1 baseline | 历史pilot保留，新引用待重测 | 15:48 UTC CUDA smoke确认为完整H800；后续校准实际设备与该引用不一致，重新建立pilot/worker/service |
| M2 graph/pipeline | 实现及历史验证保留，新scope待重验 | 两slot固定地址、显式事件和1000请求检查；新硬件防护在加载权重前拒绝MIG |
| M3 调度/冻结 | 旧113组MIG成功校准已剔除，当前0/252 | 硬件身份进入候选hash；冻结前逐项核对实际CUDA设备与pilot，未冻结 |
| M4 实验 | 未开始 | 尚未进行held-out推理，无正式serving结论 |
| M5 报告 | 生成器完成，待真实结果 | 曲线、消融、质量表、profile与英文Markdown/PDF代码齐备；暂无报告PDF |

## 当前硬件阻塞与恢复

GPU0：MIG mode=Enabled，但没有GPU/compute instance、没有计算进程。绑定其物理UUID时CUDA返回`No CUDA GPUs are available`。数字`CUDA_VISIBLE_DEVICES=0`实际枚举GPU1上的`H800 MIG 2g.20gb`（30 SM、21,072,183,296 bytes），不是nvidia-smi物理GPU0。校准manifest证明此前113个当前时钟版本成功配置均在MIG上运行，因此不进入完整卡比较。旧Full smoke与部分baseline可以保留为历史，不能与这些MIG校准组合成正式结论。

已把179个旧校准attempt及752个关联文件按SHA-256保存在本地`artifacts/hardware_scope_attempts`，事件记录为`artifacts/hardware_scope_incident_v2.json`。原始成本照计，当前2.367/30 GPU-hour，其中0.25为未计量smoke保守预留；存储10.60/30 GB。未修改MIG、驱动或其他用户任务。

当前账号无免密码sudo。需要管理员在确认GPU0仍无实例、无计算进程后执行：

```bash
sudo nvidia-smi -i 0 -mig 0
```

依据：[NVIDIA MIG说明](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/getting-started-with-mig.html)指出没有GPU/compute instance时不能运行CUDA。只针对GPU0；GPU1已有任务不在修改范围内。

恢复入口：激活项目环境后运行`CVPR_GPU=0 bash scripts/resume_experiments.sh`。入口绑定物理UUID，等待MIG Disabled及整机连续60秒无其他GPU进程；依次重验模型、pilot、编译与校准，冻结后进行12×4000图像质量评估及300个正式配置（若compile无效为8×4000与264配置）。不会自动修改GPU设置。已有freeze不会重新调参。

10项CPU协议/硬件边界测试通过。GPU数值/性能的重新验证尚未执行，不能把CPU测试当作GPU结果。

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
