# 项目设计与完成计划

**推荐完成：Attention-Protected Token Merging for Efficient Image Classification: An Accuracy–Latency Study on NVIDIA H800。**

这是一个可控的计算机视觉课程研究：输入图像，以分类准确率为质量指标，通过合并视觉 token 减少计算，同时测量 H800 的真实收益。课程允许非 SOTA 的自提方法，适合用完整研究过程取得扎实交付。

目录：1.问题与范围；2.方法；3.实验；4.日程与预算；5.风险和交付。

## 1. 研究问题与范围

### 1.1 为什么选择它

| 维度 | 选择理由 |
|---|---|
| 课程匹配 | 原生图像输入、视觉分类任务、明确算法与实验 |
| 与你的方向连接 | token 数、注意力后端、GPU 利用率、实际延迟，而非纯模型调参 |
| 可完成性 | 单 H800、约 22M 参数的 DeiT-Small、公开的小规模数据集；不训练大模型 |
| 可解释性 | token 合并图、重要区域保护图、准确率–延迟曲线适合论文式报告 |
| 风险控制 | 基线复现即能形成完整实证研究；扩展失败不会使课程项目归零 |

### 1.2 三个问题

1. **RQ1**：固定每层 token 数时，保护分类注意力较高的 token，是否比普通 ToMe 保留更多分类信息？
2. **RQ2**：这类保护带来的分数计算、排序和数据重排开销，在 H800 上是否值得？
3. **RQ3**：结论是否随 batch size 和 attention backend 改变？

主方法命名 **AP-ToMe（Attention-Protected Token Merging）**。这是课程工作名称，不是声称首创的论文方法名。EViT、ToMe、SAD-TM 等已覆盖相关思想；本项目贡献限于明确定义的变体、受控比较及 H800 上的证据。详见 `docs/RELATED_WORK.md`。

### 1.3 固定范围

- 主数据：CIFAR-100；45,000 train / 5,000 validation / 官方 10,000 test。
- 主模型：非蒸馏版 DeiT-Small/16，224×224 输入，12 blocks，196 image tokens + 1 CLS token。
- 方法：Dense、ToMe、AP-ToMe、Random-Protect-ToMe；统一使用同一训练 seed 的 dense checkpoint。
- 不做多 GPU 通信优化、LLM serving、检测/分割、自定义 CUDA kernel；可选扩展只在主实验完成后做。
- CIFAR-100 原图为 32×32，放大至 224×224 不增加细节；不能据此宣称对高分辨率真实场景或 ImageNet 都有效。

## 2. 方法：精确定义

### 2.1 先理解基线

ViT 把 224×224 图像分成 14×14 个 patch，形成 196 个图像 token，加上分类 token（CLS）。Dense 在所有层保留它们。ToMe 在注意力残差更新之后、MLP 之前，把相似 token 合并，以减少后续处理数量。

主基线采用官方 ToMe 的 bipartite soft matching 逻辑与 size-weighted average，使用本文共同指定的层间合并计划。它属于 **将 ToMe 迁移到 CIFAR-100 的受控评估**，不称为精确复现原论文 ImageNet 数字。

### 2.2 AP-ToMe：保留重要 token，再合并其余 token

在第 l 个 block，用当前层 Q、K 和 token mass 定义 CLS 对第 j 个 token 的分数：

`a_j = mean_h softmax_j(q_CLS,h · k_j,h / sqrt(d) + log(s_j))`。

其中 s_j 是此 token 已聚合的原始 token 数；初始为 1。主实验所有 merging 方法统一 `prop_attn=true`。先对含 CLS 的完整序列 softmax，再提取图像 token 分数。排名是启发式，不等于真实因果重要性，也不等于 segmentation mask。

具体流程：

1. 完成当前层 attention 和第一条 residual；获取注意力里的 K，以及 CLS attention row。
2. **保持官方 ToMe 的偶数/奇数序列分区 A/B**，不要先重排全序列。CLS 始终位于 A，既不能合并出去，也不能成为合并目的地。
3. 在 A 的图像 token 和 B 的图像 token 内分别保护分数最高的 p 比例；保护 token 在本次 merge 中既不能作 source，也不能作 destination。
4. 对剩余 A/B token 用跨 head 平均 K 的归一化余弦相似度匹配，每个候选 A 选择一个 B，再取最佳 r 条 source edge；允许多个 A 合入同一 B。
5. 按 token mass 加权聚合；保留官方输出顺序和 CLS 首位；将新序列输入 MLP，更新 mass 后进入下一层。

这是 **分区内保护**，不是全局 top-p 保护。分区选择保证 p=0 时能退化为相同实现的普通 ToMe，避免把不同二分图误当成单纯保护消融。

### 2.3 固定 shape 与可行性

设 A 中非 CLS 数为 nA，B 中图像数为 nB，计划合并 r 个：

`kA = min(ceil(p*nA), nA-r)`；`kB = min(ceil(p*nB), nB-1)`。

要求 `0 <= r <= nA` 且在 r>0 时 nB>=1。保护 kA/kB 个 token，保证足够合法 source 和至少一个 destination。p 是配置常数；各样本的保护对象不同，保护数量一致。每层实际长度由计划决定，不按样本动态裁剪。对 r=0 直接 no-op；禁止为满足保护规则静默减少 r，非法配置应报错。

聚合一个 destination j 及其 sources I 时：

`s'_j = s_j + sum_i s_i`；`x'_j = (s_j*x_j + sum_i s_i*x_i) / s'_j`。

同时更新 provenance（仅用于可视化），计时路径关闭 provenance tracking。保护仅约束 merge，不禁止 token 在正常 attention/MLP 中更新。浮点 ties 固定原序号优先，不能用 CPU 排序破坏 GPU 测速。

### 2.4 合并预算

主实验前两层不合并，从第 3 层到第 12 层每层合并 r∈{4,8,12} 个。对应最后的 image tokens 是 156、116、76（另外始终有 1 个 CLS）。

| r | 每层计划 | 最后图像 token | 用途 |
|---|---|---:|---|
| 0 | 12 层均为 0 | 196 | Dense / 零合并一致性 |
| 4 | [0,0,4,4,4,4,4,4,4,4,4,4] | 156 | 温和压缩 |
| 8 | [0,0,8,8,8,8,8,8,8,8,8,8] | 116 | 中等压缩 |
| 12 | [0,0,12,12,12,12,12,12,12,12,12,12] | 76 | 较强压缩 |

AP-ToMe 在验证集选择 p∈{0.10,0.20}；p=0 由 ToMe 代表。Random-Protect 在同样分区随机保护同样数量，隔离“保留数量”和“保护重要性”的效果。详细选择与随机数规则见实验协议。

### 2.5 防止 attention 后端造成假结论

分两条实验轨：

- **算法轨**：Dense/ToMe/AP/Random 均使用相同 explicit attention 实现，便于验证数值及抽取完整 attention；不作为“击败优化 Dense”的唯一证据。
- **部署轨**：Dense 使用已验证的 PyTorch SDPA（scaled dot-product attention）路径；ToMe/AP 尽量使用 SDPA，AP 另外只计算 CLS 的一行概率。为 token mass 加的 mask 可能改变实际 kernel，必须用 profiler 核实，不能把 SDPA API 等同 FlashAttention。

两条轨均保留所有运行开销。先做 eager，`torch.compile` 只作为可选独立实验，编译时间单列且双方公平优化。若 AP-ToMe 未能超过最快有效 Dense，就报告未实现部署加速。

## 3. 实验设计

主实验流程和精确默认值见 `docs/EXPERIMENT_PROTOCOL.md`；参数集中在 `configs/project.json`。

### 3.1 必须完成

1. 训练同一 recipe 的 3 个 dense seeds：17、42、2026；验证选择 checkpoint，测试集封存。
2. 在 seed 17 checkpoint 的完整验证集筛选保护比例；冻结每个 r 的 p 和所有比较配置。
3. 在三个 dense checkpoints 上运行固定方法，得到准确率、分样本预测和成对差值。
4. 在独占 H800 上对 batch 1/16/64 测延迟、吞吐、显存，分析 scoring/merge/attention/MLP 开销。
5. 从真实结果生成主表、Pareto 图、消融表和 token 可视化，再形成论文式报告。

### 3.2 正向假设与合格交付分开

**研究目标（不是承诺）：** 在至少一个固定 token 预算下，AP 比 ToMe 保留更多准确率；或相对最快 dense 在精度下降不超过 1.0 个百分点时提高吞吐 10% 以上。

**完成标准：** 即使两个目标都未达到，只要实现正确、比较公平、结果可复现并解释失败原因，仍完成项目。不能把目标数字填入结果表。

主要比较以百分点（percentage points, pp）衡量 accuracy 差异，注明置信区间及 seed 方差。对“最好”的操作点，只能在验证集和预设性能协议上选择。

### 3.3 最小版本与可选扩展

最小可交付：1 个 dense seed、Dense + ToMe 三种预算、batch 1/16/64、两条后端轨、完整真实性说明及局限。必须注明单 seed，不得当作完整实验。

完整版本：3 seeds + AP + Random 消融。

可选项最多选一个：CIFAR-10 的外部任务复核、DeiT-Tiny 的模型泛化，或对双方 `torch.compile` 的附加比较。扩展前先计算剩余 GPU 预算；不能并行堆满所有扩展。

## 4. 四周完成计划

以 2026-10-09 开始为例；真实课程 deadline 未提供。每个阶段先通过验收再扩大规模。

| 阶段 | 建议日期 | 主要动作 | 退出条件 |
|---|---|---|---|
| M0 设定 | 10/09–10/10 | 建环境、核实数据/预训练来源、冻结划分、补读近邻方法 | 能载入模型，输出100类，split无交集，预检记录齐全 |
| M1 Dense | 10/11–10/14 | tiny-set过拟合、10分钟pilot、训练seed17 | 验证曲线合理、可恢复checkpoint、真实成本估计 |
| M2 ToMe | 10/15–10/18 | 接入merge、正确性测试、验证三种预算 | r=0等价，mass守恒，长度正确，与官方算法对齐 |
| M3 AP | 10/19–10/22 | 实现保护/随机消融、验证筛选与配置冻结 | 保护不被merge，p=0等价，冻结记录已保存 |
| M4 证据 | 10/23–10/29 | 补齐3 seeds、正式test和性能实验 | 原始证据完备，主表/曲线可脚本再生成 |
| M5 写作 | 10/30–11/05 | 分析负面结果、写报告、完成可复现检查 | 报告所有数值能追溯到run，限制与引用明确 |

这些日期是可调整的内部安排，不代表教师要求。建议本人投入约 25–40 小时用于文献理解、抽查代码/实验及报告修订；Agent 自动工作时间另计。

### 4.1 H800 预算（待 pilot 修正）

| 工作 | 规划 GPU-hours |
|---|---:|
| 环境/正确性/pilot | 1–3 |
| 3个dense seeds，默认各30 epochs | 12–30 |
| 验证筛选、test评估、性能和profiling | 6–15 |
| 失败重跑与预留 | 5–12 |

合计约 **24–60 GPU-hours**；上界为规划余量，非实测预测。自动执行总上限 80 GPU-hours；存储软上限 50 GB。两卡不能把 GPU-hours 除二，只能在独立任务上缩短 wall time。

pilot 以 `ceil(45000/effective_batch) × measured_step_seconds × epochs` 估算训练，并加上验证、数据启动和存盘成本。若预计超预算，先减可选项，再减 recipe 搜索，最后降级为单 seed 最小版本并披露；不能缩小test却不说明。

## 5. 失败处理和最终交付

| 风险 | 处理 |
|---|---|
| 预训练下载或数据被阻断 | 支持用户提供本地路径和哈希；不能静默以随机权重替代 |
| 老ToMe与新timm不兼容 | 在隔离环境移植最小merge/attention adapter，保留来源和测试，不降级系统驱动 |
| Dense收敛差 | 先查标签、normalize、train/eval、head与权重；只在validation上有限调整 |
| AP不改善准确率 | 保留负结果，随机保护检验假设；将报告聚焦真实accuracy–latency trade-off |
| 有FLOP减少但无加速 | 用profile说明选择开销、矩阵尺寸及后端差异，避免选择性省略batch1 |

最终提交包建议包含：英文论文式报告、源码与锁定环境、实验配置/小型结果文件、复现步骤、5类关键图表。短演示材料仅在课程要求或本人需要时制作。用户负责最终审阅和提交。
