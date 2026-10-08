# 项目计划 v2：面向时延要求的目标检测推理系统

**将项目主线从视觉压缩算法改为推理系统：用预训练目标检测器研究CUDA Graph、CPU–GPU流水线与请求调度。**

英文题目：Deadline-Aware Object Detection Serving on NVIDIA H800 with CUDA Graphs and CPU–GPU Pipelining。

目录：1.定位；2.问题；3.运行时；4.实验；5.进度与退路。

## 1. 为什么重新选题

上一版AP-ToMe需要研究视觉token重要性、模型精度和微调，与你的MLSys/AI Infra积累不够直接。新版直接用到你已有的GPU推理、CUDA Graph、CPU–GPU同步和serving经验：模型权重冻结，把研究变量放到runtime。

| 方向 | 在本项目的位置 |
|---|---|
| 视觉任务 | 输入RGB图片，输出类别、置信度与bounding boxes |
| 研究对象 | 单GPU目标检测请求的执行和调度 |
| 方法 | 预捕获batch graph +固定buffer池 +异步流水线 +deadline-aware batching |
| 质量约束 | 相同模型/输入分辨率/精度下，优化前后检测结果和COCO AP基本一致 |
| 主要指标 | 端到端p50/p95/p99、按时完成的请求率、吞吐、资源占用 |

这属于视觉推理系统课程项目。课程PDF第8页允许视觉主题及非SOTA自提方案，因此按文字范围匹配；不能宣称教师已经批准系统侧选题。最终报告需明确目标检测问题和视觉质量证据，不能只交CUDA微基准。

无需新训练、人工标注或机器人环境。第一版不实现HTTP/RPC服务、不接摄像头、不做追踪；在本机用真实图片和可复现到达序列构造serving harness。称为目标检测serving实验，不能冒称真实视频或生产流量。

## 2. 研究问题和有边界的贡献

**RQ1：** H800运行小批次视觉检测时，瓶颈是GPU计算、host launch gap、预处理，还是H2D/D2H与同步？

**RQ2：** CUDA Graph与固定缓冲区/异步copy能否减少真实请求延迟？二者单独效果和组合效果是什么？

**RQ3：** 到达速率变化、请求具有不同deadline时，使用实测batch服务时间选择graph bucket，能否比调优后的固定等待批处理获得更高SLO goodput？

课程方法简称 **deadline-aware graph-bucket batching**，只指本仓库的组合与评估。CUDA Graph、流水线、EDF和dynamic batching都是已有技术；不宣称首创调度器或OSDI级创新。有效贡献是明确策略、正确实现、消融和真实瓶颈分析。

### 2.1 视觉载体

主模型：`facebook/detr-resnet-50`，加载预训练COCO checkpoint，不更新权重。模型/processor使用同一不可变revision，记录下载文件SHA-256。候选来源支持`no_timm`分支，但在服务器先核验并解析为具体commit，不能浮动引用。

数据：COCO2017 validation 5,000图像及instances标注。固定排序后划分1,000 calibration、4,000 held-out evaluation；不训练、不用calibration结论调held-out结果。细节见实验协议。

为固定shape，第一版统一processor resize：shortest_edge=480、longest_edge=640，保持比例，再右/下pad至640×640，携带正确pixel_mask。这是部署预处理设置，不是原论文的800/1333设置，因此不以论文AP作为复现目标。所有runtime必须共用这一视觉设置，不通过降低精度或分辨率赢得速度。

## 3. 运行时设计

### 3.1 数据路径

原始图像字节到达 → CPU decode/resize/normalize → ready queue → batching → pinned host buffer → H2D → DETR forward → D2H logits/boxes → CPU后处理与坐标恢复 → 结果交付。

计时从计划到达时刻开始，终点是对应request_id的CPU结果可用。既包含排队和预处理，也包含拷贝、GPU、输出同步及后处理。另测GPU-resident forward解释瓶颈，但不能将其称为端到端延迟。

### 3.2 Graph bucket和双缓冲

固定batch候选{1,2,4,8}，所有tensor shape/address在每个graph内保持稳定；按实际可用显存排除OOM bucket并冻结共同集合。batch不足时使用最小可容纳bucket、dummy lane和valid_count；dummy lane必须是合法图像/有效mask，不能全masked制造NaN，且绝不输出为真实检测结果。

每个(bucket,slot)预分配自己的host/device输入及输出存储；默认2 slots、1个compute stream。独立graph实例绑定各slot地址，共享只读权重。严格禁止在上次D2H和CPU消费完成前覆盖output或复用slot。显存受限时共同减bucket/slot并说明，不以隐藏串行化伪称双缓冲。

在capture前warmup；模型设eval/inference_mode，固定backend/精度。capture只包含已验证可捕获的forward或明确的静态子图；CPU图像处理、queue逻辑、文件I/O放在capture外。不得为capture删除pixel_mask或改变检测数学语义。

copy stream与compute stream通过events建立H2D完成→forward、forward完成→D2H的依赖。copy/compute是否真的重叠由trace证明。第一版一个compute stream，不做并发graph kernels，避免对graph pool并发安全的额外假设。

### 3.3 截止时间感知调度：精确定义

请求i带有arrival `a_i`、deadline `d_i=a_i+D_i`、图像ID和ready时间。ready queue按deadline排序（相同deadline按arrival、request_id）。在compute可接受下一批时做决策，不允许无限预提交；最多一批执行、一批已确定并预拷贝。

在calibration上测每个batch bucket的保守剩余服务时间 `R_b`：从dispatch到CPU结果可用的p95，涵盖pack/H2D/forward/D2H/postprocess。另估计已排队GPU工作的剩余时间 `G`；不使用未来到达信息。保守估计不扣除尚未验证的overlap收益。

对于n=1..min(ready_count,max_bucket)：取EDF前n项，令b为能容纳n的最小bucket。预计完成时间 `F(n)=now+G+R_b`。只有F(n)不晚于这n个请求的最早deadline时，n才是可行候选。优先选择`n/R_b`最大的可行候选，tie取较小完成时间，再取较小bucket。没有可行候选时立即服务最早deadline请求；**不通过丢弃困难请求美化goodput**。

允许有限等待以聚合：仅在有空闲slot、当前候选bucket未满或存在更大bucket、且更大bucket满足 `now+wait+G+R_next <= earliest_deadline` 时等待。wait不超过冻结的tau，且对最老ready请求的累计人为等待不超过tau；下一到达或timer先触发就重新判断，不能每次重置timer无限等待。tau在calibration从{0,1,2}ms选，冻结后固定。不存在可行更大bucket就立即dispatch。

策略是可检验启发式，不保证deadline。服务时间估计不准也必须报告miss，不能回填真实未来运行时间作在线决策。

### 3.4 基线和必要消融

| ID | 方法 | 用途 |
|---|---|---|
| E0 | eager、batch1、pageable/synchronous copy、串行pipeline | 常规朴素起点 |
| E1 | eager、batch1、pinned预分配、串行pipeline | 单独隔离buffer/copy方式 |
| G0 | CUDA Graph、batch1、pinned、串行pipeline | 隔离graph收益 |
| P0 | CUDA Graph、batch1、双缓冲异步pipeline | 隔离overlap收益 |
| R0 | eager + pipeline + 调优固定等待dynamic batching | 强非graph基线 |
| F0 | graph + pipeline + 调优固定等待dynamic batching | 主要对手 |
| D0 | graph + pipeline + 本节deadline-aware策略 | 课程方法 |

R0/F0使用同一bucket集合，FIFO、max_batch和max_wait均在相同calibration搜索预算内调优，不能故意选差参数。必须在关键高负载场景增加A0：EDF + 与F0相同的固定等待，隔离单纯EDF与服务时间可行性判断。

另外对`torch.compile`优化基线给最多2小时兼容/正确性尝试；成功则加入C0（compile + pipeline + 固定等待），与最快有效基线比较。编译mode可能自动使用CUDAGraph，要记录实际设置，不能把它叫纯kernel fusion。失败保留证据并明确结论限于已实现基线。

## 4. 实验范围和成功标准

主实验覆盖平稳Poisson、等间隔、突发三类**合成到达轨迹**；每类4个负载，3个种子，各方法共享相同真实图像ID/到达/deadline。模型权重不变，三个种子是流量种子，不是训练seed。

主结果必须给COCO AP/AP50/AP75、低延迟/高负载曲线、SLO goodput、拒绝/超时数量、stage时间、profile和资源开销。AP在所有held-out图像上独立离线评估，不能只算在线按时返回的简单样本。

研究目标（非结果）：在检测AP变化≤0.1个百分点的条件下，相对调优F0提高至少10%的按时完成率，或在完成率与SLO成功率均不下降的可复现场景降低p95至少10%。仅完成请求的p95降低不能单独证明服务能力改善；必须一并报告拒绝/unfinished和goodput。不达到仍可完成课程项目，结论写瓶颈和无收益的条件。

不要求GH200、不做CPU内存offload：DETR权重较小，不应人为制造不真实的容量瓶颈。H800上的PCIe/synchronization实验只能支持该平台，不宣称C2C可获得同样收益。

## 5. 进度、资源与停止条件

不再按固定日历日期排任务，按验收门槛推进；建议2–3周。官方deadline未知，后续得到截止日期时逆推收尾时间。

| 阶段 | 主要动作 | 验收 | 人工工作估计 |
|---|---|---|---|
| M0 | 环境、模型、图像预处理与质量参考 | 真实框可视化、坐标正确、hash、reference AP | 3–5小时 |
| M1 | 串行baseline及端到端profile | 1000次pilot与阶段时间；瓶颈证据 | 3–5小时 |
| M2 | graph/buffer/pipeline | 结果一致、无race、capture边界、真实overlap trace | 5–8小时 |
| M3 | 到达生成器、调度器、冻结配置 | 无丢失/重复、open-loop可靠、deterministic trace | 4–6小时 |
| M4 | 完整比较、消融和统计 | request日志、AP、图表与可复现汇总 | 3–5小时 |
| M5 | 写论文式报告 | 数值都能追溯，局限和来源明确 | 4–6小时 |

规划GPU-hours：模型/正确性1–3、runtime调试2–5、正式负载/quality/profile4–9、重跑预留1–3，总计8–20；不是性能实测。自动上限30 GPU-hours、存储30GB。首轮≤10分钟pilot重估；正式benchmark独占一张GPU，不在第二张卡并发干扰共享CPU/PCIe。

### 5.1 早期决策门槛

- 若host launch gap很小：保留graph负结果，把重点放到排队/批处理；不能宣称必有同步瓶颈。
- 若H2D/D2H占比<5%：降低copy优化优先级，仍给完整分解；不用人为放大传输数据。
- 若CPU预处理占主导：先固定且公平调优worker数，再比较GPU runtime；CPU worker数量必须各方法相同。
- 若完整forward无法capture：限定2小时排查；允许可解释的部分capture并给coverage及fallback成本，不换成随机模型。
- 若M3不能如期完成：交付E0/E1/G0/P0/R0/F0的完整分析，标记未实现deadline策略。不要拖延基础结果来追求“创新”。

最小项目：单模型、真实检测AP、串行/graph/pipeline对比及profile；完整项目增加F0/D0、三类负载与EDF消融。可选扩展最多一个：第二个检测模型或真实视频帧trace；不要同时扩展两者。
