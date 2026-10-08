# 实验协议 v1

本文件是待执行的预注册设计，不含已测得的效果。首次 test 前保存冻结版本的 Git commit 与 `artifacts/frozen_protocol.json`。修改协议必须说明时间、原因及是否已查看 test。

目录：1.数据；2.训练；3.方法与筛选；4.性能；5.统计和记录。

## 1. 数据与泄漏控制

### 1.1 CIFAR-100

- 使用官方 Python 数据或 torchvision CIFAR100 loader；目标为100个 fine labels，范围0–99，不能错用20个 coarse labels。
- 官方50,000训练样本每类500个。对每类按原始 index 排序，使用 NumPy `Generator(PCG64(20261008))`，按 class 0..99 顺序打乱，各取50个作validation，其余450个作train。
- 得到45,000 train / 5,000 validation；保存原始样本index、标签、split脚本版本、数据校验和与split JSON的SHA-256。三个训练seed共用同一划分。
- 官方10,000 test在冻结配置前不参与准确率评估、选择checkpoint或可视化筛选。可以核验文件校验和，不能用标签指导决策。
- validation/test固定Resize((224,224), bicubic, antialias=True)、ToTensor、ImageNet mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]；禁止不同方法不同预处理。

训练默认：在原始32×32图上 RandomCrop(32,padding=4)，RandomHorizontalFlip(0.5)，随后Resize到224、ToTensor、同样normalize。验证不随机增强。记录torchvision版本与实际操作顺序。

原图尺寸小且ImageNet预训练可能含语义重叠，报告将结论限定为迁移设置；未审计预训练集重复时不能宣称无数据重叠。不得把此结果直接等同ImageNet或真实高分辨率任务。

### 1.2 样本标识

`dataset:split:index` 作为稳定 sample_id。predictions保存sample_id、label、pred、正确性以及可选概率；评估顺序固定。不依赖文件系统遍历顺序。

## 2. 模型与训练

### 2.1 主模型

候选timm模型标识 `deit_small_patch16_224.fb_in1k`；先通过当前安装版本的 `list_models` / pretrained config验证。若版本仅支持旧别名 `deit_small_patch16_224`，明确记录实际权重来源并核对等价性。使用非distilled版，只有CLS，不带distillation token。用ImageNet预训练backbone，分类head改100类并重新初始化；日志列出被替换的键，不能掩盖其他missing/unexpected keys。

统一seed17、42、2026。每seed只训练dense一次，同seed的所有压缩方法共享该checkpoint；主实验不为AP单独微调。

### 2.2 默认recipe（起点，不是已验证最优值）

| 项目 | 默认 |
|---|---|
| epochs | 30 |
| optimizer | AdamW，betas=(0.9,0.999)，eps=1e-8 |
| LR | backbone=5e-5，head=5e-4；按effective batch256固定，不自动缩放 |
| weight decay | 0.05；bias、norm、CLS和pos embedding设0并记录参数组 |
| schedule | 3 epochs线性warmup，从0.1×目标LR开始；随后cosine到0.01×目标LR |
| batch | effective256，优先micro256；OOM时micro128累积2，或micro64累积4 |
| precision | BF16 autocast，权重和optimizer FP32；不在BF16下无理由启用GradScaler |
| loss | cross entropy，label smoothing=0.1 |
| regularization | dropout=0，drop_path=0.1，gradient clipping norm=1.0 |
| selection | 每epoch完整validation；最高top1，tie取较早epoch |

训练data loader `drop_last=False`；末尾不完整累积组按真实样本数归一化并执行optimizer step。每epoch shuffle，worker种子可复现。断点保存model、optimizer、scheduler、epoch/step、RNG states和config。记录AMP/TF32设置；方法比较不能悄悄改变精度。

先通过64张train图像的过拟合检查：关闭增强/label smoothing，FP32或BF16记录清楚，≤500steps应能显著降低loss并达到≥95%训练准确率；未达标先诊断而非直接完整训练。这不是泛化性能门槛。

再跑≤10分钟pilot估算成本。若seed17收敛异常，先排查实现。允许最多2次额外短recipe试跑（每次≤5epochs）：backbone LR 2e-5或1e-4，head/backbone比10；只用validation选择，再从同一预训练权重重训。其成本计入预算，所有其他seed用选定recipe。不得根据test扩大超参搜索。

### 2.3 环境

复用服务器可工作的PyTorch/CUDA组合；先隔离再装依赖。Python 3.10/3.11为候选，不盲目锁旧torch/timm。需求包括torch、torchvision、timm、numpy、pandas、matplotlib、pytest；安装后写精确锁文件和库版本。不要将含凭据的pip索引URL写入公开日志。

## 3. 方法、筛选与冻结

### 3.1 相同预算

固定12层、前2层r=0，后10层r∈{4,8,12}。ToMe/AP/Random必须有相同每层输出长度。保存每层trace，不能只对齐最后层token数。统一mass-weighted merge与 `prop_attn=true`。Dense使用同checkpoint的原始无merge路径。

| 名称 | 保护逻辑 | 作用 |
|---|---|---|
| Dense | 无压缩 | 准确率及最快实际部署基线 |
| ToMe | p=0，仅保留CLS | 压缩基线 |
| AP-ToMe | 分区内CLS attention top-p | 提议扩展 |
| Random-Protect | 相同数量随机保护 | 检验注意力分数是否有价值 |

### 3.2 验证搜索（只在seed17 checkpoint）

先实现算法轨。每个r评估ToMe、AP(p=.10)、AP(p=.20)，在完整5,000 validation中选AP最高top1的p，tie取较小p。这个p固定给所有训练seeds和所有batch/backend；不在test或其他seed上重新择优。

Random的主对照用同r所选p。初步验证可固定p=.10检查正确性，但正式主表必须使用最终选择的p。随机保护不能在每次相同输入时乱变：以protection_seed、sample_id、layer产生固定GPU随机分数；记录种子。计时必须包括随机生成/索引开销，不能偷偷缓存随机index而让AP实时计算。

成本：算法轨筛选共有1 Dense + 3 ToMe + 6 AP =10个配置；另3个Random(p=.10)是开发诊断，总共13配置由`make_matrix.py`生成。冻结后是1 Dense +3×(ToMe、所选AP、Random)=10配置，每个训练seed执行。

之后验证SDPA轨与explicit轨的预测一致性；若出现差异，分别报告backend，不将显著不同的模型输出混为同配置。所有改变与数值误差记录入freeze。

### 3.3 可比性检查

1. r=0：adapter与原dense FP32 logits `allclose(atol=1e-5,rtol=1e-4)`；误差不通过先诊断，不能直接放宽。
2. AP p=0：在相同backend和tie规则下与ToMe输出、mass、index一致。
3. 每层mass总和197（196 patch+1 CLS）；CLS mass恒1，长度精确符合计划；被保护向量在merge前后不变。
4. 全masked/零norm输入、r边界、batch1和batch>1、奇偶长度均有测试；确保不会选中-inf edge、产生NaN或跨样本scatter。
5. 与固定官方ToMe revision的merge逻辑作随机小tensor对照；移植差异（tie规则等）明确列出。r>0的数值一致性使用同精度，BF16容差另经FP32参照确认。

冻结文件至少包含：Git SHA、split哈希、三个checkpoint哈希/对应recipe（若后两seed尚在训练则先冻结recipe，test前补齐checkpoint哈希）、每r所选p、layer schedule、backend、precision、batch列表、允许的统计与图表、benchmark样本ID。

## 4. 性能实验

### 4.1 两条轨和三种计时

两条轨各自给完整结果，不能跨轨混算速度：

- 算法轨：所有方法explicit attention、eager、BF16。
- 部署轨：原dense最佳已验证SDPA eager；merging方法SDPA + 必要操作。记录实际kernel，mask导致fallback也属于方法成本。

每轨对batch1/16/64分别测量（OOM则所有方法共同降至最大可行batch并披露，不能只给AP更大batch）。正式性能测量使用seed17的冻结checkpoint；seed间主要变化是权重，性能重复与训练seed重复分开处理。

| 模式 | 范围 | 方法 |
|---|---|---|
| GPU forward | GPU-resident输入到GPU logits，含scoring/merge | CUDA events；记录GPU时间，不冒称用户端E2E |
| Host-wall forward | GPU-resident输入，同步完成推理 | perf_counter前后同步；包含CPU提交开销 |
| Pipeline | 内存中原始图像→CPU transform/batching→H2D→model→D2H预测 | batch1串行请求与大batch吞吐分别测；磁盘读取不计入，明确说明 |

保存main forward的p50/p95（单位ms/batch），吞吐以总样本/总时间统计；不把ms/batch写成ms/image。Pipeline的预处理、copy及输出同步成本单列。

### 4.2 时间样本

使用冻结的1,024张validation样本ID，不使用挑选的最快样本。预先准备GPU输入时只用于GPU/host-forward；pipeline从CPU原始图像开始。每个方法和batch：50次warmup（编译另计），300个计时batch，重复5个round，round间随机化方法顺序。batch构造固定并循环这些ID。

GPU event timing和host-wall timing分开执行。GPU可连续记录成对events，测量末同步读取；host-wall为串行请求时每batch显式同步。保存所有原始时间数组及测量方式。median speedup使用同round成对基线，不用最快一次；报告round间变异。总吞吐独立测一个稳态循环，用实际处理的样本数除以完整循环时间。

不在正式计时同时训练另一张卡：PCIe、CPU、磁盘/电源共享会造成偏差。记录GPU型号、可用显存、利用率、功耗状态、驱动/软件版本，不能据硬件名称假设服务器是80GB或空闲。

### 4.3 显存和profile

warmup后同步，reset_peak_memory_stats，测量 forward期间peak allocated及peak reserved；减去baseline常驻量可以额外给activation delta，但必须同时给绝对值。不要混合CPU内存和GPU显存。

用torch.profiler/NVTX分段：attention、CLS score、matching/topk、gather/scatter、MLP、H2D/D2H。profile运行与速度运行分离。若有Nsight则分析代表性配置即可，不要求全矩阵ncu。provenance tracing、hooks debug、大量logging不能保留在测速路径中。

MAC/FLOP估算声明 convention（例如1 multiply-add=2 FLOPs），包含attention与MLP主要矩阵算子，注明topk/gather等非算术成本不能由FLOPs完整描述。

## 5. 统计、结果和审计

### 5.1 指标

主要质量：top1（%）和压缩方法减dense的差值（pp）；补充top5、per-class accuracy。三个训练seed分别报告，并给mean±sample SD（ddof=1）。置信区间不能只用3个seed当作可靠正态样本。

按固定sample_id对齐同seed预测，用2,000次paired bootstrap估计AP-ToMe与ToMe准确率差值的95%区间；可对每seed分别给区间，并给同一重采样index作用于全部seed后的平均差值区间。此区间主要衡量test样本不确定性，训练不确定性用跨seedSD另报。固定bootstrap seed=20261008。

性能给5个round的分布和原始样本数；近似持平时不要仅凭1–2%的差异宣称优势。Pareto图同时保留慢/差的点，主结论相对最快有效dense部署基线给出。

### 5.2 结果记录

每次run保存：`manifest.json`、`config.json`、`metrics.json`、实际command、Git SHA/dirty diff哈希、checkpoint哈希、split哈希、seed、backend、dtype、GPU index及型号、起止时间、运行状态、GPU-hours、失败原因；另按需要保存`predictions.csv`、`timings.csv`、token trace。

`run_id`唯一且不覆盖已有目录。只有`status=completed`且证据完整的run可进入结果聚合。部分完成或失败写明状态，聚合器拒绝缺字段或串接不同split/checkpoint的结果。

上传Git的小文件：聚合表、配置、复现命令、匿名环境摘要、图、必要sample-level预测；大checkpoint/full traces留服务器并给SHA-256。公开日志不要包含私钥、账号token、个人目录名/主机信息；内部记录硬件信息不等于可以整份环境上传。

### 5.3 必须生成的图表

1. 主表：accuracy、pp差、p50/p95、throughput、memory，分backend和batch。
2. Accuracy–latency Pareto图与batch scaling图。
3. AP vs Random vs p=0消融及每层token数图。
4. 代表性配置的耗时分解。
5. Token组映射/保护图，以及固定抽样规则选出的成功与失败案例。

图中文字和caption清楚区分“验证筛选”“最终test”和“synthetic correctness test”。可视化attention不是可证明的视觉解释，报告须保留此限制。
