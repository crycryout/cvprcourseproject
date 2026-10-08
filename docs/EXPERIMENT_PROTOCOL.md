# 实验协议 v2：真实图像的目标检测serving

本文件规定待执行实验，不含效果。先calibration后freeze，最后held-out评估。结果输出必须携带protocol_version=2，不能混入旧AP-ToMe配置。

目录：1.视觉正确性；2.流量；3.基线/调优；4.计时与统计；5.证据。

## 1. 模型、数据和视觉质量

冻结`facebook/detr-resnet-50`模型和processor的不可变revision与文件SHA-256。禁止随机权重替代或额外微调。服务器先验证torch/transformers兼容性，不盲目跟随网页latest版本升级。

COCO2017 `val2017` + `instances_val2017.json`，5,000张。对image_id数值排序，用`numpy.Generator(PCG64(20261008)).permutation`打乱，前1000为calibration，其余4000为held-out。保存列表及SHA-256。全部属于COCO官方validation，本项目自行留出两部分，**不要称为官方test集**。

calibration用于预处理/数值验证、worker数、batch/等待参数、service table、SLO与load标定。held-out不用于这些选择。所有模型权重都已固定；这不是训练数据划分。

### 1.1 固定视觉输入契约

processor按shortest_edge480、longest_edge640保持比例resize；normalize使用checkpoint规范；右侧/下侧pad到640×640，并输入pixel_mask。用当前版本正式API或独立已验证的adapter实现固定pad。不要把pad视为真实内容。

DETR输出boxes的坐标语义必须通过官方processor对照：使用原图(H,W)和官方postprocess恢复，并在横图、竖图和不同pad量样本上验证。**不能自行假定boxes一定相对于640×640画布再减padding**，以实际模型/processor规范为准。变换/逆变换需单元测试和框叠加可视化。

COCO category_id不连续，必须使用checkpoint id2label和官方映射核实，不能简单`label+1`。排除no-object类。AP评估保留100个query的有效类别预测，threshold=0（仅可视化使用0.7），不得以0.9阈值过滤来计算AP；不要私自增加NMS。

### 1.2 精度和一致性

先FP32 eager建立参照，再检查BF16与参照在calibration上的AP差异。主轨默认BF16，若差异>0.3 AP百分点则所有方法主轨统一回到FP32，冻结后不再根据held-out改变精度。FP32时TF32设置也统一并记录。

冻结前还检查同精度各batch bucket相对batch1的calibration AP变化；若BF16不满足0.1个百分点runtime质量门槛，统一回到FP32并重新标定所有policy。此额外稳定性检查仅使用calibration，并在观察到held-out结果前确定；保留失败的BF16证据。FP32若也失败，先修复正确性，不开始正式矩阵。

runtime优化前后使用**同精度/权重/后端/输入**：FP32 raw logits/normalized boxes以atol=1e-5,rtol=1e-4起步；BF16以与FP32参照观察到的数值误差制定并记录容差，不能为了让测试过关无限放宽。主要正确性验收为同精度AP变化≤0.1百分点及1000次request_id/output完整性压力测试；超出先查race、mask、dummy、decode和精度。

编译兼容性记录初始FP32严格容差的全部失败。融合改变浮点运算顺序时，在held-out之前额外冻结一次明确的语义误差界：logits绝对差≤0.01；由cxcywh误差界推导的原图xyxy角点位移≤0.5像素（normalized box atol=0.5/(1.5×calibration最大原图边长)，rtol=0）。仍须通过1000请求压力测试及每个bucket的完整calibration AP≤0.1门槛，最后独立验证held-out AP。不得在此误差界失败后继续放宽。Graph与eager的同bucket比较保留初始严格容差。初始失败与最终误差界都写入报告。

独立离线遍历4000 held-out图像，使用pycocotools COCOeval、IoU0.50:0.95，报告AP/AP50/AP75/AP_small/medium/large。对所有实际backend和batch bucket验证；在线超时样本仍参与离线AP。不同纯调度策略若共享同一已验证executor，可复用该executor的离线AP，但要验证在线request_id→输出匹配。报告这是4000图像subset AP，不能与原论文全5000图像分数直接比较。

## 2. 开放环流量与SLO

使用open-loop：到达序列预先生成，不等待上一请求完成。每个请求从计划到达时刻起算，以免发生coordinated omission（负载发生器等服务完成后才发下一请求，掩盖排队）。分离load generator和服务线程，记录scheduled_arrival、actual_enqueue及generator_lag。

若实际发送lag的p99>max(1ms,0.05×最小deadline)，本轮标记loadgen-limited并修复后重测；不能把发生器瓶颈当成系统极限。尽量单独CPU core/进程，固定worker数。线程/进程调度选择记录但不修改其他用户任务。

### 2.1 负载内容

主轨从内存缓存的**压缩图像字节**开始，到CPU可用检测结果结束，包括JPEG decode/预处理，排除磁盘与网络。所有方法同样缓存字节，不能只给某些方法缓存预处理tensor。另设GPU-resident forward微基准解释kernel/launch开销，两个scope分开报告。

图片ID在held-out集合中按trace seed确定的顺序循环；每次request_id唯一，相同图像可反复请求，**禁止缓存模型结果**。固定语料的复用要在报告披露。calibration用其独立1000图像。

三种到达：

- periodic：间隔1/lambda；
- poisson：独立指数间隔，mean=1/lambda；
- burst：1秒周期内前200ms以5lambda的Poisson率到达，后800ms为0，使期望均值lambda；记录实际到达数，不强称每轮精确相同。

流量种子17/42/2026，每种策略共享同seed完整轨迹。deadline类别比例50%/30%/20%，相对期限分别为{2,4,8}×S_base，独立于图像内容、采用相同固定随机抽样，不按方法改变。

S_base为calibration上E1串行batch1完整pipeline延迟p95，转换成毫秒后冻结。这些是归一化研究SLO，不代表业务真实要求。另报告具体毫秒值，不笼统称real-time。

### 2.2 负载强度

先在calibration上调优F0，持续backlog运行其pipeline，以90秒实测完成数量/时间得到共同lambda_ref；不包含warmup。所有方法的offered rates统一为{0.3,0.6,0.9,1.1}×lambda_ref。不得每种方法用自己的容量归一化后比较同一load标签。

每run预热10秒、测量窗口60秒，末尾最多30秒drain。只对测量窗口内到达的request_id统计，但不能从分母删掉未完成请求。预热请求可能影响测量起始状态，必须保留固定策略并披露，不在边界选择性清空队列。

所有方法共同max_pending=512（含preprocessing/ready/in-flight），超额时拒绝最新到达请求；所有拒绝计入SLO失败。不得D0单独提前丢弃过期请求。drain结束仍未完成的请求标记unfinished并计入失败；不要把这类请求的延迟虚构成deadline。

## 3. 基线、校准与冻结

方法ID和完整定义见PROJECT_PLAN。E0/E1/G0/P0固定batch1。R0/F0均搜索max_batch∈{1,2,4,8可行子集}，wait_ms∈{0,1,2}。R0/F0用同calibration traces和相同搜索预算，选择deadline满足数最高的配置，tie优先更低p95，再更小max_batch；不能只调D0。

先用10秒warmup+20秒窗口的短calibration轨迹搜索；用相同trace seeds但不同于最终评测的图像集合。标定初始负载用E1饱和吞吐的{0.5,1,2,4}倍，先固定S_base。选出F0后测lambda_ref；在最终相同rates上做一次所有候选的calibration复核并冻结，不无限循环优化lambda_ref。

D0 tau也只搜索{0,1,2}ms，service table `R_b`由calibration实测dispatch→CPU结果p95给出。只使用过去记录估计in-flight剩余G，不能访问未来真实completion。在提交prefetch batch后它变成locked，EDF不能凭未来到达重新覆盖其buffers；最多一个locked next batch，准确记录prefetch限制带来的迟到。

A0（EDF fixed timer）在poisson和burst、load0.9/1.1、3 seeds必须执行。A0与F0共同executor/bucket/timer，只有队列顺序不同，进而与D0隔离deadline可行性预测的作用。优先完成该消融，不扩大模型数。

C0用最多2小时尝试torch.compile：如果能够正确运行，纳入最终表和最快有效baseline；记录compile mode、是否自动cudagraph、编译/预热成本。失败需真实日志和边界说明，不宣称已超过编译/TensorRT等未测框架。

冻结文件包括数据/图像IDs、模型/processor哈希、runtime版本、所有参数、common buckets、precision、service table、S_base、lambda_ref、实际SLO与rates、worker数、buffer数、队列限制、计时语义及calibration选择依据。正式held-out在冻结后开始。

## 4. 指标与统计

- `latency_i = CPU_result_ready_i - scheduled_arrival_i`。报告全部完成请求p50/p95/p99，并同时报告未完成/拒绝率；percentile不能隐藏未完成者。
- `SLO_success = 在自身deadline前成功返回且request_id正确的请求数 / 测量窗口全部offered请求数`。
- `goodput = 上述按时完成数 / 60秒`，迟到、拒绝、失败、unfinished均不能进入分子。
- throughput明确采用窗口到达cohort完成数/含drain观察时长，另给稳态窗口完成速率；不能将drain完成数直接除60冒称持续吞吐。
- GPU显存峰值含所有graphs/private pools、输入输出slots，另给pinned host memory、CPU占用、graph初始化时间、实际dummy/padding比例及batch分布。

每种trace三seed构成独立负载重复，同seed跨方法成对比较；不同请求不是独立系统运行重复。给每seed值及均值/范围，对小差异追加3个负载seed只在预先定义“相对差<5%且影响结论”条件触发，各方法一起重测。不要用3个seed宣称高置信显著性。

正式测量不带profiler；另对代表性低负载/高负载配置抓trace。阶段时间以host monotonic clock及CUDA events分别记录；GPU/CPU时钟不能直接相减，不能把重叠阶段总和等同E2E。统计submit成本、等待、H2D、forward、D2H、postprocess，给timeline解释重叠。

固定硬件、精度、backend、CPU workers、输入契约、bucket集合。每个trace seed随机方法执行顺序，记录温度/利用率/共享任务情况，不改GPU功耗设置。第二张卡也不并发共享CPU/PCIe负载。

## 5. 证据和图表

每run生成manifest、frozen config快照、request_events.csv、metrics.json、失败原因与累计GPU-hours。request记录含ID/image_id、计划/实际到达、preprocess/ready/dispatch/GPU/d2h/postprocess/complete各阶段、deadline、bucket/valid_count、status。只写真实可采集时间，没有的填null及原因，不拼接GPU与host时间域。

汇总只接受completed且语义一致的run，允许completed_with_failures明确服务失败数；基础设施崩溃标记failed不能悄悄替换。重新运行使用新run_id并链接原因。

最终图表：offered load—p95/p99；offered load—goodput/成功率；E0→E1→G0→P0消融；F0/A0/D0对比；profile timeline；离线AP表与检测框实例。所有图都从小型结果文件再生成，不使用估计性能充数。
