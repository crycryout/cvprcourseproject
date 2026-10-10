# CISC8005: Deadline-Aware Object Detection Serving on H800

**项目已完成：300组真实图像serving实验、12组完整留出集精度评估、6组Torch profile与Nsight实测。**

英文报告：[9页PDF](results/report.pdf) · [Markdown全文](docs/REPORT.md)。证据索引见[results/README.md](results/README.md)，验收记录见[docs/STATUS.md](docs/STATUS.md)。

以固定预训练DETR为视觉工作负载，在单张完整NVIDIA H800 PCIe上实现CUDA Graph bucket、固定buffer pool、CPU–GPU流水线和截止时间感知批处理。输入为真实COCO图像，输出为原图坐标检测框；请求流量是合成的开放环periodic/Poisson/burst，不是生产流量。项目不训练新模型，不声称Graph、EDF或流水线本身具有新颖性。

## 已完成的交付

| 内容 | 实测证据 |
|---|---|
| Serving矩阵 | [300组结果](results/summary.csv)：252主实验、12 EDF控制、36有效Inductor基线；主策略各36组（三种到达模式×四个负载×三个seed），A0仅12组高负载控制 |
| 检测质量 | [12组×4000图像](results/quality.csv)，eager/graph/compile各4个bucket；FP32参考AP=36.311754，最大绝对变化0.000112个百分点 |
| 机制分析 | [实际阶段耗时及batch占用](results/stage_summary.csv)、[六组拷贝重叠证据](results/profile_overlap.json)、[Nsight kernel及计数器](results/nsight_v2.json) |
| 英文报告和图表 | [PDF](results/report.pdf)、[英文全文](docs/REPORT.md)、[曲线/消融/时间线](results/figures)、[真实检测及图片许可](results/detections/README.md) |
| 验收与成本 | [交付清单](results/delivery.json)：12.473 GPU预留小时（含失败及0.250小时保守预留），分析导出时存储约16.77 GB；均低于30小时/30 GB上限 |

## 主要结果

在36个共享主实验设置上，各策略按时goodput的算术均值如下。每个设置使用相同请求trace，goodput为测量窗口内到达且最终按时完成的请求数除以60秒。

| 策略 | 平均按时goodput（请求/秒） |
|---|---:|
| F0：调优后的Graph FIFO批处理 | 33.955 |
| D0：截止时间可行性启发式 | 19.580 |
| C0：有效Inductor批处理基线 | 20.027 |

**D0未超过调优后的F0：配对均值低14.375请求/秒，即42.3%。** D0实际平均batch占用为1.027，F0为2.832；这与批处理效率损失一致，但不能单独证明唯一原因。负结果及4次GPU干扰失败均保留，未使用留出集结果重新调参。D0/C0的小差异仅作描述，不声称统计显著。

最低共同到达率50.267请求/秒已高于校准E1串行容量33.663请求/秒，因此该网格没有未饱和E1点。串行E0/E1/G0的零按时goodput描述此过载网格，不能解释为其未饱和时延。正式计时、独立profile和GPU-resident forward微基准的范围分别报告；不外推至GH200/NVLink-C2C。

## 安装与验证

实测环境为Python 3.10、PyTorch 2.14.0+cu130、CUDA 13.0、Transformers 4.46.3；完整依赖固定在[requirements.lock.txt](configs/requirements.lock.txt)。在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/requirements.lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python scripts/verify_results.py
```

10项CPU协议/硬件边界测试通过。发布证据校验不需要GPU、COCO或权重；它核对300组配置、共同trace、全请求分母、12组AP、profile硬件身份、Nsight实测元数据以及12个冻结runtime文件的SHA。

## 实验复现

**完整重做请使用新checkout及空artifacts目录。** 激活上述环境，确认一张完整H800和共享主机GPU空闲后执行：

```bash
CVPR_GPU=0 bash scripts/reproduce.sh
python scripts/verify_results.py
```

脚本下载官方COCO validation和固定DETR权重，依次验证、校准、冻结、独立质量评估、运行矩阵和profiling，最后生成英文报告。模型revision为`70120ba84d68ca1211e007c4fb61d0cd5424be54`。Nsight阶段需要已安装的`nsys`和`ncu`及可访问GPU计数器；本次两项实际采集均成功。数据、权重、原始请求CSV和完整profile只保留本地，不进入Git。

已有本地实验epoch需要恢复时，先确认没有活动队列，再执行`CVPR_GPU=0 bash scripts/resume_experiments.sh`。已完成配置按manifest和SHA复用，已有freeze不会重新调参；GPU干扰尝试保留后等待空闲重试。恢复详情见[H800_HANDOFF.md](docs/H800_HANDOFF.md)。

| 设计文件 | 用途 |
|---|---|
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 原始选题、设计、阶段验收和预算 |
| [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 质量、开放环负载、基线、计时与统计协议 |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | 已实现CLI、模块接口与验收要求 |
| [CUDA_RUNTIME.md](docs/CUDA_RUNTIME.md) | 固定地址、stream/event及slot生命周期 |
| [项目Skill](.agents/skills/run-cvpr-course-project/SKILL.md) | 分阶段执行、证据规则与恢复流程 |

代码与报告使用Codex辅助完成；课程提交由用户审阅后自行处理。旧AP-ToMe项目仅保留Git历史。
