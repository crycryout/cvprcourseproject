# CISC8005: Deadline-Aware Object Detection Serving on H800

**当前项目：面向时延要求的目标检测推理系统——CUDA Graph、CPU–GPU流水线与截止时间感知批处理。**

English title: **Deadline-Aware Object Detection Serving on NVIDIA H800 with CUDA Graphs and CPU–GPU Pipelining**.

以预训练DETR为视觉工作负载，研究提交、传输、同步和排队开销。重点是MLSys / AI Infra运行时设计，不训练新视觉模型，不研究token pruning/merging。真实图像输入、检测框输出和COCO检测精度评测保留，满足课程的视觉任务边界。

> v2运行时、校准/冻结、矩阵恢复、profiling和报告生成器已实现，10项CPU测试通过。当前发现校准误选MIG分区，旧记录已保留并剔除；完整GPU0处于MIG开启但无实例状态，等待管理员恢复。正式实验与报告尚未完成，见[当前状态](docs/STATUS.md)。旧AP-ToMe只保留Git历史。

## 为什么贴近你的背景

| 既有能力 | 项目中对应的工作 |
|---|---|
| LLM/VLA serving | 请求队列、batching、尾延迟和SLO goodput |
| CUDA Graph / CPU–GPU同步 | 固定地址capture/replay、异步完成事件和结果生命周期 |
| PCIe数据移动 | pinned memory、H2D/D2H、固定buffer pool及copy/compute overlap |
| GPU profiling | NVTX、torch.profiler/Nsight，定位host gap、copy和kernel成本 |

单H800即可完成；不依赖两卡互联，不把PCIe结果外推为GH200/NVLink-C2C结果。DETR只作为可复现的目标检测载体，项目名称不构成新颖性声明。

## 阅读与执行

| 文件 | 用途 |
|---|---|
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 完整选题、运行时设计、进度门槛、预算和退路 |
| [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 视觉质量、开放环负载、基线、goodput、计时和统计 |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | 模块接口、CUDA生命周期规则和验收 |
| [H800_HANDOFF.md](docs/H800_HANDOFF.md) | 给服务器Codex的完整提示词 |
| [项目Skill](.agents/skills/run-cvpr-course-project/SKILL.md) | 分阶段执行与恢复 |

其他：[课程映射](docs/COURSE_REQUIREMENTS.md)、[相关资料](docs/RELATED_WORK.md)、[报告提纲](docs/REPORT_OUTLINE.md)、[状态](docs/STATUS.md)、[配置](configs/project.json)。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r configs/requirements.lock.txt
python -m pip install --no-deps -e .
python -m pytest -q
python -m cvpr_project --help
python3 scripts/preflight.py --output artifacts/preflight_v2.json
python3 scripts/make_matrix.py --output artifacts/serving_matrix_v2.json
```

完整复现（新checkout/artifacts）：确认一张完整H800可用且整机空闲后运行`CVPR_GPU=0 bash scripts/reproduce.sh`。需要官方COCO下载及固定DETR权重，环境锁为实测Python 3.10/PyTorch 2.14.0/CUDA 13.0；数据、权重、原始请求日志和完整profile仅保留本地，不上传Git。CPU协议检查不需要GPU。

已完成的配置按manifest和hash恢复；GPU干扰会保存失败并等待空闲后重试。已有freeze与新配置冲突时拒绝覆盖，重做实验使用新checkout/evidence目录。完整矩阵包括252主实验、12 EDF控制及36有效编译基线，计时窗口每个60秒；预算上限30 GPU-hours/30 GB。详细操作见[H800交接](docs/H800_HANDOFF.md)，CUDA生命周期见[CUDA_RUNTIME](docs/CUDA_RUNTIME.md)。

当前实验恢复：激活环境后执行`CVPR_GPU=0 bash scripts/resume_experiments.sh`。程序先将nvidia-smi物理序号解析为GPU UUID；runtime以真实CUDA属性和UUID hash核对设备，拒绝把MIG分区冒充完整卡。GPU0当前需管理员关闭空配置MIG模式，见[恢复命令](docs/STATUS.md)。
