# CISC8005: Deadline-Aware Object Detection Serving on H800

**当前项目：面向时延要求的目标检测推理系统——CUDA Graph、CPU–GPU流水线与截止时间感知批处理。**

English title: **Deadline-Aware Object Detection Serving on NVIDIA H800 with CUDA Graphs and CPU–GPU Pipelining**.

以预训练DETR为视觉工作负载，研究提交、传输、同步和排队开销。重点是MLSys / AI Infra运行时设计，不训练新视觉模型，不研究token pruning/merging。真实图像输入、检测框输出和COCO检测精度评测保留，满足课程的视觉任务边界。

> 2026-10-08按用户反馈重新选题。旧AP-ToMe方案已退出执行计划，保留在Git历史提交 `a9491421aa1d607b34fe10894ea8b021544f8144`。所有当前docs、config、skill和工具以本版为准。当前仅完成研究设计与交接，尚无H800实验结果。

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
python3 scripts/preflight.py --output artifacts/preflight_v2.json
python3 scripts/make_matrix.py --output artifacts/serving_matrix_v2.json
```

这两条命令已实现，仅检查环境和生成实验计划；`cvpr_project`推理主体仍需服务器Codex实现。详细启动流程见交接文件。预计2–3周、8–20 GPU-hours，需实测pilot修正；自动执行上限30 GPU-hours。
