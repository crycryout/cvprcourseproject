# H800交接 v2：视觉推理系统

本版已替代AP-ToMe。首次使用克隆repo；已有checkout先保留本地修改再拉取最新main。不要运行旧validation_matrix或旧训练提示词。

```bash
git clone https://github.com/crycryout/cvprcourseproject.git
cd cvprcourseproject
python3 scripts/preflight.py --output artifacts/preflight_v2.json
python3 scripts/make_matrix.py --output artifacts/serving_matrix_v2.json
codex
```

默认preflight只收集环境，不做GPU计算。确认空闲卡后，由Codex显式指定`--cuda-smoke --device <可见设备序号> --require-cuda`完成小规模GPU检查；尖括号部分需替换为核实过的设备号。

给Codex的提示词：

```text
使用 $run-cvpr-course-project 完成当前v2项目。
若未自动发现skill，直接读取 .agents/skills/run-cvpr-course-project/SKILL.md。
先读AGENTS.md、PROJECT_PLAN.md、docs/STATUS.md、docs/EXPERIMENT_PROTOCOL.md。

这是H800上的目标检测推理系统项目，重点是CUDA Graph、CPU–GPU异步流水线、deadline-aware batching。
旧AP-ToMe/token merging/CIFAR训练方案已取消，不要继续旧方案，也不要重新设计选题。
用预训练DETR和COCO真实图像，固定权重、预处理、精度和后处理，验证检测质量。
请按M0→M5实现缺失代码、运行真实实验、分析并写英文论文式报告，不要只给计划。
先做环境与模型正确性，运行不超过10分钟的pilot定位真实瓶颈，然后推进Graph/pipeline/调度。
采用开放环到达，延迟包含排队与CPU预处理到CPU检测结果的完整路径；所有offered请求进入SLO分母。
调优固定等待batching基线，完成EDF消融；尝试可行的compile基线并据实报告。
保留负结果，不伪造加速，不把GPU forward时间冒称E2E，也不把PCIe结果说成GH200结果。
单卡优先，正式计时避免另一张卡共享CPU/PCIe干扰；不修改系统驱动、不结束其他任务。
每阶段更新STATUS、manifest和累计GPU-hours，预算8–20、自动上限30 GPU-hours和30 GB存储。
上限内自主继续普通实现步骤；下载受限时给出确切所需本地文件，禁止随机权重替代。
最后给我质量表、p95/p99/goodput曲线、消融、profile、真实框图、报告和复现指令。
由我最终审阅和提交课程材料。
```

中断恢复：先检查active process、run manifest和freeze版本，续跑第一个未完成gate；不重复启动、不覆盖结果、不重新用held-out调参。

当前已有两个工具只做环境检查/计划生成。`python -m cvpr_project ...`是IMPLEMENTATION_PLAN规定的待实现CLI，先实现并验证`--help`后才能用。
