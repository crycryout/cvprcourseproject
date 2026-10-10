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

`cvpr_project`全部CLI已实现，10项CPU协议/硬件边界测试通过；完整实测进度以STATUS为准。创建Python 3.10独立环境，安装`configs/requirements.lock.txt`后`pip install --no-deps -e .`。完整执行使用`CVPR_GPU=0 bash scripts/reproduce.sh`，已完成的同hash配置自动恢复。正式运行使用完整H800，不使用MIG smoke替代性能测量。

`scripts/run_when_idle.py`先解析物理GPU UUID并检查MIG Disabled，再等候连续60秒无其他GPU任务后启动各GPU阶段，只重试明确的GPU干扰失败；其他错误保持退出状态供检查。矩阵每个配置还会每5秒检查GPU任务。不得结束其他人的任务。

在已有实验目录中恢复时不要重跑verify-model/pilot覆盖校准引用；检查active lease和freeze后，从未完成命令恢复。修改被冻结的runtime、数据或权重会被拒绝。完整重新复现应从新checkout与独立artifacts开始。

硬件事件：数字CUDA序号与nvidia-smi物理序号不能在MIG环境中混用。runtime必须使用实际CUDA device name、SM数、可见内存、UUID hash和软件版本核对；pilot/calibration/freeze/quality/main一致才接受。此前GPU0开启MIG但无实例的阻塞已解除，完整H800的新校准与冻结已完成。恢复前仍须核查实际设备及active process，再使用`CVPR_GPU=0 bash scripts/resume_experiments.sh`；该入口保留freeze并从未完成阶段续跑。详见STATUS。
