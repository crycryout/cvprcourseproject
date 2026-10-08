# H800 服务器交接

## 第一次启动

```bash
git clone https://github.com/crycryout/cvprcourseproject.git
cd cvprcourseproject
python3 scripts/preflight.py --output artifacts/preflight.json
python3 scripts/make_matrix.py --output artifacts/validation_matrix.json
codex
```

若已有checkout，只需在确认工作区改动已妥善保留后拉取最新main。`codex`须由服务器现有客户端提供；上述工具不会安装或升级它。

给 Codex 的完整提示词：

```text
使用 $run-cvpr-course-project 完成这个仓库的 CISC8005 course project。
如果没有自动发现 skill，请直接读取 .agents/skills/run-cvpr-course-project/SKILL.md。
先读取 AGENTS.md、PROJECT_PLAN.md、docs/STATUS.md 和 docs/EXPERIMENT_PROTOCOL.md。

我在 H800 服务器运行你。请按照 M0→M5 实现缺失代码、运行真实实验、分析并撰写英文论文式报告，
不要只重新生成计划。当前仓库是研究设计和工具，训练/模型/评估主体尚待实现。
先检查已有环境、GPU占用与已有run，选择空闲GPU，完成隔离环境和不超过10分钟的pilot。
单卡优先；第二张卡可用于独立seed，正式benchmark期间避免任何并行任务干扰。

严格执行固定数据划分、validation调参、test冻结后评测、同checkpoint/每层token预算对比。
同时报告explicit算法对照和SDPA部署对照，计入所有保护/合并开销。
不伪造实验成绩，不把目标值填进表，不将FLOPs下降当成加速；保留负结果。
每个阶段更新docs/STATUS.md、run manifest和累计GPU-hours，保存可恢复checkpoint。
自动工作上限80 GPU-hours和50 GB项目存储；在上限内继续，不反复询问普通实现选择。
若数据/权重下载受限，精确列出所需本地文件和路径；不得随机权重替代预训练后继续冒充结果。
完成后给我主表、关键图、报告、复现指令与剩余限制。由我审阅并提交课程材料。
```

## 中断后继续

```text
使用 $run-cvpr-course-project 继续项目。先读取docs/STATUS.md并检查正在运行的进程、
最新manifest和checkpoint，以实际证据恢复第一个未完成阶段。
保留已有实验、失败记录和未提交修改，不重复启动仍在运行的job，不重置数据划分，
不重新使用test调参。报告当前阶段、已花GPU-hours和下一步，然后继续执行。
```

## 已有与待实现命令

已可运行：`preflight.py`只做环境检查和小CUDA smoke，`make_matrix.py`只生成验证配置并检查token预算。它们都不训练模型。

待实现：`python -m cvpr_project ...`各子命令，定义在`IMPLEMENTATION_PLAN.md`。不要直接执行尚不存在的模块后把报错当成环境故障。

默认将data、checkpoints、artifacts放在repo下并由`.gitignore`排除；可通过后续CLI改到服务器数据盘。上传Git前检查是否含大文件、课程PDF或敏感环境输出。
