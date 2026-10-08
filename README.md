# CISC8005 Course Project: Attention-Protected Token Merging

**项目：面向图像分类的注意力保护 Token 合并与 H800 实测评估。**

English title: **Attention-Protected Token Merging for Efficient Image Classification: An Accuracy–Latency Study on NVIDIA H800**.

以预训练 DeiT-Small 为图像分类模型，先完成 CIFAR-100 迁移学习及 ToMe 基线，再实现一个小范围的 attention-protected 扩展（AP-ToMe），研究保留重要图像 token 能否改善准确率，以及额外选择开销会不会抵消计算节省。

> 当前交付是完整研究设计、实验协议、Codex skill 与启动工具。训练、AP-ToMe 模型实现和 H800 实验尚未执行；仓库没有真实实验成绩。此项目不承诺新颖性、SOTA 或正向加速。

## 开始

1. 在 H800 服务器克隆仓库：`git clone https://github.com/crycryout/cvprcourseproject.git`。
2. 进入仓库：`cd cvprcourseproject`。
3. 运行只读检查：`python3 scripts/preflight.py --output artifacts/preflight.json`。
4. 启动 Codex，将 [H800_HANDOFF.md](docs/H800_HANDOFF.md) 中的启动提示词交给它。

Skill 位于 [`.agents/skills/run-cvpr-course-project/SKILL.md`](.agents/skills/run-cvpr-course-project/SKILL.md)。如果客户端未自动发现该目录，直接要求 Codex 读取此文件和根目录 `AGENTS.md`；不依赖全局安装。

## 阅读顺序

| 文件 | 内容 |
|---|---|
| [PROJECT_PLAN.md](PROJECT_PLAN.md) | 选题理由、方法定义、范围、日程、预算与失败退路 |
| [COURSE_REQUIREMENTS.md](docs/COURSE_REQUIREMENTS.md) | PDF 要求逐项对应与未规定事项 |
| [EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) | 数据划分、训练、基线、计时、统计及禁止泄漏的规则 |
| [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | 模块接口、实现顺序、验收标准与未来 CLI |
| [RELATED_WORK.md](docs/RELATED_WORK.md) | 已核实资料、相关工作重叠、需补读部分 |

后续入口：[执行状态](docs/STATUS.md)、[H800 启动与恢复](docs/H800_HANDOFF.md)、[报告提纲](docs/REPORT_OUTLINE.md)、[机器可读配置](configs/project.json)。

## 项目边界

- 课程为 **CISC8005 Advanced Computer Vision and Pattern Recognition**，不是 CISC8006 StreamingLLM 项目。
- 主任务：CIFAR-100 图像分类；单张 H800 足够设计所需规模。第二张卡仅用于独立训练 seed，正式计时应避免共享资源干扰。
- 最小版本：Dense + ToMe 的受控迁移评估、性能分析与论文式报告；完整版本增加 AP-ToMe 和随机保护消融。
- 正向效果是待验证假设；准确率下降、没有加速、保护无效都可以构成有证据的结论。
- 课程原始讲义、数据集、预训练权重、训练 checkpoint 不上传此公开仓库。保留来源、哈希及可再生成指令。

规划基准日期：2026-10-08。建议 4 周完成；这是项目安排，不是课程官方截止日期。
