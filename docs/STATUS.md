# 状态 v2

更新日期：2026-10-08。用户认为AP-ToMe偏离其MLSys背景，因此当前项目已改为DETR目标检测serving；旧计划仅保留Git历史，不再执行。

| 阶段 | 状态 | 证据 |
|---|---|---|
| v2研究设计/交接 | 已编写并通过本地配置/工具检查 | 当前docs/config/skill |
| M0 模型/数据/质量 | 未开始 | 无H800环境或模型结果 |
| M1 baseline/profile | 未开始 | 无pilot |
| M2 graph/pipeline | 未开始 | 无runtime实现 |
| M3 调度/冻结 | 未开始 | 无service table/trace/freeze |
| M4 实验 | 未开始 | 无AP/serving实测 |
| M5 报告 | 未开始 | 仅有提纲 |

本次对话启动的H800任务：无。由本交付产生的H800 GPU-hours：0；服务器此前独立任务状态未知，交接时必须检查。

下一步：服务器运行preflight_v2，然后完成M0模型/processor/COCO split与质量校验；不要开始训练。

每次推进更新实际run、证据、active process、累计成本、blocking issue与下一步。旧v1本地工具测试不等于v2 runtime实测，旧artifact不得混入统计。

## 交付验证

新版matrix实际生成252主run与12消融，旧协议、closed-loop、仅completed分母、选择性过期丢弃和训练设置均被拒绝。Skill结构及Markdown链接校验通过。preflight默认无GPU计算，显式smoke在当前无CUDA环境准确报不可用。这些不是H800性能证据。
