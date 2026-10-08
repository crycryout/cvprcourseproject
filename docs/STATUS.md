# 项目状态

更新日期：2026-10-08。

## 已完成

- 已阅读CISC8005课程说明并核对project report要求。
- 已定义AP-ToMe研究问题、数学规则、对照实验、资源预算与交接流程。
- 已编写repository-scoped Codex skill和两个启动工具。

## 实验状态

| 阶段 | 状态 | 证据 |
|---|---|---|
| 设计与交接材料 | 已编写并通过本地工具/配置检查 | 本仓库docs/config/skill；下方验证记录 |
| M0 环境和数据 | 未开始 | 尚无H800预检或split |
| M1 Dense | 未开始 | 尚无训练或checkpoint |
| M2 ToMe | 未开始 | 尚无模型适配代码 |
| M3 AP | 未开始 | 尚无方法实现或validation结果 |
| M4 最终评测 | 未开始 | 尚无test或性能数据 |
| M5 报告 | 未开始 | 仅有提纲 |

累计H800 GPU-hours：0。当前没有由本次交付启动的服务器任务。

## 交付前验证记录

- Skill frontmatter/目录结构校验通过；所有Markdown本地链接存在；模板占位符已移除。
- `make_matrix.py`真实执行生成13个planned配置，逐层预算和最终token数符合设计。
- 验证器拒绝test调参、超额合并、额外special token三个错误配置。
- `preflight.py`在当前无torch/CUDA的环境真实运行并报告不可训练；`--require-cuda`正确退出2，没有冒称H800已验证。
- Skill交接检查能够区分设计与实验，能正确处理AP比ToMe快但比SDPA Dense慢的结论；检查时发现的启动工具缺失已补齐。

这些是交付材料的检查，**不构成模型正确性、训练成功或H800性能证据**。后续服务器必须重新收集自身环境和run状态。

**下一步：在H800服务器运行preflight，然后按M0实现数据/模型/运行记录。**

## 后续每次更新填写

填写日期、当前milestone、完成的task、证据路径、active run/是否仍在运行、最新可恢复checkpoint、累计GPU-hours、遇到的blocker及下一个可执行动作。没有真实证据不得改为completed。
