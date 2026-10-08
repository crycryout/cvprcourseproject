# 课程要求与v2映射

来源：用户提供的`Introduction_2026.pdf`，12页，2026-10-08已核对文本与第8页视觉内容。原PDF不公开上传。SHA-256：`8d4d53f87b5d834ef435e1c339604da43266ca1828fabe3e35d9c2169a6817cc`。

| PDF要求（转述） | 本项目满足方式 |
|---|---|
| 第1页：CISC8005 Advanced Computer Vision and Pattern Recognition | 本仓库不是CISC8006 StreamingLLM项目 |
| 第3–4页：研究问题、文献、设计实现、数据分析与结论 | 用目标检测任务研究推理runtime并给受控实验 |
| 第6页：project report占40% | 主要交付为论文式报告和可复现证据 |
| 第8页：视觉/模式识别/图像或多媒体主题，输入或输出含图像/音频/视频 | 输入COCO图像，输出检测框/类别，并给检测框图和COCO AP |
| 第8页：可做survey或自己的方法，不要求SOTA | 实现graph/pipeline/deadline batching组合并分析收益边界 |

**范围判断：** 按讲义文字，视觉推理系统有合理匹配；没有教师已批准此具体选题的证据。保持视觉问题、真实输出和质量评估，不能将项目缩成纯GPU微基准。若教师另有“必须模型算法创新”的新要求，再调整范围；原PDF未写该限制。

第7页近5年/CCF A/B等针对paper reading report，不能自动套用到project。PDF未规定project页数、截止日期、组队人数、语言、AI使用政策。本计划建议英文8–10页，但这是安排，不是课程要求。后续按真实Moodle通知调整；不编造教师授权。
