# 最终报告提纲

建议英文论文式正文8–10页；这不是课程规定页数。若教师提供模板，以实际要求为准。报告正文在真实实验后生成，以下仅为写作提纲。

## 1. Abstract 与 Introduction

Abstract最后写，约150–200词：问题、方法、实验设置、真实关键结果、限制。不能预写“显著优于”。Introduction说明视觉token冗余、为什么算术节省不保证H800加速，列出RQ1–RQ3和限定范围的贡献。

## 2. Related Work

视觉Transformer、token pruning/merging、实际GPU推理性能三个方向。引用DeiT、ToMe、EViT以及全文核验的近期近邻工作。直接指出AP是课程变体，已有研究覆盖importance/saliency，不制造首创叙事。

## 3. Method

给图像→patch→attention→protection→merge→MLP→分类的结构图。说明score公式、分区保护数量、mass加权、固定token长度、p=0退化以及CLS保护。给复杂度与额外memory/matching开销；不能只列attention的O(N²)。

## 4. Experimental Setup

写数据划分与hash、ImageNet预训练迁移设置、32→224放大、recipe、seeds、selection/test隔离、所有方法与backend、GPU及计时范围、统计方法。区分复现算法与复现原论文数字。

## 5. Results 与 Analysis

| 小节 | 所需真实证据 | 回答 |
|---|---|---|
| Quality under matched budgets | 各seed准确率、paired差值、CI | AP保护是否有效 |
| Practical performance | batch1/16/64、两轨、GPU/host/pipeline timing | H800是否真加速 |
| Ablation | p=0、selected p、Random | 是attention还是保护数量在起作用 |
| Profiling | scoring/matching/scatter/attention/MLP成本 | 解释有无收益 |
| Qualitative/failure cases | token映射、固定抽样案例 | 哪里成功、哪里丢失信息 |

## 6. Limitations 与 Conclusion

限制包括单主数据/模型、低分辨率放大、预训练潜在重叠、单类硬件、有限seed、attention不等于因果解释。结论只回答测到的问题。如果无加速，应清楚写出未达部署目标和成本归因。

附录给运行命令、环境锁、配置表、原始run索引和失败记录。记录AI辅助的实际范围；如课程有披露要求据实填写，不能声明教师已批准。不要自动向教师发送或提交。
