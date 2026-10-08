# 相关工作与来源记录

核对日期：2026-10-08。本文件区分已经读到的官方信息和仍需补读的内容。不要把搜索摘要或他人结果写成自己的实验。

| 工作/资源 | 已核实的信息 | 本项目的使用方式 |
|---|---|---|
| [DeiT, ICML 2021](https://proceedings.mlr.press/v139/touvron21a.html)；[官方仓库](https://github.com/facebookresearch/deit) | 论文与官方模型代码可访问 | 使用非蒸馏DeiT-Small作为图像分类backbone；不是复现其完整ImageNet训练 |
| [ToMe, ICLR 2023](https://arxiv.org/abs/2210.09461)；[官方仓库](https://github.com/facebookresearch/ToMe) | 官方实现和merge/patch源码可读；仓库已归档 | 主压缩基线；迁移算法而不抄其速度数字 |
| [EViT, ICLR 2022](https://arxiv.org/abs/2202.07800)；[作者代码](https://github.com/youweiliang/evit) | 官方仓库说明其注意力相关token重组方法 | 明确重要token保留并非本项目新概念；全文作为M0补读 |
| [SAD-TM, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Xie_Saliency-Driven_Token_Merging_for_Vision_Transformers_CVPR_2026_paper.html) | CVF搜索索引给出标题、作者、摘要，提及saliency与延后合并；直接页面抓取403，尚未全文核验 | 最接近的近期工作之一；M0必须获取论文补查差异，不能宣称AP首创 |
| [CIFAR官方页面](https://cave.cs.toronto.edu/kriz/cifar.html) | CIFAR-100原始数据规格与官方下载入口 | 数据来源和fine labels定义 |
| [PyTorch SDPA文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) | 存在多种backend与适用限制 | 服务器上以安装版本文档和profiler核实，不能因API名推定Flash kernel |

## ToMe 的具体参考与版本

核实官方main commit为 `af95e4b1befa172dadccd8c81e223b10090f9579`。执行前按该revision取得源码，避免浮动依赖。

- [merge.py](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/merge.py)：二分匹配、token mass加权聚合及来源追踪。
- [timm adapter](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/tome/patch/timm.py)：attention残差后/MLP前合并；导出跨head平均K。
- [LICENSE](https://github.com/facebookresearch/ToMe/blob/af95e4b1befa172dadccd8c81e223b10090f9579/LICENSE)：CC-BY-NC-4.0。如果后续复制改写源码，保留完整license和notice，列出改动文件。

当前仓库只写研究计划和原创启动工具，未打包ToMe源码或预训练权重。后续适配需单独记录upstream来源。

## 创新性边界

AP-ToMe的课程级研究点是：固定每层token预算的分区保护规则，以及包含选择成本的H800实证评估。**“注意力衡量重要性”“保护高分token”“合并冗余token”“前几层不合并”都不能单独宣称首创。**

主张应为“we evaluate an attention-protected variant under controlled budgets”，而非“we introduce the first importance-aware merging method”。如果补读发现定义完全重复，直接将其标记为已知方法的复现/变体，研究问题和实验仍成立。

## M0 的有限文献任务

1. 精读ToMe的matching、weighted merge、实验协议和附录；形成300–500词方法笔记。
2. 对照EViT和SAD-TM，表格记录importance信号、是否训练、哪些token合并、部署开销。
3. 补充至少2篇2024–2026相关视觉token compression论文，只读与本方法区别直接相关的部分；找不到全文则标记unverified。
4. 更新报告引用及scope，保留链接/DOI/venue；不要因此扩大成全面顶会综述或增加多个实现基线。

阅读报告近5年要求不自动套用于project；如果后续课程另有规定再调整。
