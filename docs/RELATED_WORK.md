# v2 资料与已有技术边界

核对日期2026-10-08。仅使用可核实的官方/作者来源。M0需在服务器读与安装版本对应的API文档；网页stable版本不构成升级环境指令。

| 来源 | 已核实用途 |
|---|---|
| [DETR作者仓库](https://github.com/facebookresearch/detr)；[论文](https://arxiv.org/abs/2005.12872) | 目标检测算法与参考实现，使用已有权重而非重训 |
| [facebook/detr-resnet-50模型页](https://huggingface.co/facebook/detr-resnet-50) | COCO预训练、固定object queries、processor/模型用法与模型卡限制 |
| [Transformers DETR文档](https://huggingface.co/docs/transformers/model_doc/detr) | pixel_mask、resize/padding、boxes后处理，需版本核实 |
| [PyTorch CUDA语义](https://docs.pytorch.org/docs/stable/notes/cuda.html) | stream/event、pinned memory、CUDA Graph及静态地址/操作约束 |
| [NVIDIA Triton batcher文档](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html) | dynamic batching、preferred sizes及queue delay是已有服务功能 |
| [COCO官方](https://cocodataset.org/#download)；[COCO API](https://github.com/cocodataset/cocoapi) | 图像/标注来源、检测AP实现 |

当前设计没有复现整个Triton server；F0只是明确实现的固定等待batching对照，不能称作“Triton基线”。不引用未跑过的TensorRT/Triton数字来当实验结果。

## 不宣称哪些新颖性

CUDA Graph降低重复提交开销、pinned memory、异步copy、双缓冲、EDF、SLO-aware batching均有已有工作。课程研究目标是把这些机制放进视觉检测pipeline，给出明确的batch可行性策略和可靠评估。不得声称首个deadline-aware serving系统或OSDI级贡献。

## M0有限补读任务

1. 阅读DETR论文与processor说明，确认目标检测输出、pixel_mask和坐标含义。
2. 阅读当前PyTorch版本capture/stream规则，整理一页实现约束。
3. 检索并精读至少两篇与deadline/SLO-aware GPU inference batching直接相关的系统论文，使用原论文/作者代码验证；把本策略与已有策略逐项比较。
4. 记录选择DETR的原因（输出结构可控、官方权重可用），以及它不是最新或最快检测器的限制；可选第二模型只能在主实验完成后增加。

不要求先完成大规模文献综述才能开始M0。报告引用必须有核实过的title/authors/venue/year/URL；检索未完成的内容不得伪造。
