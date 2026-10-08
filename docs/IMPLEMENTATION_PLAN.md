# Codex 实现任务与验收

以下训练/评估模块是 **待实现接口**，不是已经存在的命令。当前可运行工具仅为`scripts/preflight.py`和`scripts/make_matrix.py`。按M0→M5完成；每步把真实证据写入`docs/STATUS.md`。

## M0：环境和数据

| Task | 实现 | 验收证据 |
|---|---|---|
| T00 | preflight，隔离环境，检查现有torch与timm | environment摘要；CUDA smoke通过；锁定依赖 |
| T01 | `src/cvpr_project/data.py`，prepare-data CLI | 45k/5k/10k，标签0–99，split无交集，class各450/50/100 |
| T02 | `src/cvpr_project/models.py`，加载DeiT-Small | 非distilled、100类head、预训练来源/哈希、1×100输出 |
| T03 | `src/cvpr_project/runs.py` | run_id不覆盖，状态、配置、成本、恢复路径可追溯 |

数据download失败时输出所需文件名、可信URL、目标路径与已知校验方法，允许`--data-root`本地输入；不尝试无来源镜像。预训练checkpoint也支持显式路径。

## M1：训练基础

实现`train.py`、`evaluate.py`、checkpoint恢复和CLI。先tiny-set overfit，再≤10分钟pilot，检查真实epoch成本并更新预算。运行seed17，保存best-validation checkpoint和完整curve。若全训练不可行，执行明确标注的最小版本，不能称为三个seed完成。

检查优化器累积、末尾partial batch、标签平滑的train/eval区别。test CLI在`frozen_protocol.json`不存在或不匹配时拒绝执行正式test。

## M2：ToMe 基线

实现 `src/cvpr_project/merging.py` 和 `attention.py`。固定官方ToMe revision：`af95e4b1befa172dadccd8c81e223b10090f9579`。

建议接口（可保持语义后调整名称）：

```python
class MergePlan:
    # tensors on input device; all dimensions documented
    src_index: Tensor
    dst_index: Tensor
    unmerged_index: Tensor

def build_plan(metric, r, protected_mask=None) -> MergePlan:
    # metric [B,N,Dh]; full-sequence parity A/B; CLS masked
    ...

def apply_plan(x, mass, plan):
    # x [B,N,C], mass [B,N,1]; weighted sums, no cross-batch scatter
    # returns x_out [B,N-r,C], mass_out [B,N-r,1]
    ...
```

patch方式不要依赖`module.__class__`的未经验证替换；在当前timm API上用清晰wrapper或subclass，保留原layer norm、residual、drop path和checkpoint参数。r=0必须走真正no-op，不能重排token。每次forward重新初始化mass，不能泄漏上个batch状态。

必要测试：官方小tensor对照、weighted mass守恒、CLS位置、重复destination、r边界、奇偶N、多样本隔离、r=0 logits、eval dropout关闭。只测试与结论相关的正确性，不堆纯实现镜像测试。

## M3：AP 与消融

实现 `protection.py`：从相同attention得到CLS row，在分区内执行固定k保护、屏蔽source与destination；注意不得只禁止source却让高分token被当作destination污染。计算kA/kB用shape及配置常量，不用`.item()`/`.cpu()`进行逐样本GPU→CPU控制。

Random使用稳定sample_id和layer的确定性随机分数。生成器的seed来自配置，不用Python进程随机hash。mask/topk/tie处理均在GPU完成。p=0应绕过score计算并退化为ToMe。

AP测试包括：protected向量与mass在merge阶段不变、p=0等价、足够合法edge、全batch固定shape、改变样本顺序不改变随机保护身份。通过后执行validation矩阵；保存所选p并冻结，不先偷看test。

## M4：后端、最终实验与统计

实现`benchmark.py`、`profile.py`、`analyze.py`。SDPA适配时显式设eval `dropout_p=0.0`；将mass bias正确广播。不要假定存在mask就仍会用Flash kernel。保存API路径和profiler所见kernel。

main score额外计算CLS row时复用Q/K，避免物化全部attention矩阵；这种实现优化不改变AP算法。两条轨都保留；explicit结果作为解释对照而不是隐藏dense的最佳实现。

实现命令示意（实现完成后才能运行）：

```bash
python -m cvpr_project prepare-data --config configs/project.json
python -m cvpr_project train --config configs/project.json --seed 17
python -m cvpr_project validate --config configs/project.json --matrix artifacts/validation_matrix.json
python -m cvpr_project freeze --config configs/project.json
python -m cvpr_project evaluate --split test --frozen artifacts/frozen_protocol.json
python -m cvpr_project benchmark --frozen artifacts/frozen_protocol.json
python -m cvpr_project analyze --runs artifacts/runs --output results
```

将其实现为可安装的`src/` package和真实`pyproject.toml`，添加`--help`、显式checkpoint/run选择和错误输出。以上命令不强制所有配置“一键全部跑”；支持单run和恢复，预算核算先于启动。

补齐训练seeds42/2026。冻结配置后完整test，跑性能矩阵，aggregate只接纳completed且一致的run。至少一次从manifest重建主表，确认图与表没有人工改数。

## M5：报告与收尾

按`docs/REPORT_OUTLINE.md`从结果写报告。创建`results/summary.csv`、figures、`results/reproduce.md`、`report/report.md`（后续若课程指定则导出PDF/LaTeX）。报告草稿不能把缺失结果写成事实。写完成的限制、失败案例、贡献边界和真实AI辅助记录，格式由课程政策决定。

交付验收：所有报告数字可追溯；readme命令真实可用；checkpoint/data均有来源和哈希；一次最小eval能够恢复；没有大权重或私密文件进入Git。

## 中断恢复

每阶段记录active run、PID/进程管理方式、checkpoint、日志、已消耗GPU-hours。恢复时先确认已有进程是否仍在执行，避免重复启动；对未完成run标记interrupted并在新run记录parent_run_id。无证据时状态为unknown，不能根据文件夹名推断成功。
