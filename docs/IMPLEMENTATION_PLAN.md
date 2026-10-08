# 实现计划 v2

当前仅`preflight.py`与`make_matrix.py`已实现。下列`cvpr_project`模块/CLI是待实现契约，禁止把文件名当作已有程序。

## M0：检测正确性

创建`pyproject.toml`与src package。模块：`data.py`（COCO IDs/bytes/annotations/split）、`model.py`（DETR/processor静态adapter）、`quality.py`（COCOeval、label映射）、`runs.py`（manifest/成本/冻结）。

验证模型revision和权重，不做训练。实现标准processor路径与静态pad路径的小样本对照，检查pixel_mask，避免不同pad形状改变检测语义。FP32→BF16选择仅在calibration完成。输出至少横图、竖图、多人小目标三类框图。坐标归一化和category_id通过官方行为对照，不凭直觉。

数据下载不可用时支持本地`--data-root`/`--weights`，列出确切所需文件；不绕过下载许可，不用随机权重。COCO图像与权重不进Git。

## M1：Eager与真实计时

实现`executor.py`、`preprocess.py`、`benchmark.py`、`profile.py`。E0/E1跑1000请求pilot并记录raw events、质量和内存。先判断瓶颈，不预设PCIe一定重要。

preflight默认只收集环境，不运行GPU算子。检查空闲卡后显式用`--cuda-smoke --device <核实设备号> --require-cuda`执行CUDA检查；通过仍不自动证明runtime可用。CPU worker数量先calibration并固定；loadgen不与preprocess共用会阻塞的调用线程。计时采用monotonic_ns；`request_id`贯穿所有阶段。

## M2：Graph与pipeline

实现`graph_pool.py`与`buffer_pool.py`，候选bucket{1,2,4,8}、2 slots、1 compute stream。初始化warmup/capture与在线replay分离，提供显式fallback状态与coverage。

slot状态机：FREE→FILLING→H2D→READY_GPU→COMPUTING→D2H→CPU_CONSUMING→FREE。events建立happens-before。host buffer在H2D完成前不可写；device input在forward完成前不可写；output在D2H和CPU消费完成前不可覆盖。不能用一份graph output给多个未完成请求。

每slot capture到其固定地址，不可假设调用replay时传新tensor会替换地址。graph私有pool并发复用需证明安全；第一版各slot独立capture，不开多compute streams。batch dummy使用合法tensor/mask，valid_count控制结果拆包。

测试：1000交替不同图片请求与eager对照；乱序CPU ready；所有batch包括不足batch；故意延迟D2H/CPU消费验证slot不会早复用；重复ID/遗漏检测；多轮内存不增长；no-grad/eval一致。仅用与正确性结论相关的测试。

capture失败最多2小时定位，优先包装静态tensor forward、移出Python控制/CPU同步。允许静态子图但必须记录capture区域；不删mask/改变输入来“成功”。profile证明overlap；没有重叠则报告实际串行路径。

## M3：负载与调度

实现`trace.py`（开放环到达/期限/样本）、`scheduler.py`（FIFO timer、EDF timer、deadline feasibility）、`serve.py`（本机harness）。没有真实网络传输，不叫完整在线服务部署。

建议对象：

```python
# 以下是待实现接口描述，不是可运行代码
Request(id, image_id, scheduled_arrival_ns, deadline_ns, ready_ns)
Decision(request_ids, bucket_size, slot_id, dispatch_or_wait, wakeup_ns)
ServiceTable(bucket_to_dispatch_to_result_p95_ns)
```

调度函数不读未来completion/未来arrival，最多一执行批、一locked预取批。任何wait受最老ready的绝对wait截止时间限制；没有可行候选也要处理，不能丢弃迟到图像。全局max_pending=512，overflow统一reject newest。

必要逻辑测试：相同trace确定性、earliest deadline顺序、无候选立即处理、已等待请求不重置timer、dummy不输出、locked批不被新到达覆盖、overflow进失败分母、drain后unfinished进失败分母。用synthetic fixture测逻辑不计为GPU实测。

## M4：实测矩阵

先calibration freeze，再跑脚本计划中的252个主run（7 policies×3 traces×4 loads×3 seeds）、12个A0消融；C0若可用按完整36个trace配置追加。每run60秒测量，另有预热/drain，按pilot预算分组执行并支持resume。

正式命令的预期接口（实现与`--help`通过后使用）：

```bash
python -m cvpr_project prepare-data --config configs/project.json
python -m cvpr_project verify-model --config configs/project.json
python -m cvpr_project calibrate --config configs/project.json
python -m cvpr_project freeze --config configs/project.json
python -m cvpr_project evaluate-quality --frozen artifacts/frozen_v2.json
python -m cvpr_project run-matrix --matrix artifacts/serving_matrix_v2.json --frozen artifacts/frozen_v2.json
python -m cvpr_project analyze --runs artifacts/runs_v2 --output results
```

冻结前不得执行held-out主结果。控制run walltime/cumulative budget，所有失败也算成本。C0最多2小时兼容尝试计入总预算；不能因不如eager就隐藏。比较SLO goodput时将所有offered计入分母。

## M5：分析、报告、恢复

从真实events产生summary.csv、图和英文report，按REPORT_OUTLINE写。记录source/version/quality限制。状态每阶段落盘；恢复先查active processes/manifest，不重复启动运行中的矩阵。

最小版可只完成单模型质量及E0/E1/G0/P0分析，明确F0/D0尚未实现。完整结果要求264个主/消融run，或逐项解释实际缩减。用户最终审阅提交，不自动发教师。
