# v2论文式报告提纲

建议英文8–10页，非官方页数要求；真实结果后写全文。

1. **Introduction**：目标检测在线推理的时延和吞吐问题，host/GPU/queue的相互作用，三项RQ和有限范围贡献。
2. **Background and Related Work**：DETR检测、GPU graph/streams、dynamic batching/EDF/SLO serving；说明所有基础技术已有，不虚构首创。
3. **System Design**：图像字节→预处理→队列→buffer/copy/graph→后处理的流程；slot生命周期；batch bucket与deadline feasibility公式；fallback与成本。
4. **Evaluation**：COCO held-out subset AP、开放环合成到达、deadline/load的calibration定义、调优基线、E0/E1/G0/P0消融、F0/A0/D0对比、编译基线状态。
5. **Discussion and Conclusion**：真实瓶颈、负结果、质量等价性、固定pad/单模型/单卡限制；与真实视频流、生产服务、GH200平台的差异。

关键图表：检测框样例和AP表；load-latency曲线；load-goodput曲线；graph/pipeline消融；EDF/可行性消融；NVTX时间线；graph启动/显存/pinned memory开销。AP不能只统计按时请求；吞吐不能混用drain与测量窗口。

Abstract最后写，性能数值只来自真实run。附录包含环境锁、配置、trace/ID生成方式、原始结果索引和复现命令。准确记录AI辅助范围并遵循实际课程政策，不声明教师已批准。用户最终提交。
