---
name: run-cvpr-course-project
description: Implement and resume the CISC8005 H800 computer vision serving project using pretrained DETR, CUDA Graph buckets, CPU-GPU pipelining, deadline-aware batching, COCO quality checks, and reproducible latency/goodput evaluation. Use for this repository's runtime implementation, profiling, scheduling, experiments, debugging, report writing, or recovery. The previous AP-ToMe training plan is superseded.
---

# Run the visual inference systems course project

Complete the current v2 project, not another plan. Use the user's existing H800 resources and systems background. Do not train a vision model or implement token merging. This is a repository-scoped skill.

## Establish current state

1. Locate the repository with `AGENTS.md`, `PROJECT_PLAN.md`, and `configs/project.json`. Read them plus `docs/STATUS.md`; check active jobs, local modifications, and evidence before resuming.
2. Require protocol_version=2 on artifacts. Do not reuse a v1 AP-ToMe matrix, checkpoint, or claimed metric as current evidence.
3. Read `docs/EXPERIMENT_PROTOCOL.md` before experiments and `docs/IMPLEMENTATION_PLAN.md` for the current milestone. Read `references/evidence-gates.md` before advancing.
4. Run the existing `scripts/preflight.py` and `scripts/make_matrix.py` when necessary. These probe environment and emit plans only. Implement future `cvpr_project` CLI before executing its examples.
5. State the current gate, real evidence, uncertainty, and next executable step in concise Chinese, then do the work.

## Execute M0 through M5

- **M0:** Verify pretrained DETR/processor revision and weights; implement COCO's 1000 calibration / 4000 held-out split; validate resizing, fixed padding, pixel_mask, label mapping, and original-coordinate box decoding. Establish a same-precision quality reference. Read `docs/RELATED_WORK.md` for prior-art boundaries.
- **M1:** Implement serial eager E0/E1 and complete <=10-minute real-image pilot. Measure the entire pipeline, identify actual bottlenecks, and revise compute estimates. Do not assume PCIe or launch dominates.
- **M2:** Implement batch graph buckets and fixed buffer pools with safe slot/event lifetimes; validate G0 and P0. Capture static forward or a justified partial subgraph. Verify outputs across repeated requests, partial batches, mixed image shapes, and delayed CPU consumption.
- **M3:** Implement open-loop trace replay, tuned fixed-delay batching, EDF control, and the explicit deadline-feasibility heuristic. Calibrate all methods fairly; freeze service tables, common buckets, SLOs, rates, precision and configs before held-out evaluation.
- **M4:** Execute 252 main and 12 EDF-control planned runs within budget; attempt a bounded compile baseline. Collect independent offline COCO quality plus online timing/goodput, rejects, misses and unfinished requests. Generate results from raw evidence.
- **M5:** Write the English report with `docs/REPORT_OUTLINE.md`; produce reproducibility commands, figures and limitations for the user's review. Use `docs/H800_HANDOFF.md` for startup/recovery instructions.

## Enforce execution correctness

Keep weights, input processing, precision and postprocessing common. Never remove pixel_mask or change image quality to make graph capture succeed. Graph replay uses captured addresses; new tensors do not automatically replace them.

Give each bucket/slot correct static buffers and a valid graph instance. Do not overwrite host input before H2D completes, device input before consumption, or output before D2H and CPU consumption complete. Use explicit CUDA event dependencies; keep one compute stream initially. Dummy lanes must be valid tensors/masks and never appear as real outputs. Trace actual overlap rather than assuming async means overlap.

Do not add host-wide synchronization for every stage in the optimized pipeline; use events and record the residual necessary waits. Debug synchronizations must not silently remain in performance runs. Include legitimate pack/copy/queue/postprocess costs and graph pool memory.

## Enforce experimental meaning

Use real COCO image bytes. Label periodic/Poisson/burst arrivals as synthetic and do not describe this as real video or production traffic. Start E2E timing at scheduled arrival, finish when CPU detection results are available. Keep GPU-event microbenchmarks separate.

Count every offered request in the SLO denominator, including overflow rejection, failure and unfinished drain. Report completed-request percentiles alongside completion ratio. Never drop expired images only in the proposed policy. Prevent coordinated omission and detect load-generator lag.

Calibrate R0/F0 as well as D0; compare F0/A0/D0 to separate EDF from service-time feasibility. Try `torch.compile` for at most two hours, report whether it includes automatic graphs, and keep it if valid even if it beats the proposal. Never claim superiority over unmeasured TensorRT/Triton systems.

Evaluate offline AP over all 4000 held-out images, not just successful online requests. Preserve source/revision hashes and label/output mappings. Do not tune using held-out outcomes or invent literature/accuracy/speedup. Existing Graph, pipelining and EDF are not novel claims.

## Control resources and resume

Inspect server environment and idle GPUs. Use isolated dependencies, no driver/global package replacement, no GPU power changes, no killing unrelated jobs. Use one H800 and avoid second-card CPU/PCIe interference during timing. Do not claim GH200/C2C results.

Plan 8–20 GPU-hours, cap automated jobs at 30 GPU-hours and 30 GB project storage. Count failed jobs and compatibility attempts across sessions. Limit capture debugging to two hours before an evidence-backed partial-capture fallback. Stop new costly work at caps and deliver measured scope with omissions explicit.

Resume from manifests and actual active processes, not directory names. Preserve interrupted/failed runs. If download is blocked, request exact local files with verified sources; never substitute random weights. Proceed autonomously with ordinary authorized implementation choices.

## Complete the handoff

Update STATUS after each verified gate. Publish only source and sanitized small evidence, not model weights, datasets, instructor PDF, secrets, private paths, or full traces. Preserve upstream licenses. Do not submit coursework or contact the instructor. Report results honestly even if no optimization improves the strongest baseline.
