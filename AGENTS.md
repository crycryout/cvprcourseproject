# Codex instructions: CISC8005 visual inference systems project

Current protocol version is 2. Complete `PROJECT_PLAN.md` using `.agents/skills/run-cvpr-course-project/SKILL.md`. Read `docs/STATUS.md` and `docs/EXPERIMENT_PROTOCOL.md` before work. The old AP-ToMe/CIFAR-100 training project is superseded and remains only in Git history. Do not execute old artifacts or mix old metrics with v2.

## Goal

Implement and evaluate a pretrained DETR object detection serving runtime on a single H800: CUDA Graph buckets, fixed buffer pools, CPU–GPU pipelining, and deadline-aware batching. Do the implementation and real experiments, not another plan. Explain progress in Chinese; write the paper in English unless requested otherwise.

## Evidence and fairness

1. Keep model weights, precision, preprocessing, output decoding, and actual hardware common across policies. Verify COCO AP and request/output identity; this is a computer vision project, not just a GEMM benchmark.
2. Use the 1000 calibration images for tuning; freeze before the 4000 held-out subset. Both subsets come from official COCO validation, not an official test split. Do not train models or tune using held-out results.
3. Use open-loop scheduled arrivals. Include queue, decode/transform, H2D, forward, D2H, and CPU postprocessing in E2E latency. Count rejects, misses, failures, and unfinished requests in the SLO denominator.
4. Compare against tuned fixed-delay batching and an optimized backend when viable. Do not claim novelty for CUDA Graph/EDF/pipelining or infer PCIe bottlenecks without profile evidence.
5. Never fabricate run IDs, AP, latency, gains, or milestone completion. Preserve negative/failed runs. Synthetic tests validate logic only.

## Operations

- Inspect existing server PyTorch/CUDA and isolate dependencies. Do not replace drivers, global packages, power settings, or other jobs.
- Default one idle H800; no second-card work during timing because CPU/PCIe may be shared. Do not claim GH200/NVLink-C2C evidence.
- Plan 8–20 GPU-hours; stop new costly jobs at 30 cumulative GPU-hours or 30 GB project storage. Count failed runs and compile/capture work. Use <=10-minute pilot to revise cost.
- Track active jobs, checkpoint-equivalent artifacts (graph/calibration metadata, configs, partial run outputs), and actual cumulative cost across sessions. Resume without overwriting evidence.
- Limit full-forward capture debugging to 2 hours; record a valid partial capture fallback if needed. Limit compile-baseline compatibility work to 2 hours and disclose success/failure.
- Follow fixed-address/event lifetime rules. No reuse of input/output slots before relevant copy and CPU consumption finish. Never erase mask semantics to enable capture.
- Keep datasets, weights, course PDF, full traces, credentials, personal hostnames/paths out of public Git. Commit source, sanitized small metrics, hashes and reproduction instructions. Preserve upstream licenses when copying code.

## Delivery

Implement missing CLI before running its examples. Update STATUS at each evidence gate. Generate quality table, latency/goodput curves, ablations, profile timeline, real detection examples, and report. Report synthetic arrival traces as synthetic; do not call COCO image replay live video or production traffic. User reviews/submits coursework; do not contact instructors automatically.
