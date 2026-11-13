# Distributed DETR serving research project

The user authorized end-to-end implementation, real multi-host experiments, reports,
and publication to crycryout/CISC8003Project and crycryout/cvprcourseproject.
Use protocol version 3 for this extension. The completed CVPR v2 experiment is
historical evidence and must retain its source hashes and results.

1. Read docs/PROTOCOL.md and docs/STATUS.md before continuing experiments.
2. src/cvpr_project is an attributed, unmodified snapshot of CVPR commit 91ccf56.
   Put all extension behavior in src/batchserve. Do not weaken the v2 H800 guard.
3. Use real JPEG bytes over TCP (remote connections travel through SSH forwards),
   immutable DETR FP32 weights, and the original 1000/4000 COCO validation split.
4. Tune only with calibration images; freeze source/config before held-out runs.
   Count every offered request, including rejects, errors, and unfinished work.
5. Work in the established renjie account, inspect GPUs before use, never alter
   drivers/MIG/clocks or other users' work. Stop only this project's processes.

Cap new aggregate GPU reservation time at 16 hours (including failures and idle
worker uptime), keeping the previous 12.474-hour CVPR project below 30 hours total.
Keep aggregate project storage below 30 GB; reuse local data/weights/environment.
Never publish credentials, endpoints, hostnames, raw JPEGs, weights, or full traces.
Publish source, sanitized metrics, evidence hashes, figures, and English report.
Explain progress in Chinese. User handles coursework submission.
