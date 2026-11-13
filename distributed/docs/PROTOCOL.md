# Protocol 3: global versus local batching

Research question: how do batching location and heterogeneous end-to-end service
estimation affect deadline goodput in a two-machine object-detection service?
The project is an implementation and empirical extension of existing mechanisms,
not a claim to invent global batching, EDF, predictable serving, or deferred batching.

## Fixed visual and execution contract

Reuse the immutable CVPR commit 91ccf56 runtime. FP32, TF32 off, pretrained DETR
revision 70120ba84d68ca1211e007c4fb61d0cd5424be54, resize 480/640, pad 640x640,
and pixel_mask are common. COCO official validation is partitioned by the v2
PCG64 seed into 1000 calibration and 4000 evaluation images. No training.
All actual worker/bucket combinations undergo independent AP evaluation on all
4000 evaluation images; the quality tolerance is 0.1 AP percentage point against
the corresponding same-worker eager b1 reference. Validate raw Graph/eager
outputs on calibration images, partial batches and request/output identity first.

## Transport and timing

The driver caches compressed JPEG bytes in RAM. An independent CPU process emits
synthetic open-loop Poisson/burst arrival notifications over a nonblocking atomic
pipe, with producer monotonic timestamps; router JSON work cannot block emission.
Producer lag and ingress pickup lag are reported separately, and both remain in
scheduled-arrival E2E. A full notification pipe is charged to producer lag.
The driver sends actual bytes over persistent framed TCP. The
remote connection uses an authenticated SSH forward. This encryption/transport
cost is part of the measured platform. Workers bind loopback, never public ports.
Outputs contain complete decoded detections at threshold 0. No output/input tensor
cache. CPU decode/transform is repeated for every request with 8 CPU threads.
E2E starts at the scheduled arrival and ends when the driver has decoded the
matching detections. These two timestamps share one monotonic clock; worker
durations stay in their own clock domain. Do not sum overlapping stage durations
to reconstruct E2E. Bytes, round trips, preprocessing, queueing and GPU spans are
recorded. The logical client and router share one process; the emitter is separate.

## Policies and fairness

RR, capacity-weighted RR (WRR), and shortest predicted completion (SECT) route
individual arrivals and use worker-local FIFO ready-queue batching. Local batch
size and timeout are tuned on calibration with the same candidate budget per
worker. GLOBAL_GPU and GLOBAL_E2E batch globally in EDF order, then choose a
worker/bucket. They use the same algorithm, waiting cap and outstanding-batch
limit; only the service table changes (GPU forward span versus observed RPC
through result). Service estimates are past calibration measurements, not future
arrivals or completions. Router-side reservations update synchronously on dispatch
and remove completed work even while another batch remains outstanding. A regression
test checks that repeated early completions cannot create a phantom backlog.
No policy drops expired requests. All share a 512-request admitted pending limit,
reject-newest overflow, and a 30-second drain cap.
Global groups fix membership before worker preprocessing and wait for all members
to become ready. Local batches form from completed preprocessing jobs. The global
versus local comparison therefore includes this readiness-barrier difference;
it does not isolate a pure GPU batch-size effect. The two global estimators share
the same readiness semantics and differ only in their service table.
The global cap of two outstanding RPC batches can limit CPU admission at small
batch sizes; worker-local input queues may keep more preprocessing jobs available.
This is a comparison of complete bounded policies, not an isolated pooling effect.

Tune and freeze source hashes, service curves, SLOs, rates, local batch settings,
global wait cap, software/hardware identities and trace definitions before evaluation.
Use calibration-only pilot to fix the precise matrix, durations and resource estimate.
Preserve failed attempts. Do not retune from evaluation outcomes.
After measuring the saturated WRR reference, recheck every local candidate once
at 0.5/0.85/1.3 of the observed per-worker completion rate. Keep the common load
reference fixed after this recheck; do not iterate normalization to favor a policy.

## Metrics and planned evidence

Report on-time completions / measurement seconds (deadline goodput), SLO success
over ALL offered cohort requests, completed p50/p95/p99 alongside completion ratio,
reject/error/unfinished counts, actual batch occupancy, dispatch/CPU/GPU time, and
per-worker traffic. Use three paired trace seeds, randomized policy order, and
means plus ranges; requests are not independent experimental replicates.
Check loadgen lag p99 <= max(1ms, 5% of the smallest deadline); failing runs are
not silently presented as valid capacity measurements. Include unsaturated load
and overload. Main results are true two-host inference. Single-host controls and
optional application-injected delay are separately labeled. No production/video
traffic claims, no unmeasured commercial-backend superiority claims.

## Resource and publication boundary

New workers reserve at most 16 aggregate GPU-hours, including failed runs,
capture and idle time; historical CVPR cost is 12.474 hours. Count worker lifetime,
not just CUDA event time. Respect 30GB aggregate project storage. Stop only
project-owned workers/tunnels. Publish sanitized per-run metrics, small stage
statistics, hashes, code, report and figures; raw JPEGs, model weights, endpoints,
private hostnames, full request traces, and credentials remain ignored locally.
