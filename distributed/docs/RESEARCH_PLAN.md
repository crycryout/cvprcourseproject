# Research plan fixed before evaluation

Chinese title: 面向异构 GPU 集群的目标检测服务：全局组批与节点内组批的端到端比较。

English title: Where Should We Batch? Coordinating Request Routing and Batching
for Heterogeneous Object Detection Serving.

## Scope

Replicate the frozen DETR detector on one H800 and one RTX 5090 physical machine.
Study routing/local batching versus global batching, and GPU-only versus E2E
service-time estimation. Reuse the existing graph executor without modifying its
v2 source. New implementation covers actual JPEG RPC, per-worker preprocessing,
causal reservations, calibration, open-loop arrivals and all-offered accounting.

No claim of a first batch-aware scheduler is intended. Nexus (SOSP 2019), Clockwork
(OSDI 2020), and Symphony (arXiv:2308.07470) provide direct prior mechanisms.
This is a controlled empirical research extension on a real heterogeneous visual
workload. The official PhD-course rubric and deadline were not supplied. An
unrelated locally found undergraduate course document was excluded from requirements.

## Evidence gates

1. Protocol tests and complete JPEG -> matching detections pilot on both hosts.
2. >=1024 same-worker eager/Graph numerical checks, all buckets, partial and
   concurrent slots. Equal model, preprocessing, FP32 and TF32 policy.
3. Calibration-only service profiles, equal local sweeps, shared global wait sweep,
   measured saturated capacity, immutable source/config freeze.
4. Ten independent complete 4000-image AP evaluations, 120 paired two-host cases,
   and 48 single-host controls. Preserve failed/invalidated attempts.
5. Request-level audit, figures, English report/presentation, reproduction check,
   and publication to the two repositories authorized by the user.

## Interpretation fixed before results

Show all five methods, all loads and all seeds. Report goodput, SLO success,
completed latency plus completion ratio, actual batch occupancy and stage evidence.
Do not promise a positive result. A global policy that loses is retained, explained
as a bounded implementation result and never retuned using evaluation outcomes.
The GPU-only/E2E comparison is a service-table ablation within the same scheduler;
it is not a comparison against full implementations of cited prior systems.

The single-node controls use the same total offered traces and locally calibrated
settings. They evaluate scale-out on this platform; they do not imply elastic
provisioning or router fault tolerance has been implemented.
