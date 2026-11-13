# Final evidence status: protocol 3

All experimental and delivery checks passed. The original CVPR protocol 2 source,
300 serving cases, 12 quality evaluations and report remain intact.

1. **Implementation and correctness:** real JPEG RPC and complete detections on
   two physical machines (H800 + RTX 5090); 12 CPU regression tests passed. Each
   worker passed 1024 eager/Graph comparisons covering all buckets, partial batches
   and concurrent slots, plus 64 additional checks after rebuilding the workers.
2. **Calibration and freeze:** calibration-only service curves, balanced local
   sweeps and a shared global wait sweep. Final timing freeze is
   `9b55b5e12c731b20682cc2d80fdbe66d35d3052dccbd2e90517d15c5809c42c0`.
   Reference capacity 339.733333 rps; H800 b2/wait2 ms, RTX 5090 b8/wait2 ms;
   shared global wait5 ms. No evaluation-based retuning.
3. **Real experiments and quality:** all 120 paired two-host cases and 48 matched
   single-host controls passed the producer-lag rule, with zero request errors.
   Rejections and misses remain in the all-offered denominator. Ten full
   4000-image AP evaluations passed; maximum within-worker change is
   0.000115779 pp. Genuine AP/stress measurements preserve their origin freeze
   `7aa056c109b2b94b8cdd818221d28b6f5fbadbda9a16871ba23a2afbcc1adc50`;
   the published unchanged-runtime certificate verifies their compatibility.
4. **Independent audit and resources:** 1,429,561 raw request records across all
   168 timing configurations were recomputed before export. The CPU-only evidence
   validator passed. New aggregate GPU reservation cost is 7.086649 hours including
   failed/idle time; combined with CVPR it is 19.559739 hours. All project workers
   and forwards were stopped. Storage accounting is in `results/storage.json`.
5. **Deliverables:** [6-page English report](../results/report.pdf),
   [Markdown report](../results/report.md), [12-slide presentation](../results/slides.pdf),
   [figures and evidence index](../results/README.md), full source and reproduction
   scripts. The CISC8003 repository contains the standalone project; the CVPR
   repository contains the same files under `distributed/`. Coursework submission
   remains with the author.

## Measured conclusion

Across the 24 common trace settings per policy, WRR has mean deadline goodput
112.524 requests/s; GLOBAL_GPU has 31.338 and GLOBAL_E2E has 13.000. GLOBAL_E2E is
88.4% below WRR and averages 18.337 requests/s below GLOBAL_GPU. It wins zero,
loses seven and ties 17 of their paired cases. This is a negative result for the
frozen bounded heuristic, not a general limit on global batching. Completed-batch
and dispatch diagnostics are observational; there is no causal fallback ablation.

The original cooperative-generator timing epoch and the independent-emitter run
that failed its predeclared lag rule are excluded, preserved, and described in
[attempts.json](../results/attempts.json). Valid runs were not selected by goodput.
Shared laboratory CPU noise, synthetic arrivals, asymmetric transport and three
seeds limit the conclusions. No production or large-fleet scalability is claimed.
